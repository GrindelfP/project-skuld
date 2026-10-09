"""
trop_sweep.py — multi-GPU resumable grid sweep for TROP v2.

Sweeps:
  - n_pieces: [8, 16, 32, 64]
  - lr: [1e-3, 1e-2]
  - tau_start: [1.0]
  - tau_end: [0.01]
  - epochs: [16000]

Total: 4 * 2 = 8 configs x 5 seeds = 40 runs.

Multi-GPU: each GPU processes a subset of configs. Resumable via CSV
checkpointing — if a config already has results for a seed, it is skipped.

GPU index is auto-detected from SLURM_ARRAY_TASK_ID (SLURM job arrays) or
LOCAL_RANK (torchrun/srun). No manual --gpu flag needed in those cases.

Usage (on CUDA):
    python trop_sweep.py --num-gpus 4              # auto-detect from SLURM
    python trop_sweep.py --gpu 0 --num-gpus 4      # manual override
    python trop_sweep.py --gpu 1 --num-gpus 4
    ...
"""
import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch

from physics import integrand_transformed, reference_scipy, PARAM_SETS


##############################################################################
#  TROP v2 — same as flat/22 but parameterized for sweeping
##############################################################################

class TropicalAntiderivativeV2:
    """Tropical max-plus antiderivative with log-sum-exp smoothing."""

    def __init__(self, n_params: int = 4, n_int_vars: int = 3, n_pieces: int = 16,
                 tau_start: float = 1.0, tau_end: float = 0.01):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.n_pieces = n_pieces
        self.tau_start = tau_start
        self.tau_end = tau_end

        self.param_layer = torch.nn.Linear(n_params, n_pieces * (n_int_vars + 1))

        with torch.no_grad():
            bound = 1.0 / math.sqrt(n_params)
            self.param_layer.weight.uniform_(-bound, bound)
            self.param_layer.bias.uniform_(-bound, bound)

        self.corners = list(itertools.product([0.0, 1.0], repeat=n_int_vars))
        self.signs = [(-1.0) ** sum(c) for c in self.corners]

    def get_weights(self, theta: torch.Tensor) -> tuple:
        out = self.param_layer(theta)
        out = out.view(-1, self.n_pieces, self.n_int_vars + 1)
        w = out[:, :, :self.n_int_vars]
        b = out[:, :, self.n_int_vars]
        return w, b

    def forward(self, u: torch.Tensor, theta: torch.Tensor, tau: float) -> torch.Tensor:
        w, b = self.get_weights(theta)
        u_expanded = u.unsqueeze(1)
        linear = (w * u_expanded).sum(dim=2) + b

        max_linear = linear.max(dim=1, keepdim=True).values
        exp_shifted = torch.exp((linear - max_linear) / tau)
        lse = max_linear + tau * torch.log(exp_shifted.sum(dim=1, keepdim=True))

        return lse.squeeze(1)

    def corner_sum(self, theta: torch.Tensor, tau: float) -> torch.Tensor:
        corner_sum = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
        for corner, sign in zip(self.corners, self.signs):
            u_corner = torch.tensor(corner, device=theta.device, dtype=theta.dtype)
            u_corner = u_corner.unsqueeze(0).expand(theta.shape[0], -1)
            n_corner = self.forward(u_corner, theta, tau)
            corner_sum += sign * n_corner
        return corner_sum

    def train(self, param_sets: list, targets: dict, n_epochs: int,
              lr: float, device: torch.device, verbose_every: int = 0) -> list:
        self.param_layer.to(device)
        optimizer = torch.optim.Adam(self.param_layer.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        history = []
        for epoch in range(n_epochs):
            tau = self.tau_start * (self.tau_end / self.tau_start) ** (epoch / n_epochs)

            total_loss = 0.0
            n_batches = 0

            for params_tuple in param_sets:
                theta = torch.tensor(params_tuple, dtype=torch.float32,
                                      device=device).unsqueeze(0)
                cs = self.corner_sum(theta, tau)
                target = torch.tensor(targets[params_tuple], dtype=torch.float32,
                                       device=device)
                loss = ((cs - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / n_batches
            history.append(avg_loss)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"    Epoch {epoch + 1:>6d}/{n_epochs}  "
                      f"tau={tau:.4f}  loss = {avg_loss:.6e}")

        return history

    def integrate(self, params_tuple: tuple, device: torch.device,
                  tau: float = 0.01) -> float:
        self.param_layer.to(device)
        theta = torch.tensor(params_tuple, dtype=torch.float32,
                              device=device).unsqueeze(0)
        with torch.no_grad():
            result = self.corner_sum(theta, tau).item()
        return result

    @property
    def n_weights(self) -> int:
        return sum(p.numel() for p in self.param_layer.parameters())


##############################################################################
#  Grid sweep
##############################################################################

def digits_of(abs_err: float) -> int:
    return max(0, -math.floor(math.log10(abs_err + 1e-30)))


def _detect_gpu_index() -> int:
    """Auto-detect GPU index from SLURM or torchrun environment."""
    # SLURM job array: each task gets a unique ID
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    # torchrun / srun: LOCAL_RANK is set per process
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    # SLURM without array (single job, multiple GPUs via SLURM_NTASKS)
    if "SLURM_NODEID" in os.environ and "SLURM_NTASKS" in os.environ:
        # Fallback: use 0 if we can't determine rank
        return 0
    return 0


def main():
    parser = argparse.ArgumentParser(description="TROP v2 grid sweep")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (0-based). Auto-detected from SLURM if omitted.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="total number of GPUs")
    parser.add_argument("--epochs", type=int, default=16000)
    parser.add_argument("--seeds", type=str, default="42,137,2024,999,7",
                        help="comma-separated list of seeds")
    args = parser.parse_args()

    # Auto-detect GPU index if not provided
    if args.gpu is None:
        args.gpu = _detect_gpu_index()

    # Set the specific GPU device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    seeds = [int(s) for s in args.seeds.split(",")]

    # Grid definition — tuned around champion v5 (12 pieces, lr=1e-3)
    grid = {
        "n_pieces": [10, 12, 14, 16],
        "lr": [5e-4, 1e-3, 2e-3],
        "tau_start": [1.0],
        "tau_end": [0.01],
    }

    # Generate all configs
    keys = list(grid.keys())
    all_configs = list(itertools.product(*[grid[k] for k in keys]))

    # Distribute configs across GPUs
    my_configs = [c for i, c in enumerate(all_configs) if i % args.num_gpus == args.gpu]

    print(f"{'=' * 78}")
    print(f"  TROP v2 GRID SWEEP — GPU {args.gpu}/{args.num_gpus}")
    print(f"  device={device}  epochs={args.epochs}  seeds={seeds}")
    print(f"  total configs: {len(all_configs)}  my configs: {len(my_configs)}")
    print(f"{'=' * 78}\n")

    # Compute reference integrals
    print("Computing scipy reference integrals ...")
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    targets = {p: refs[p][0] for p in PARAM_SETS}
    print("Done.\n")

    # Results directory
    results_dir = get_mirror_path(__file__, "results")
    csv_file = os.path.join(results_dir, f"results_gpu{args.gpu}.csv")

    # Load existing results for resumability
    existing = set()
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as fh:
            header = fh.readline()
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    config_key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"
                    existing.add(config_key)
        print(f"  Found {len(existing)} existing results, will skip.\n")

    # CSV header
    header = ("n_pieces,lr,tau_start,tau_end,seed,"
              "I1,I2,I3,I4,I5,I6,I7,I8,"
              "min_dig,mean_dig,max_dig,mean_rel_err,max_rel_err,"
              "final_loss,min_loss,elapsed_s,n_weights")

    # Append mode
    write_header = not os.path.exists(csv_file)
    csv_fh = open(csv_file, "a", encoding="utf-8")
    if write_header:
        csv_fh.write(header + "\n")

    total = len(my_configs) * len(seeds)
    done = 0

    for config in my_configs:
        n_pieces, lr, tau_start, tau_end = config

        for seed in seeds:
            config_key = f"{n_pieces}_{lr}_{tau_start}_{tau_end}_{seed}"
            if config_key in existing:
                done += 1
                continue

            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = time.time()
            trop = TropicalAntiderivativeV2(
                n_params=4, n_int_vars=3, n_pieces=n_pieces,
                tau_start=tau_start, tau_end=tau_end)

            history = trop.train(
                param_sets=PARAM_SETS, targets=targets,
                n_epochs=args.epochs, lr=lr, device=device,
                verbose_every=max(1, args.epochs // 20))

            # Evaluate
            digits_list = []
            rel_errs_list = []
            per_integral = []
            for params in PARAM_SETS:
                nni_val = trop.integrate(params, device=device, tau=tau_end)
                ref_val, _ = refs[params]
                abs_err = abs(nni_val - ref_val)
                rel_err = abs_err / (abs(ref_val) + 1e-30)
                d = digits_of(abs_err)
                digits_list.append(d)
                rel_errs_list.append(rel_err)
                per_integral.append(d)

            elapsed = time.time() - t0

            row = (
                f"{n_pieces},{lr},{tau_start},{tau_end},{seed},"
                f"{per_integral[0]},{per_integral[1]},{per_integral[2]},"
                f"{per_integral[3]},{per_integral[4]},{per_integral[5]},"
                f"{per_integral[6]},{per_integral[7]},"
                f"{min(digits_list)},{sum(digits_list)/len(digits_list):.2f},"
                f"{max(digits_list)},{sum(rel_errs_list)/len(rel_errs_list):.6e},"
                f"{max(rel_errs_list):.6e},"
                f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},{trop.n_weights}"
            )
            csv_fh.write(row + "\n")
            csv_fh.flush()

            done += 1
            print(f"  [{done}/{total}] pieces={n_pieces} lr={lr} seed={seed}  "
                  f"mean_dig={sum(digits_list)/len(digits_list):.2f}  "
                  f"max_dig={max(digits_list)}  loss={history[-1]:.2e}  "
                  f"({elapsed:.1f}s)")

    csv_fh.close()
    print(f"\n  GPU {args.gpu} done. Results -> {csv_file}\n")


if __name__ == "__main__":
    main()
