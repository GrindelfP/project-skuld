"""
trop_debug.py — verbose diagnostic for the TROP grid sweep hang.

Runs a single config with per-epoch timing, GPU memory tracking, and
RSS monitoring. Designed to answer: "Is training progressing or hung?"

Usage (remote, CUDA):
    python trop_debug.py --gpu 0 --pieces 8 --lr 1e-3 --epochs 100

Usage (quick smoke test, 20 epochs):
    python trop_debug.py --epochs 20
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
#  Memory / timing helpers
##############################################################################

def gpu_mem_allocated_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def gpu_mem_reserved_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2
    return 0.0


def gpu_mem_max_allocated_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def cuda_info():
    if not torch.cuda.is_available():
        return "CUDA not available"
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append(f"GPU {i}: {props.name}, "
                    f"{props.total_memory / 1024**3:.1f} GB, "
                    f"cc {props.major}.{props.minor}")
    return "\n  ".join(info)


##############################################################################
#  TROP v2 — same as sweep but with verbose training
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
              lr: float, device: torch.device, verbose_every: int = 1) -> list:
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
                      f"tau={tau:.6f}  loss = {avg_loss:.6e}  "
                      f"gpu_alloc={gpu_mem_allocated_mb():.1f}MB  "
                      f"gpu_reserved={gpu_mem_reserved_mb():.1f}MB  "
                      f"gpu_max={gpu_mem_max_allocated_mb():.1f}MB  "
                      f"rss={rss_mb():.1f}MB")

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
#  Diagnostic
##############################################################################

def digits_of(abs_err: float) -> int:
    return max(0, -math.floor(math.log10(abs_err + 1e-30)))


def main():
    parser = argparse.ArgumentParser(description="TROP diagnostic")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pieces", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.01)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"{'=' * 78}")
    print(f"  TROP DIAGNOSTIC")
    print(f"{'=' * 78}")
    print(f"  device: {device}")
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  {cuda_info()}")
    print(f"  config: n_pieces={args.pieces} lr={args.lr} "
          f"epochs={args.epochs} seed={args.seed}")
    print(f"  tau: {args.tau_start} -> {args.tau_end}")
    print(f"  param_sets: {len(PARAM_SETS)}")
    print(f"  corners: {2 ** 3} = 8")
    print(f"  forward passes per epoch: {len(PARAM_SETS) * 8}")
    print(f"  total forward passes: {args.epochs * len(PARAM_SETS) * 8}")
    print(f"{'=' * 78}\n")

    # Step 0: Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[SEED] seed={args.seed}\n")

    # Step 1: Reference integrals
    print("[STEP 1] Computing scipy reference integrals ...")
    t_ref = time.time()
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    targets = {p: refs[p][0] for p in PARAM_SETS}
    t_ref_elapsed = time.time() - t_ref
    print(f"  Done in {t_ref_elapsed:.2f}s")
    print(f"  Reference values:")
    for p in PARAM_SETS:
        print(f"    {p} -> {refs[p][0]:.10e}")
    print()

    # Step 2: Build model
    print("[STEP 2] Building model ...")
    t_model = time.time()
    trop = TropicalAntiderivativeV2(
        n_params=4, n_int_vars=3, n_pieces=args.pieces,
        tau_start=args.tau_start, tau_end=args.tau_end)
    t_model_elapsed = time.time() - t_model
    print(f"  n_weights: {trop.n_weights}")
    print(f"  Built in {t_model_elapsed:.4f}s")
    print(f"  gpu_alloc={gpu_mem_allocated_mb():.1f}MB  "
          f"gpu_reserved={gpu_mem_reserved_mb():.1f}MB  "
          f"rss={rss_mb():.1f}MB\n")

    # Step 3: Train
    verbose_every = max(1, args.epochs // 20)  # ~20 progress lines
    print(f"[STEP 3] Training ({args.epochs} epochs, verbose_every={verbose_every}) ...")
    print(f"  {'Epoch':>8s}  {'Tau':>8s}  {'Loss':>12s}  "
          f"{'GPU alloc':>10s}  {'GPU reserved':>12s}  {'GPU max':>10s}  {'RSS':>10s}")
    print(f"  {'':>8s}  {'':>8s}  {'':>12s}  "
          f"{'(MB)':>10s}  {'(MB)':>12s}  {'(MB)':>10s}  {'(MB)':>10s}")

    t_train = time.time()
    history = trop.train(
        param_sets=PARAM_SETS, targets=targets,
        n_epochs=args.epochs, lr=args.lr, device=device,
        verbose_every=verbose_every)
    t_train_elapsed = time.time() - t_train

    print(f"\n  Training done in {t_train_elapsed:.2f}s "
          f"({t_train_elapsed / args.epochs:.4f}s/epoch)")
    print(f"  Final loss: {history[-1]:.6e}")
    print(f"  Min loss: {min(history):.6e}")
    print(f"  gpu_alloc={gpu_mem_allocated_mb():.1f}MB  "
          f"gpu_reserved={gpu_reserved_mb():.1f}MB  "
          f"gpu_max={gpu_mem_max_allocated_mb():.1f}MB  "
          f"rss={rss_mb():.1f}MB\n")

    # Step 4: Evaluate
    print("[STEP 4] Evaluating ...")
    t_eval = time.time()
    digits_list = []
    rel_errs_list = []
    for params in PARAM_SETS:
        nni_val = trop.integrate(params, device=device, tau=args.tau_end)
        ref_val, _ = refs[params]
        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)
        d = digits_of(abs_err)
        digits_list.append(d)
        rel_errs_list.append(rel_err)
        print(f"  I_{PARAM_SETS.index(params) + 1}: nni={nni_val:.10e}  "
              f"ref={ref_val:.10e}  abs_err={abs_err:.2e}  "
              f"rel_err={rel_err:.2e}  digits={d}")

    t_eval_elapsed = time.time() - t_eval

    mean_dig = sum(digits_list) / len(digits_list)
    print(f"\n  min_dig={min(digits_list)}  mean_dig={mean_dig:.2f}  "
          f"max_dig={max(digits_list)}")
    print(f"  mean_rel_err={sum(rel_errs_list) / len(rel_errs_list):.6e}  "
          f"max_rel_err={max(rel_errs_list):.6e}")
    print(f"  Eval done in {t_eval_elapsed:.2f}s\n")

    # Step 5: Summary
    total_elapsed = t_ref_elapsed + t_model_elapsed + t_train_elapsed + t_eval_elapsed
    print(f"{'=' * 78}")
    print(f"  SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"    Reference:  {t_ref_elapsed:.2f}s")
    print(f"    Model init: {t_model_elapsed:.4f}s")
    print(f"    Training:   {t_train_elapsed:.2f}s "
          f"({t_train_elapsed / args.epochs:.4f}s/epoch)")
    print(f"    Evaluation: {t_eval_elapsed:.2f}s")
    print(f"  Epochs: {args.epochs}")
    print(f"  n_pieces: {args.pieces}")
    print(f"  lr: {args.lr}")
    print(f"  seed: {args.seed}")
    print(f"  n_weights: {trop.n_weights}")
    print(f"  Final loss: {history[-1]:.6e}")
    print(f"  Min loss: {min(history):.6e}")
    print(f"  Min/mean/max digits: {min(digits_list)}/{mean_dig:.2f}/{max(digits_list)}")
    print(f"  GPU peak allocated: {gpu_mem_max_allocated_mb():.1f}MB")
    print(f"  GPU final allocated: {gpu_mem_allocated_mb():.1f}MB")
    print(f"  GPU final reserved: {gpu_mem_reserved_mb():.1f}MB")
    print(f"  RSS: {rss_mb():.1f}MB")

    # Extrapolate full run
    if args.epochs < 16000:
        est_16k = (t_train_elapsed / args.epochs) * 16000
        print(f"\n  Extrapolated time for 16000 epochs: {est_16k:.0f}s "
              f"({est_16k / 60:.1f} min)")
        est_total = est_16k * 10  # 2 configs x 5 seeds per GPU
        print(f"  Extrapolated time for full GPU sweep (10 runs): "
              f"{est_total:.0f}s ({est_total / 3600:.1f} hours)")

    print(f"\n  Peak GPU memory: {gpu_mem_max_allocated_mb():.1f}MB")
    print(f"  Final GPU memory: {gpu_mem_allocated_mb():.1f}MB")
    print(f"  Memory leak check: "
          f"{'OK' if gpu_mem_allocated_mb() < 100 else 'SUSPECTED LEAK'}")
    print(f"\n  DIAGNOSTIC COMPLETE\n")


if __name__ == "__main__":
    main()
