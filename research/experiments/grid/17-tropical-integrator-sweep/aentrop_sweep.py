"""
aentrop_sweep.py — multi-GPU resumable grid sweep for AenTrop (tropical max-plus).

Sweeps:
  - n_pieces: [10, 12, 14, 16]
  - lr: [5e-4, 1e-3, 2e-3]
  - tau_start: [1.0]
  - tau_end: [0.01]
  - epochs: [8000, 16000] (configurable)

Total: 4 * 3 * 2 = 24 configs x 5 seeds = 120 runs per GPU partition.

Multi-GPU: each GPU processes a subset of configs. Resumable via CSV
checkpointing — if a config already has results for a seed, it is skipped.

GPU index is auto-detected from SLURM_ARRAY_TASK_ID (SLURM job arrays) or
LOCAL_RANK (torchrun/srun). No manual --gpu flag needed in those cases.

Run tag namespaces results for concurrent runs on different GPU partitions
(volta/ampere/hopper).

Usage (on CUDA):
    python aentrop_sweep.py --num-gpus 4              # auto-detect from SLURM
    python aentrop_sweep.py --gpu 0 --num-gpus 4      # manual override
    python aentrop_sweep.py --run-tag volta           # namespace results
"""
import argparse
import itertools
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch

from physics import integrand_transformed, reference_scipy, PARAM_SETS

from skuld.aentrop import AenTropIntegrator


##############################################################################
#  Grid sweep
##############################################################################

def digits_of(abs_err: float) -> int:
    return max(0, -math.floor(math.log10(abs_err + 1e-30)))


def _sanitize_tag(tag: str) -> str:
    """Sanitize run tag to safe filesystem characters."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', tag)


def _detect_gpu_index() -> int:
    """Auto-detect GPU index from SLURM or torchrun environment."""
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_NODEID" in os.environ and "SLURM_NTASKS" in os.environ:
        return 0
    return 0


def main():
    parser = argparse.ArgumentParser(description="AenTrop grid sweep")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (0-based). Auto-detected from SLURM if omitted.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="total number of GPUs")
    parser.add_argument("--epochs", type=int, default=8000,
                        help="epochs per config (default: 8000, v5 champion used 16000)")
    parser.add_argument("--seeds", type=str, default="42,137,2024,999,7",
                        help="comma-separated list of seeds")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="namespace results directory (e.g., volta, ampere, hopper). "
                             "Defaults to SLURM_JOB_PARTITION or SKULD_RUN_TAG.")
    args = parser.parse_args()

    # Auto-detect GPU index if not provided
    if args.gpu is None:
        args.gpu = _detect_gpu_index()

    # Resolve run tag: --run-tag > SKULD_RUN_TAG > SLURM_JOB_PARTITION > ""
    run_tag = (args.run_tag or os.environ.get("SKULD_RUN_TAG")
               or os.environ.get("SLURM_JOB_PARTITION") or "")
    run_tag = _sanitize_tag(run_tag)

    # When using SLURM job arrays with --gres=gpu:1, each job sees only 1 GPU.
    # CUDA_VISIBLE_DEVICES remaps it to cuda:0, so always use device 0.
    # The array task ID (args.gpu) is used only for config distribution.
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    # Grid definition — centered on champion v5 (12 pieces, lr=1e-3)
    grid = {
        "n_pieces": [10, 12, 14, 16],
        "lr": [5e-4, 1e-3, 2e-3],
        "tau_start": [1.0],
        "tau_end": [0.01],
        "epochs": [args.epochs],  # single value from CLI
    }

    # Generate all configs
    keys = list(grid.keys())
    all_configs = list(itertools.product(*[grid[k] for k in keys]))

    # Distribute configs across GPUs
    my_configs = [c for i, c in enumerate(all_configs) if i % args.num_gpus == args.gpu]

    tag_str = f" [{run_tag}]" if run_tag else ""
    print(f"{'=' * 78}")
    print(f"  AenTROP GRID SWEEP — GPU {args.gpu}/{args.num_gpus}{tag_str}")
    print(f"  device={device}  epochs={args.epochs}  seeds={seeds}")
    print(f"  total configs: {len(all_configs)}  my configs: {len(my_configs)}")
    print(f"{'=' * 78}\n")

    # Compute reference integrals
    print("Computing scipy reference integrals ...")
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    targets = {p: refs[p][0] for p in PARAM_SETS}
    print("Done.\n")

    # Results directory with optional run-tag namespacing
    base_results_dir = get_mirror_path(__file__, "results")
    if run_tag:
        results_dir = base_results_dir / run_tag
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = base_results_dir
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

    # CSV header — per-integral digits + aggregate
    header = ("n_pieces,lr,tau_start,tau_end,epochs,seed,"
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
        n_pieces, lr, tau_start, tau_end, epochs = config

        for seed in seeds:
            config_key = f"{n_pieces}_{lr}_{tau_start}_{tau_end}_{epochs}_{seed}"
            if config_key in existing:
                done += 1
                continue

            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = time.time()
            aentrop = AenTropIntegrator(
                n_params=4, n_int_vars=3, n_pieces=n_pieces,
                tau_start=tau_start, tau_end=tau_end)

            history = aentrop.train(
                integrand_fn=integrand_transformed,
                param_sets=PARAM_SETS, targets=targets,
                n_epochs=epochs, n_per_param=1024, lr=lr, device=device,
                verbose_every=max(1, epochs // 20))

            # Evaluate
            digits_list = []
            rel_errs_list = []
            per_integral = []
            for params in PARAM_SETS:
                nni_val = aentrop.integrate(params, device=device, tau=tau_end)
                ref_val, _ = refs[params]
                abs_err = abs(nni_val - ref_val)
                rel_err = abs_err / (abs(ref_val) + 1e-30)
                d = digits_of(abs_err)
                digits_list.append(d)
                rel_errs_list.append(rel_err)
                per_integral.append(d)

            elapsed = time.time() - t0

            row = (
                f"{n_pieces},{lr},{tau_start},{tau_end},{epochs},{seed},"
                f"{per_integral[0]},{per_integral[1]},{per_integral[2]},"
                f"{per_integral[3]},{per_integral[4]},{per_integral[5]},"
                f"{per_integral[6]},{per_integral[7]},"
                f"{min(digits_list)},{sum(digits_list)/len(digits_list):.2f},"
                f"{max(digits_list)},{sum(rel_errs_list)/len(rel_errs_list):.6e},"
                f"{max(rel_errs_list):.6e},"
                f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},{aentrop.n_weights}"
            )
            csv_fh.write(row + "\n")
            csv_fh.flush()

            done += 1
            print(f"  [{done}/{total}] pieces={n_pieces} lr={lr} epochs={epochs} seed={seed}  "
                  f"mean_dig={sum(digits_list)/len(digits_list):.2f}  "
                  f"max_dig={max(digits_list)}  loss={history[-1]:.2e}  "
                  f"({elapsed:.1f}s)")

    csv_fh.close()
    print(f"\n  GPU {args.gpu}{tag_str} done. Results -> {csv_file}\n")


if __name__ == "__main__":
    main()