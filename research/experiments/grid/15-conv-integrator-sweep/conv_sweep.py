"""
conv_sweep.py — multi-GPU hyperparameter sweep for ConvIntegrator.

Sweeps 4 hyperparameters in a full factorial grid:
  hidden_sizes, n_epochs, lr, n_per_param

Multi-GPU via SLURM job array:
  Each job reads SLURM_ARRAY_TASK_ID and processes a contiguous block
  of the (config x seed) list. Results are written to per-GPU CSVs.

Resumable:
  Reads existing per-GPU CSV at startup, skips completed runs.

Usage:
    python conv_sweep.py                    # full grid, 5 seeds, 1 GPU
    python conv_sweep.py --num-gpus 4       # 4 GPUs (SLURM array)
    python conv_sweep.py --quick            # tiny grid for testing
    python conv_sweep.py --seeds 42         # single seed
"""
import argparse
import csv
import itertools
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../skuld-lib'))

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch

from skuld.conv import ConvIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "hidden_sizes": [
        [16, 32, 64, 128],
        [32, 64, 128, 256],
        [48, 96, 192, 384],
        [64, 128, 256, 512],
    ],
    "n_epochs": [2000, 4000, 8000],
    "lr": [5e-4, 1e-3, 2e-3],
    "n_per_param": [256, 512, 1024],
}

QUICK_GRID = {
    "hidden_sizes": [[16, 32, 64, 128]],
    "n_epochs": [500],
    "lr": [1e-3],
    "n_per_param": [128],
}

SEEDS = [42, 137, 2024, 7, 999]

CSV_FIELDS = [
    "hidden_sizes", "lr", "n_epochs", "n_per_param", "seed",
    "n_params_total", "final_loss", "min_loss", "mean_rel_err", "max_rel_err",
    "I_1_dig", "I_2_dig", "I_3_dig", "I_4_dig",
    "I_5_dig", "I_6_dig", "I_7_dig", "I_8_dig",
    "min_dig", "mean_dig", "max_dig", "elapsed_s",
]


def make_configs(grid: dict) -> list:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


def make_key(cfg: dict, seed: int) -> tuple:
    """Hashable key for deduplication / resumability."""
    return (
        str(cfg["hidden_sizes"]),
        str(cfg["lr"]),
        str(cfg["n_epochs"]),
        str(cfg["n_per_param"]),
        str(seed),
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. ONE TRAIN + EVAL RUN
# ─────────────────────────────────────────────────────────────────────────
def run_one(cfg: dict, seed: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    integrator = ConvIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=cfg["hidden_sizes"],
    )
    n_params_total = integrator.n_weights

    t0 = time.time()
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=cfg["n_epochs"],
        n_per_param=cfg["n_per_param"],
        lr=cfg["lr"],
        device=device,
        verbose_every=verbose_every if verbose_every > 0 else cfg["n_epochs"] + 1,
    )
    elapsed = time.time() - t0

    # ── accuracy vs. precomputed scipy reference ──────────────────────────
    abs_errs, rel_errs, digits = [], [], []
    for (a, b, m, n) in PARAM_SETS:
        nni_val = integrator.integrate((a, b, m, n),
                                        norm_cache=norm_cache, device=device)
        ref_val, _ = refs[(a, b, m, n)]
        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)
        d = max(0, -math.floor(math.log10(abs_err + 1e-30)))
        abs_errs.append(abs_err)
        rel_errs.append(rel_err)
        digits.append(d)

    return {
        "hidden_sizes": str(cfg["hidden_sizes"]),
        "lr": cfg["lr"],
        "n_epochs": cfg["n_epochs"],
        "n_per_param": cfg["n_per_param"],
        "seed": seed,
        "n_params_total": n_params_total,
        "final_loss": history[-1],
        "min_loss": min(history),
        "mean_rel_err": sum(rel_errs) / len(rel_errs),
        "max_rel_err": max(rel_errs),
        "I_1_dig": digits[0],
        "I_2_dig": digits[1],
        "I_3_dig": digits[2],
        "I_4_dig": digits[3],
        "I_5_dig": digits[4],
        "I_6_dig": digits[5],
        "I_7_dig": digits[6],
        "I_8_dig": digits[7],
        "min_dig": min(digits),
        "mean_dig": sum(digits) / len(digits),
        "max_dig": max(digits),
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. RESUMABILITY — load completed runs from existing CSV
# ─────────────────────────────────────────────────────────────────────────
def load_completed_runs(csv_path: str) -> set:
    """Return set of (hidden_sizes, lr, n_epochs, n_per_param, seed) tuples."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (
                row["hidden_sizes"],
                row["lr"],
                row["n_epochs"],
                row["n_per_param"],
                row["seed"],
            )
            completed.add(key)
    return completed


def append_result(csv_path: str, result: dict):
    """Append one result row to CSV, writing header if file is new."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
        fh.flush()


# ─────────────────────────────────────────────────────────────────────────
# 4. MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU hyperparameter sweep for ConvIntegrator")
    parser.add_argument("--quick", action="store_true",
                        help="use a tiny hyperparameter grid for testing")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="total number of GPUs in the job array (default: 1)")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU ID to process (default: SLURM_ARRAY_TASK_ID env var, or 0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"seeds to run (default: {SEEDS})")
    parser.add_argument("--verbose-every", type=int, default=0,
                        help="print training progress every N epochs (0 = silent)")
    args = parser.parse_args()

    # ── determine GPU ID ──────────────────────────────────────────────────
    gpu_id = args.gpu_id if args.gpu_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_gpus = args.num_gpus

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    grid = QUICK_GRID if args.quick else GRID
    configs = make_configs(grid)
    all_runs = list(itertools.product(configs, args.seeds))

    # ── split into contiguous blocks for each GPU ─────────────────────────
    block_size = (len(all_runs) + num_gpus - 1) // num_gpus
    start_idx = gpu_id * block_size
    end_idx = min(start_idx + block_size, len(all_runs))
    my_runs = all_runs[start_idx:end_idx]

    print(f"\n{'=' * 72}")
    print(f"  CONV INTEGRATOR SWEEP  —  {len(configs)} configs × {len(args.seeds)} seeds "
          f"= {len(all_runs)} total runs")
    print(f"  GPU {gpu_id}/{num_gpus}  —  processing runs [{start_idx}:{end_idx}] "
          f"= {len(my_runs)} runs, device={device}")
    print(f"{'=' * 72}\n")

    if not my_runs:
        print(f"  GPU {gpu_id}: no runs to process (block empty).")
        return

    # ── reference values (shared across all configs) ──────────────────────
    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    # ── resumability: load completed runs ─────────────────────────────────
    results_dir = get_mirror_path(__file__, "results")
    csv_path = os.path.join(results_dir, f"results_gpu{gpu_id}.csv")
    completed = load_completed_runs(csv_path)
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, skipping them.\n")

    # ── main loop ─────────────────────────────────────────────────────────
    results = []
    for i, (cfg, seed) in enumerate(my_runs, 1):
        key = make_key(cfg, seed)
        if key in completed:
            continue

        tag = (f"[{i}/{len(my_runs)}] hidden={cfg['hidden_sizes']} "
               f"lr={cfg['lr']} epochs={cfg['n_epochs']} npp={cfg['n_per_param']} seed={seed}")
        print(f"{tag} ...", flush=True)
        try:
            res = run_one(cfg, seed, device, refs, args.verbose_every)
            results.append(res)
            append_result(csv_path, res)
            print(f"    -> I1={res['I_1_dig']} I2={res['I_2_dig']} I3={res['I_3_dig']} "
                  f"I4={res['I_4_dig']} I5={res['I_5_dig']} I6={res['I_6_dig']} "
                  f"I7={res['I_7_dig']} I8={res['I_8_dig']} "
                  f"min={res['min_dig']} mean={res['mean_dig']:.1f} max={res['max_dig']} "
                  f"({res['elapsed_s']:.1f}s)")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")

    if not results:
        print(f"\n  GPU {gpu_id}: no new runs completed (all were already done).")
        return

    # ── rank: best = highest min_dig, tiebreak by mean_rel_err, tiebreak by fewer epochs ──
    results.sort(key=lambda r: (-r["min_dig"], r["mean_rel_err"], r["n_epochs"]))

    SEP = "=" * 140
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first) — GPU {gpu_id}")
    print(SEP)
    header = (f"  {'#':>3} {'hidden':<20} {'lr':>8} {'epochs':>6} "
              f"{'npp':>5} {'seed':>5} {'min_loss':>10} {'mean_relerr':>11} "
              f"{'I1':>3} {'I2':>3} {'I3':>3} {'I4':>3} {'I5':>3} {'I6':>3} {'I7':>3} {'I8':>3} "
              f"{'min':>4} {'mean':>5} {'max':>4} {'time':>6}")
    print(header)
    print("-" * 140)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {r['hidden_sizes']:<20} {r['lr']:>8.1e} "
              f"{r['n_epochs']:>6} {r['n_per_param']:>5} {r['seed']:>5} "
              f"{r['min_loss']:>10.3e} {r['mean_rel_err']:>11.3e} "
              f"{r['I_1_dig']:>3} {r['I_2_dig']:>3} {r['I_3_dig']:>3} {r['I_4_dig']:>3} "
              f"{r['I_5_dig']:>3} {r['I_6_dig']:>3} {r['I_7_dig']:>3} {r['I_8_dig']:>3} "
              f"{r['min_dig']:>4} {r['mean_dig']:>5.1f} {r['max_dig']:>4} "
              f"{r['elapsed_s']:>6.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG (GPU {gpu_id}):")
    print(f"    hidden_sizes = {best['hidden_sizes']}")
    print(f"    lr           = {best['lr']}")
    print(f"    n_epochs     = {best['n_epochs']}")
    print(f"    n_per_param  = {best['n_per_param']}")
    print(f"    seed         = {best['seed']}")
    print(f"    -> min {best['min_dig']} digits, "
          f"mean {best['mean_dig']:.1f} digits, "
          f"max {best['max_dig']} digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    # ── per-config seed-averaged summary ─────────────────────────────────
    print(f"{SEP}")
    print(f"  PER-CONFIG SUMMARY (seed-averaged) — GPU {gpu_id}")
    print(SEP)
    by_cfg = {}
    for r in results:
        key = (r["hidden_sizes"], r["lr"], r["n_epochs"], r["n_per_param"])
        by_cfg.setdefault(key, []).append(r)
    for key, rs in sorted(by_cfg.items(),
                          key=lambda kv: (-min(r["min_dig"] for r in kv[1]),
                                          sum(r["mean_rel_err"] for r in kv[1]) / len(kv[1]))):
        hs, lr, ep, npp = key
        min_d = min(r["min_dig"] for r in rs)
        mean_d = sum(r["mean_dig"] for r in rs) / len(rs)
        max_d = max(r["max_dig"] for r in rs)
        mean_rel = sum(r["mean_rel_err"] for r in rs) / len(rs)
        print(f"  hidden={hs} lr={lr} epochs={ep} npp={npp}")
        print(f"      min_dig={min_d}  mean_dig={mean_d:.1f}  max_dig={max_d}  "
              f"mean_relerr={mean_rel:.3e}  n_seeds={len(rs)}")
    print(SEP)
    print(f"  Results saved -> {csv_path}\n")


if __name__ == "__main__":
    main()
