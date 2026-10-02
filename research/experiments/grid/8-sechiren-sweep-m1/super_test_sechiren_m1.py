"""
super_test_sechiren_m1.py — hyperparameter sweep for SechirenIntegrator.

Grid sweep around the best-known SECHIREN config:
  omega_0=15, hidden=[128,128,128], epochs=8000, lr=5e-4, n_per_param=1024

Varies 4 key hyperparameters in a full factorial grid:
  omega_0, hidden_sizes, n_epochs, lr

Fixed at best known:
  n_per_param=1024, output_scale=1/omega_0^3, weight_decay=0, corner=off

3 seeds per config. Results to CSV.

Usage:
    python super_test_sechiren_m1.py                # full grid, 3 seeds
    python super_test_sechiren_m1.py --quick        # smaller grid
    python super_test_sechiren_m1.py --seeds 1      # single seed
"""
import argparse
import csv
import itertools
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../skuld-lib'))

import numpy as np
import torch

from skuld.sechiren import SechirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "hidden_sizes": [
        [64, 64, 64],
        [128, 128, 128],
    ],
    "omega_0": [10.0, 15.0, 20.0],
    "n_epochs": [4000, 8000],
    "lr": [1e-4, 5e-4],
}

QUICK_GRID = {
    "hidden_sizes": [[128, 128, 128]],
    "omega_0": [10.0, 15.0, 20.0],
    "n_epochs": [8000],
    "lr": [5e-4],
}

SEEDS = [42, 137, 2024]


def make_configs(grid: dict) -> list:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


# ─────────────────────────────────────────────────────────────────────────
# 2. ONE TRAIN + EVAL RUN
# ─────────────────────────────────────────────────────────────────────────
def run_one(cfg: dict, seed: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    omega_0 = cfg["omega_0"]
    output_scale = 1.0 / (omega_0 ** 3)

    integrator = SechirenIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=cfg["hidden_sizes"],
        omega_0=omega_0,
        output_scale=output_scale,
    )
    n_params_total = integrator.n_weights

    t0 = time.time()
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=cfg["n_epochs"],
        n_per_param=1024,
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
        "omega_0": omega_0,
        "lr": cfg["lr"],
        "n_epochs": cfg["n_epochs"],
        "n_per_param": 1024,
        "seed": seed,
        "n_params_total": n_params_total,
        "final_loss": history[-1],
        "min_loss": min(history),
        "mean_rel_err": sum(rel_errs) / len(rel_errs),
        "max_rel_err": max(rel_errs),
        "min_correct_digits": min(digits),
        "mean_correct_digits": sum(digits) / len(digits),
        "max_correct_digits": max(digits),
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for SechirenIntegrator")
    parser.add_argument("--quick", action="store_true",
                        help="use a smaller hyperparameter grid")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"seeds to run (default: {SEEDS})")
    parser.add_argument("--verbose-every", type=int, default=0,
                        help="print training progress every N epochs (0 = silent)")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV output path")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    grid = QUICK_GRID if args.quick else GRID
    configs = make_configs(grid)
    runs = list(itertools.product(configs, args.seeds))

    print(f"\n{'=' * 72}")
    print(f"  SECHIREN SWEEP m1  —  {len(configs)} configs × {len(args.seeds)} seeds "
          f"= {len(runs)} runs, device={device}")
    print(f"{'=' * 72}\n")

    # Reference values depend only on (a,b,m,n) — compute ONCE, reuse everywhere.
    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    results = []
    for i, (cfg, seed) in enumerate(runs, 1):
        tag = (f"[{i}/{len(runs)}] hidden={cfg['hidden_sizes']} omega0={cfg['omega_0']} "
               f"lr={cfg['lr']} epochs={cfg['n_epochs']} seed={seed}")
        print(f"{tag} ...", flush=True)
        try:
            res = run_one(cfg, seed, device, refs, args.verbose_every)
            results.append(res)
            print(f"    -> min_loss={res['min_loss']:.3e}  "
                  f"mean_rel_err={res['mean_rel_err']:.3e}  "
                  f"min_dig={res['min_correct_digits']}  "
                  f"mean_dig={res['mean_correct_digits']:.1f}  "
                  f"max_dig={res['max_correct_digits']}  "
                  f"({res['elapsed_s']:.1f}s)")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")

    if not results:
        print("\nNo successful runs.")
        return

    # ── rank: best = highest min_correct_digits, tiebreak by mean_rel_err,
    #    tiebreak by fewer epochs ─────────────────────────────────────────
    results.sort(key=lambda r: (-r["min_correct_digits"], r["mean_rel_err"], r["n_epochs"]))

    SEP = "=" * 128
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'hidden':<20} {'omega0':>7} {'lr':>9} {'epochs':>7} "
              f"{'seed':>5} {'params':>9} {'min_loss':>11} {'mean_relerr':>12} "
              f"{'min_dig':>8} {'mean_dig':>9} {'max_dig':>8} {'time(s)':>8}")
    print(header)
    print("-" * 128)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {r['hidden_sizes']:<20} {r['omega_0']:>7.1f} {r['lr']:>9.1e} "
              f"{r['n_epochs']:>7} {r['seed']:>5} {r['n_params_total']:>9,} "
              f"{r['min_loss']:>11.3e} {r['mean_rel_err']:>12.3e} "
              f"{r['min_correct_digits']:>8} {r['mean_correct_digits']:>9.1f} "
              f"{r['max_correct_digits']:>8} {r['elapsed_s']:>8.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    hidden_sizes = {best['hidden_sizes']}")
    print(f"    omega_0      = {best['omega_0']}")
    print(f"    lr           = {best['lr']}")
    print(f"    n_epochs     = {best['n_epochs']}")
    print(f"    seed         = {best['seed']}")
    print(f"    -> min {best['min_correct_digits']} digits, "
          f"mean {best['mean_correct_digits']:.1f} digits, "
          f"max {best['max_correct_digits']} digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    # ── per-config seed-averaged summary ─────────────────────────────────
    print(f"{SEP}")
    print(f"  PER-CONFIG SUMMARY (seed-averaged)")
    print(SEP)
    by_cfg = {}
    for r in results:
        key = (r["hidden_sizes"], r["omega_0"], r["lr"], r["n_epochs"])
        by_cfg.setdefault(key, []).append(r)
    for key, rs in sorted(by_cfg.items(),
                          key=lambda kv: (-min(r["min_correct_digits"] for r in kv[1]),
                                          sum(r["mean_rel_err"] for r in kv[1]) / len(kv[1]))):
        hs, om, lr, ep = key
        min_d = min(r["min_correct_digits"] for r in rs)
        mean_d = sum(r["mean_correct_digits"] for r in rs) / len(rs)
        max_d = max(r["max_correct_digits"] for r in rs)
        mean_rel = sum(r["mean_rel_err"] for r in rs) / len(rs)
        print(f"  hidden={hs} omega0={om} lr={lr} epochs={ep}")
        print(f"      min_dig={min_d}  mean_dig={mean_d:.1f}  max_dig={max_d}  "
              f"mean_relerr={mean_rel:.3e}  n_seeds={len(rs)}")
    print(SEP)

    # ── save CSV ─────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), '../../results/grid/8-sechiren-sweep-m1')
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        results_dir, f"sweep_results_m1_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Full results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
