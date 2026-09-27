"""
super_test_siren_m2.py — hyperparameter × epoch-budget sweep for SirenIntegrator.

Runs SirenIntegrator training over a grid of (hidden_sizes, omega_0, lr,
n_per_param) combinations CROSSED with a ladder of epoch budgets
(1500 -> 10000), evaluates each trained net against the scipy reference
integrals (computed once, since they don't depend on the network or the
epoch budget), and ranks all runs by accuracy / speed.

Usage:
    python super_test_siren_m2.py                # full grid × full epoch ladder
    python super_test_siren_m2.py --quick         # smaller grid + shorter ladder
    python super_test_siren_m2.py --epochs 3000 5000 10000   # custom epoch ladder
"""
import argparse
import csv
import itertools
import math
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../flat/5-siren-integrator'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../skuld-lib'))

import torch

from skuld.siren import SirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID  — edit these lists to taste
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "hidden_sizes": [
        [64, 64, 64],
        [128, 128, 128],
        [128, 128, 128, 128],
        [256, 256],
    ],
    "omega_0": [15.0, 30.0, 45.0],
    "lr": [1e-3, 5e-4, 2e-4],
    "n_per_param": [512],
}

QUICK_GRID = {
    "hidden_sizes": [[64, 64, 64], [128, 128, 128]],
    "omega_0": [15.0, 30.0],
    "lr": [1e-3, 5e-4],
    "n_per_param": [512],
}

# Epoch ladder — every hyperparameter config is retrained from scratch at
# each of these budgets.
EPOCHS_GRID = [1500, 2000, 3000, 4500, 6500, 10000]
QUICK_EPOCHS_GRID = [1500, 3000, 10000]


def make_configs(grid: dict) -> list:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


# ─────────────────────────────────────────────────────────────────────────
# 2. ONE TRAIN + EVAL RUN
# ─────────────────────────────────────────────────────────────────────────
def run_one(cfg: dict, n_epochs: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(42)

    omega_0 = cfg["omega_0"]
    output_scale = 1.0 / (omega_0 ** 3)

    integrator = SirenIntegrator(
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
        n_epochs=n_epochs,
        n_per_param=cfg["n_per_param"],
        lr=cfg["lr"],
        device=device,
        verbose_every=verbose_every if verbose_every > 0 else n_epochs + 1,
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
        "hidden_sizes": cfg["hidden_sizes"],
        "omega_0": omega_0,
        "lr": cfg["lr"],
        "n_per_param": cfg["n_per_param"],
        "n_epochs": n_epochs,
        "n_params_total": n_params_total,
        "final_loss": history[-1],
        "min_loss": min(history),
        "mean_rel_err": sum(rel_errs) / len(rel_errs),
        "max_rel_err": max(rel_errs),
        "min_correct_digits": min(digits),
        "mean_correct_digits": sum(digits) / len(digits),
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hyperparameter × epoch-budget sweep for SirenIntegrator")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                         help=f"epoch budgets to sweep, space-separated "
                              f"(default: {EPOCHS_GRID}, or {QUICK_EPOCHS_GRID} with --quick)")
    parser.add_argument("--quick", action="store_true",
                         help="use a smaller hyperparameter grid AND a shorter epoch ladder")
    parser.add_argument("--verbose-every", type=int, default=0,
                         help="print training progress every N epochs (0 = silent per-epoch)")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path (default: sweep_results_m2_<timestamp>.csv)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda:1" if torch.cuda.is_available() else
        "cpu"
    )

    grid = QUICK_GRID if args.quick else GRID
    epochs_grid = args.epochs if args.epochs is not None else (
        QUICK_EPOCHS_GRID if args.quick else EPOCHS_GRID
    )
    configs = make_configs(grid)
    runs = list(itertools.product(configs, epochs_grid))

    print(f"\n{'═' * 72}")
    print(f"  SUPER-TEST m2  —  {len(configs)} configs × {len(epochs_grid)} epoch "
          f"budgets {epochs_grid} = {len(runs)} runs, device={device}")
    print(f"{'═' * 72}\n")

    # Reference values depend only on (a,b,m,n) — compute ONCE, reuse everywhere.
    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    results = []
    for i, (cfg, n_epochs) in enumerate(runs, 1):
        tag = (f"[{i}/{len(runs)}] hidden={cfg['hidden_sizes']} omega0={cfg['omega_0']} "
               f"lr={cfg['lr']} N={cfg['n_per_param']} epochs={n_epochs}")
        print(f"{tag} ...", flush=True)
        try:
            res = run_one(cfg, n_epochs, device, refs, args.verbose_every)
            results.append(res)
            print(f"    -> min_loss={res['min_loss']:.3e}  "
                  f"mean_rel_err={res['mean_rel_err']:.3e}  "
                  f"min_digits={res['min_correct_digits']}  "
                  f"({res['elapsed_s']:.1f}s)")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")

    if not results:
        print("\nNo successful runs.")
        return

    # ── rank: best = highest min_correct_digits, tiebreak by mean_rel_err,
    #    tiebreak by fewer epochs ─────────────────────────────────────────
    results.sort(key=lambda r: (-r["min_correct_digits"], r["mean_rel_err"], r["n_epochs"]))

    SEP = "═" * 128
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'hidden':<20} {'omega0':>7} {'lr':>9} {'N/param':>8} {'epochs':>7} "
              f"{'params':>9} {'min_loss':>11} {'mean_relerr':>12} {'min_dig':>8} {'time(s)':>8}")
    print(header)
    print("-" * 128)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {str(r['hidden_sizes']):<20} {r['omega_0']:>7.1f} {r['lr']:>9.1e} "
              f"{r['n_per_param']:>8} {r['n_epochs']:>7} {r['n_params_total']:>9,} "
              f"{r['min_loss']:>11.3e} {r['mean_rel_err']:>12.3e} "
              f"{r['min_correct_digits']:>8} {r['elapsed_s']:>8.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    hidden_sizes = {best['hidden_sizes']}")
    print(f"    omega_0      = {best['omega_0']}")
    print(f"    lr           = {best['lr']}")
    print(f"    n_per_param  = {best['n_per_param']}")
    print(f"    n_epochs     = {best['n_epochs']}")
    print(f"    -> min {best['min_correct_digits']} correct digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    # ── per-config epoch-scan ────────────────────────────────────────────
    print(f"{SEP}")
    print(f"  ACCURACY vs EPOCHS, per hyperparameter config")
    print(SEP)
    by_cfg = {}
    for r in results:
        key = (str(r["hidden_sizes"]), r["omega_0"], r["lr"], r["n_per_param"])
        by_cfg.setdefault(key, []).append(r)
    for key, rs in by_cfg.items():
        rs_sorted = sorted(rs, key=lambda r: r["n_epochs"])
        hs, om, lr, npp = key
        print(f"  hidden={hs} omega0={om} lr={lr} N={npp}")
        for r in rs_sorted:
            print(f"      epochs={r['n_epochs']:>6}  min_dig={r['min_correct_digits']:>2}  "
                  f"mean_relerr={r['mean_rel_err']:.3e}  time={r['elapsed_s']:.1f}s")
    print(SEP)

    # ── save CSV to mirroring results directory ──────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), '../../results/grid/6-siren-wire-kan-sweep-m2')
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(results_dir, f"sweep_results_m2_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Full results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
