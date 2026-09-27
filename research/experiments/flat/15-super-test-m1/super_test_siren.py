"""
super_test.py — hyperparameter sweep for siren_nni.py

Runs SirenPrimitiveNet training over a grid of (hidden_sizes, omega_0, lr,
n_per_param) combinations, evaluates each trained net against the scipy
reference integrals (computed once, since they don't depend on the network),
and ranks configs by accuracy / speed.

Usage:
    python super_test.py                      # default grid, 1500 epochs/config
    python super_test.py --epochs 3000
    python super_test.py --epochs 800 --quick  # smaller grid, fast sanity pass

Place this file in the SAME directory as siren_nni.py.
"""

import argparse
import csv
import itertools
import math
import time
from datetime import datetime

import torch

import siren_nni as base  # reuse model / training / integral code as-is


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
    "n_per_param": [512],   # kept fixed by default; add e.g. 1024 to sweep it too
}

QUICK_GRID = {
    "hidden_sizes": [[64, 64, 64], [128, 128, 128]],
    "omega_0": [15.0, 30.0],
    "lr": [1e-3, 5e-4],
    "n_per_param": [512],
}


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

    net = base.SirenPrimitiveNet(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=cfg["hidden_sizes"],
        omega_0=omega_0,
        output_scale=output_scale,
    )
    n_params_total = sum(p.numel() for p in net.parameters())

    t0 = time.time()
    history, norm_cache = base.train(
        net,
        param_sets=base.PARAM_SETS,
        n_epochs=n_epochs,
        n_per_param=cfg["n_per_param"],
        lr=cfg["lr"],
        device=device,
        verbose_every=verbose_every if verbose_every > 0 else n_epochs + 1,
    )
    elapsed = time.time() - t0

    # ── accuracy vs. precomputed scipy reference ──────────────────────────
    abs_errs, rel_errs, digits = [], [], []
    for (a, b, m, n) in base.PARAM_SETS:
        nni_val = base.compute_integral(net, a, b, m, n,
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
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for siren_nni.py")
    parser.add_argument("--epochs", type=int, default=1500,
                         help="epochs per config (default 1500; full run in siren_nni.py uses 8000)")
    parser.add_argument("--quick", action="store_true",
                         help="use a smaller grid for a fast sanity pass")
    parser.add_argument("--verbose-every", type=int, default=0,
                         help="print training progress every N epochs (0 = silent per-epoch)")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path (default: sweep_results_<timestamp>.csv)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda:" if torch.cuda.is_available() else
        "cpu"
    )
    torch.set_default_dtype(base.FLOATING_POINT_PRECISION)

    grid = QUICK_GRID if args.quick else GRID
    configs = make_configs(grid)

    print(f"\n{'═' * 72}")
    print(f"  SUPER-TEST  —  {len(configs)} configs × {args.epochs} epochs, device={device}")
    print(f"{'═' * 72}\n")

    # Reference values depend only on (a,b,m,n) — compute ONCE, reuse everywhere.
    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in base.PARAM_SETS:
        r, e = base.reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    results = []
    for i, cfg in enumerate(configs, 1):
        tag = f"[{i}/{len(configs)}] hidden={cfg['hidden_sizes']} omega0={cfg['omega_0']} lr={cfg['lr']} N={cfg['n_per_param']}"
        print(f"{tag} ...", flush=True)
        try:
            res = run_one(cfg, args.epochs, device, refs, args.verbose_every)
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

    # ── rank: best = highest min_correct_digits, tiebreak by mean_rel_err ──
    results.sort(key=lambda r: (-r["min_correct_digits"], r["mean_rel_err"]))

    SEP = "═" * 118
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'hidden':<20} {'omega0':>7} {'lr':>9} {'N/param':>8} "
              f"{'params':>9} {'min_loss':>11} {'mean_relerr':>12} {'min_dig':>8} {'time(s)':>8}")
    print(header)
    print("-" * 118)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {str(r['hidden_sizes']):<20} {r['omega_0']:>7.1f} {r['lr']:>9.1e} "
              f"{r['n_per_param']:>8} {r['n_params_total']:>9,} {r['min_loss']:>11.3e} "
              f"{r['mean_rel_err']:>12.3e} {r['min_correct_digits']:>8} {r['elapsed_s']:>8.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    hidden_sizes = {best['hidden_sizes']}")
    print(f"    omega_0      = {best['omega_0']}")
    print(f"    lr           = {best['lr']}")
    print(f"    n_per_param  = {best['n_per_param']}")
    print(f"    -> min {best['min_correct_digits']} correct digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    # ── save CSV ─────────────────────────────────────────────────────────
    out_path = args.out or f"sweep_results_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Full results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
