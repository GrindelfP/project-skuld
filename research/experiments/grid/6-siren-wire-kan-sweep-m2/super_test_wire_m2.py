"""
super_test_wire_m2.py — hyperparameter + epoch-budget sweep for wire_nni.py

Same idea as super_test_wire.py, grid tailored to Wire (Gabor/Morlet)
specifics: omega_0 (frequency) AND sigma_0 (envelope bandwidth) interact,
plus entry_width / n_blocks / lr.

NEW in m2: instead of a single fixed --epochs value, each hyperparameter
config is now trained and evaluated at MULTIPLE epoch budgets, so you can
see how accuracy scales with training length for the same architecture.
Default epoch budgets: 1500, 2000, 3000, 4000, 5500, 7000, 8500, 10000
(edit EPOCHS_GRID below, or pass --epochs-list to override).

Usage:
    python super_test_wire_m2.py                          # default grid x default epoch budgets
    python super_test_wire_m2.py --epochs-list 1500,3000,10000
    python super_test_wire_m2.py --quick --epochs-list 500,1000  # fast sanity pass

Place this file in the SAME directory as wire_nni.py.
"""

import argparse
import csv
import itertools
import math
import time
from datetime import datetime
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import torch

import wire_nni as base  # reuse model / training / integral code as-is


# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID — edit these lists to taste
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "omega_0": [5.0, 10.0, 15.0],
    "sigma_0": [5.0, 10.0, 15.0],   # envelope bandwidth — couples with omega_0
    "entry_width": [32, 64],
    "n_blocks": [2, 3, 4],
    "lr": [5e-4],                   # kept fixed by default; add e.g. 1e-3 to sweep
}

QUICK_GRID = {
    "omega_0": [5.0, 10.0],
    "sigma_0": [5.0, 10.0],
    "entry_width": [32, 64],
    "n_blocks": [2, 3],
    "lr": [5e-4],
}

# Epoch budgets each config is trained/evaluated at (1500 -> 10000).
EPOCHS_GRID = [1500, 2000, 3000, 4000, 5500, 7000, 8500, 10000]
QUICK_EPOCHS_GRID = [500, 1000]


def make_configs(grid: dict) -> list:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


# ─────────────────────────────────────────────────────────────────────────
# 2. ONE TRAIN + EVAL RUN
# ─────────────────────────────────────────────────────────────────────────
def run_one(cfg: dict, n_epochs: int, n_per_param: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(42)

    omega_0 = cfg["omega_0"]
    output_scale = 1.0 / (omega_0 ** 3)   # same compensation logic as main()

    net = base.WirePrimitiveNet(
        n_params=4,
        n_int_vars=3,
        entry_width=cfg["entry_width"],
        n_blocks=cfg["n_blocks"],
        omega_0=omega_0,
        sigma_0=cfg["sigma_0"],
        output_scale=output_scale,
    )
    n_params_total = sum(p.numel() for p in net.parameters())

    t0 = time.time()
    history, norm_cache = base.train(
        net,
        param_sets=base.PARAM_SETS,
        n_epochs=n_epochs,
        n_per_param=n_per_param,
        lr=cfg["lr"],
        device=device,
        verbose_every=verbose_every if verbose_every > 0 else n_epochs + 1,
    )
    elapsed = time.time() - t0

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
        "omega_0": omega_0,
        "sigma_0": cfg["sigma_0"],
        "entry_width": cfg["entry_width"],
        "n_blocks": cfg["n_blocks"],
        "lr": cfg["lr"],
        "epochs": n_epochs,
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
    parser = argparse.ArgumentParser(description="Hyperparameter + epoch-budget sweep for wire_nni.py")
    parser.add_argument("--epochs-list", type=str, default=None,
                         help="comma-separated epoch budgets to test per config "
                              "(default: 1500,2000,3000,4000,5500,7000,8500,10000; "
                              "500,1000 with --quick)")
    parser.add_argument("--n-per-param", type=int, default=512,
                         help="training points per (a,b,m,n) set per epoch (default 512)")
    parser.add_argument("--quick", action="store_true",
                         help="use a smaller grid for a fast sanity pass")
    parser.add_argument("--verbose-every", type=int, default=0,
                         help="print training progress every N epochs (0 = silent per-epoch)")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path (default: sweep_wire_results_<timestamp>.csv)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda:3" if torch.cuda.is_available() else
        "cpu"
    )
    torch.set_default_dtype(base.FLOATING_POINT_PRECISION)

    grid = QUICK_GRID if args.quick else GRID
    configs = make_configs(grid)

    if args.epochs_list:
        epochs_grid = [int(e.strip()) for e in args.epochs_list.split(",") if e.strip()]
    else:
        epochs_grid = QUICK_EPOCHS_GRID if args.quick else EPOCHS_GRID

    total_runs = len(configs) * len(epochs_grid)

    print(f"\n{'═' * 72}")
    print(f"  SUPER-TEST-M2 (Wire)  —  {len(configs)} configs × {len(epochs_grid)} epoch "
          f"budgets {epochs_grid} = {total_runs} runs, device={device}")
    print(f"{'═' * 72}\n")

    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in base.PARAM_SETS:
        r, e = base.reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    results = []
    run_idx = 0
    for cfg in configs:
        for n_epochs in epochs_grid:
            run_idx += 1
            tag = (f"[{run_idx}/{total_runs}] omega0={cfg['omega_0']} sigma0={cfg['sigma_0']} "
                   f"width={cfg['entry_width']} blocks={cfg['n_blocks']} lr={cfg['lr']} "
                   f"epochs={n_epochs}")
            print(f"{tag} ...", flush=True)
            try:
                res = run_one(cfg, n_epochs, args.n_per_param, device, refs, args.verbose_every)
                results.append(res)
                print(f"    -> min_loss={res['min_loss']:.3e}  "
                      f"mean_rel_err={res['mean_rel_err']:.3e}  "
                      f"min_digits={res['min_correct_digits']}  "
                      f"params={res['n_params_total']:,}  ({res['elapsed_s']:.1f}s)")
            except Exception as exc:
                print(f"    -> FAILED: {exc}")

    if not results:
        print("\nNo successful runs.")
        return

    results.sort(key=lambda r: (-r["min_correct_digits"], r["mean_rel_err"]))

    SEP = "═" * 138
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'omega0':>7} {'sigma0':>7} {'width':>6} {'blocks':>7} {'lr':>9} "
              f"{'epochs':>7} {'params':>9} {'min_loss':>11} {'mean_relerr':>12} {'min_dig':>8} {'time(s)':>8}")
    print(header)
    print("-" * 138)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {r['omega_0']:>7.1f} {r['sigma_0']:>7.1f} {r['entry_width']:>6} "
              f"{r['n_blocks']:>7} {r['lr']:>9.1e} {r['epochs']:>7} {r['n_params_total']:>9,} "
              f"{r['min_loss']:>11.3e} {r['mean_rel_err']:>12.3e} "
              f"{r['min_correct_digits']:>8} {r['elapsed_s']:>8.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    omega_0 = {best['omega_0']}   sigma_0 = {best['sigma_0']}   "
          f"entry_width = {best['entry_width']}   n_blocks = {best['n_blocks']}   lr = {best['lr']}   "
          f"epochs = {best['epochs']}")
    print(f"    -> min {best['min_correct_digits']} correct digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    results_dir = get_mirror_path(__file__, "results")
    out_path = args.out or os.path.join(results_dir, f"sweep_wire_results_m2_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Full results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
