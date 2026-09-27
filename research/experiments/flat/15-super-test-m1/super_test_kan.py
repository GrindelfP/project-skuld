"""
super_test_kan.py — hyperparameter sweep for kan_nni.py

Same idea as super_test.py, but the grid is KAN-specific: width, depth,
B-spline grid size G, spline degree k, and lr (omega_0 doesn't exist here).

Usage:
    python super_test_kan.py                      # default grid, 1500 epochs/config
    python super_test_kan.py --epochs 3000
    python super_test_kan.py --quick --epochs 500  # small grid, fast sanity pass

Place this file in the SAME directory as kan_nni.py.
"""

import argparse
import csv
import itertools
import math
import time
from datetime import datetime

import torch

import kan_nni as base  # reuse model / training / integral code as-is


# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID — edit these lists to taste
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "width": [16, 32, 64],
    "depth": [1, 2, 3],
    "G": [5, 8, 12],
    "k": [3],              # cubic splines; add 2 or 4 to sweep smoothness too
    "lr": [2e-3, 1e-3, 5e-4],
}

QUICK_GRID = {
    "width": [16, 32],
    "depth": [1, 2],
    "G": [5, 8],
    "k": [3],
    "lr": [2e-3, 1e-3],
}


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

    net = base.KANPrimitiveNet(
        n_params=4,
        n_int_vars=3,
        width=cfg["width"],
        depth=cfg["depth"],
        G=cfg["G"],
        k=cfg["k"],
        output_scale=1.0,
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
        "width": cfg["width"],
        "depth": cfg["depth"],
        "G": cfg["G"],
        "k": cfg["k"],
        "lr": cfg["lr"],
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
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for kan_nni.py")
    parser.add_argument("--epochs", type=int, default=1500,
                         help="epochs per config (default 1500; full run in kan_nni.py uses 8000)")
    parser.add_argument("--n-per-param", type=int, default=512,
                         help="training points per (a,b,m,n) set per epoch (default 512)")
    parser.add_argument("--quick", action="store_true",
                         help="use a smaller grid for a fast sanity pass")
    parser.add_argument("--verbose-every", type=int, default=0,
                         help="print training progress every N epochs (0 = silent per-epoch)")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path (default: sweep_kan_results_<timestamp>.csv)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda:0" if torch.cuda.is_available() else
        "cpu"
    )
    torch.set_default_dtype(base.FLOATING_POINT_PRECISION)

    grid = QUICK_GRID if args.quick else GRID
    configs = make_configs(grid)

    print(f"\n{'═' * 72}")
    print(f"  SUPER-TEST (KAN)  —  {len(configs)} configs × {args.epochs} epochs, device={device}")
    print(f"{'═' * 72}\n")

    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in base.PARAM_SETS:
        r, e = base.reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    results = []
    for i, cfg in enumerate(configs, 1):
        tag = (f"[{i}/{len(configs)}] width={cfg['width']} depth={cfg['depth']} "
               f"G={cfg['G']} k={cfg['k']} lr={cfg['lr']}")
        print(f"{tag} ...", flush=True)
        try:
            res = run_one(cfg, args.epochs, args.n_per_param, device, refs, args.verbose_every)
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

    SEP = "═" * 122
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'width':>6} {'depth':>6} {'G':>4} {'k':>3} {'lr':>9} "
              f"{'params':>9} {'min_loss':>11} {'mean_relerr':>12} {'min_dig':>8} {'time(s)':>8}")
    print(header)
    print("-" * 122)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {r['width']:>6} {r['depth']:>6} {r['G']:>4} {r['k']:>3} {r['lr']:>9.1e} "
              f"{r['n_params_total']:>9,} {r['min_loss']:>11.3e} "
              f"{r['mean_rel_err']:>12.3e} {r['min_correct_digits']:>8} {r['elapsed_s']:>8.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    width = {best['width']}   depth = {best['depth']}   "
          f"G = {best['G']}   k = {best['k']}   lr = {best['lr']}")
    print(f"    -> min {best['min_correct_digits']} correct digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    out_path = args.out or f"sweep_kan_results_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Full results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
