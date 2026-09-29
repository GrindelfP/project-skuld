"""
Bethe–Salpeter_KAN_m3.py — fine hyperparameter × epoch-budget sweep for KANIntegrator.

m3 focuses on a NARROWER region around the m2 best config with FINER steps:
  m2 best: width=32, depth=3, G=8, lr=5e-4 → 3.125 mean correct digits
  m3 grid: width=[24,32,40], depth=[2,3,4], G=[6,8,10], k=[3], lr=[4e-4,5e-4,6e-4]
  epoch ladder extended to 15000 (m2 capped at 10000).

CSV is updated ONLINE (append + flush after each run) so results survive process kill.

Usage:
    python Bethe–Salpeter_KAN_m3.py                # full grid × full epoch ladder
    python Bethe–Salpeter_KAN_m3.py --quick         # smaller grid + shorter ladder
    python Bethe–Salpeter_KAN_m3.py --epochs 3000 5000 10000   # custom epoch ladder
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

import torch

from skuld.kan import KANIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

try:
    import setproctitle
    setproctitle.setproctitle("Bethe–Salpeter_KAN_m3")
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID — fine grid around m2 best
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "width": [24, 32, 40],
    "depth": [2, 3, 4],
    "G": [6, 8, 10],              # B-spline grid size
    "k": [3],                     # cubic splines
    "lr": [4e-4, 5e-4, 6e-4],
}

QUICK_GRID = {
    "width": [24, 32],
    "depth": [2, 3],
    "G": [6, 8],
    "k": [3],
    "lr": [4e-4, 5e-4],
}

# Epoch ladder — extended to 15000 (m2 capped at 10000)
EPOCHS_GRID = [2000, 3000, 4000, 5000, 7000, 10000, 15000]
QUICK_EPOCHS_GRID = [2000, 5000, 10000]


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

    integrator = KANIntegrator(
        n_params=4,
        n_int_vars=3,
        width=cfg["width"],
        depth=cfg["depth"],
        G=cfg["G"],
        k=cfg["k"],
        output_scale=1.0,
    )
    n_params_total = integrator.n_weights

    t0 = time.time()
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=n_epochs,
        n_per_param=512,
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
        "width": cfg["width"],
        "depth": cfg["depth"],
        "G": cfg["G"],
        "k": cfg["k"],
        "lr": cfg["lr"],
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
    parser = argparse.ArgumentParser(description="Fine hyperparameter × epoch-budget sweep for KANIntegrator (m3)")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                         help=f"epoch budgets to sweep, space-separated "
                               f"(default: {EPOCHS_GRID}, or {QUICK_EPOCHS_GRID} with --quick)")
    parser.add_argument("--quick", action="store_true",
                         help="use a smaller hyperparameter grid AND a shorter epoch ladder")
    parser.add_argument("--verbose-every", type=int, default=0,
                         help="print training progress every N epochs (0 = silent per-epoch)")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path (default: sweep_results_m3_<timestamp>.csv)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda:6" if torch.cuda.is_available() else
        "cpu"
    )

    grid = QUICK_GRID if args.quick else GRID
    epochs_grid = args.epochs if args.epochs is not None else (
        QUICK_EPOCHS_GRID if args.quick else EPOCHS_GRID
    )
    configs = make_configs(grid)
    runs = list(itertools.product(configs, epochs_grid))

    print(f"\n{'═' * 72}")
    print(f"  BETHE–SALPETER KAN m3  —  {len(configs)} configs × {len(epochs_grid)} epoch "
          f"budgets {epochs_grid} = {len(runs)} runs, device={device}")
    print(f"{'═' * 72}\n")

    # Reference values depend only on (a,b,m,n) — compute ONCE, reuse everywhere.
    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    # ── ONLINE CSV: open in append mode, write header if new ───────────────
    results_dir = os.path.join(os.path.dirname(__file__), '../../results/grid/7-siren-wire-kan-sweep-m3')
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(results_dir, f"sweep_results_m3_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
    fieldnames = ["width", "depth", "G", "k", "lr", "n_epochs",
                  "n_params_total", "final_loss", "min_loss", "mean_rel_err",
                  "max_rel_err", "min_correct_digits", "mean_correct_digits", "elapsed_s"]
    file_exists = os.path.exists(out_path)
    csv_fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        csv_fh.flush()
    print(f"  Results CSV (online): {out_path}\n")

    results = []
    for i, (cfg, n_epochs) in enumerate(runs, 1):
        tag = (f"[{i}/{len(runs)}] width={cfg['width']} depth={cfg['depth']} "
               f"G={cfg['G']} k={cfg['k']} lr={cfg['lr']} epochs={n_epochs}")
        print(f"{tag} ...", flush=True)
        try:
            res = run_one(cfg, n_epochs, device, refs, args.verbose_every)
            results.append(res)
            # ── ONLINE: write row + flush immediately ──────────────────────
            writer.writerow(res)
            csv_fh.flush()
            print(f"    -> min_loss={res['min_loss']:.3e}  "
                  f"mean_rel_err={res['mean_rel_err']:.3e}  "
                  f"min_digits={res['min_correct_digits']}  "
                  f"params={res['n_params_total']:,}  ({res['elapsed_s']:.1f}s)")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")

    csv_fh.close()

    if not results:
        print("\nNo successful runs.")
        return

    # ── rank: best = highest min_correct_digits, tiebreak by mean_rel_err,
    #    tiebreak by fewer epochs ─────────────────────────────────────────
    results.sort(key=lambda r: (-r["min_correct_digits"], r["mean_rel_err"], r["n_epochs"]))

    SEP = "═" * 132
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'width':>6} {'depth':>6} {'G':>4} {'k':>3} {'lr':>9} {'epochs':>7} "
              f"{'params':>9} {'min_loss':>11} {'mean_relerr':>12} {'min_dig':>8} {'time(s)':>8}")
    print(header)
    print("-" * 132)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {r['width']:>6} {r['depth']:>6} {r['G']:>4} {r['k']:>3} {r['lr']:>9.1e} "
              f"{r['n_epochs']:>7} {r['n_params_total']:>9,} {r['min_loss']:>11.3e} "
              f"{r['mean_rel_err']:>12.3e} {r['min_correct_digits']:>8} {r['elapsed_s']:>8.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    width = {best['width']}   depth = {best['depth']}   "
          f"G = {best['G']}   k = {best['k']}   lr = {best['lr']}   epochs = {best['n_epochs']}")
    print(f"    -> min {best['min_correct_digits']} correct digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    # ── per-config epoch-scan ────────────────────────────────────────────
    print(f"{SEP}")
    print(f"  ACCURACY vs EPOCHS, per hyperparameter config")
    print(SEP)
    by_cfg = {}
    for r in results:
        key = (r["width"], r["depth"], r["G"], r["k"], r["lr"])
        by_cfg.setdefault(key, []).append(r)
    for key, rs in by_cfg.items():
        rs_sorted = sorted(rs, key=lambda r: r["n_epochs"])
        w, d, G, k, lr = key
        print(f"  width={w} depth={d} G={G} k={k} lr={lr}")
        for r in rs_sorted:
            print(f"      epochs={r['n_epochs']:>6}  min_dig={r['min_correct_digits']:>2}  "
                  f"mean_relerr={r['mean_rel_err']:.3e}  time={r['elapsed_s']:.1f}s")
    print(SEP)
    print(f"  Full results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
