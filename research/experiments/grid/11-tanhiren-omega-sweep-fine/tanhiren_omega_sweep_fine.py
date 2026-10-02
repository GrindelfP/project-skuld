"""
tanhiren_omega_sweep_fine.py — fine-grained omega_0 sweep for the TanhrenIntegrator.

Sweeps omega_0 in {3.0, 4.0, 5.0, 6.0, 7.0} with 3 seeds each (42, 43, 44).
All other hyperparameters are fixed (same as flat test):
  - LR = 5e-4, CosineAnnealingLR
  - Epochs = 8000
  - Hidden sizes = [128, 128, 128]
  - N_per_param = 1024
  - Output scale = 1.0 (learnable)

CSV is updated ONLINE (append + flush after each run) so results survive process kill.

Usage:
    python tanhiren_omega_sweep_fine.py                # full sweep (5 omegas × 3 seeds = 15 runs)
    python tanhiren_omega_sweep_fine.py --quick        # single seed (5 runs)
    python tanhiren_omega_sweep_fine.py --seeds 42 43  # custom seeds
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
_here = os.path.dirname(os.path.abspath(__file__))
_research = os.path.abspath(os.path.join(_here, '../../..'))
sys.path.insert(0, _research)

import torch

from skuld.tanhiren import TanhrenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

try:
    import setproctitle
    setproctitle.setproctitle("tanhiren_omega_sweep_fine")
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID — only omega_0 is swept (fine range around optimum)
# ─────────────────────────────────────────────────────────────────────────
OMEGA_0_GRID = [3.0, 4.0, 5.0, 6.0, 7.0]
DEFAULT_SEEDS = [42, 43, 44]

# Fixed hyperparameters (same as flat test)
HIDDEN = [128, 128, 128]
N_EPOCHS = 8000
N_PER_PARAM = 1024
LR = 5e-4
OUTPUT_SCALE = 1.0


# ─────────────────────────────────────────────────────────────────────────
# 2. ONE TRAIN + EVAL RUN
# ─────────────────────────────────────────────────────────────────────────
def run_one(omega_0: float, seed: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(seed)

    integrator = TanhrenIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=HIDDEN,
        omega_0=omega_0,
        output_scale=OUTPUT_SCALE,
    )
    n_params_total = integrator.n_weights

    t0 = time.time()
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=N_EPOCHS,
        n_per_param=N_PER_PARAM,
        lr=LR,
        device=device,
        verbose_every=verbose_every if verbose_every > 0 else N_EPOCHS + 1,
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
        "omega_0": omega_0,
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
    parser = argparse.ArgumentParser(description="Fine-grained omega_0 sweep for TanhrenIntegrator")
    parser.add_argument("--quick", action="store_true",
                        help="use only seed 42 (5 runs instead of 15)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"seeds to sweep (default: {DEFAULT_SEEDS})")
    parser.add_argument("--verbose-every", type=int, default=0,
                        help="print training progress every N epochs (0 = silent per-epoch)")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV output path (default: sweep_results_<timestamp>.csv)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda:0" if torch.cuda.is_available() else
        "cpu"
    )

    seeds = [42] if args.quick else (args.seeds if args.seeds else DEFAULT_SEEDS)
    runs = list(itertools.product(OMEGA_0_GRID, seeds))

    print(f"\n{'═' * 72}")
    print(f"  TANHIREN FINE OMEGA_0 SWEEP  —  {len(OMEGA_0_GRID)} omegas × {len(seeds)} seeds "
          f"= {len(runs)} runs, device={device}")
    print(f"  omega_0 grid: {OMEGA_0_GRID}")
    print(f"  seeds: {seeds}")
    print(f"  fixed: hidden={HIDDEN}, lr={LR}, epochs={N_EPOCHS}, N={N_PER_PARAM}")
    print(f"{'═' * 72}\n")

    # Reference values depend only on (a,b,m,n) — compute ONCE, reuse everywhere.
    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    # ── ONLINE CSV: open in append mode, write header if new ───────────────
    results_dir = os.path.join(os.path.dirname(__file__), '../../results/grid/11-tanhiren-omega-sweep-fine')
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(results_dir, f"sweep_results_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
    fieldnames = ["omega_0", "seed", "n_params_total", "final_loss", "min_loss",
                  "mean_rel_err", "max_rel_err", "min_correct_digits",
                  "mean_correct_digits", "max_correct_digits", "elapsed_s"]
    file_exists = os.path.exists(out_path)
    csv_fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        csv_fh.flush()
    print(f"  Results CSV (online): {out_path}\n")

    results = []
    for i, (omega_0, seed) in enumerate(runs, 1):
        tag = f"[{i}/{len(runs)}] omega_0={omega_0} seed={seed}"
        print(f"{tag} ...", flush=True)
        try:
            res = run_one(omega_0, seed, device, refs, args.verbose_every)
            results.append(res)
            # ── ONLINE: write row + flush immediately ──────────────────────
            writer.writerow(res)
            csv_fh.flush()
            print(f"    -> min_loss={res['min_loss']:.3e}  "
                  f"mean_rel_err={res['mean_rel_err']:.3e}  "
                  f"min_dig={res['min_correct_digits']}  "
                  f"mean_dig={res['mean_correct_digits']:.1f}  "
                  f"max_dig={res['max_correct_digits']}  "
                  f"({res['elapsed_s']:.1f}s)")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")

    csv_fh.close()

    if not results:
        print("\nNo successful runs.")
        return

    # ── rank: best = highest min_correct_digits, tiebreak by mean_rel_err ─
    results.sort(key=lambda r: (-r["min_correct_digits"], r["mean_rel_err"]))

    SEP = "═" * 110
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'omega_0':>8} {'seed':>5} {'min_loss':>11} "
              f"{'mean_relerr':>12} {'min_dig':>8} {'mean_dig':>9} {'max_dig':>8} {'time(s)':>8}")
    print(header)
    print("-" * 110)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {r['omega_0']:>8.1f} {r['seed']:>5} "
              f"{r['min_loss']:>11.3e} {r['mean_rel_err']:>12.3e} "
              f"{r['min_correct_digits']:>8} {r['mean_correct_digits']:>9.1f} "
              f"{r['max_correct_digits']:>8} "
              f"{r['elapsed_s']:>8.1f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    omega_0 = {best['omega_0']}")
    print(f"    seed    = {best['seed']}")
    print(f"    -> min {best['min_correct_digits']} correct digits, "
          f"mean {best['mean_correct_digits']:.1f} digits, "
          f"max {best['max_correct_digits']} digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}\n")

    # ── per-omega_0 summary (across seeds) ───────────────────────────────
    print(f"{SEP}")
    print(f"  PER-OMEGA_0 SUMMARY (across seeds)")
    print(SEP)
    by_omega = {}
    for r in results:
        by_omega.setdefault(r["omega_0"], []).append(r)
    for omega_0 in sorted(by_omega.keys()):
        rs = by_omega[omega_0]
        min_digits = [r["min_correct_digits"] for r in rs]
        mean_digits = [r["mean_correct_digits"] for r in rs]
        max_digits = [r["max_correct_digits"] for r in rs]
        mean_rel_errs = [r["mean_rel_err"] for r in rs]
        print(f"  omega_0={omega_0:>5.1f}  "
              f"min_dig={min_digits}  "
              f"mean_dig={sum(mean_digits)/len(mean_digits):.1f}  "
              f"max_dig={max_digits}  "
              f"mean_rel_err={sum(mean_rel_errs)/len(mean_rel_errs):.3e}  "
              f"n={len(rs)}")
    print(SEP)
    print(f"  Full results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
