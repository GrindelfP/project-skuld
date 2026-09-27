"""
super_test_siren_m3.py — refined hyperparameter × seed × epoch-budget sweep
for SirenIntegrator, aimed at breaking past the 4-correct-digit plateau seen in m2.

Usage:
    python super_test_siren_m3.py                # full refined grid
    python super_test_siren_m3.py --quick         # small sanity pass
    python super_test_siren_m3.py --seeds 5       # override seed count
    python super_test_siren_m3.py --epochs 3000 6500 12000
"""
import argparse
import csv
import itertools
import math
import statistics
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../flat/5-siren'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../skuld-lib'))

import torch

from skuld.siren import SirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

# ─────────────────────────────────────────────────────────────────────────
# 1. REFINED HYPERPARAMETER GRID — centered on the m2 winning region
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "hidden_sizes": [
        [128, 128, 128],
        [128, 128, 128, 128],
        [128, 128, 128, 128, 128],
        [160, 160, 160, 160],
    ],
    "omega_0": [30.0, 37.5, 45.0, 52.5],
    "lr": [1e-4, 1.5e-4, 2e-4, 3e-4],
    "n_per_param": [512, 1024],
}

QUICK_GRID = {
    "hidden_sizes": [[128, 128, 128], [128, 128, 128, 128]],
    "omega_0": [30.0, 45.0],
    "lr": [2e-4, 3e-4],
    "n_per_param": [512],
}

EPOCHS_GRID = [3000, 4500, 6500, 10000, 15000]
QUICK_EPOCHS_GRID = [3000, 6500]

N_SEEDS_DEFAULT = 3
QUICK_N_SEEDS = 2


def make_configs(grid: dict) -> list:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


# ─────────────────────────────────────────────────────────────────────────
# 2. ONE TRAIN + EVAL RUN (single seed)
# ─────────────────────────────────────────────────────────────────────────
def run_one(cfg: dict, n_epochs: int, seed: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(seed)

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
        "seed": seed,
        "n_params_total": n_params_total,
        "final_loss": history[-1],
        "min_loss": min(history),
        "mean_rel_err": sum(rel_errs) / len(rel_errs),
        "max_rel_err": max(rel_errs),
        "min_correct_digits": min(digits),
        "mean_correct_digits": sum(digits) / len(digits),
        "elapsed_s": elapsed,
    }


def run_config_all_seeds(cfg, n_epochs, seeds, device, refs, verbose_every):
    """Runs one (cfg, n_epochs) at each seed, returns list of result dicts."""
    seed_results = []
    for seed in seeds:
        try:
            seed_results.append(run_one(cfg, n_epochs, seed, device, refs, verbose_every))
        except Exception as exc:
            print(f"    seed={seed} -> FAILED: {exc}")
    return seed_results


# ─────────────────────────────────────────────────────────────────────────
# 3. MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Refined multi-seed hyperparameter x epoch-budget sweep for SirenIntegrator (m3)")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                         help=f"epoch budgets to sweep (default: {EPOCHS_GRID}, "
                              f"or {QUICK_EPOCHS_GRID} with --quick)")
    parser.add_argument("--seeds", type=int, default=None,
                         help=f"number of seeds per (config, epoch_budget) "
                              f"(default: {N_SEEDS_DEFAULT}, or {QUICK_N_SEEDS} with --quick)")
    parser.add_argument("--quick", action="store_true",
                         help="small grid, short epoch ladder, fewer seeds")
    parser.add_argument("--verbose-every", type=int, default=0,
                         help="print training progress every N epochs (0 = silent per-epoch)")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path (default: sweep_results_m3_<timestamp>.csv)")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda:0" if torch.cuda.is_available() else
        "cpu"
    )

    grid = QUICK_GRID if args.quick else GRID
    epochs_grid = args.epochs if args.epochs is not None else (
        QUICK_EPOCHS_GRID if args.quick else EPOCHS_GRID
    )
    n_seeds = args.seeds if args.seeds is not None else (
        QUICK_N_SEEDS if args.quick else N_SEEDS_DEFAULT
    )
    seeds = [42 + i for i in range(n_seeds)]

    configs = make_configs(grid)
    runs = list(itertools.product(configs, epochs_grid))

    print(f"\n{'═' * 72}")
    print(f"  SUPER-TEST m3  —  {len(configs)} configs × {len(epochs_grid)} epoch "
          f"budgets {epochs_grid} × {n_seeds} seeds = {len(runs) * n_seeds} runs, "
          f"device={device}")
    print(f"{'═' * 72}\n")

    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    all_seed_results = []
    group_summaries = []

    total_groups = len(runs)
    for gi, (cfg, n_epochs) in enumerate(runs, 1):
        tag = (f"[group {gi}/{total_groups}] hidden={cfg['hidden_sizes']} "
               f"omega0={cfg['omega_0']} lr={cfg['lr']} N={cfg['n_per_param']} "
               f"epochs={n_epochs}  ({n_seeds} seeds)")
        print(f"{tag}", flush=True)

        seed_results = run_config_all_seeds(cfg, n_epochs, seeds, device, refs, args.verbose_every)
        if not seed_results:
            print("    -> ALL SEEDS FAILED, skipping group")
            continue

        all_seed_results.extend(seed_results)

        digits_list = [r["min_correct_digits"] for r in seed_results]
        relerr_list = [r["mean_rel_err"] for r in seed_results]
        best = min(seed_results, key=lambda r: (r["mean_rel_err"],))
        median_digits = statistics.median(digits_list)
        total_elapsed = sum(r["elapsed_s"] for r in seed_results)

        print(f"    -> digits per seed={digits_list}  "
              f"best_mean_relerr={best['mean_rel_err']:.3e}  "
              f"median_digits={median_digits}  ({total_elapsed:.1f}s total)")

        group_summaries.append({
            "hidden_sizes": cfg["hidden_sizes"],
            "omega_0": cfg["omega_0"],
            "lr": cfg["lr"],
            "n_per_param": cfg["n_per_param"],
            "n_epochs": n_epochs,
            "n_seeds_run": len(seed_results),
            "n_params_total": seed_results[0]["n_params_total"],
            "best_seed": best["seed"],
            "best_min_correct_digits": best["min_correct_digits"],
            "best_mean_rel_err": best["mean_rel_err"],
            "best_max_rel_err": best["max_rel_err"],
            "median_min_correct_digits": median_digits,
            "worst_min_correct_digits": min(digits_list),
            "total_elapsed_s": total_elapsed,
        })

    if not group_summaries:
        print("\nNo successful groups.")
        return

    # Rank groups
    group_summaries.sort(key=lambda r: (
        -r["best_min_correct_digits"],
        r["best_mean_rel_err"],
        -r["median_min_correct_digits"],
        r["n_epochs"],
    ))

    SEP = "═" * 148
    print(f"\n{SEP}")
    print("  RANKED GROUPS (best first) -- 'best' = best of N seeds, 'median' = reliability check")
    print(SEP)
    header = (f"  {'#':>3} {'hidden':<24} {'omega0':>7} {'lr':>9} {'N/param':>8} {'epochs':>7} "
              f"{'seeds':>6} {'best_dig':>9} {'med_dig':>8} {'worst_dig':>10} "
              f"{'best_relerr':>12} {'time(s)':>9}")
    print(header)
    print("-" * 148)
    for i, r in enumerate(group_summaries, 1):
        print(f"  {i:>3} {str(r['hidden_sizes']):<24} {r['omega_0']:>7.1f} {r['lr']:>9.1e} "
              f"{r['n_per_param']:>8} {r['n_epochs']:>7} {r['n_seeds_run']:>6} "
              f"{r['best_min_correct_digits']:>9} {r['median_min_correct_digits']:>8} "
              f"{r['worst_min_correct_digits']:>10} {r['best_mean_rel_err']:>12.3e} "
              f"{r['total_elapsed_s']:>9.1f}")
    print(SEP)

    best = group_summaries[0]
    print(f"\n  BEST GROUP:")
    print(f"    hidden_sizes = {best['hidden_sizes']}")
    print(f"    omega_0      = {best['omega_0']}")
    print(f"    lr           = {best['lr']}")
    print(f"    n_per_param  = {best['n_per_param']}")
    print(f"    n_epochs     = {best['n_epochs']}")
    print(f"    -> best-of-{best['n_seeds_run']} seeds: min {best['best_min_correct_digits']} "
          f"correct digits, mean rel. err {best['best_mean_rel_err']:.3e} "
          f"(median across seeds: {best['median_min_correct_digits']} digits)")
    if best["best_min_correct_digits"] >= 5:
        print("    *** 5+ digit target reached ***")
    else:
        print("    Target of 5+ digits NOT reached this run.")

    # ── n_per_param comparison ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  n_per_param COMPARISON (same hidden/omega0/lr/epochs, different sample count)")
    print(SEP)
    by_key_npp = {}
    for r in group_summaries:
        key = (str(r["hidden_sizes"]), r["omega_0"], r["lr"], r["n_epochs"])
        by_key_npp.setdefault(key, {})[r["n_per_param"]] = r
    shown = False
    for key, by_npp in by_key_npp.items():
        if len(by_npp) < 2:
            continue
        shown = True
        hs, om, lr, ep = key
        print(f"  hidden={hs} omega0={om} lr={lr} epochs={ep}")
        for npp, r in sorted(by_npp.items()):
            print(f"      n_per_param={npp:>5}  best_dig={r['best_min_correct_digits']}  "
                  f"best_relerr={r['best_mean_rel_err']:.3e}")
    if not shown:
        print("  (no directly comparable pairs found)")
    print(SEP)

    # ── save CSVs to mirroring results directory ─────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), '../../results/grid/7-super-test-m3')
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(results_dir, f"sweep_results_m3_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(group_summaries[0].keys()))
        writer.writeheader()
        for r in group_summaries:
            writer.writerow(r)
    print(f"\n  Group summary saved -> {out_path}")

    raw_path = out_path.replace(".csv", "_raw_seeds.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_seed_results[0].keys()))
        writer.writeheader()
        for r in all_seed_results:
            writer.writerow(r)
    print(f"  Raw per-seed results saved -> {raw_path}\n")


if __name__ == "__main__":
    main()
