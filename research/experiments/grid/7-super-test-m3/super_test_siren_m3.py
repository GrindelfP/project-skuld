"""
super_test_siren_m3.py — refined hyperparameter × seed × epoch-budget sweep
for siren_nni.py, aimed at breaking past the 4-correct-digit plateau seen in m2.

WHAT CHANGED FROM m2 (and why), based on sweep_results_m2_2026-09-25_21-05.csv:

  1. NARROWER, DEEPER GRID.
     Every m2 run that reached min_correct_digits=4 sat in a tight region:
     hidden_sizes in {[128,128,128], [128,128,128,128]}, omega_0 in {30, 45},
     lr = 2e-4, n_epochs >= 2000 (loss basically plateaus after ~3000-4500).
     lr=1e-3 and the [256,256] shape never got past 3 digits and were often
     WORSE than [128,128,128] despite more params -> depth/omega_0 matter
     more than width here. m3 grid is centered on the winning region instead
     of scanning the whole space blind, freeing up budget to go deeper on
     seeds/lr instead of width.

  2. MULTI-SEED BEST-OF-K PER CONFIG.
     At the plateau (mean_rel_err ~1-3e-4), the gap between 4 and 5 correct
     digits is smaller than the run-to-run noise from initialization alone
     (m2 shows configs with IDENTICAL hyperparameters landing at both 3 and
     4 digits depending on epoch count, i.e. noisy). m3 trains N_SEEDS
     independent inits per (config, epoch_budget) and reports both the best
     and the seed-median, so we can tell signal from init-noise.

  3. COSINE LR DECAY (optional, auto-detected).
     Fixed lr flatlines the loss curve well before the epoch budget is used
     up (m2's 10000-epoch runs are barely better than 6500-epoch runs at
     the same config -- wasted compute). A cosine decay from lr -> lr/50
     lets the optimizer keep taking smaller steps late in training instead
     of bouncing around the minimum at a fixed step size. This is applied
     ONLY if siren_nni.base.train accepts a `lr_schedule` or `scheduler`
     kwarg; m3 detects this via introspection and silently falls back to
     fixed-lr (m2 behavior) if not supported -- check the printed
     "[m3] LR schedule: ..." line at startup to see which mode ran.

  4. n_per_param SWEEP (512 vs 1024).
     m2 fixed n_per_param=512. If the error floor is a data-noise floor
     rather than an optimization floor, doubling samples/param should show
     up as a real digit gain; if it doesn't help, that tells you the floor
     is optimization/precision-bound, not data-bound. Kept as a real grid
     axis (not fixed) so the ranked table answers this directly.

  5. FINER omega_0 STEP.
     m2 only tried {15, 30, 45}. Since 30 and 45 both won and 15 never did,
     m3 adds 37.5 and 52.5 to bracket the winning pair more tightly.

Place this file in the SAME directory as siren_nni.py (same as m2).

Usage:
    python super_test_siren_m3.py                # full refined grid
    python super_test_siren_m3.py --quick         # small sanity pass
    python super_test_siren_m3.py --seeds 5       # override seed count
    python super_test_siren_m3.py --epochs 3000 6500 12000
"""

import argparse
import csv
import inspect
import itertools
import math
import statistics
import time
from datetime import datetime

import torch

import siren_nni as base  # reuse model / training / integral code as-is


# ─────────────────────────────────────────────────────────────────────────
# 1. REFINED HYPERPARAMETER GRID — centered on the m2 winning region
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "hidden_sizes": [
        [128, 128, 128],
        [128, 128, 128, 128],
        [128, 128, 128, 128, 128],   # one layer deeper than m2's best
        [160, 160, 160, 160],        # slightly wider than 128, still 4 deep
    ],
    "omega_0": [30.0, 37.5, 45.0, 52.5],   # bracket the m2 winners more finely
    "lr": [1e-4, 1.5e-4, 2e-4, 3e-4],       # centered on m2's best (2e-4)
    "n_per_param": [512, 1024],             # is the floor data-bound?
}

QUICK_GRID = {
    "hidden_sizes": [[128, 128, 128], [128, 128, 128, 128]],
    "omega_0": [30.0, 45.0],
    "lr": [2e-4, 3e-4],
    "n_per_param": [512],
}

# m2 showed diminishing returns past ~4500-6500 epochs at fixed lr; with
# cosine decay we push the ceiling higher since late-training steps get
# smaller (less likely to just be wasted compute / risk of divergence).
EPOCHS_GRID = [3000, 4500, 6500, 10000, 15000]
QUICK_EPOCHS_GRID = [3000, 6500]

N_SEEDS_DEFAULT = 3
QUICK_N_SEEDS = 2

# Cosine decay floor: lr_end = lr / LR_DECAY_FACTOR
LR_DECAY_FACTOR = 50.0


def make_configs(grid: dict) -> list:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


def _train_supports_kwarg(name: str) -> bool:
    try:
        sig = inspect.signature(base.train)
        return name in sig.parameters
    except (TypeError, ValueError):
        return False


# Detect once whether base.train can take a scheduler / lr_schedule hook.
# We try a couple of common kwarg spellings so this works whether
# siren_nni.py calls it `lr_schedule`, `scheduler`, or `scheduler_fn`.
_SCHEDULE_KWARG = next(
    (k for k in ("lr_schedule", "scheduler", "scheduler_fn") if _train_supports_kwarg(k)),
    None,
)


def _cosine_schedule_factory(lr: float, n_epochs: int):
    """Returns a callable epoch -> lr implementing cosine decay lr -> lr/LR_DECAY_FACTOR."""
    lr_end = lr / LR_DECAY_FACTOR

    def schedule(epoch: int) -> float:
        t = min(epoch, n_epochs) / max(n_epochs, 1)
        cos_factor = 0.5 * (1 + math.cos(math.pi * t))
        return lr_end + (lr - lr_end) * cos_factor

    return schedule


# ─────────────────────────────────────────────────────────────────────────
# 2. ONE TRAIN + EVAL RUN (single seed)
# ─────────────────────────────────────────────────────────────────────────
def run_one(cfg: dict, n_epochs: int, seed: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(seed)

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

    train_kwargs = dict(
        param_sets=base.PARAM_SETS,
        n_epochs=n_epochs,
        n_per_param=cfg["n_per_param"],
        lr=cfg["lr"],
        device=device,
        verbose_every=verbose_every if verbose_every > 0 else n_epochs + 1,
    )
    if _SCHEDULE_KWARG is not None:
        train_kwargs[_SCHEDULE_KWARG] = _cosine_schedule_factory(cfg["lr"], n_epochs)

    t0 = time.time()
    history, norm_cache = base.train(net, **train_kwargs)
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
    """Runs one (cfg, n_epochs) at each seed, returns list of result dicts
    plus a synthetic 'best' and 'median' summary row for this group."""
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
        description="Refined multi-seed hyperparameter x epoch-budget sweep for siren_nni.py (m3)")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                         help=f"epoch budgets to sweep (default: {EPOCHS_GRID}, "
                              f"or {QUICK_EPOCHS_GRID} with --quick)")
    parser.add_argument("--seeds", type=int, default=None,
                         help=f"number of seeds per (config, epoch_budget) "
                              f"(default: {N_SEEDS_DEFAULT}, or {QUICK_N_SEEDS} with --quick)")
    parser.add_argument("--quick", action="store_true",
                         help="small grid, short epoch ladder, fewer seeds -- sanity pass")
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
    torch.set_default_dtype(base.FLOATING_POINT_PRECISION)

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
    print(f"  LR schedule: "
          f"{'cosine decay via base.train(' + _SCHEDULE_KWARG + '=...)' if _SCHEDULE_KWARG else 'NOT SUPPORTED by siren_nni.train -- falling back to fixed lr (m2 behavior)'}")
    print(f"{'═' * 72}\n")

    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in base.PARAM_SETS:
        r, e = base.reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    all_seed_results = []   # every individual (config, epochs, seed) run
    group_summaries = []    # one row per (config, epochs) group: best + median

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

    # Rank groups: best min_correct_digits (from best seed), tiebreak by
    # best_mean_rel_err, then by median (reward configs that are reliably
    # good, not just lucky once), then cheaper (fewer epochs).
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
        print("    Target of 5+ digits NOT reached this run -- see notes at bottom of file "
              "for what to try next.")

    # ── n_per_param comparison: is the floor data-bound? ────────────────
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

    # ── save CSVs ────────────────────────────────────────────────────────
    out_path = args.out or f"sweep_results_m3_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
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

    # ── notes for next iteration if target not hit ──────────────────────
    if best["best_min_correct_digits"] < 5:
        print(f"{SEP}")
        print("  IF STILL STUCK AT 4 DIGITS, TRY NEXT (m4 candidates):")
        print(SEP)
        print("  - Check float precision: base.FLOATING_POINT_PRECISION -- if it's float32,")
        print("    switching the whole pipeline (incl. scipy refs) to float64 may be required")
        print("    to even represent a 5th correct digit reliably.")
        print("  - Look at max_rel_err vs mean_rel_err in the raw CSV: if a single (a,b,m,n)")
        print("    param set is dragging min_correct_digits down, it may need its own")
        print("    architecture (e.g. per-param-set output scale) rather than a shared net.")
        print("  - Try weight decay / gradient clipping if base.train supports it -- cosine")
        print("    decay alone may not be enough if late training is noisy rather than flat.")
        print("  - Consider an ensemble (average predictions of the top-K seeds) instead of")
        print("    best-of-K -- averaging independent-noise models often buys a digit for free.")
        print(SEP)


if __name__ == "__main__":
    main()
