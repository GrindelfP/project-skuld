"""
merge_results.py — combine per-GPU CSVs of the CONV m1.2 sweep into a
single ranked summary.

Reads all results_gpu*.csv files from the results directory, merges them,
and prints a ranked summary. Also writes a combined CSV.

Usage:
    python merge_results.py
"""
import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path


CSV_FIELDS = [
    "hidden_sizes", "lr", "n_epochs", "n_per_param", "seed",
    "n_params_total", "final_loss", "min_loss", "mean_rel_err", "max_rel_err",
    "I_1_dig", "I_2_dig", "I_3_dig", "I_4_dig",
    "I_5_dig", "I_6_dig", "I_7_dig", "I_8_dig",
    "min_dig", "mean_dig", "max_dig", "elapsed_s",
]


def main():
    results_dir = get_mirror_path(__file__, "results")
    pattern = os.path.join(results_dir, "results_gpu*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No per-GPU CSVs found in {results_dir}")
        return

    print(f"\n{'=' * 72}")
    print(f"  MERGING {len(csv_files)} per-GPU CSV files")
    for f in csv_files:
        print(f"    {os.path.basename(f)}")
    print(f"{'=' * 72}\n")

    # ── read all results ──────────────────────────────────────────────────
    all_results = []
    for csv_file in csv_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                all_results.append(row)

    if not all_results:
        print("No results found in any CSV.")
        return

    # ── deduplicate (in case of overlapping blocks) ──────────────────────
    seen = set()
    unique_results = []
    for r in all_results:
        key = (r["hidden_sizes"], r["lr"],
               r["n_epochs"], r["n_per_param"], r["seed"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    print(f"  Total runs: {len(all_results)} (deduplicated: {len(unique_results)})\n")

    # ── sort: best = highest min_dig, tiebreak by mean_rel_err ───────────
    unique_results.sort(key=lambda r: (-int(r["min_dig"]), float(r["mean_rel_err"]), int(r["n_epochs"])))

    # ── ranked table ──────────────────────────────────────────────────────
    SEP = "=" * 140
    print(SEP)
    print("  RANKED RESULTS (best first)")
    print(SEP)
    header = (f"  {'#':>3} {'hidden':<20} {'lr':>8} {'epochs':>6} "
              f"{'npp':>5} {'seed':>5} {'min_loss':>10} {'mean_relerr':>11} "
              f"{'I1':>3} {'I2':>3} {'I3':>3} {'I4':>3} {'I5':>3} {'I6':>3} {'I7':>3} {'I8':>3} "
              f"{'min':>4} {'mean':>5} {'max':>4} {'time':>6}")
    print(header)
    print("-" * 140)
    for i, r in enumerate(unique_results, 1):
        print(f"  {i:>3} {r['hidden_sizes']:<20} {float(r['lr']):>8.1e} "
              f"{int(r['n_epochs']):>6} "
              f"{int(r['n_per_param']):>5} {int(r['seed']):>5} "
              f"{float(r['min_loss']):>10.3e} {float(r['mean_rel_err']):>11.3e} "
              f"{int(r['I_1_dig']):>3} {int(r['I_2_dig']):>3} {int(r['I_3_dig']):>3} "
              f"{int(r['I_4_dig']):>3} {int(r['I_5_dig']):>3} {int(r['I_6_dig']):>3} "
              f"{int(r['I_7_dig']):>3} {int(r['I_8_dig']):>3} "
              f"{int(r['min_dig']):>4} {float(r['mean_dig']):>5.1f} "
              f"{int(r['max_dig']):>4} {float(r['elapsed_s']):>6.1f}")
    print(SEP)

    best = unique_results[0]
    print(f"\n  BEST CONFIG:")
    print(f"    hidden_sizes = {best['hidden_sizes']}")
    print(f"    lr           = {best['lr']}")
    print(f"    n_epochs     = {best['n_epochs']}")
    print(f"    n_per_param  = {best['n_per_param']}")
    print(f"    seed         = {best['seed']}")
    print(f"    -> min {best['min_dig']} digits, "
          f"mean {float(best['mean_dig']):.1f} digits, "
          f"max {best['max_dig']} digits, "
          f"mean rel. err {best['mean_rel_err']}\n")

    # ── per-config seed-averaged summary ─────────────────────────────────
    print(SEP)
    print("  PER-CONFIG SUMMARY (seed-averaged)")
    print(SEP)
    by_cfg = {}
    for r in unique_results:
        key = (r["hidden_sizes"], r["lr"], r["n_epochs"], r["n_per_param"])
        by_cfg.setdefault(key, []).append(r)
    for key, rs in sorted(by_cfg.items(),
                          key=lambda kv: (-min(int(r["min_dig"]) for r in kv[1]),
                                          sum(float(r["mean_rel_err"]) for r in kv[1]) / len(kv[1]))):
        hs, lr, ep, npp = key
        min_d = min(int(r["min_dig"]) for r in rs)
        mean_d = sum(float(r["mean_dig"]) for r in rs) / len(rs)
        max_d = max(int(r["max_dig"]) for r in rs)
        mean_rel = sum(float(r["mean_rel_err"]) for r in rs) / len(rs)
        print(f"  hidden={hs} lr={lr} epochs={ep} npp={npp}")
        print(f"      min_dig={min_d}  mean_dig={mean_d:.1f}  max_dig={max_d}  "
              f"mean_relerr={mean_rel:.3e}  n_seeds={len(rs)}")
    print(SEP)

    # ── write combined CSV ───────────────────────────────────────────────
    combined_path = os.path.join(results_dir, "conv_sweep_m12_combined.csv")
    with open(combined_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in unique_results:
            writer.writerow(r)
    print(f"  Combined results saved -> {combined_path}\n")


if __name__ == "__main__":
    main()
