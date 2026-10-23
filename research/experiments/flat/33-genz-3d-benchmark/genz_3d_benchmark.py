"""
genz_3d_benchmark.py — all seven architectures on the 3-D Genz benchmark.

Trains each architecture at its best known configuration on all five Genz
functions and scores the corner-sum estimate against each function's closed-form
exact integral. Every cell records absolute error, relative error and correct
digits.

The functions use the **tuned** difficulty tier (shape parameter a = 2.0), chosen
by calibration against the meson Feynman target, which reaches ~5 correct digits.
The canonical Genz1984 tier (a = 5) puts every architecture at ~2.7 digits and
compresses the field until it stops discriminating between architectures.

This is a characterisation benchmark: synthetic functions with analytically
known integrals. It is complementary to the meson Feynman target benchmark and
says nothing about performance there.

Independent of the 6-D companion script (flat/34); the two can run concurrently.

Run:
    python research/experiments/flat/33-genz-3d-benchmark/genz_3d_benchmark.py \\
        &> research/results/flat/33-genz-3d-benchmark/genz_3d_benchmark.log &
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # research/
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.genz_runner import BUILDERS, quadrature_reference, run_architecture  # noqa: E402
from utils.paths import get_mirror_path  # noqa: E402

ARCH_ORDER = ("wire", "siren", "sechiren", "tanhiren", "mixiren",
              "cosiren_ps", "gaussian")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="50 epochs per cell; plumbing only")
    parser.add_argument("--difficulty", choices=("canonical", "tuned"),
                        default="tuned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archs", nargs="*", default=list(ARCH_ORDER),
                        help="subset of architectures, by key")
    args = parser.parse_args()

    results_dir = get_mirror_path(__file__, "results")

    rows = []
    for key in args.archs:
        rows.extend(
            run_architecture(
                key, results_dir=None, write=False,
                smoke=args.smoke, seed=args.seed,
                n=3, difficulty=args.difficulty,
            )
        )

    suffix = "_smoke" if args.smoke else ""
    out = results_dir / f"genz_3d_{args.difficulty}{suffix}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # per-architecture aggregates
    print("\n" + "=" * 78)
    print(f"Genz 3-D benchmark ({args.difficulty}), seed {args.seed}")
    print("=" * 78)
    by_arch: dict[str, list[float]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(float(r["digits"]))

    for arch, digs in sorted(by_arch.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {arch:<12} min={min(digs):6.2f}  mean={sum(digs) / len(digs):6.2f}"
              f"  max={max(digs):6.2f}")

    ref = quadrature_reference(n=3, difficulty=args.difficulty)
    print("\n  quadrature reference (calibration, not truth):")
    for r in ref:
        print(f"    {r['function']:<15} rel_err={r['rel_err']}  {r['status']}")

    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()