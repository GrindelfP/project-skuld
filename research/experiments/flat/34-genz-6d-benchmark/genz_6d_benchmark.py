"""
genz_6d_benchmark.py — all seven architectures on the 6-D Genz benchmark.

The 6-D companion to flat/33. Same five functions, same metrics, same tuned
difficulty tier, one dimension higher. Independent of the 3-D script: the two
can run concurrently on separate GPUs.

Why this needed library work
----------------------------
Every integrator's loss called a module-level `mixed_partial_3`, which took
exactly three autograd passes over columns 0, 1, 2 regardless of the true
dimension. At n_int_vars=6 a network was trained to match a third-order partial
derivative and then evaluated with a 64-corner sum: no exception, loss
decreasing happily, result confidently wrong. `mixed_partial_n` in skuld-lib
fixes this and is verified by research/check_mixed_partial.py.

Cost
----
An n-th derivative chain is more expensive than a third-order one, but far less
dramatically than first assumed: measured at 256 points per parameter set,

    SIREN    0.074 s/epoch at n=6   (0.260 at n=3, warmup-affected)
    MIXIREN  0.118 s/epoch at n=6   (0.065 at n=3)
    WIRE     0.561 s/epoch at n=6   (0.148 at n=3)

WIRE is the outlier at ~3.8x because it is genuinely compute-bound in its
residual blocks, while the others are launch-overhead-bound and largely
insensitive to batch size. At the default 1000 epochs the full seven-architecture
sweep is roughly 40-60 minutes, WIRE dominating.

Run:
    python research/experiments/flat/34-genz-6d-benchmark/genz_6d_benchmark.py \\
        &> research/results/flat/34-genz-6d-benchmark/genz_6d_benchmark.log &
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # research/
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.genz_runner import run_architecture  # noqa: E402
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
    parser.add_argument("--epochs", type=int, default=1000,
                        help="epochs per cell")
    parser.add_argument("--archs", nargs="*", default=list(ARCH_ORDER))
    args = parser.parse_args()

    results_dir = get_mirror_path(__file__, "results")

    rows = []
    for key in args.archs:
        rows.extend(
            run_architecture(
                key, results_dir=None, write=False,
                smoke=args.smoke, seed=args.seed,
                n=6, difficulty=args.difficulty,
                epochs_6d_override=args.epochs,
            )
        )

    suffix = "_smoke" if args.smoke else ""
    out = results_dir / f"genz_6d_{args.difficulty}{suffix}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print(f"Genz 6-D benchmark ({args.difficulty}), seed {args.seed}")
    print("=" * 78)
    by_arch: dict[str, list[float]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(float(r["digits"]))

    for arch, digs in sorted(by_arch.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {arch:<12} min={min(digs):6.2f}  mean={sum(digs) / len(digs):6.2f}"
              f"  max={max(digs):6.2f}")

    # Adaptive cubature is intractable above n~4: at n=6 all five Genz functions
    # ran past 20 s without finishing. The calibration row therefore belongs to
    # the 3-D benchmark only; computing it here would triple this script's
    # runtime for five timeouts.
    print("\n  quadrature reference: not computed at n=6 "
          "(adaptive cubature is intractable above n~4; see flat/33)")

    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()