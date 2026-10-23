"""
sechiren_genz.py — SECHIREN on the Genz characterisation benchmark.

Trains SECHIREN once per Genz function at its best known configuration and scores
the corner-sum estimate against that function's closed-form exact integral.

This is a flat experiment: one architecture, one configuration, no sweep. The
functions vary, not the hyperparameters. It is a *characterisation* benchmark
and is complementary to the meson Feynman integral experiment — it says nothing
about performance on the target problem.

Configuration: grid/18 seed-averaged best: [80,80,80], omega_0=15, lr=2e-4, npp=2048

Run:
    python research/experiments/flat/28-28-sechiren-genz/sechiren_genz.py \
        &> research/results/flat/28-28-sechiren-genz/sechiren_genz.log &

Smoke test (50 epochs per function, checks plumbing only):
    python research/experiments/flat/28-28-sechiren-genz/sechiren_genz.py --smoke
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # research/
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.genz_runner import run_architecture  # noqa: E402
from utils.paths import get_mirror_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="50 epochs per function; validates plumbing, not accuracy",
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="number of integration variables (3 or 6)",
    )
    parser.add_argument(
        "--difficulty", choices=("canonical", "tuned"), default="canonical",
        help="canonical: Genz1984 a=5 (hard). tuned: softer, meson-comparable",
    )
    args = parser.parse_args()

    results_dir = get_mirror_path(__file__, "results")
    run_architecture(
        "sechiren", results_dir,
        smoke=args.smoke, n=args.n, difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
