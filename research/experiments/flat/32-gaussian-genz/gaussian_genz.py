"""
gaussian_genz.py — GAUSSIAN on the Genz characterisation benchmark.

Trains GAUSSIAN once per Genz function at its best known configuration and scores
the corner-sum estimate against that function's closed-form exact integral.

This is a flat experiment: one architecture, one configuration, no sweep. The
functions vary, not the hyperparameters. It is a *characterisation* benchmark
and is complementary to the meson Feynman integral experiment — it says nothing
about performance on the target problem.

Configuration: flat/14 fixed settings: [128,128,128], omega_0=30, lr=5e-4

Run:
    python research/experiments/flat/32-32-gaussian-genz/gaussian_genz.py \
        &> research/results/flat/32-32-gaussian-genz/gaussian_genz.log &

Smoke test (50 epochs per function, checks plumbing only):
    python research/experiments/flat/32-32-gaussian-genz/gaussian_genz.py --smoke
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # research/
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from genz_runner import run_architecture  # noqa: E402
from utils.paths import get_mirror_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="50 epochs per function; validates plumbing, not accuracy",
    )
    args = parser.parse_args()

    results_dir = get_mirror_path(__file__, "results")
    run_architecture("gaussian", results_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
