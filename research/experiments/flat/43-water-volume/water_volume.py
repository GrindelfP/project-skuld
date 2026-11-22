"""Measure the complement of learned water occupancy for a 2D surface."""

import csv
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from skuld._water_volume import WaterVolumeNetwork
from utils.paths import get_mirror_path

FloatArray = NDArray[np.float64]
SEEDS = (42, 123, 2024, 2026, 90210)
N_TRAIN = 256
N_HOLDOUT = 4096
WIDTH = 2


def surface(points: FloatArray) -> FloatArray:
    x, y = points.T
    return 0.43 + 0.18 * np.sin(2.35 * x + 3.10 * y + 0.4) + 0.13 * np.cos(
        4.70 * x - 1.80 * y - 0.7
    )


def reference() -> mp.mpf:
    """Integrate the fixture independently by its mixed antiderivative."""
    real = mp.mpf
    sine_corners = real(0)
    cosine_corners = real(0)
    for x, y, sign in ((0, 0, 1), (1, 0, -1), (0, 1, -1), (1, 1, 1)):
        sine_corners += sign * mp.sin(real("2.35") * x + real("3.10") * y + real("0.4"))
        cosine_corners += sign * mp.cos(real("4.70") * x - real("1.80") * y - real("0.7"))
    return (
        real("0.43")
        - real("0.18") * sine_corners / (real("2.35") * real("3.10"))
        + real("0.13") * cosine_corners / (real("4.70") * real("1.80"))
    )


def main() -> None:
    holdout = np.random.default_rng(918).random((N_HOLDOUT, 2))
    rows = []
    with mp.workdps(60):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            points = rng.random((N_TRAIN, 2))
            heights = surface(points)

            started = perf_counter()
            model = WaterVolumeNetwork.fit(
                points, heights, width=WIDTH, frequency_limit=8.0, restarts=12, rng=rng
            )
            fit_seconds = perf_counter() - started
            started = perf_counter()
            water = model.water_volume()
            estimate = 1 - water
            readout_microseconds = 1e6 * (perf_counter() - started)

            exact = reference()
            error = abs(mp.mpf(estimate) - exact)
            relative = error / abs(exact)
            digits = float(-mp.log10(relative))
            holdout_error = np.max(np.abs(1 - model.depth(holdout) - surface(holdout)))
            lower, upper = model.depth_bounds()
            rows.append({
                "integral": "I_1",
                "seed": seed,
                "training_sites": N_TRAIN,
                "holdout_sites": N_HOLDOUT,
                "parameters": model.frequencies.size + model.coefficients.size,
                "water_volume": water,
                "estimate": estimate,
                "reference": mp.nstr(exact, 55),
                "absolute_error": float(error),
                "relative_error": float(relative),
                "correct_digits": digits,
                "holdout_max_absolute_surface_error": holdout_error,
                "depth_lower_bound": lower,
                "depth_upper_bound": upper,
            })
            print(
                f"I_1 seed={seed:5d}  A={water:.17g}  I={estimate:.17g}  "
                f"relative error={float(relative):.3e}  digits={digits:.3f}  "
                f"holdout max={holdout_error:.3e}  "
                f"fit={fit_seconds:.3f}s  readout={readout_microseconds:.0f}us",
                flush=True,
            )

    scores = [row["correct_digits"] for row in rows]
    print(f"Correct digits (min / mean / max): {min(scores):.3f} / {mean(scores):.3f} / {max(scores):.3f}")
    results_path = get_mirror_path(__file__, "results") / "water_volume.csv"
    with results_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {results_path}")


if __name__ == "__main__":
    main()
