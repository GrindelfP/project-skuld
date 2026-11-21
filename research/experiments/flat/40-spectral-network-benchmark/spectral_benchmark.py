"""Measure the spectral network on seven integrand families in dimensions 1–6."""

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.genz import GENZ_FUNCTIONS
from skuld._spectral_network import SpectralNetwork
from utils.paths import get_mirror_path

SEEDS = (1, 42, 2026)
# (Chebyshev degree, readout sites per axis, training fibers per axis)
CONFIGS = {
    1: (47, 56, 1),
    2: (47, 56, 384),
    3: (47, 56, 384),
    4: (19, 22, 384),
    5: (9, 12, 384),
    6: (6, 8, 384),
}
FAMILIES = tuple(fn.name for fn in GENZ_FUNCTIONS) + ("coupled_polynomial", "coupled_exponential")


def integrand_values(*coordinates: np.ndarray) -> np.ndarray:
    """Return seven channels from point values alone."""
    axes = np.broadcast_arrays(*coordinates)
    points = np.stack(axes, axis=-1)
    flat_points = points.reshape(-1, len(axes))
    genz = [fn.numpy_fn(flat_points, *fn.params).reshape(points.shape[:-1]) for fn in GENZ_FUNCTIONS]
    cyclic = np.prod(1 + points * np.roll(points, -1, axis=-1), axis=-1)
    exponential = np.exp(2 * points.prod(axis=-1))
    return np.stack([*genz, cyclic, exponential], axis=-1)


def exact_integrals(dimensions: int) -> dict[str, mp.mpf]:
    """Integrate the canonical formulas in 50-digit arithmetic."""
    with mp.workdps(50):
        two = mp.mpf(2)
        five = mp.mpf(5)
        oscillation = five / 2
        # Expand the cyclic product. Each selected edge adds one power
        # to each of its endpoints; the monomial integral then factorizes.
        polynomial = mp.mpf(0)
        for subset in range(1 << dimensions):
            exponents = [0] * dimensions
            for axis in range(dimensions):
                if subset & (1 << axis):
                    exponents[axis] += 1
                    exponents[(axis + 1) % dimensions] += 1
            polynomial += 1 / mp.mpf(math.prod(p + 1 for p in exponents))
        # The exponential series converges uniformly over the cube.
        exponential = mp.fsum(two**k / (mp.factorial(k) * (k + 1)**dimensions) for k in range(100))
        return {
            "oscillatory": -mp.cos(oscillation * dimensions) * (mp.sin(oscillation) / oscillation)**dimensions,
            "product_peak": (2 * five * mp.atan(1 / (2 * five)))**dimensions,
            "corner_peak": 1 / (mp.factorial(dimensions) * (dimensions + 1)),
            "gaussian": (mp.sqrt(mp.pi) * mp.erf(oscillation) / five)**dimensions,
            "c0_continuous": (2 * (1 - mp.exp(-oscillation)) / five)**dimensions,
            "coupled_polynomial": polynomial,
            "coupled_exponential": exponential,
        }


def benchmark() -> tuple[list[dict], list[dict]]:
    results = []
    timings = []
    for dimensions, (degree, nodes, fibers) in CONFIGS.items():
        print(f"Dimension {dimensions}, degree {degree}, readout grid {nodes}^{dimensions}", flush=True)
        for seed in SEEDS:
            start = perf_counter()
            model = SpectralNetwork.fit(
                integrand_values, degree=degree, fibers=fibers, readout_nodes=nodes,
                rank_tolerance=1e-15, seed=seed, dimensions=dimensions,
            )
            fit_seconds = perf_counter() - start
            with torch.inference_mode():
                start = perf_counter()
                estimates = model.integrate().tolist()
                readout_seconds = perf_counter() - start
            timings.append({
                "dimension": dimensions, "seed": seed, "degree": degree, "nodes_per_axis": nodes,
                "training_sites": dimensions * fibers * (degree + 1) + nodes**dimensions,
                "parameters": sum(p.numel() for p in model.parameters()),
                "fit_seconds": fit_seconds, "readout_seconds": readout_seconds,
                "ready_seconds": fit_seconds + readout_seconds,
            })
            reference = exact_integrals(dimensions)
            with mp.workdps(50):
                for name, estimate in zip(FAMILIES, estimates, strict=True):
                    exact = reference[name]
                    error = abs(mp.mpf(estimate) - exact)
                    relative_error = error / abs(exact)
                    results.append({
                        "dimension": dimensions, "seed": seed, "function": name, "estimate": estimate,
                        "exact": mp.nstr(exact, 35), "absolute_error": float(error),
                        "relative_error": float(relative_error), "correct_digits": float(-mp.log10(relative_error)),
                    })
        scores = [row["correct_digits"] for row in results if row["dimension"] == dimensions]
        print(f"  digits min / mean / max: {min(scores):.2f} / {mean(scores):.2f} / {max(scores):.2f}", flush=True)
    return results, timings


def main() -> None:
    results, timings = benchmark()
    dest = get_mirror_path(__file__, "results")
    for filename, rows in (("integrals.csv", results), ("timings.csv", timings)):
        path = dest / filename
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(path)


if __name__ == "__main__":
    main()
