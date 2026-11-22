"""Sweep sine water-volume networks on bounded surfaces of increasing difficulty."""

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import product
from math import comb, factorial, prod
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks import genz
from skuld._water_volume import UncertifiedWaterDepthError, WaterVolumeNetwork
from utils.paths import get_mirror_path

FloatArray = NDArray[np.float64]
DIMENSIONS = (2, 3, 4, 5, 6)
SEEDS = (42, 123, 2026)
N_HOLDOUT = 4096
TARGET_DIGITS = 12


@dataclass(frozen=True)
class Budget:
    name: str
    width: int
    training_sites: int
    frequency_limit: float
    restarts: int
    max_nfev: int


BUDGETS = (
    Budget("pilot", 2, 256, 8, 12, 300),
    Budget("more_samples", 2, 1024, 8, 12, 300),
    Budget("wider", 8, 1024, 8, 4, 100),
    Budget("wide_band", 8, 1024, 24, 4, 100),
)


@dataclass(frozen=True)
class Surface:
    name: str
    evaluate: Callable[[FloatArray], FloatArray]
    exact: mp.mpf


def ridge_surface(
    name: str, frequencies: FloatArray, phases: FloatArray, amplitudes: FloatArray,
    offset: str, scale: str = "1",
) -> Surface:
    """Create a sine mixture and score it via vertex differences in mp arithmetic."""
    dimension = frequencies.shape[1]
    vertices = tuple(product((0, 1), repeat=dimension))
    integral = mp.mpf(offset)
    for frequency, phase, amplitude in zip(frequencies, phases, amplitudes, strict=True):
        omega = [mp.mpf(float(w)) for w in frequency]
        # The n-fold mixed primitive of sin(s) is sin(s - n*pi/2) / prod(w).
        corner_sum = mp.fsum(
            (-1)**(dimension - sum(vertex)) *
            mp.sin(mp.mpf(float(phase)) + mp.fdot(omega, vertex) - dimension * mp.pi / 2)
            for vertex in vertices
        )
        integral += mp.mpf(float(amplitude)) * corner_sum / mp.fprod(omega)

    def evaluate(points: FloatArray) -> FloatArray:
        return float(scale) * (float(offset) + np.sin(points @ frequencies.T + phases) @ amplitudes)

    return Surface(name, evaluate, mp.mpf(scale) * integral)


def surfaces(dimension: int) -> tuple[Surface, ...]:
    """Return eleven analytic fixtures, all with heights in the unit interval."""
    frequencies = np.array([
        [2.35, 3.10, -2.65, 1.75, 2.85, -1.55],
        [4.70, -1.80, 3.25, -2.15, 1.45, 2.25],
    ])[:, :dimension]
    phases = np.array([0.4, np.pi / 2 - 0.7])
    amplitudes = np.array([0.18, 0.13])
    ridge = ridge_surface("two_ridges", frequencies, phases, amplitudes, ".43")
    tiny = ridge_surface("tiny_integral", frequencies, phases, amplitudes, ".43", "1e-6")

    rng = np.random.default_rng(8305)
    # Generate all six coordinates before slicing so the 2D modes extend to 6D.
    signs = rng.choice([-1, 1], size=(6, 6))
    fast = (9 + 14 * rng.random((6, 6))) * signs
    multiscale = ridge_surface(
        "six_fast_ridges", fast[:, :dimension], rng.uniform(-3, 3, size=6),
        np.array([.095, .076, .062, .048, .037, .025]), ".48",
    )

    def polynomial(points: FloatArray) -> FloatArray:
        return np.prod(1 + points * np.roll(points, -1, axis=1), axis=1) / 2**dimension

    polynomial_exact = mp.mpf(0)
    for edges in product((0, 1), repeat=dimension):
        polynomial_exact += 1 / mp.mpf(prod(1 + edges[i] + edges[i - 1] for i in range(dimension)))

    def exponential(points: FloatArray) -> FloatArray:
        return np.exp(4 * (points.prod(axis=1) - 1))

    exponential_exact = mp.exp(-4) * mp.fsum(
        mp.mpf(4)**k / (mp.factorial(k) * (k + 1)**dimension) for k in range(96)
    )

    phases = np.array([np.pi / 2 + 0.78])
    oscillation = ridge_surface(
        "genz_oscillatory", np.linspace(3.8, 7.2, 6)[None, :dimension],
        phases, np.array([0.43]), ".5",
    )
    offsets = np.linspace(.23, .77, 6)[:dimension]
    u = [mp.mpf(float(value)) for value in offsets]
    lorentz_width = np.linspace(.07, .16, 6)[:dimension]
    gaussian_rate = np.linspace(7, 13, 6)[:dimension]
    kink_rate = np.linspace(3, 7, 6)[:dimension]
    corner_rate = np.linspace(.9, 1.9, 6)[:dimension]

    def high_precision(values: FloatArray) -> list[mp.mpf]:
        return [mp.mpf(float(value)) for value in values]

    lorentz_exact = mp.fprod(
        a * (mp.atan((1 - x) / a) + mp.atan(x / a))
        for a, x in zip(high_precision(lorentz_width), u, strict=True)
    )
    gaussian_exact = mp.fprod(
        mp.sqrt(mp.pi) / (2 * a) * (mp.erf(a * (1 - x)) + mp.erf(a * x))
        for a, x in zip(high_precision(gaussian_rate), u, strict=True)
    )
    kink_exact = mp.fprod(
        (2 - mp.exp(-a * x) - mp.exp(-a * (1 - x))) / a
        for a, x in zip(high_precision(kink_rate), u, strict=True)
    )
    a = high_precision(corner_rate)
    corner_exact = mp.fsum(
        (-1)**sum(vertex) / (1 + mp.fdot(a, vertex))
        for vertex in product((0, 1), repeat=dimension)
    ) / (factorial(dimension) * mp.fprod(a))

    threshold = mp.mpf(dimension) / 2 + mp.mpf(".17")
    below_jump = mp.fsum(
        (-1)**k * comb(dimension, k) * (threshold - k)**dimension
        for k in range(int(threshold) + 1)
    ) / factorial(dimension)

    def jump(points: FloatArray) -> FloatArray:
        return np.where(points.sum(axis=1) < float(threshold), .08, .92)

    return (
        ridge, multiscale,
        Surface("coupled_polynomial", polynomial, polynomial_exact / 2**dimension),
        Surface("coupled_exponential", exponential, exponential_exact),
        oscillation,
        Surface("genz_product_peak", lambda x: genz.product_peak_np(x, lorentz_width, offsets), lorentz_exact),
        Surface("genz_corner_peak", lambda x: genz.corner_peak_np(x, corner_rate), corner_exact),
        Surface("genz_gaussian", lambda x: genz.gaussian_np(x, gaussian_rate, offsets), gaussian_exact),
        Surface("genz_c0", lambda x: genz.c0_continuous_np(x, kink_rate, offsets), kink_exact),
        Surface("oblique_jump", jump, mp.mpf(".92") - mp.mpf(".84") * below_jump),
        tiny,
    )


def digits(estimate: float, exact: mp.mpf) -> tuple[float, float]:
    relative = abs(mp.mpf(estimate) - exact) / abs(exact)
    return float(relative), float(-mp.log10(relative))


def evaluate(
    name: str, exact: mp.mpf, train_x: FloatArray, train_y: FloatArray,
    test_x: FloatArray, test_y: FloatArray, seed: int, budget: Budget,
) -> dict:
    dimension = train_x.shape[1]
    row = {
        "integral": "I_1", "function": name, "dimension": dimension, "seed": seed,
        **asdict(budget), "parameters": 1 + budget.width * (dimension + 2),
        "reference": mp.nstr(exact, 45), "max_sampled_height": float(np.max(train_y)),
        "sample_mean_digits": digits(float(np.mean(train_y)), exact)[1],
        "status": "uncertified_depth", "water_volume": None, "estimate": None,
        "relative_error": None, "correct_digits": None, "reached_12": False,
        "holdout_rmse": None, "holdout_max_abs_error": None,
        "depth_lower": None, "depth_upper": None, "fit_seconds": None,
        "readout_seconds": None,
    }
    started = perf_counter()
    try:
        model = WaterVolumeNetwork.fit(
            train_x, train_y, width=budget.width, frequency_limit=budget.frequency_limit,
            restarts=budget.restarts, max_nfev=budget.max_nfev,
            rng=np.random.default_rng(seed),
        )
    except UncertifiedWaterDepthError:
        row["fit_seconds"] = perf_counter() - started
        return row
    row["fit_seconds"] = perf_counter() - started

    started = perf_counter()
    volume = model.water_volume()
    estimate = 1 - volume
    readout_seconds = perf_counter() - started
    relative, score = digits(estimate, exact)
    errors = (1 - model.depth(test_x)) - test_y
    lower, upper = model.depth_bounds()
    row.update(
        status="ok", water_volume=volume, estimate=estimate, relative_error=relative,
        correct_digits=score, reached_12=score >= TARGET_DIGITS,
        holdout_rmse=float(np.sqrt(np.mean(errors**2))),
        holdout_max_abs_error=float(np.max(np.abs(errors))),
        depth_lower=lower, depth_upper=upper, readout_seconds=readout_seconds,
    )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], **labels: int | str) -> dict:
    scores = [row["correct_digits"] for row in rows if row["status"] == "ok"]
    return {
        **labels, "attempts": len(rows), "returned_estimates": len(scores),
        "uncertified": len(rows) - len(scores), "reached_12": sum(row["reached_12"] for row in rows),
        "min_digits": min(scores) if scores else None,
        "mean_digits": mean(scores) if scores else None,
        "max_digits": max(scores) if scores else None,
        "fit_seconds": sum(row["fit_seconds"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use one seed, dimension and small budget")
    args = parser.parse_args()
    dimensions = DIMENSIONS[:1] if args.smoke else DIMENSIONS
    seeds = SEEDS[:1] if args.smoke else SEEDS
    budgets = (Budget("smoke", 2, 64, 8, 1, 15),) if args.smoke else BUDGETS
    suffix = "_smoke" if args.smoke else ""
    destination = get_mirror_path(__file__, "results")
    rows = []
    summary = []
    with mp.workdps(60), threadpool_limits(limits=1, user_api="blas"):
        for dimension in dimensions:
            fixtures = surfaces(dimension)
            test_x = np.random.default_rng(918).random((N_HOLDOUT, dimension))
            first = len(rows)
            for seed in seeds:
                train_x = np.random.default_rng(seed).random(
                    (max(budget.training_sites for budget in budgets), dimension)
                )
                for surface in fixtures:
                    train_y = surface.evaluate(train_x)
                    test_y = surface.evaluate(test_x)
                    for budget in budgets:
                        n = budget.training_sites
                        rows.append(evaluate(
                            surface.name, surface.exact, train_x[:n], train_y[:n],
                            test_x, test_y, seed, budget,
                        ))
                print(f"{dimension}D seed={seed} complete", flush=True)
            current = rows[first:]
            summary.extend(
                summarize(
                    [row for row in current if row["function"] == surface.name and row["name"] == budget.name],
                    dimension=dimension, function=surface.name, budget=budget.name,
                )
                for surface in fixtures for budget in budgets
            )
            print(summarize(current, dimension=dimension), flush=True)
            write_csv(destination / f"integrals{suffix}.csv", rows)
            write_csv(destination / f"summary{suffix}.csv", summary)
    print(summarize(rows), flush=True)


if __name__ == "__main__":
    main()
