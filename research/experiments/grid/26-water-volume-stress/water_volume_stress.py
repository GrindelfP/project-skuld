"""Sweep sine water-volume networks on bounded surfaces of increasing difficulty."""

import argparse
import csv
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.water_volume_surfaces import digits, surfaces
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


def run_dimension(
    dimension: int, seeds: tuple[int, ...], budgets: tuple[Budget, ...],
) -> tuple[int, list[dict], list[dict]]:
    rows = []
    with mp.workdps(60), threadpool_limits(limits=1, user_api="blas"):
        fixtures = surfaces(dimension)
        test_x = np.random.default_rng(918).random((N_HOLDOUT, dimension))
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
    summary = [
        summarize(
            [row for row in rows if row["function"] == surface.name and row["name"] == budget.name],
            dimension=dimension, function=surface.name, budget=budget.name,
        )
        for surface in fixtures for budget in budgets
    ]
    return dimension, rows, summary


def run_dimensions(
    dimensions: tuple[int, ...], seeds: tuple[int, ...], budgets: tuple[Budget, ...],
    workers: int,
) -> Iterator[tuple[int, list[dict], list[dict]]]:
    arguments = (dimensions, repeat(seeds), repeat(budgets))
    if workers == 1:
        yield from map(run_dimension, *arguments)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(run_dimension, *arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use one seed, dimension and small budget")
    parser.add_argument("--workers", type=int, default=1, help="Run this many dimensions in parallel")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    dimensions = DIMENSIONS[:1] if args.smoke else DIMENSIONS
    seeds = SEEDS[:1] if args.smoke else SEEDS
    budgets = (Budget("smoke", 2, 64, 8, 1, 15),) if args.smoke else BUDGETS
    suffix = "_smoke" if args.smoke else ""
    destination = get_mirror_path(__file__, "results")
    rows = []
    summary = []
    for dimension, current, current_summary in run_dimensions(dimensions, seeds, budgets, args.workers):
        rows.extend(current)
        summary.extend(current_summary)
        print(summarize(current, dimension=dimension), flush=True)
        write_csv(destination / f"integrals{suffix}.csv", rows)
        write_csv(destination / f"summary{suffix}.csv", summary)
    print(summarize(rows), flush=True)


if __name__ == "__main__":
    main()
