"""Compare the completed range repair with a bounded multiplicative neural layer."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import mpmath as mp
import numpy as np

from bounded_water_volume import (
    BUDGETS, DIMENSIONS, N_HOLDOUT, N_VALIDATION, PASS_DIGITS, SEEDS, Model,
    get_mirror_path, rmse, score_model, summary, surfaces, write_csv,
)
from skuld._fiber_water_volume import FiberSurface


def extend(row: dict[str, str], fixture, dimension: int, validation_x, test_x) -> dict:
    sites, seed = int(row["random_training_sites"]), int(row["seed"])
    points = np.random.default_rng(seed).random((sites, dimension))
    heights = fixture.evaluate(points)
    baseline_rmse = float(row["validation_rmse"])
    probe_sites = 0
    fiber_score = None
    selected = False
    seconds = 0.0
    if baseline_rmse > 2e-14 * np.max(heights):
        degree, splits, tolerance = (8, 40, 8e-14) if sites == BUDGETS[0].sites else (10, 75, 4e-16)
        started = perf_counter()
        fit, probe_sites = FiberSurface.fit(
            fixture.evaluate, points, heights, degree=degree, max_splits=splits, tolerance=tolerance,
        )
        seconds = perf_counter() - started
        if fit is not None:
            validation_error = rmse(fit.height(validation_x), fixture.evaluate(validation_x))
            model = Model("fiber", fit.height, fit.integrate, fit.parameter_count, validation_error)
            selected = validation_error < baseline_rmse
            estimate, relative, fiber_score, test_rmse = score_model(model, fixture, test_x, fixture.evaluate(test_x))

    base_score = float(row["correct_digits"])
    result = dict(row)
    result.update(
        dimension=dimension, seed=seed, random_training_sites=sites,
        total_fit_sites=int(row["total_fit_sites"]) + probe_sites, fiber_query_sites=probe_sites,
        fiber_fit_seconds=seconds, base_model=row["model"], base_digits=base_score,
        fiber_digits=fiber_score, fiber_selected=selected,
        fit_seconds=float(row["fit_seconds"]) + seconds, ridge_digits=float(row["ridge_digits"]),
        correct_digits=base_score, reached_12=base_score >= PASS_DIGITS,
    )
    if selected:
        result.update(
            model="fiber", parameters=model.parameters, water_volume=1 - estimate,
            estimate=estimate, relative_error=relative, correct_digits=fiber_score, reached_12=fiber_score >= PASS_DIGITS,
            holdout_rmse=test_rmse, validation_rmse=validation_error,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    directory = get_mirror_path(__file__, "results")
    source = Path(directory) / f"integrals{suffix}.csv"
    with source.open(encoding="utf-8") as data:
        source_rows = list(csv.DictReader(data))
    dimensions = DIMENSIONS[:1] if args.smoke else DIMENSIONS
    expected = 11 if args.smoke else len(DIMENSIONS) * len(SEEDS) * len(BUDGETS) * 11
    if len(source_rows) != expected:
        raise ValueError("Complete the bounded water-volume sweep first")

    rows = []
    started = perf_counter()
    with mp.workdps(60):
        for dimension in dimensions:
            fixtures = {fixture.name: fixture for fixture in surfaces(dimension)}
            validation_x = np.random.default_rng(707).random((N_VALIDATION, dimension))
            test_x = np.random.default_rng(918).random((N_HOLDOUT, dimension))
            current = [
                extend(row, fixtures[row["function"]], dimension, validation_x, test_x)
                for row in source_rows if int(row["dimension"]) == dimension
            ]
            rows.extend(current)
            scores = [row["correct_digits"] for row in current]
            print(
                f"{dimension}D: {sum(row['fiber_selected'] for row in current)} fiber models selected; "
                f"digits min/mean/max {min(scores):.3f}/{np.mean(scores):.3f}/{max(scores):.3f}; "
                f"{sum(row['reached_12'] for row in current)} passing", flush=True,
            )
    write_csv(directory / f"extended_integrals{suffix}.csv", rows)
    write_csv(directory / f"extended_summary{suffix}.csv", summary(rows))
    print(f"Fiber evaluation: {perf_counter() - started:.2f}s wall time", flush=True)


if __name__ == "__main__":
    main()
