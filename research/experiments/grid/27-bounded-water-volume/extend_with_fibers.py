"""Add bounded multiplicative layers and spectral ridge starts learned on fibers."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import mpmath as mp
import numpy as np

from bounded_water_volume import (
    BUDGETS, DIMENSIONS, MAX_CELLS, PASS_DIGITS, SEEDS, Model, evaluation_points,
    get_mirror_path, ridge_model, rmse, score_model, summary, surfaces, write_csv,
)
from skuld._bounded_water_volume import certify_ridge
from skuld._fiber_water_volume import FiberSurface
from skuld._ridge_line_initialization import infer_frequencies
from skuld._water_volume import WaterVolumeNetwork, _ridge_candidates


def fit_line_ridges(fixture, points, heights, validation_x, validation_y) -> tuple[Model | None, int]:
    dimension = points.shape[1]
    rng = np.random.default_rng(13)
    directions = np.unique(rng.choice([-0.8, 0.8], size=(12, dimension)), axis=0)
    scale = float(np.max(heights)) or 1.0
    samples = 0
    best = None
    for direction in directions:
        initial, count = infer_frequencies(fixture.evaluate, direction, 24)
        samples += count
        if initial is None:
            continue
        fit, _ = next(_ridge_candidates(
            points, heights / scale, len(initial), 24, 1, 65, rng, initial_frequencies=initial,
        ))
        ridge = WaterVolumeNetwork(fit.frequencies, fit.coefficients * scale)
        if certify_ridge(ridge, MAX_CELLS).status not in {"amplitude", "subdivision"}:
            continue
        candidate = ridge_model("line_ridge", ridge, validation_x, validation_y)
        if best is None or candidate.validation_rmse < best.validation_rmse:
            best = candidate
        if best.validation_rmse <= 2e-14 * scale:
            break
    return best, samples


def extend(row: dict[str, str], fixture, dimension: int, validation_x, test_x) -> dict:
    sites, seed = int(row["random_training_sites"]), int(row["seed"])
    points = np.random.default_rng(seed).random((sites, dimension))
    heights = fixture.evaluate(points)
    baseline_rmse = float(row["validation_rmse"])
    probe_sites = line_sites = 0
    model = None
    fiber_seconds = line_seconds = 0.0
    if baseline_rmse > 2e-14 * np.max(heights):
        degree, splits, tolerance = (8, 40, 8e-14) if sites == BUDGETS[0].sites else (10, 75, 4e-16)
        started = perf_counter()
        validation_y = fixture.evaluate(validation_x)
        fit, probe_sites = FiberSurface.fit(
            fixture.evaluate, points, heights, degree=degree, max_splits=splits, tolerance=tolerance,
        )
        fiber_seconds = perf_counter() - started
        if fit is not None:
            validation_error = rmse(fit.height(validation_x), validation_y)
            model = Model("fiber", fit.height, fit.integrate, fit.parameter_count, validation_error)
        else:
            started = perf_counter()
            model, line_sites = fit_line_ridges(fixture, points, heights, validation_x, validation_y)
            line_seconds = perf_counter() - started

    base_score = float(row["correct_digits"])
    selected = model is not None and model.validation_rmse < baseline_rmse
    result = dict(row)
    result.update(
        dimension=dimension, seed=seed, random_training_sites=sites,
        total_fit_sites=int(row["total_fit_sites"]) + probe_sites + line_sites,
        fiber_query_sites=probe_sites, line_query_sites=line_sites,
        fiber_fit_seconds=fiber_seconds, line_fit_seconds=line_seconds,
        base_model=row["model"], base_digits=base_score,
        fit_seconds=float(row["fit_seconds"]) + fiber_seconds + line_seconds,
        ridge_digits=float(row["ridge_digits"]),
        correct_digits=base_score, reached_12=base_score >= PASS_DIGITS,
    )
    if selected:
        estimate, relative, score, test_rmse = score_model(model, fixture, test_x, fixture.evaluate(test_x))
        result.update(
            model=model.kind, parameters=model.parameters, water_volume=1 - estimate,
            estimate=estimate, relative_error=relative, correct_digits=score, reached_12=score >= PASS_DIGITS,
            holdout_rmse=test_rmse, validation_rmse=model.validation_rmse,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--confirmation", action="store_true", help="Extend the independent-seed sweep")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else "_confirmation" if args.confirmation else ""
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
            validation_x, test_x = evaluation_points(dimension, args.confirmation)
            current = [
                extend(row, fixtures[row["function"]], dimension, validation_x, test_x)
                for row in source_rows if int(row["dimension"]) == dimension
            ]
            rows.extend(current)
            scores = [row["correct_digits"] for row in current]
            print(
                f"{dimension}D: {sum(row['model'] == 'fiber' for row in current)} product models and "
                f"{sum(row['model'] == 'line_ridge' for row in current)} line-initialized models; "
                f"digits min/mean/max {min(scores):.3f}/{np.mean(scores):.3f}/{max(scores):.3f}; "
                f"{sum(row['reached_12'] for row in current)} passing", flush=True,
            )
    write_csv(directory / f"extended_integrals{suffix}.csv", rows)
    write_csv(directory / f"extended_summary{suffix}.csv", summary(rows))
    print(f"Fiber and line evaluation: {perf_counter() - started:.2f}s wall time", flush=True)


if __name__ == "__main__":
    main()
