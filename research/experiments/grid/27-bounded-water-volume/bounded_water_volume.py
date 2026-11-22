"""Audit water-depth rejections and sweep bounded neural occupancy models."""

import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Callable

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.water_volume_surfaces import Surface, digits, surfaces
from skuld._bounded_water_volume import (
    BernsteinSurface, ConvexRidgeSurface, certify_ridge, refit_at_violations,
)
from skuld._water_volume import FloatArray, WaterVolumeNetwork, _ridge_candidates
from utils.paths import get_mirror_path

DIMENSIONS = (2, 3, 4, 5, 6)
SEEDS = (42, 123, 2026)
CONFIRMATION_SEEDS = (17, 1701, 98765)
N_VALIDATION = 512
N_HOLDOUT = 4096
MAX_CELLS = 1023
PASS_DIGITS = 12


@dataclass(frozen=True)
class Budget:
    sites: int
    band_width: int
    max_nfev: int


BUDGETS = (Budget(256, 3, 85), Budget(1024, 6, 110))


@dataclass(frozen=True)
class Model:
    kind: str
    height: Callable[[FloatArray], FloatArray]
    integrate: Callable[[], float]
    parameters: int
    validation_rmse: float


def rmse(predictions: FloatArray, target: FloatArray) -> float:
    return float(np.sqrt(np.mean((predictions - target)**2)))


def ridge_model(
    kind: str, ridge: WaterVolumeNetwork, validation_x: FloatArray, validation_y: FloatArray,
) -> Model:
    return Model(
        kind, ridge.depth, ridge._cube_mean,
        ridge.frequencies.size + ridge.coefficients.size,
        rmse(ridge.depth(validation_x), validation_y),
    )


def fit_ridge(
    points: FloatArray, heights: FloatArray, validation_x: FloatArray, validation_y: FloatArray,
    seed: int, budget: Budget,
) -> tuple[Model, Counter]:
    scale = float(np.max(heights)) or 1.0
    training_target = heights / scale
    rng = np.random.default_rng(seed)
    best = None
    raw = []
    stats = Counter()
    for width, limit, starts in ((1, 8, 5), (2, 8, 8), (budget.band_width, 24, 2)):
        for fit, _ in _ridge_candidates(
            points, training_target, width, limit, starts, budget.max_nfev, rng,
        ):
            ridge = WaterVolumeNetwork(fit.frequencies, fit.coefficients * scale)
            candidate = ridge_model("ridge", ridge, validation_x, validation_y)
            check = certify_ridge(ridge, MAX_CELLS)
            stats[check.status] += 1
            if check.status in {"amplitude", "subdivision"}:
                candidate = Model(
                    check.status, candidate.height, candidate.integrate, candidate.parameters,
                    candidate.validation_rmse,
                )
                if best is None or candidate.validation_rmse < best.validation_rmse:
                    best = candidate
                if best.validation_rmse <= 2e-14 * scale:
                    return best, stats
            else:
                raw.append((candidate.validation_rmse, ridge, check))

    for _, ridge, check in sorted(raw, key=lambda item: item[0])[:3]:
        if check.status == "violation":
            local, verified = refit_at_violations(ridge, check, points, heights, MAX_CELLS)
            if verified.status in {"amplitude", "subdivision"}:
                stats["locally_repaired"] += 1
                candidate = ridge_model("local_repair", local, validation_x, validation_y)
                if best is None or candidate.validation_rmse < best.validation_rmse:
                    best = candidate

        convex = ConvexRidgeSurface.fit(ridge.frequencies, points, heights)
        candidate = Model(
            "convex_ridge", convex.height, convex.integrate,
            convex.frequencies.size + convex.weights.size,
            rmse(convex.height(validation_x), validation_y),
        )
        if best is None or candidate.validation_rmse < best.validation_rmse:
            best = candidate
    if best is None:
        raise RuntimeError("Expected at least one feasible neural surface")
    return best, stats


def fit_bernstein(
    fixture: Surface, validation_x: FloatArray, validation_y: FloatArray,
    budget: Budget, dimension: int,
) -> tuple[Model, int]:
    degrees = tuple(p for p in (1, 2, 4, 8, 16) if (p + 1)**dimension <= budget.sites)
    maximum = max(degrees)
    nodes = (1 - np.cos(np.linspace(0, np.pi, maximum + 1))) / 2
    nodes[[0, -1]] = [0, 1]
    grid = np.stack(np.meshgrid(*[nodes] * dimension, indexing="ij"), axis=-1)
    values = fixture.evaluate(grid.reshape(-1, dimension)).reshape(grid.shape[:-1])
    models = []
    for degree in degrees:
        step = maximum // degree
        fitted = BernsteinSurface.fit(values[(slice(None, None, step),) * dimension], nodes[::step])
        models.append(Model(
            f"bernstein_{degree}", fitted.height, fitted.integrate, fitted.coefficients.size,
            rmse(fitted.height(validation_x), validation_y),
        ))
    return min(models, key=lambda model: model.validation_rmse), values.size


def audit(
    fixture: Surface, seed: int, points: FloatArray, heights: FloatArray,
    smoke: bool,
) -> dict:
    """Replay the original 2-neuron budget at 256 sites as a paired diagnostic."""
    best = None
    best_rms = float("inf")
    rejected = []
    started = perf_counter()
    for ridge, train_rms in _ridge_candidates(
        points[:256], 1 - heights[:256], 2, 8,
        1 if smoke else 12, 12 if smoke else 300, np.random.default_rng(seed),
    ):
        lower, upper = ridge.depth_bounds()
        if 0 <= lower <= upper <= 1 and train_rms < best_rms:
            best, best_rms = ridge, train_rms
            if best_rms <= 5e-15:
                break
        elif lower < 0 or upper > 1:
            rejected.append((train_rms, ridge))

    counts = Counter()
    rescued = None
    if best is None:
        for _, ridge in sorted(rejected, key=lambda item: item[0]):
            check = certify_ridge(ridge, MAX_CELLS)
            counts[check.status] += 1
            if rescued is None and check.status in {"amplitude", "subdivision"}:
                rescued = ridge
    improved = best if best is not None else rescued
    estimate = None if improved is None else 1 - improved._cube_mean()
    return {
        "function": fixture.name, "dimension": points.shape[1], "seed": seed,
        "integral": "I_1", "original_accepts": best is not None,
        "subdivision_rescues": rescued is not None,
        "violating_candidates": counts["violation"], "unresolved_candidates": counts["unresolved"],
        "certified_candidates": counts["amplitude"] + counts["subdivision"],
        "estimate": estimate, "correct_digits": None if estimate is None else digits(estimate, fixture.exact)[1],
        "seconds": perf_counter() - started,
    }


def score_model(model: Model, fixture: Surface, test_x: FloatArray, test_y: FloatArray) -> tuple:
    estimate = model.integrate()
    predictions = model.height(test_x)
    if not 0 <= estimate <= 1 or not np.all((predictions >= -1e-12) & (predictions <= 1 + 1e-12)):
        raise RuntimeError(f"Range check failed on {model.kind} for {fixture.name}")
    relative, score = digits(estimate, fixture.exact)
    return estimate, relative, score, rmse(predictions, test_y)


def fit_case(
    fixture: Surface, budget: Budget, seed: int, points: FloatArray, heights: FloatArray,
    validation_x: FloatArray, validation_y: FloatArray, test_x: FloatArray, test_y: FloatArray,
) -> dict:
    n = budget.sites
    started = perf_counter()
    ridge, stats = fit_ridge(points[:n], heights[:n], validation_x, validation_y, seed, budget)
    bernstein_sites = 0
    model = ridge
    if ridge.validation_rmse > 2e-14 * float(np.max(heights[:n])):
        polynomial, bernstein_sites = fit_bernstein(fixture, validation_x, validation_y, budget, points.shape[1])
        model = min((ridge, polynomial), key=lambda item: item.validation_rmse)
    fit_seconds = perf_counter() - started
    estimate, relative, score, error = score_model(model, fixture, test_x, test_y)
    ridge_estimate, _, ridge_digits, ridge_error = score_model(ridge, fixture, test_x, test_y)
    return {
        "function": fixture.name, "dimension": points.shape[1], "seed": seed, "integral": "I_1",
        "random_training_sites": n, "validation_sites": len(validation_x), "bernstein_sites": bernstein_sites,
        "total_fit_sites": n + len(validation_x) + bernstein_sites,
        "holdout_sites": len(test_x), "reference": mp.nstr(fixture.exact, 45),
        "model": model.kind, "parameters": model.parameters, "water_volume": 1 - estimate,
        "estimate": estimate, "relative_error": relative, "correct_digits": score,
        "reached_12": score >= PASS_DIGITS, "holdout_rmse": error,
        "validation_rmse": model.validation_rmse, "ridge_model": ridge.kind,
        "ridge_estimate": ridge_estimate, "ridge_digits": ridge_digits, "ridge_holdout_rmse": ridge_error,
        "sample_mean_digits": digits(float(np.mean(heights[:n])), fixture.exact)[1],
        "amplitude_candidates": stats["amplitude"], "subdivision_candidates": stats["subdivision"],
        "violating_candidates": stats["violation"], "unresolved_candidates": stats["unresolved"],
        "locally_repaired_candidates": stats["locally_repaired"], "fit_seconds": fit_seconds,
    }


def evaluation_points(dimension: int, confirmation: bool) -> tuple[FloatArray, FloatArray]:
    validation_seed, test_seed = (1229, 6133) if confirmation else (707, 918)
    validation = np.random.default_rng(validation_seed).random((N_VALIDATION, dimension))
    test = np.random.default_rng(test_seed).random((N_HOLDOUT, dimension))
    return validation, test


def run_dimension(dimension: int, smoke: bool, confirmation: bool) -> tuple[int, list[dict], list[dict]]:
    rows, audits = [], []
    seeds = CONFIRMATION_SEEDS if confirmation else SEEDS
    seeds = seeds[:1] if smoke else seeds
    budgets = (Budget(128, 2, 12),) if smoke else BUDGETS
    validation_x, test_x = evaluation_points(dimension, confirmation)
    with mp.workdps(60), threadpool_limits(limits=1, user_api="blas"):
        fixtures = surfaces(dimension)
        for seed in seeds:
            points = np.random.default_rng(seed).random((max(256, budgets[-1].sites), dimension))
            for fixture in fixtures:
                heights = fixture.evaluate(points)
                validation_y = fixture.evaluate(validation_x)
                test_y = fixture.evaluate(test_x)
                if not confirmation:
                    audits.append(audit(fixture, seed, points, heights, smoke))
                for budget in budgets:
                    rows.append(fit_case(
                        fixture, budget, seed, points, heights, validation_x, validation_y, test_x, test_y,
                    ))
            print(f"{dimension}D seed={seed} completed", flush=True)
    return dimension, rows, audits


def summary(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = row["dimension"], row["function"], row["random_training_sites"]
        groups.setdefault(key, []).append(row)
    results = []
    for (dimension, name, sites), group in groups.items():
        scores = [row["correct_digits"] for row in group]
        ridge = [row["ridge_digits"] for row in group]
        results.append({
            "dimension": dimension, "function": name, "random_training_sites": sites,
            "attempts": len(group), "reached_12": sum(score >= PASS_DIGITS for score in scores),
            "min_digits": min(scores), "mean_digits": mean(scores), "max_digits": max(scores),
            "median_digits": median(scores), "seed_std_digits": pstdev(scores),
            "ridge_min_digits": min(ridge), "ridge_mean_digits": mean(ridge), "ridge_max_digits": max(ridge),
        })
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_results(destination: Path, dims, suffix: str) -> None:
    rows, audits = [], []
    for dimension, current, diagnosis in dims:
        rows.extend(current)
        audits.extend(diagnosis)
        scores = [row["correct_digits"] for row in current]
        print(f"{dimension}D: {len(current)} estimates; digits min/mean/max "
              f"{min(scores):.3f}/{mean(scores):.3f}/{max(scores):.3f}; "
              f"{sum(row['reached_12'] for row in current)} passing", flush=True)
        for filename, table in (("integrals", rows), ("summary", summary(rows)), ("range_audit", audits)):
            if table:
                write_csv(destination / f"{filename}{suffix}.csv", table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--confirmation", action="store_true", help="Use independent seeds and evaluation points")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    destination = get_mirror_path(__file__, "results")
    dimensions = DIMENSIONS[:1] if args.smoke else DIMENSIONS
    arguments = (dimensions, repeat(args.smoke), repeat(args.confirmation))
    suffix = "_smoke" if args.smoke else "_confirmation" if args.confirmation else ""
    if args.workers == 1:
        save_results(destination, map(run_dimension, *arguments), suffix)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            save_results(destination, executor.map(run_dimension, *arguments), suffix)


if __name__ == "__main__":
    main()
