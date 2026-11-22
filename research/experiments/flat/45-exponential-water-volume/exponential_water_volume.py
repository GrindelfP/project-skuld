"""Integrate eight bounded meson surfaces with trainable exponential features."""

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.meson_reference import physical_references
from physics import D_R2, L_LOG, PARAM_SETS, T_MAX
from skuld._exponential_water_volume import ExponentialWaterSurface
from utils.paths import get_mirror_path

STARTS = ((42, (1.5, -1.5)), (123, (2.0, -2.0)), (2026, (3.0, -1.0)))
TRAINING_SITES = 384
HOLDOUT_SITES = 4096
TARGET_DIGITS = 12
WEIGHTS = tuple((a, b, m) for a, b, m, _ in PARAM_SETS)


def known_weights(points: np.ndarray, a: int, b: int, m: int) -> np.ndarray:
    x, y, z = points.T
    return x**a * (1 - x)**(b + 1) * y**b * (-np.expm1(-L_LOG * z))**m


def meson_samples(points: np.ndarray, a: int, b: int, m: int) -> np.ndarray:
    """Evaluate a full meson integrand after simplex and log-time substitutions."""
    x, y, z = points.T
    t = np.expm1(L_LOG * z)
    d, r2 = D_R2(x, (1 - x) * y)
    return L_LOG * known_weights(points, a, b, m) * np.exp(-t * d - t / (1 + t) * r2)


def training_values(points: np.ndarray) -> np.ndarray:
    """Use only pointwise values of one channel after removing its known envelope."""
    return meson_samples(points, 0, 0, 1) / (L_LOG * known_weights(points, 0, 0, 1))


def check_by_gauss(model: ExponentialWaterSurface, order: int) -> np.ndarray:
    """Integrate only the fitted model for a readout check and the frozen control."""
    nodes, weights = roots_legendre(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    points = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    measure = np.einsum("i,j,k->ijk", weights, weights, weights).ravel()
    field = np.exp(model.log_surface(points)) * measure
    return L_LOG * np.array([math.fsum(field * known_weights(points, *case)) for case in WEIGHTS])


def fit_and_predict(seed: int, start: tuple[float, float], method: str) -> tuple:
    learned = method == "learned"
    rng = np.random.default_rng(seed)
    points = rng.random((TRAINING_SITES, 3))
    model = ExponentialWaterSurface.fit(
        points, training_values(points), np.asarray(start), learn_rates=learned,
    )
    if learned and np.any(np.abs(model.rates - start) < 0.1):
        raise RuntimeError("The learned temporal features did not move from initialization")

    log_bound = model.log_upper_bound()
    scale = L_LOG * math.exp(log_bound)
    holdout = np.random.default_rng(94).random((HOLDOUT_SITES, 3))
    normalized_field = np.exp(model.log_surface(holdout) - log_bound)
    surface_height = normalized_field * known_weights(holdout, 0, 0, 1)
    water_depth = 1 - surface_height
    if np.any((water_depth < 0) | (water_depth > 1)):
        raise RuntimeError("Model exceeded its unit-height normalization")
    target_height = meson_samples(holdout, 0, 0, 1) / scale
    relative_l2 = float(np.linalg.norm(surface_height - target_height) / np.linalg.norm(target_height))

    low_check, high_check = (check_by_gauss(model, order) for order in (32, 48))
    if np.max(np.abs(low_check - high_check) / np.abs(high_check)) > 2e-13:
        raise RuntimeError("Independent network-output check did not converge")
    estimations_by_series = [None] * len(WEIGHTS)
    estimates = high_check
    if learned:
        values = L_LOG * np.asarray(model.integrate_weighted(WEIGHTS, L_LOG))
        estimates = L_LOG * np.asarray(model.integrate_weighted(WEIGHTS, L_LOG, series_tolerance=1e-24))
        estimations_by_series = np.abs(values - estimates) / np.abs(estimates)
        if np.max(estimations_by_series) > 2e-13:
            raise RuntimeError("Activation series did not converge")
    delta = np.abs(estimates - high_check) / np.abs(high_check)
    if np.max(delta) > 2e-13:
        raise RuntimeError("Activation series disagrees with the independent network-output check")
    occupied = estimates / scale
    if not np.all((occupied >= 0) & (occupied <= 1)):
        raise RuntimeError("Learned occupied volume left the unit cube")
    # This is scale * (1 - A), with the complement cancelled before rounding A.
    return model, estimates, scale, occupied, relative_l2, estimations_by_series, high_check, delta


def main() -> int:
    if any(n != m + 1 for _, _, m, n in PARAM_SETS):
        raise ValueError("This envelope requires n = m + 1")
    predictions = []
    for seed, start in STARTS:
        methods = (("learned", start), ("frozen", start))
        if seed == STARTS[0][0]:
            methods += (("physics_fixed", (L_LOG, -L_LOG)),)
        for name, rates in methods:
            started = perf_counter()
            result = fit_and_predict(seed, rates, name)
            model, _, _, _, holdout, _, _, _ = result
            print(f"{seed} {name}: rates={model.rates}, holdout relative L2={holdout:.3e}, "
                  f"fit and readout {perf_counter() - started:.3f}s", flush=True)
            predictions.append((seed, rates, name, result))

    rows = []
    summaries = []
    with mp.workdps(45):
        coarse = physical_references(26)
        exact = physical_references(36)
        deltas = [abs((low - value) / value) for low, value in zip(coarse, exact, strict=True)]
        if max(deltas) > mp.mpf("1e-15"):
            raise RuntimeError("Independent references did not converge")
        for seed, start, method, result in predictions:
            model, estimates, scale, occupied, holdout, changes, checks, deltas_readout = result
            scores = []
            for i, (params, prediction, reference) in enumerate(zip(PARAM_SETS, estimates, exact, strict=True)):
                relative = abs(mp.mpf(float(prediction)) - reference) / abs(reference)
                digits = float(-mp.log10(relative))
                a, b, m, n = params
                rows.append({
                    "integral": f"I_{i+1}", "a": a, "b": b, "m": m, "n": n, "t_max": T_MAX,
                    "seed": seed, "method": method, "initial_positive_rate": start[0],
                    "initial_negative_rate": start[1], "positive_rate": model.rates[0],
                    "negative_rate": model.rates[1], "training_sites": TRAINING_SITES,
                    "optimized_parameters": model.coefficients.size + (model.rates.size if method == "learned" else 0),
                    "estimate": prediction, "reference": mp.nstr(reference, 38),
                    "relative_error": float(relative), "correct_digits": digits, "holdout_relative_l2": holdout,
                    "scale": scale, "occupied_volume": occupied[i], "water_volume": 1 - occupied[i],
                    "network_gauss_check": checks[i], "network_gauss_relative_difference": deltas_readout[i],
                    "series_relative_change": changes[i], "reference_relative_change": float(deltas[i]),
                })
                scores.append(digits)
            summaries.append({
                "seed": seed, "method": method, "reached_12": sum(value >= TARGET_DIGITS for value in scores),
                "min_digits": min(scores), "mean_digits": mean(scores), "max_digits": max(scores),
            })
            print(f"{seed} {method}: min/mean/max digits "
                  f"{min(scores):.3f}/{mean(scores):.3f}/{max(scores):.3f}", flush=True)

    directory = get_mirror_path(__file__, "results")
    for filename, data in (("integrals.csv", rows), ("summary.csv", summaries)):
        with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    return 0 if all(
        row["reached_12"] == len(PARAM_SETS) for row in summaries if row["method"] == "learned"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
