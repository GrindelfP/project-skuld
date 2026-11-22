"""Test which parts of the meson water-volume pipeline matter at 27 oracle sites."""

import csv
import math
import sys
from pathlib import Path
from statistics import mean

import mpmath as mp
import numpy as np
import torch
from numpy.polynomial.polynomial import polyvander2d, polyvander3d
from scipy.optimize import least_squares
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.meson_reference import physical_references
from benchmarks.meson_surfaces import known_weights, training_values
from physics import L_LOG, PARAM_SETS, T_MAX
from skuld._exponential_water_volume import ExponentialWaterSurface
from skuld._neural_water_volume import fit_neural_surface
from utils.paths import get_mirror_path

STARTS = ((42, (3.7, -3.3)), (123, (5.1, -3.8)), (2026, (3.2, -5.2)))
TARGET = 12.0
WEIGHTS = tuple((a, b, m) for a, b, m, _ in PARAM_SETS)


def mesh(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    points = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    measure = np.einsum("i,j,k->ijk", weights, weights, weights).ravel()
    return points, measure


def weighted_integrals(points: np.ndarray, measure: np.ndarray, surface: np.ndarray) -> np.ndarray:
    field = measure * surface
    return L_LOG * np.array([math.fsum(field * known_weights(points, *case)) for case in WEIGHTS])


def integrated_surrogate(log_field) -> np.ndarray:
    # Applies the same high-order check to each fitted surrogate, with no extra oracle calls.
    estimates = []
    for order in (32, 48):
        points, measure = mesh(order)
        estimates.append(weighted_integrals(points, measure, np.exp(log_field(points))))
    first, second = estimates
    if np.max(np.abs((first - second) / second)) > 2e-13:
        raise RuntimeError("Surrogate quadrature did not converge")
    return second


def anchored_features(points: np.ndarray, rates: np.ndarray) -> np.ndarray:
    xy = 2 * points[:, :2] - 1
    spatial = polyvander2d(xy[:, 0], xy[:, 1], (2, 2))
    temporal = np.exp(-rates / 2) * np.expm1(points[:, 2:3] * rates)
    return (temporal[:, :, None] * spatial[:, None, :]).reshape(len(points), -1)


def regression(
    points: np.ndarray, values: np.ndarray, start: tuple[float, float], *, learn_rates: bool,
) -> ExponentialWaterSurface:
    """Non-neural variable projection with exactly the same anchored function class."""
    target = np.log(values)

    def project(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = anchored_features(points, rates)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        return coefficients, design @ coefficients - target

    rates = np.array(start)
    if learn_rates:
        result = least_squares(
            lambda r: project(r)[1], rates, bounds=([.15, -8], [8, -.15]),
            gtol=None, ftol=1e-14, xtol=1e-14,
        )
        if not result.success:
            raise RuntimeError(result.message)
        rates = result.x
    coefficients = project(rates)[0].reshape(2, 3, 3)
    static = -np.einsum("i,ipq->pq", np.exp(-rates / 2), coefficients)
    return ExponentialWaterSurface(rates, np.concatenate((static[None], coefficients)))


def analytic_integrals(model: ExponentialWaterSurface) -> np.ndarray:
    values = L_LOG * np.asarray(model.integrate_weighted(WEIGHTS, L_LOG, series_tolerance=1e-24))
    # Algebraically evaluate scale*(1-A) to retain the small occupied volume.
    scale = L_LOG * np.exp(model.log_upper_bound())
    occupied = values / scale
    if not np.all((0 <= occupied) & (occupied <= 1)):
        raise RuntimeError("Occupied volume outside the unit cube")
    return values


def fit_polynomial(points: np.ndarray, values: np.ndarray):
    def features(probes):
        x, y, z = probes.T
        return polyvander3d(2 * x - 1, 2 * y - 1, 2 * z - 1, (2, 2, 2)) * z[:, None]

    coefficients = np.linalg.lstsq(features(points), np.log(values), rcond=None)[0]
    return lambda probes: features(probes) @ coefficients


def main() -> int:
    torch.set_num_threads(1)
    if any(n != m + 1 for _, _, m, n in PARAM_SETS):
        raise ValueError("Expected the shared meson envelope")
    points, measure = mesh(3)
    values = training_values(points)
    methods = [
        ("direct_gauss", None, weighted_integrals(points, measure, values)),
        ("polynomial", None, integrated_surrogate(fit_polynomial(points, values))),
        ("physics_fixed", None, analytic_integrals(regression(
            points, values, (L_LOG, -L_LOG), learn_rates=False,
        ))),
    ]
    for seed, start in STARTS:
        fit = fit_neural_surface(points, values, seed=seed, start_rates=start, anchored=True)
        if min(fit.layer_updates.values()) <= 1e-6:
            raise RuntimeError("A layer did not learn")
        network = fit.network.moment_representation()
        estimates = analytic_integrals(network)
        check = integrated_surrogate(network.log_surface)
        if np.max(np.abs((estimates - check) / check)) > 2e-13:
            raise RuntimeError("Analytic readout disagrees with quadrature of the trained model")
        methods.extend((
            ("neural", seed, estimates),
            ("frozen_initial_rates", seed, integrated_surrogate(
                regression(points, values, start, learn_rates=False).log_surface
            )),
            ("anchored_regression", seed, analytic_integrals(
                regression(points, values, start, learn_rates=True)
            )),
        ))

    rows, summaries = [], []
    with mp.workdps(45):
        low, exact = physical_references(26), physical_references(36)
        if max(abs((a - b) / b) for a, b in zip(low, exact, strict=True)) > mp.mpf("1e-15"):
            raise RuntimeError("Independent references did not converge")
        for method, seed, estimates in methods:
            scores = []
            for index, (params, prediction, ref) in enumerate(zip(PARAM_SETS, estimates, exact, strict=True), 1):
                relative = abs((mp.mpf(float(prediction)) - ref) / ref)
                digits = float(-mp.log10(relative))
                a, b, m, n = params
                rows.append({
                    "integral": f"I_{index}", "method": method, "seed": seed,
                    "a": a, "b": b, "m": m, "n": n, "t_max": T_MAX, "oracle_sites": len(points),
                    "estimate": prediction, "reference": mp.nstr(ref, 38),
                    "relative_error": float(relative), "correct_digits": digits,
                })
                scores.append(digits)
            summaries.append({
                "method": method, "seed": seed, "oracle_sites": len(points),
                "min_digits": min(scores), "mean_digits": mean(scores), "max_digits": max(scores),
                "reached_12": sum(digits >= TARGET for digits in scores),
            })
            print(f"{method} {seed}: {min(scores):.3f}/{mean(scores):.3f}/{max(scores):.3f} digits", flush=True)
    dest = get_mirror_path(__file__, "results")
    for name, data in (("integrals.csv", rows), ("summary.csv", summaries)):
        with (dest / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    return 0 if all(row["reached_12"] == 8 for row in summaries if row["method"] == "neural") else 1


if __name__ == "__main__":
    raise SystemExit(main())
