"""Fit one compact common meson field, then integrate it against distinct weights."""

import csv
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from physics import D_R2, L_LOG, PARAM_SETS, reference_scipy
from skuld._legendre_tucker import LegendreTucker, mode_product
from utils.paths import get_mirror_path

ORDER = 24
MOMENT_ORDER = 80
COEFFICIENT_BUDGET = 2000
HOLDOUT_PARAMS = (
    (2, 0, 1, 2), (0, 2, 1, 2), (2, 1, 2, 3), (1, 2, 2, 3),
    (0, 0, 3, 4), (1, 1, 3, 4), (2, 2, 3, 4), (1, 2, 2, 4),
)


def common_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    t = np.expm1(L_LOG * z)
    d, r2 = D_R2(x, (1 - x) * y)
    return np.exp(-t * d - t / (1 + t) * r2)


def factors(points: np.ndarray, params: list[tuple[int, int, int, int]]) -> np.ndarray:
    fraction = -np.expm1(-L_LOG * points)
    # Includes the Jacobians of the simplex-to-square and the log substitution.
    return np.array([
        (points**a * (1 - points)**(b + 1), points**b,
         L_LOG * fraction**m * np.exp((1 + m - n) * L_LOG * points))
        for a, b, m, n in params
    ]).transpose(1, 0, 2)


def normalized_legendre(points: np.ndarray) -> np.ndarray:
    return legvander(points, ORDER - 1) * np.sqrt(2 * np.arange(ORDER) + 1)


def estimates(params: list[tuple[int, int, int, int]]) -> dict:
    nodes, weights = roots_legendre(ORDER)
    points = (1 + nodes) / 2
    weights = weights / 2
    field = common_field(*np.ix_(points, points, points))
    transform = normalized_legendre(nodes).T * weights
    coefficients = field
    for axis in range(3):
        coefficients = mode_product(coefficients, transform, axis)

    learned = LegendreTucker.fit(coefficients, COEFFICIENT_BUDGET)
    fixed_rank = LegendreTucker.fit(
        coefficients, COEFFICIENT_BUDGET, learned=False, same_ranks=learned.core.shape,
    )
    fixed_budget = LegendreTucker.fit(coefficients, COEFFICIENT_BUDGET, learned=False)
    for model in (learned, fixed_rank, fixed_budget):
        if model.n_coefficients > COEFFICIENT_BUDGET:
            raise RuntimeError("Model exceeded the coefficient budget")

    # This quadrature evaluates only known weights and basis functions.
    # It never samples the common meson field.
    moment_nodes, moment_weights = roots_legendre(MOMENT_ORDER)
    moment_design = normalized_legendre(moment_nodes)
    moments = factors((1 + moment_nodes) / 2, params) @ (moment_design * (moment_weights[:, None] / 2))
    check_nodes, check_weights = roots_legendre(2 * MOMENT_ORDER)
    check = factors((1 + check_nodes) / 2, params) @ (
        normalized_legendre(check_nodes) * (check_weights[:, None] / 2)
    )
    np.testing.assert_allclose(moments, check, atol=2e-12, rtol=0)

    scores = {
        name: (model.integrate_weighted(moments), model.core.shape, model.n_coefficients)
        for name, model in (("learned", learned), ("fixed_rank", fixed_rank), ("fixed_budget", fixed_budget))
    }
    gauss = np.einsum("ci,cj,ck,ijk->c", *(factors(points, params) * weights), field, optimize=True)
    scores["direct_gauss"] = (gauss, (), 0)
    return scores


def main() -> None:
    params = [*PARAM_SETS, *HOLDOUT_PARAMS]
    predictions = estimates(params)
    references = [reference_scipy(*parameters, tol=1e-12)[0] for parameters in params]
    rows = []
    for method, (values, ranks, size) in predictions.items():
        for index, (parameters, estimate, reference) in enumerate(zip(params, values, references, strict=True), 1):
            relative_error = abs(estimate - reference) / abs(reference)
            rows.append({
                "split": "original" if index <= len(PARAM_SETS) else "holdout",
                "integral": f"I_{index}", "method": method,
                "a": parameters[0], "b": parameters[1], "m": parameters[2], "n": parameters[3],
                "estimate": estimate, "reference": reference, "relative_error": relative_error,
                "correct_digits": -math.log10(relative_error) if relative_error else math.inf,
                "ranks": "x".join(map(str, ranks)), "coefficients": size, "field_sites": ORDER**3,
            })
        for split in ("original", "holdout"):
            digits = [row["correct_digits"] for row in rows if row["method"] == method and row["split"] == split]
            print(f"{method:12s} {split:8s} {str(ranks):13s} {size:4d} coefficients  "
                  f"digits min/mean/max {min(digits):.2f}/{mean(digits):.2f}/{max(digits):.2f}", flush=True)

    dest = get_mirror_path(__file__, "results") / "factored_meson_tucker.csv"
    with dest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(dest)


if __name__ == "__main__":
    main()
