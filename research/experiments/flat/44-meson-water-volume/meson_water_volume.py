"""Compute all eight finite-cutoff meson integrals with bounded water surfaces."""

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from physics import D_R2, L_LOG, M1, M2, M3, PARAM_SETS, P1P1, P2P2, PP, T_MAX
from skuld._piecewise_bernstein_volume import PiecewiseBernsteinSurface, collocation_points
from utils.paths import get_mirror_path

CELLS = (7, 5, 12)
DEGREE = 12
VALIDATION_SITES = 4096
TARGET_DIGITS = 12


def height_scales() -> dict[int, float]:
    # On the simplex, alpha1*alpha2 and alpha3*(alpha1+alpha2) are at most 1/4.
    # The mixed term in R² is nonnegative with the current kinematics.
    if PP > P1P1 + P2P2:
        raise ValueError("Re-derive the surface scale for these kinematics")
    p_bound = max(abs(P1P1), abs(P2P2))
    d_lower = min(M1**2, M2**2, M3**2) - (abs(PP) + p_bound) / 4
    exponent_upper = T_MAX * max(0.0, -d_lower) + p_bound
    return {m: L_LOG**(m + 1) * math.exp(exponent_upper) for m in {p[2] for p in PARAM_SETS}}


def smooth_heights(x, y, z, scales: dict[int, float]) -> dict[int, np.ndarray]:
    """Remove the known boundary factors before fitting the bounded surface."""
    t = np.expm1(L_LOG * z)
    d, r2 = D_R2(x, (1 - x) * y)
    common = L_LOG * np.exp(-t * d - t / (1 + t) * r2)
    fraction_per_z = np.divide(
        -np.expm1(-L_LOG * z), z, out=np.full_like(z, L_LOG), where=z != 0,
    )
    return {m: common * fraction_per_z**m / scale for m, scale in scales.items()}


def powers_for(params: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    a, b, m, n = params
    if n != m + 1:
        raise ValueError("This polynomial factorization requires n = m + 1")
    return (a, b + 1), (b, 0), (m, 0)


def predict_integrals(models: dict[int, PiecewiseBernsteinSurface], scales: dict[int, float]):
    for params in PARAM_SETS:
        m = params[2]
        occupied = models[m].integrate_weighted(powers_for(params))
        # A = 1 - occupied. Evaluate scale*(1-A) without rounding A first.
        yield scales[m] * occupied, occupied


def physical_references(order: int) -> list[mp.mpf]:
    """Independent Gauss integration on the physical simplex and linear t panels."""
    nodes, weights = mp.gauss_quadrature(order, "legendre01")
    panels = (0, 1, 4, 16, 40, T_MAX)
    time_nodes = []
    for lo, hi in zip(panels[:-1], panels[1:], strict=True):
        for z, w in zip(nodes, weights, strict=True):
            t = lo + (hi - lo) * z
            ratio = t / (1 + t)
            time_nodes.append((t, ratio, (hi - lo) * w * ratio / (1 + t)))

    totals = [mp.mpf(0) for _ in PARAM_SETS]
    for x, wx in zip(nodes, weights, strict=True):
        for y, wy in zip(nodes, weights, strict=True):
            alpha1, alpha2 = x, (1 - x) * y
            d, r2 = D_R2(alpha1, alpha2)
            m1 = m2 = mp.mpf(0)
            for t, ratio, wt in time_nodes:
                value = wt * mp.exp(-t * d - ratio * r2)
                m1 += value
                m2 += value * ratio
            spatial = wx * wy * (1 - x)
            for index, (a, b, m, n) in enumerate(PARAM_SETS):
                if (m, n) not in {(1, 2), (2, 3)}:
                    raise ValueError("Reference time weights require the standard meson exponents")
                totals[index] += spatial * alpha1**a * alpha2**b * (m1 if m == 1 else m2)
    return totals


def main() -> None:
    scales = height_scales()
    started = perf_counter()
    axes = collocation_points(CELLS, DEGREE)
    sites = math.prod(map(len, axes))
    sample = smooth_heights(*np.ix_(*axes), scales)
    models = {m: PiecewiseBernsteinSurface.fit(values, CELLS, DEGREE) for m, values in sample.items()}
    fit_seconds = perf_counter() - started

    started = perf_counter()
    estimates = list(predict_integrals(models, scales))
    readout_seconds = perf_counter() - started
    points = np.random.default_rng(44).random((VALIDATION_SITES, 3))
    targets = smooth_heights(*points.T, scales)
    holdout = {}
    for m, model in models.items():
        prediction = model.height(points)
        np.testing.assert_array_less(-1e-14, prediction)
        np.testing.assert_array_less(prediction, 1 + 1e-14)
        holdout[m] = float(np.linalg.norm(prediction - targets[m]) / np.linalg.norm(targets[m]))

    n_coefficients = sum(model.coefficients.size for model in models.values())
    print(f"{sites:,} fit sites, {n_coefficients:,} output weights, fit {fit_seconds:.3f}s, "
          f"readout {readout_seconds:.3f}s", flush=True)
    print(f"Validating references on the physical domain, 0 <= t <= {T_MAX:g}", flush=True)
    rows = []
    with mp.workdps(45):
        coarse = physical_references(26)
        fine = physical_references(36)
        for index, (params, values, low, exact) in enumerate(
            zip(PARAM_SETS, estimates, coarse, fine, strict=True), 1,
        ):
            a, b, m, n = params
            estimate, occupied = values
            reference_delta = abs((low - exact) / exact)
            if reference_delta > mp.mpf("1e-15"):
                raise RuntimeError(f"I_{index} reference did not converge: {reference_delta}")
            relative = abs((mp.mpf(estimate) - exact) / exact)
            digits = float(-mp.log10(relative))
            row = {
                "integral": f"I_{index}", "a": a, "b": b, "m": m, "n": n, "t_max": T_MAX,
                "surface_scale": scales[m], "water_volume": 1 - occupied,
                "occupied_volume": occupied, "estimate": estimate, "reference": mp.nstr(exact, 38),
                "ref_relative_delta": float(reference_delta), "relative_error": float(relative),
                "correct_digits": digits, "holdout_relative_l2": holdout[m], "training_sites": sites,
                "output_weights_total": n_coefficients,
            }
            rows.append(row)
            print(f"I_{index}  {estimate:.16g}  rel={float(relative):.3e}  digits={digits:.3f}", flush=True)

    scores = [row["correct_digits"] for row in rows]
    print(f"Correct digits (min / mean / max): {min(scores):.3f} / {mean(scores):.3f} / {max(scores):.3f}")
    print(f"Target {TARGET_DIGITS} digits reached on {sum(d >= TARGET_DIGITS for d in scores)}/{len(scores)}")
    path = get_mirror_path(__file__, "results") / "meson_water_volume.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
