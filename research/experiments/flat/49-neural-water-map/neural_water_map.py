"""Verify the neural water map on held-out integrals and a discontinuity stress case."""

import csv
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.meson_reference import physical_references
from benchmarks.meson_surfaces import meson_samples
from benchmarks.water_volume_surfaces import surfaces
from physics import PARAM_SETS
from skuld.neural_water import Array, fit_water_map
from utils.paths import get_mirror_path

TARGET = 15
ORDERS = (56, 72, 88, 104, 120)


@dataclass
class PointOracle:
    evaluate: Callable[[Array], Array]
    active: bool = True
    queries: int = 0

    def __call__(self, points: Array) -> Array:
        if not self.active:
            raise AssertionError("Volume readout attempted to query the integrand")
        self.queries += len(points)
        return self.evaluate(points)


@dataclass(frozen=True)
class Measurement:
    estimate: float
    fit_order: int
    fit_change: float
    readout_change: float
    queries: int
    weights: int
    seconds: float


def measure(
    evaluate: Callable[[Array], Array],
    bounds: Sequence[tuple[float, float]],
    breaks: Sequence[Sequence[float]],
    orders: Sequence[int],
) -> Measurement:
    """Choose training resolution without using the scoring reference."""
    started = perf_counter()
    previous = None
    total_queries = 0
    relative_fit = float("inf")
    for order in orders:
        oracle = PointOracle(evaluate)
        network = fit_water_map(oracle, bounds, order=order, breaks=breaks)
        oracle.active = False
        if oracle.queries != network.sample_count:
            raise AssertionError("Point evaluation accounting does not match the training grids")
        total_queries += oracle.queries
        parameters = network.parameter_count
        coarse_readout = network.integrate()
        estimate = network.integrate(order=order + 12)
        scale = max(abs(estimate), np.finfo(float).tiny)
        relative_readout = abs(estimate - coarse_readout) / scale
        if previous is not None:
            relative_fit = abs(estimate - previous) / scale
            if relative_fit <= 8e-16 and relative_readout <= 5e-16:
                break
        previous = estimate
        del network
    return Measurement(
        estimate, order, relative_fit, relative_readout, total_queries, parameters,
        perf_counter() - started,
    )


def score(group, name, dimension, measurement, exact, note=""):
    relative = abs(mp.mpf(measurement.estimate) - exact) / abs(exact)
    digits = float(-mp.log10(relative))
    print(f"{group:8s} {dimension}D {name:<25s} {digits:7.3f} digits "
          f"(order {measurement.fit_order}, {measurement.seconds:.2f}s)", flush=True)
    return {
        "group": group,
        "integral": name,
        "dimension": dimension,
        "estimate": repr(measurement.estimate),
        "reference": mp.nstr(exact, 35),
        "relative_error": mp.nstr(relative, 12),
        "correct_digits": digits,
        "fit_order": measurement.fit_order,
        "fit_relative_change": measurement.fit_change,
        "readout_relative_change": measurement.readout_change,
        "training_point_evaluations": measurement.queries,
        "network_weights": measurement.weights,
        "wall_seconds": round(measurement.seconds, 6),
        "note": note,
    }


def check_volume_geometry():
    oracle = PointOracle(lambda points: points[:, 0] - 0.25)
    network = fit_water_map(oracle, [(-2.0, 3.0)], order=32)
    oracle.active = False
    points = np.array([[-1.5], [0.25], [2.5]])
    np.testing.assert_allclose(network.depths(points), [-1.75, 0, 2.25], atol=2e-15, rtol=0)
    assert abs(network.integrate() - 1.25) < 1e-15
    layer_type = type(network._tanks[0].network)
    unavailable = RuntimeError("Neural depth evaluation disabled")
    with patch.object(layer_type, "_latent_grid", side_effect=unavailable):
        try:
            network.integrate()
        except RuntimeError as error:
            assert error is unavailable
        else:
            raise AssertionError("Volume readout bypassed its depth network")
    zero = fit_water_map(lambda points: np.zeros(len(points)), [(0, 1)], order=4)
    assert zero.integrate() == 0


def main():
    check_volume_geometry()
    with mp.workdps(40):
        meson = []
        for params in PARAM_SETS:
            a, b, m, n = params
            assert n == m + 1
            meson.append(measure(
                lambda points: meson_samples(points, a, b, m),
                ((0, 1),) * 3, ((),) * 3, ORDERS,
            ))
        references_low = physical_references(28)
        references_high = physical_references(36)
        reference_change = max(abs((a - b) / b) for a, b in zip(
            references_low, references_high, strict=True,
        ))
        if reference_change >= mp.mpf("1e-18"):
            raise AssertionError(f"Meson scoring references changed by {mp.nstr(reference_change, 6)}")
        print(f"Meson reference refinement: {mp.nstr(reference_change, 6)} relative", flush=True)
        rows = [
            score("meson", f"I_{i}", 3, result, exact, "physical cutoff t=80")
            for i, (result, exact) in enumerate(zip(meson, references_high, strict=True), 1)
        ]
        for dimension in (2, 3):
            bounds = ((0, 1),) * dimension
            for case in surfaces(dimension):
                group, note = "general", "uniform midpoint panels"
                breaks = ((0.5,),) * dimension
                orders = ORDERS
                if case.name == "genz_c0":
                    breaks = tuple((float(x),) for x in np.linspace(.23, .77, 6)[:dimension])
                    note = "supplied kink locations"
                if case.name == "oblique_jump":
                    group, note = "stress", "undisclosed interface"
                    orders = ORDERS[:2]
                result = measure(case.evaluate, bounds, breaks, orders)
                rows.append(score(group, case.name, dimension, result, case.exact, note))
            exact = mp.mpf(4) / 2**dimension - mp.mpf("0.7")
            result = measure(
                lambda points: 4 * np.prod(points, axis=1) - 0.7,
                bounds, ((0.5,),) * dimension, ORDERS,
            )
            rows.append(score("general", "signed_product", dimension, result, exact))

        smooth_4d = {"coupled_polynomial", "coupled_exponential", "genz_oscillatory", "genz_corner_peak"}
        for case in surfaces(4):
            if case.name in smooth_4d:
                result = measure(case.evaluate, ((0, 1),) * 4, ((),) * 4, (28, 32))
                rows.append(score("general", case.name, 4, result, case.exact, "one panel"))

    for group in ("meson", "general", "stress"):
        digits = [row["correct_digits"] for row in rows if row["group"] == group]
        print(f"{group} relative digits (min / mean / max): "
              f"{min(digits):.3f} / {mean(digits):.3f} / {max(digits):.3f}; "
              f"{sum(value >= TARGET for value in digits)}/{len(digits)} at {TARGET}+")
    path = get_mirror_path(__file__, "results") / "integrals.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return int(any(row["group"] != "stress" and row["correct_digits"] < TARGET for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
