"""Measure neural meson water volumes beyond float64 resolution."""

import csv
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np
import torch
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.meson_reference import physical_references
from physics import D_R2, PARAM_SETS, T_MAX
from skuld._neural_water_volume import fit_neural_surface
from skuld._precision_water_volume import NeuralWaterGeometry, PrecisionSurface
from utils.paths import get_mirror_path

SEED = 42
START_RATES = (3.7, -3.3)
PRECISION = 70
TARGET = 12
WEIGHTS = tuple((a, b, m) for a, b, m, _ in PARAM_SETS)


def point_value(point, extent):
    """One pointwise meson sample with its known positive envelope factored out."""
    x, y, z = point
    t = mp.expm1(extent * z)
    d, r2 = D_R2(x, (1 - x) * y)
    return mp.exp(-t * d - (t / (1 + t)) * r2)


def main() -> int:
    torch.set_num_threads(1)
    if any(n != m + 1 for _, _, m, n in PARAM_SETS):
        raise ValueError("Expected the shared meson envelope")
    nodes = (roots_legendre(3)[0] + 1) / 2
    inputs = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    with mp.workdps(PRECISION):
        started = perf_counter()
        extent = mp.log(1 + mp.mpf(T_MAX))
        points = [tuple(mp.mpf(float(value)) for value in point) for point in inputs]
        samples = [point_value(point, extent) for point in points]
        learned = fit_neural_surface(
            inputs, np.asarray(samples, dtype=float), seed=SEED, start_rates=START_RATES, anchored=True,
        )
        if min(learned.layer_updates.values()) < 1e-6:
            raise RuntimeError("A neural layer did not learn")
        surface = PrecisionSurface.from_network(learned.network)
        probe = tuple(mp.mpf(value) for value in (".13", ".77", ".65"))
        with torch.inference_mode():
            before = float(learned.network(torch.tensor([list(map(float, probe))], dtype=torch.float64), log=True)[0])
        if abs(mp.mpf(before) - surface.log_and_gradient(probe)[0]) > mp.mpf("1e-13"):
            raise RuntimeError("Precision refinement changed the network architecture")
        initial_residual, final_residual = surface.refine(points, samples)
        print(f"Training max log residual {mp.nstr(initial_residual, 5)} -> {mp.nstr(final_residual, 5)} "
              f"in {perf_counter() - started:.2f}s; {len(points)} point evaluations", flush=True)

        geometry = NeuralWaterGeometry(surface, extent)
        if not all(0 <= geometry.water_depth(probe, *weights) <= 1 for weights in WEIGHTS):
            raise RuntimeError("Water depth must lie in the unit cube")
        started = perf_counter()
        scale = geometry.scale()
        coarse = [scale * (1 - volume) for volume in geometry.water_volumes(WEIGHTS, tolerance=mp.mpf("1e-38"))]
        water = geometry.water_volumes(WEIGHTS, tolerance=mp.mpf("1e-46"))
        estimates = [scale * (1 - volume) for volume in water]
        readout_delta = max(abs((a-b) / b) for a, b in zip(coarse, estimates, strict=True))
        if readout_delta > mp.mpf("1e-37"):
            raise RuntimeError("Volume readout did not converge")
        print(f"Moment readout converged within {mp.nstr(readout_delta, 5)} in "
              f"{perf_counter() - started:.2f}s", flush=True)

        ref_low, ref_high = physical_references(48), physical_references(60)
        ref_delta = max(abs((a-b) / b) for a, b in zip(ref_low, ref_high, strict=True))
        if ref_delta > mp.mpf("1e-37"):
            raise RuntimeError(f"Reference refinement changed by {mp.nstr(ref_delta, 5)}")
        rows = []
        for index, (params, estimate, volume, ref) in enumerate(
            zip(PARAM_SETS, estimates, water, ref_high, strict=True), 1,
        ):
            error = abs(estimate - ref)
            relative = error / abs(ref)
            digits = float(-mp.log10(relative))
            a, b, m, n = params
            rows.append({
                "integral": f"I_{index}", "a": a, "b": b, "m": m, "n": n, "seed": SEED,
                "t_max": T_MAX, "point_evaluations": len(points), "working_decimal_precision": PRECISION,
                "estimate": mp.nstr(estimate, 62), "reference": mp.nstr(ref, 62),
                "absolute_error": mp.nstr(error, 8), "relative_error": mp.nstr(relative, 8),
                "correct_digits": digits, "water_volume": mp.nstr(volume, 62), "scale": mp.nstr(scale, 62),
                "training_log_max_residual": mp.nstr(final_residual, 8),
                "readout_relative_change": mp.nstr(readout_delta, 8),
                "reference_relative_change": mp.nstr(ref_delta, 8),
            })
            print(f"I_{index} relative error {mp.nstr(relative, 5)}   digits {digits:.3f}", flush=True)

    scores = [row["correct_digits"] for row in rows]
    print(f"Correct digits (min / mean / max): {min(scores):.3f} / {mean(scores):.3f} / {max(scores):.3f}")
    path = get_mirror_path(__file__, "results") / "integrals.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return 0 if min(scores) >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
