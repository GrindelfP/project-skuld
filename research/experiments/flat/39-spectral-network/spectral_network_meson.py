"""Fit a polynomial neural network to meson integrands and integrate its outputs."""

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from physics import D_R2, L_LOG, PARAM_SETS, T_MAX, reference_scipy
from skuld._spectral_network import SpectralNetwork
from utils.paths import get_mirror_path

DEGREE = 47
FIBERS_PER_AXIS = 384
READOUT_NODES = 56
RANK_TOLERANCE = 1e-15
VALIDATION_POINTS = 4096
REFERENCE_TOLERANCE = 1e-12
TARGET_RELATIVE_ERROR = 1e-12
FloatArray = NDArray[np.float64]


def meson_values(x: FloatArray, y: FloatArray, z: FloatArray) -> FloatArray:
    """Sample all eight full integrands after mapping the simplex to the cube."""
    alpha1 = x
    alpha2 = (1 - x) * y
    t = np.expm1(L_LOG * z)
    d, r2 = D_R2(alpha1, alpha2)
    jacobian = (1 - x) * L_LOG * (1 + t)
    common = jacobian * np.exp(-t * d - t / (1 + t) * r2)
    return np.stack([
        common * alpha1**a * alpha2**b * t**m / (1 + t)**n
        for a, b, m, n in PARAM_SETS
    ], axis=-1)


def fit_meson_network(seed: int) -> SpectralNetwork:
    return SpectralNetwork.fit(
        meson_values,
        degree=DEGREE,
        fibers=FIBERS_PER_AXIS,
        readout_nodes=READOUT_NODES,
        rank_tolerance=RANK_TOLERANCE,
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the training sites")
    args = parser.parse_args()

    samples = 3 * FIBERS_PER_AXIS * (DEGREE + 1) + READOUT_NODES**3
    print(f"Fitting the spectral network, seed {args.seed}, {samples:,} sites", flush=True)
    start = perf_counter()
    model = fit_meson_network(args.seed)
    fit_seconds = perf_counter() - start
    widths = [w.shape[1] for w in model.branches]
    parameters = sum(w.numel() for w in model.parameters())
    print(f"Hidden widths {widths}, {parameters:,} parameters, fitted in {fit_seconds:.3f} s", flush=True)

    with torch.inference_mode():
        start = perf_counter()
        estimates = model.integrate().tolist()
        readout_seconds = perf_counter() - start

        points = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(1,))).random(
            (VALIDATION_POINTS, 3)
        )
        exact = meson_values(*points.T)
        predicted = model(torch.from_numpy(points)).numpy()
        relative_l2 = np.linalg.norm(predicted - exact, axis=0) / np.linalg.norm(exact, axis=0)
    print(f"Analytic integral readout for all outputs took {readout_seconds:.3e} s")
    print(f"Scoring against independent adaptive integrals on 0 <= t <= {T_MAX:g}", flush=True)

    rows = []
    for index, (params, estimate, holdout_l2) in enumerate(
        zip(PARAM_SETS, estimates, relative_l2, strict=True), start=1
    ):
        reference, outer_error = reference_scipy(*params, tol=REFERENCE_TOLERANCE)
        error_abs = abs(estimate - reference)
        error_rel = error_abs / abs(reference)
        digits = -math.log10(error_rel) if error_rel else math.inf
        label = f"I_{index}"
        print(f"{label}: estimate={estimate:.16g}, relative error={error_rel:.3e}, "
              f"digits={digits:.2f}, holdout relative L2={holdout_l2:.3e}", flush=True)
        a, b, m, n = params
        rows.append({
            "integral": label, "a": a, "b": b, "m": m, "n": n, "seed": args.seed,
            "network": estimate, "scipy": reference, "scipy_outer_error": outer_error,
            "err_abs": error_abs, "err_rel": error_rel, "correct_digits": digits,
            "holdout_relative_l2": holdout_l2,
        })

    scores = [r["correct_digits"] for r in rows]
    print(f"Correct digits (min / mean / max): {min(scores):.2f} / {mean(scores):.2f} / {max(scores):.2f}")
    results_path = get_mirror_path(__file__, "results") / f"spectral_network_meson_seed_{args.seed}.csv"
    with results_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {results_path}")

    passed = all(r["err_rel"] <= TARGET_RELATIVE_ERROR for r in rows)
    print(f"Relative error <= {TARGET_RELATIVE_ERROR:g} for all integrals: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
