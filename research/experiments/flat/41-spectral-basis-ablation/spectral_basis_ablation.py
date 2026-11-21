"""Measure integral accuracy attributable to learning a compact spectral basis."""

import csv
import math
import sys
from collections.abc import Callable
from pathlib import Path
from runpy import run_path
from statistics import mean

import numpy as np
import torch
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.genz import GENZ_FUNCTIONS
from skuld._spectral_network import SpectralNetwork
from utils.paths import get_mirror_path


DIMENSIONS = 3
DEGREE = 31
MAX_RANK = 8
FIBERS = 64
GRID_NODES = 36
GAUSS_NODES = 37
SEEDS = (1, 42, 1729, 2024, 2026)
FIBER_SITES = DIMENSIONS * FIBERS * (DEGREE + 1)
GRID_SITES = GRID_NODES**DIMENSIONS
TRAINING_SITES = FIBER_SITES + GRID_SITES


def genz_values(*coordinates: np.ndarray) -> np.ndarray:
    points = np.stack(np.broadcast_arrays(*coordinates), axis=-1)
    flat_points = points.reshape(-1, DIMENSIONS)
    channels = [family.numpy_fn(flat_points, *family.params) for family in GENZ_FUNCTIONS]
    return np.stack(channels, axis=-1).reshape(*points.shape[:-1], len(channels))


def estimates_for_seed(function: Callable, seed: int) -> tuple[tuple[int, ...], dict]:
    samples = None

    def capture(*coordinates: np.ndarray) -> np.ndarray:
        nonlocal samples
        values = function(*coordinates)
        samples = coordinates, values
        return values

    learned = SpectralNetwork.fit(
        capture, degree=DEGREE, fibers=FIBERS, readout_nodes=GRID_NODES,
        rank_tolerance=1e-15, seed=seed, dimensions=DIMENSIONS, max_rank=MAX_RANK,
    )
    if samples is None:
        raise RuntimeError("Fit did not return sampled readout data")
    # The fit's last call samples the shared readout grid.
    coordinates, grid_values = samples
    grid = np.stack([axis.ravel() for axis in coordinates])

    widths = tuple(branch.shape[1] for branch in learned.branches)
    identity = np.eye(DEGREE + 1)
    compact_bases = [np.ascontiguousarray(identity[:, :width]) for width in widths]
    fixed = SpectralNetwork.fit_readout(compact_bases, grid, grid_values)
    full = SpectralNetwork.fit_readout([identity] * DIMENSIONS, grid, grid_values)
    if sum(p.numel() for p in learned.parameters()) != sum(p.numel() for p in fixed.parameters()):
        raise RuntimeError("Compact models must have the same number of stored coefficients")
    with torch.inference_mode():
        return widths, {
            "learned": (learned.integrate().tolist(), TRAINING_SITES),
            "fixed_compact": (fixed.integrate().tolist(), GRID_SITES),
            "fixed_full": (full.integrate().tolist(), GRID_SITES),
        }


def direct_gauss(function: Callable) -> tuple[list[float], int]:
    nodes, weights = roots_legendre(GAUSS_NODES)
    axes = [(nodes + 1) / 2] * DIMENSIONS
    weights = weights / 2
    values = function(*np.ix_(*axes))
    integrals = np.einsum("i,j,k,ijkc->c", weights, weights, weights, values, optimize=True)
    return integrals.tolist(), GAUSS_NODES**DIMENSIONS


def compare(
    dataset: str,
    function: Callable,
    references: Callable,
) -> list[dict]:
    predictions = {seed: estimates_for_seed(function, seed) for seed in SEEDS}
    gauss = direct_gauss(function)
    exact = references()
    rows = []
    for seed, (widths, methods) in predictions.items():
        for method, (estimates, sites) in {**methods, "direct_gauss": gauss}.items():
            digits = []
            for (label, reference), estimate in zip(exact, estimates, strict=True):
                error = abs(estimate - reference) / abs(reference)
                score = -math.log10(error) if error else math.inf
                digits.append(score)
                rows.append({
                    "dataset": dataset, "seed": seed, "method": method, "integral": label,
                    "estimate": estimate, "reference": reference, "relative_error": error,
                    "correct_digits": score, "sample_sites": sites,
                    "basis_widths": "x".join(map(str, widths)) if method in ("learned", "fixed_compact") else "",
                })
            print(
                f"{dataset:5s} seed={seed:4d} {method:13s} "
                f"digits min/mean/max {min(digits):6.2f}/{mean(digits):6.2f}/{max(digits):6.2f}",
                flush=True,
            )
    return rows


def main() -> None:
    experiments = Path(__file__).resolve().parents[1]
    (meson_script,) = experiments.glob("*-spectral-network/spectral_network_meson.py")
    meson = run_path(str(meson_script))

    def meson_references() -> list[tuple[str, float]]:
        return [
            (f"I_{index}", meson["reference_scipy"](*params, tol=meson["REFERENCE_TOLERANCE"])[0])
            for index, params in enumerate(meson["PARAM_SETS"], start=1)
        ]

    def genz_references() -> list[tuple[str, float]]:
        return [(family.name, family.exact(DIMENSIONS)) for family in GENZ_FUNCTIONS]

    rows = compare("meson", meson["meson_values"], meson_references)
    rows.extend(compare("genz", genz_values, genz_references))
    dest = get_mirror_path(__file__, "results") / "spectral_basis_ablation.csv"
    with dest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(dest)


if __name__ == "__main__":
    main()
