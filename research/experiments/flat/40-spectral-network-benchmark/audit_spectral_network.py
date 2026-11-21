"""Compare learned and fixed spectral bases on identical meson samples."""

import csv
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebvander

from spectral_benchmark import SEEDS, SpectralNetwork, get_mirror_path


def sample_weights(design: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """Compose the least-squares fit and analytic integral into sample weights."""
    weights, _, _, _ = np.linalg.lstsq(design.T, moments, rcond=None)
    return weights


def main() -> None:
    experiments = Path(__file__).resolve().parents[1]
    meson = next(experiments.glob("*-spectral-network/spectral_network_meson.py"))
    original = run_path(str(meson))
    function = original["meson_values"]
    degree = original["DEGREE"]
    rows = []
    for seed in SEEDS:
        training_calls = []

        def capture(*coordinates):
            samples = function(*coordinates)
            training_calls.append((coordinates, samples))
            return samples

        started = perf_counter()
        model = SpectralNetwork.fit(
            capture, degree=degree, fibers=original["FIBERS_PER_AXIS"], readout_nodes=original["READOUT_NODES"],
            rank_tolerance=original["RANK_TOLERANCE"], seed=seed,
        )
        with torch.inference_mode():
            learned = model.integrate().numpy()
            branches = [b.numpy() for b in model.branches]
            moments = model.moments.numpy()
        learned_seconds = perf_counter() - started

        # Same final training sites and point values used by both variants.
        coordinates, values = training_calls[-1]
        started = perf_counter()
        designs = [chebvander(2 * axis.ravel() - 1, degree) for axis in coordinates]
        fixed_weights = [sample_weights(v, moments) for v in designs]
        fixed = np.einsum("a,b,c,abcl->l", *fixed_weights, values, optimize=True)
        fixed_seconds = perf_counter() - started
        learned_weights = [sample_weights(v @ w, moments @ w) for v, w in zip(designs, branches, strict=True)]
        weighted_sum = np.einsum("a,b,c,abcl->l", *learned_weights, values, optimize=True)
        np.testing.assert_allclose(weighted_sum, learned, rtol=1e-13, atol=0)

        for label, params, trained, fixed_basis, weighted in zip(
            range(1, 9), original["PARAM_SETS"], learned, fixed, weighted_sum, strict=True
        ):
            exact, _ = original["reference_scipy"](*params, tol=original["REFERENCE_TOLERANCE"])
            rows.append({
                "seed": seed, "integral": f"I_{label}", "reference": exact,
                "learned": trained, "fixed_basis": fixed_basis, "learned_weighted_sum": weighted,
                "learned_relative_error": abs(trained - exact) / abs(exact),
                "fixed_relative_error": abs(fixed_basis - exact) / abs(exact),
                "weighted_sum_relative_difference": abs(weighted - trained) / abs(trained),
                "learned_seconds_including_sampling": learned_seconds,
                "fixed_seconds_given_samples": fixed_seconds,
            })
        group = rows[-len(original["PARAM_SETS"]):]
        print(f"seed {seed}: learned worst error {max(r['learned_relative_error'] for r in group):.3e}; "
              f"fixed-basis worst error {max(r['fixed_relative_error'] for r in group):.3e}; "
              f"weighted-sum reconstruction {max(r['weighted_sum_relative_difference'] for r in group):.3e}",
              flush=True)

    dest = get_mirror_path(__file__, "results") / "meson_ablation.csv"
    with dest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(dest)


if __name__ == "__main__":
    main()
