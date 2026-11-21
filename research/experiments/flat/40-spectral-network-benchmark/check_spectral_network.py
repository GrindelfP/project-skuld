"""Check point predictions, analytic moments, and the original meson experiment."""

from pathlib import Path
from runpy import run_path
from unittest.mock import patch

import numpy as np
import torch

from spectral_benchmark import GENZ_FUNCTIONS, SpectralNetwork, exact_integrals


def main() -> None:
    rng = np.random.default_rng(1729)
    with torch.inference_mode():
        for dimension in range(1, 7):
            coefficients = np.arange(1, dimension + 1, dtype=np.float64)

            def polynomial(*coordinates: np.ndarray) -> np.ndarray:
                points = np.stack(np.broadcast_arrays(*coordinates), axis=-1)
                first = (1 + coefficients * points**2).prod(axis=-1)
                second = 1 - (coefficients * points).sum(axis=-1) + points.prod(axis=-1)
                return np.stack((first, second), axis=-1)

            model = SpectralNetwork.fit(
                polynomial, degree=3, fibers=20, readout_nodes=5,
                rank_tolerance=1e-12, seed=19, dimensions=dimension,
            )
            points = rng.random((128, dimension))
            expected = [(1 + coefficients / 3).prod(), 1 - coefficients.sum() / 2 + 0.5**dimension]
            np.testing.assert_allclose(
                model(torch.from_numpy(points)).numpy(), polynomial(*points.T), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(model.integrate().numpy(), expected, rtol=1e-12, atol=1e-12)
            exact = exact_integrals(dimension)
            for family in GENZ_FUNCTIONS:
                np.testing.assert_allclose(float(exact[family.name]), family.exact(dimension), rtol=1e-12)
            print(f"{dimension}D: point predictions, moments, and Genz formulas passed")

        experiments = Path(__file__).resolve().parents[1]
        meson = next(experiments.glob("*-spectral-network/spectral_network_meson.py"))
        original = run_path(str(meson))
        fit = original["fit_meson_network"]
        with patch.dict(fit.__globals__, {"reference_scipy": None}):
            estimates = fit(42).integrate().numpy()
        references = [original["reference_scipy"](*p, tol=1e-12)[0] for p in original["PARAM_SETS"]]
        np.testing.assert_allclose(estimates, references, rtol=1e-12, atol=0)
        print(f"Meson regression passed, {len(references)} outputs")


if __name__ == "__main__":
    main()
