"""Independent check of the canonical six-dimensional Genz exact values.

Uses 2**20 scrambled Sobol points for the two non-separable functions and
independent one-dimensional mpmath quadrature for the separable functions.
This is a reference check only; exact values are never passed to training.

Validated locally (jasher, 2026-11-02): absolute discrepancies were 4.465e-6
(oscillatory), 4.441e-16 (product_peak), 1.339e-8 (corner_peak), 1.301e-18
(gaussian), and 8.674e-19 (c0_continuous), all within the assertions below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.genz import GENZ_FUNCTIONS, separable_reference_1d  # noqa: E402

N = 6
SOBOL_POWER = 20
SOBOL_SEED = 20261102
QMC_FUNCTIONS = {"oscillatory", "corner_peak"}


def main() -> None:
    print(f"Independent canonical n={N} reference check")
    for fn in GENZ_FUNCTIONS:
        params = fn.params_for("canonical")
        exact = fn.exact(N, params)
        if fn.name in QMC_FUNCTIONS:
            points = qmc.Sobol(N, scramble=True, seed=SOBOL_SEED).random_base2(SOBOL_POWER)
            estimate = float(np.mean(fn.numpy_fn(points, *params), dtype=np.float64))
            method = f"scrambled Sobol 2^{SOBOL_POWER}"
            tolerance = 2e-5 if fn.name == "oscillatory" else 1e-5
        else:
            estimate = separable_reference_1d(fn, N, dps=35)
            method = "independent mpmath 1-D factors"
            tolerance = 5e-14
        error = abs(estimate - exact)
        print(f"{fn.name:15s} exact={exact:.17g} independent={estimate:.17g} "
              f"abs_diff={error:.3e} method={method}")
        if error > tolerance:
            raise AssertionError(f"{fn.name}: disagreement {error:.3e} exceeds {tolerance:.3e}")
    print("all five canonical n=6 references agree within stated tolerances")


if __name__ == "__main__":
    main()
