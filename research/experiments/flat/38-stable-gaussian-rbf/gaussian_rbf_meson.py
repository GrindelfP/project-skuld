"""Approximate the Skuld meson exponent with a stable Gaussian RBF interpolant."""

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.special import roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from physics import D_R2, L_LOG, PARAM_SETS, T_MAX, korobov, korobov_weight, reference_scipy
from skuld.stable_gaussian_rbf import StableGaussianRBF
from utils.paths import get_mirror_path

RBF_NODES = 23
RBF_GAMMA = 2.0
QUADRATURE_ORDER = 48
REFERENCE_TOL = 1e-12
FloatArray = NDArray[np.float64]


def exponent(u1: FloatArray, u2: FloatArray, u3: FloatArray) -> FloatArray:
    """Shared exponent sampled by the RBF model, before taking the exponential."""
    alpha1 = korobov(u1)
    alpha2 = (1 - alpha1) * korobov(u2)
    t = np.expm1(L_LOG * u3)
    d, r2 = D_R2(alpha1, alpha2)
    return -(t * d + t / (1 + t) * r2)


def estimate_integrals(model: StableGaussianRBF) -> list[float]:
    """Integrate the reconstructed functions using tensor Gauss–Legendre."""
    nodes, weights = roots_legendre(QUADRATURE_ORDER)
    nodes, weights = (nodes + 1) / 2, weights / 2
    u1, u2, u3 = nodes[:, None, None], nodes[None, :, None], nodes[None, None, :]

    alpha1 = korobov(u1)
    alpha2 = (1 - alpha1) * korobov(u2)
    t = np.expm1(L_LOG * u3)
    j_alpha = (1 - alpha1) * korobov_weight(u1) * korobov_weight(u2)
    j_t = L_LOG * (1 + t)
    tensor_weights = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    weighted_common = tensor_weights * j_alpha * j_t * np.exp(
        model.evaluate_grid(nodes, nodes, nodes)
    )
    return [
        math.fsum((weighted_common * alpha1**a * alpha2**b * t**m / (1 + t)**n).ravel())
        for a, b, m, n in PARAM_SETS
    ]


def main() -> None:
    started = perf_counter()
    print("Skuld meson integrals | stable Gaussian RBF", flush=True)
    print(f"NumPy {np.__version__}, SciPy {scipy.__version__} | float64", flush=True)
    print(
        f"[1/4] Sampling the shared exponent on {RBF_NODES}³ "
        f"transformed Chebyshev–Lobatto sites (gamma={RBF_GAMMA:g})...",
        flush=True,
    )
    stage = perf_counter()
    model = StableGaussianRBF(RBF_NODES, RBF_GAMMA)
    model.fit(exponent)
    print(f"      RBF fit completed in {perf_counter() - stage:.3f} s.", flush=True)

    print(f"[2/4] Integrating with {QUADRATURE_ORDER} Gauss–Legendre nodes per axis...", flush=True)
    stage = perf_counter()
    estimates = estimate_integrals(model)
    print(f"      All {len(estimates)} RBF estimates ready in {perf_counter() - stage:.3f} s.", flush=True)

    print(
        f"[3/4] Computing independent SciPy references in physical coordinates "
        f"(t ∈ [0, {T_MAX:g}], epsabs=epsrel={REFERENCE_TOL:g})...",
        flush=True,
    )
    rows = []
    for index, (params, estimate) in enumerate(zip(PARAM_SETS, estimates, strict=True), 1):
        stage = perf_counter()
        label = f"I_{index}"
        print(f"      {label} {params}: running adaptive quadrature...", flush=True)
        reference, reference_error = reference_scipy(*params, tol=REFERENCE_TOL)
        err_abs = abs(estimate - reference)
        err_rel = err_abs / abs(reference)
        digits = math.floor(-math.log10(err_rel)) if err_rel else math.inf
        print(f"      {label}: relative difference {err_rel:.3e} ({perf_counter() - stage:.3f} s).",
              flush=True)
        rows.append({
            "integral": label, "a": params[0], "b": params[1], "m": params[2], "n": params[3],
            "rbf": estimate, "scipy": reference, "scipy_outer_error": reference_error,
            "err_abs": err_abs, "err_rel": err_rel, "correct_digits": digits,
        })

    print("\n[4/4] Final comparison")
    print(f"{'I_n':<5} {'RBF':>20} {'SciPy':>20} {'Err_abs':>12} {'Err_rel':>12} {'Correct digits':>14}")
    for row in rows:
        print(
            f"{row['integral']:<5} {row['rbf']:20.16f} {row['scipy']:20.16f} "
            f"{row['err_abs']:12.3e} {row['err_rel']:12.3e} {row['correct_digits']:>14}"
        )
    scores = [row["correct_digits"] for row in rows]
    worst = max(rows, key=lambda row: row["err_rel"])
    print(f"Correct digits (min / mean / max): {min(scores)} / {mean(scores):.2f} / {max(scores)}")
    print(f"Maximum relative difference: {worst['err_rel']:.3e} ({worst['integral']})")
    print("Correct digits = floor(-log10(Err_rel)). inf denotes identical float64 values.")
    print("The differences measure agreement with SciPy on the finite integration domain.")

    results_path = get_mirror_path(__file__, "results") / "gaussian_rbf_meson.csv"
    with results_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {results_path}")
    print(f"Total elapsed: {perf_counter() - started:.3f} s", flush=True)


if __name__ == "__main__":
    main()
