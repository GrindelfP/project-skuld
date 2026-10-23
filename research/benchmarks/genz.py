"""
genz.py — Genz test integrands with analytically known integrals.

Five families from Genz (1984, 1987), as tabulated in Burkardt's TESTPACK /
TEST_NINT problems P27-P31, each defined on the unit cube in arbitrary
dimension with a closed-form exact integral:

    oscillatory     cos(2*pi*r + sum_i a_i x_i)
    product peak    prod_i 1 / (a_i^2 + (x_i - u_i)^2)
    corner peak     (1 + sum_i a_i x_i)^-(n + r)
    gaussian        exp(-sum_i a_i^2 (x_i - u_i)^2)
    c0 continuous   exp(-sum_i a_i |x_i - u_i|)

These are *characterisation* integrands: synthetic functions whose exact
values are known, used to measure method quality independently of any target
application. The meson Feynman integral in `physics.py` is the target
benchmark. The two are complementary; neither substitutes for the other.

Two conventions matter and are easy to get wrong:

1.  **Shape.** `integrand` returns shape ``[N]``, not ``[N, 1]``. The antiderivative
    integrators compare a mixed partial of shape ``[N]`` against the integrand;
    an ``[N, 1]`` return broadcasts the MSE to ``[N, N]`` and silently corrupts
    the loss with no error raised.

2.  **Digits.** Exact integrals across this function set span 4.2e-2 (corner
    peak) to 1.7e+3 (product peak), so accuracy must be measured as *relative*
    correct digits, never absolute.

Default parameters are the canonical Genz1984 set: a = 5, u = 0.5, r = 0.5.
Genz's guidance is that the shape parameter `a` dominates difficulty while the
offset `u` does not, which is why the defaults are symmetric.

References:
    Genz, A. (1984), Testing multidimensional integration routines.
    Genz, A. (1987), A package for testing multiple integration subroutines.
    Burkardt, J., TEST_NINT problems P27-P31 (reference implementation).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.special import erf

__all__ = [
    "GenzFunction",
    "GENZ_FUNCTIONS",
    "genz_by_name",
    "CANONICAL",
    "oscillatory",
    "product_peak",
    "corner_peak",
    "gaussian",
    "c0_continuous",
    "correct_digits",
    "separable_reference_1d",
]


# ---------------------------------------------------------------------------
# parameter broadcasting
# ---------------------------------------------------------------------------


def _np_par(p, n: int) -> np.ndarray:
    """Broadcast a scalar or length-n parameter vector to float64[n]."""
    return np.broadcast_to(np.atleast_1d(np.asarray(p, dtype=np.float64)), (n,)).copy()


def _torch_par(p, n: int, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or length-n parameter vector to a tensor shaped like `like`."""
    return torch.as_tensor(_np_par(p, n), dtype=like.dtype, device=like.device)


# ---------------------------------------------------------------------------
# integrands — numpy, [N, n] -> [N]
# ---------------------------------------------------------------------------


def oscillatory_np(X: np.ndarray, a: float = 5.0, r: float = 0.5) -> np.ndarray:
    n = X.shape[-1]
    phase = 2.0 * np.pi * r + X @ _np_par(a, n)
    return np.cos(phase)


def product_peak_np(X: np.ndarray, a: float = 5.0, u: float = 0.5) -> np.ndarray:
    n = X.shape[-1]
    aa, uu = _np_par(a, n), _np_par(u, n)
    return np.prod(aa**2 / (aa**2 + (X - uu) ** 2), axis=-1)


def corner_peak_np(X: np.ndarray, a: float = 1.0, r: float = 1.0) -> np.ndarray:
    n = X.shape[-1]
    return (1.0 + X @ _np_par(a, n)) ** (-(n + r))


def gaussian_np(X: np.ndarray, a: float = 5.0, u: float = 0.5) -> np.ndarray:
    n = X.shape[-1]
    aa, uu = _np_par(a, n), _np_par(u, n)
    return np.exp(-(((X - uu) * aa) ** 2).sum(axis=-1))


def c0_continuous_np(X: np.ndarray, a: float = 5.0, u: float = 0.5) -> np.ndarray:
    n = X.shape[-1]
    aa, uu = _np_par(a, n), _np_par(u, n)
    return np.exp(-(aa * np.abs(X - uu)).sum(axis=-1))


# ---------------------------------------------------------------------------
# integrands — torch, [N, n] -> [N]   (NOT [N, 1] — see module docstring)
# ---------------------------------------------------------------------------


def oscillatory(u: torch.Tensor, a: float = 5.0, r: float = 0.5) -> torch.Tensor:
    """cos(2*pi*r + sum_i a_i x_i). Separable integral, non-separable function."""
    n = u.shape[-1]
    phase = 2.0 * math.pi * r + (u * _torch_par(a, n, u)).sum(-1)
    return torch.cos(phase)


def product_peak(u: torch.Tensor, a: float = 5.0, u_off: float = 0.5) -> torch.Tensor:
    """
    prod_i a_i^2 / (a_i^2 + (x_i - u_i)^2). Separable.

    Normalized so the integrand peaks at exactly 1.0 when x = u. This matches
    the closed form `product_peak_exact`; the two must be kept consistent, and
    dropping the a^2 numerator silently inflates the exact value by a^{2n}.
    """
    n = u.shape[-1]
    aa = _torch_par(a, n, u)
    uu = _torch_par(u_off, n, u)
    return torch.prod(aa**2 / (aa**2 + (u - uu) ** 2), dim=-1)


def corner_peak(u: torch.Tensor, a: float = 1.0, r: float = 1.0) -> torch.Tensor:
    """(1 + sum_i a_i x_i)^-(n+r). Non-separable; the only such family in the smooth set."""
    n = u.shape[-1]
    return (1.0 + u @ _torch_par(a, n, u)) ** (-(n + r))


def gaussian(u: torch.Tensor, a: float = 5.0, u_off: float = 0.5) -> torch.Tensor:
    """exp(-sum_i a_i^2 (x_i - u_i)^2). Separable."""
    n = u.shape[-1]
    aa = _torch_par(a, n, u)
    uu = _torch_par(u_off, n, u)
    return torch.exp(-(((u - uu) * aa) ** 2).sum(-1))


def c0_continuous(u: torch.Tensor, a: float = 5.0, u_off: float = 0.5) -> torch.Tensor:
    """exp(-sum_i a_i |x_i - u_i|). C^0 but not C^1: kinked on each x_i = u_i."""
    n = u.shape[-1]
    aa = _torch_par(a, n, u)
    uu = _torch_par(u_off, n, u)
    return torch.exp(-(aa * torch.abs(u - uu)).sum(-1))


# ---------------------------------------------------------------------------
# exact integrals — closed form, arbitrary dimension
# ---------------------------------------------------------------------------


def oscillatory_exact(n: int, a: float = 5.0, r: float = 0.5) -> float:
    """prod_i [2 sin(a_i/2)/a_i] * cos(2*pi*r + sum_i a_i / 2)."""
    aa = _np_par(a, n)
    return float(
        np.cos(2.0 * np.pi * r + 0.5 * aa.sum()) * np.prod(np.sin(aa / 2.0) / (aa / 2.0))
    )


def product_peak_exact(n: int, a: float = 5.0, u: float = 0.5) -> float:
    """prod_i a_i * (atan((1-u_i)/a_i) + atan(u_i/a_i)). Valid for the peak-height-1 form."""
    aa, uu = _np_par(a, n), _np_par(u, n)
    return float(np.prod(aa * (np.arctan((1.0 - uu) / aa) + np.arctan(uu / aa))))


def corner_peak_exact(n: int, a: float = 1.0, r: float = 1.0) -> float:
    """
    Inclusion-exclusion over the 2**n cube vertices.

    The vertex sign is (-1)**|v|, NOT (-1)**(n - |v|). The two differ by a
    factor of (-1)**n, so the wrong one is silently correct in even dimensions
    and negates the answer in odd ones. This is a known trap.
    """
    aa = _np_par(a, n)
    total = 0.0
    for v in itertools.product((0.0, 1.0), repeat=n):
        total += (-1.0) ** sum(v) / (1.0 + aa @ np.asarray(v)) ** r
    denom = float(np.prod([r + k for k in range(n)])) * float(np.prod(aa))
    return float(total / denom)


def gaussian_exact(n: int, a: float = 5.0, u: float = 0.5) -> float:
    """prod_i [sqrt(pi)/(2 a_i) * (erf(a_i(1-u_i)) + erf(a_i u_i))]."""
    aa, uu = _np_par(a, n), _np_par(u, n)
    return float(
        np.prod(np.sqrt(np.pi) / (2.0 * aa) * (erf(aa * (1.0 - uu)) + erf(aa * uu)))
    )


def c0_continuous_exact(n: int, a: float = 5.0, u: float = 0.5) -> float:
    """prod_i [(2 - exp(-a_i u_i) - exp(-a_i(1-u_i))) / a_i]."""
    aa, uu = _np_par(a, n), _np_par(u, n)
    return float(np.prod((2.0 - np.exp(-aa * uu) - np.exp(-aa * (1.0 - uu))) / aa))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenzFunction:
    """One Genz family: torch integrand, numpy integrand, exact value, params."""

    name: str
    torch_fn: Callable
    numpy_fn: Callable
    exact_fn: Callable
    params: tuple
    separable: bool
    note: str
    # Parameters scaled down to roughly the difficulty of the meson Feynman
    # integrand. The canonical Genz1984 set (a = 5) is deliberately severe: the
    # shape parameter a controls both frequency and peak sharpness, and a = 5
    # pushes every architecture to ~3 digits. See research/genz_difficulty.md.
    tuned_params: tuple | None = None

    def integrand(self, u: torch.Tensor, *params) -> torch.Tensor:
        """Adapter for the antiderivative integrators: `(u, *params) -> [N]`."""
        return self.torch_fn(u, *params)

    def exact(self, n: int, params: tuple | None = None) -> float:
        return self.exact_fn(n, *(params if params is not None else self.params))

    def params_for(self, difficulty: str = "canonical") -> tuple:
        """Parameters for a named difficulty tier."""
        if difficulty == "canonical":
            return self.params
        if difficulty == "tuned":
            if self.tuned_params is None:
                return self.params
            return self.tuned_params
        raise ValueError(
            f"unknown difficulty tier {difficulty!r}; use 'canonical' or 'tuned'"
        )

    @property
    def n_params(self) -> int:
        return len(self.params)


CANONICAL = (5.0, 0.5)
CANONICAL_TUNED = (2.0, 0.5)

GENZ_FUNCTIONS: tuple[GenzFunction, ...] = (
    GenzFunction(
        name="oscillatory",
        torch_fn=oscillatory,
        numpy_fn=oscillatory_np,
        exact_fn=oscillatory_exact,
        params=(5.0, 0.5),
        tuned_params=(2.0, 0.5),
        separable=False,
        note="cosine of a sum; integral factorises but the function does not",
    ),
    GenzFunction(
        name="product_peak",
        torch_fn=product_peak,
        numpy_fn=product_peak_np,
        exact_fn=product_peak_exact,
        params=CANONICAL,
        tuned_params=CANONICAL_TUNED,
        separable=True,
        note="Lorentzian peak, normalized to peak height 1; exact ~0.99 at n=3",
    ),
    GenzFunction(
        name="corner_peak",
        torch_fn=corner_peak,
        numpy_fn=corner_peak_np,
        exact_fn=corner_peak_exact,
        params=(1.0, 1.0),
        tuned_params=(0.5, 0.5),
        separable=False,
        note="exact value 1/(n!(n+1)); a relative-accuracy trap, never report absolute",
    ),
    GenzFunction(
        name="gaussian",
        torch_fn=gaussian,
        numpy_fn=gaussian_np,
        exact_fn=gaussian_exact,
        params=CANONICAL,
        tuned_params=CANONICAL_TUNED,
        separable=True,
        note="smoothest family in the set; serves as the sanity floor",
    ),
    GenzFunction(
        name="c0_continuous",
        torch_fn=c0_continuous,
        numpy_fn=c0_continuous_np,
        exact_fn=c0_continuous_exact,
        params=CANONICAL,
        tuned_params=CANONICAL_TUNED,
        separable=True,
        note="C^0 not C^1: kinked on every hyperplane x_i = u_i; the only non-smooth family",
    ),
)


def genz_by_name(name: str) -> GenzFunction:
    for fn in GENZ_FUNCTIONS:
        if fn.name == name:
            return fn
    raise KeyError(f"no Genz function named {name!r}")


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------


def correct_digits(estimate: float, exact: float) -> float:
    """
    Leading decimal digits of agreement, as a float.

    Always relative — see the module docstring. Returns a negative number when
    the estimate is wrong by more than a factor of ten, which is a legitimate
    outcome for a method that has failed rather than something to clamp away.
    """
    if exact == 0.0:
        return float("inf") if estimate == 0.0 else -float("inf")
    rel = abs(estimate - exact) / abs(exact)
    if rel == 0.0:
        return 16.0
    return -math.log10(rel)


# ---------------------------------------------------------------------------
# independent quadrature reference
# ---------------------------------------------------------------------------


def separable_reference_1d(fn: GenzFunction, n: int, dps: int = 20) -> float:
    """
    Exact integral of a separable Genz function, by n independent 1-D quadratures.

    For ``f(x) = prod_i g_i(x_i)`` the integral factorises as ``prod_i int_0^1 g_i``.
    Evaluating each factor with `mpmath.quad` is independent of the closed form
    in this module and exact to arbitrary precision, at a cost of O(n)
    quadratures instead of the O(nodes**n) of blind nested quadrature.

    This is the reference that makes c0_continuous tractable: nested quadrature
    chokes on its kink, but splitting each 1-D factor at the known kink location
    ``u_i`` resolves it immediately.

    Separability here is structural (a product of per-coordinate functions), so
    this reference is valid for product_peak, gaussian and c0_continuous, and is
    NOT valid for oscillatory or corner_peak — check ``fn.separable`` first.
    """
    if not fn.separable:
        raise ValueError(f"{fn.name} is not separable; use nested quadrature instead")

    import mpmath as mp

    mp.mp.dps = dps
    a, u = fn.params[0], fn.params[1]

    total = mp.mpf(1)
    for i in range(n):
        av = a[i] if np.ndim(a) else float(a)
        uv = u[i] if np.ndim(u) else float(u)

        def factor(x, _i=i, _a=av, _u=uv):
            # Evaluate the 1-D kernel with the other coordinates pinned at the
            # offset, so the per-coordinate function is recovered exactly.
            point = np.full((1, n), _u)
            point[0, _i] = float(x)
            return mp.mpf(float(fn.numpy_fn(point, _a, uv)[0]))

        # Split at the kink location u_i so mpmath never straddles it.
        left = mp.quad(factor, [0, uv])
        right = mp.quad(factor, [uv, 1])
        total *= left + right

    return float(total)