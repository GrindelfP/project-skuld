"""Stable evaluation of a tensor-product Gaussian RBF interpolant on [0, 1]³."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
GridFunction = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]


class StableGaussianRBF:
    """Fit point samples and evaluate the interpolant on Cartesian grids.

    Equally spaced Gaussian centres give a common envelope times a polynomial
    in an exponential coordinate along each axis. Store its barycentric data
    at Chebyshev–Lobatto sites instead of solving for raw Gaussian coefficients.
    """

    def __init__(self, nodes_per_axis: int, gamma: float) -> None:
        if nodes_per_axis < 2 or not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("Expected at least two nodes per axis and a positive finite gamma")

        self._gamma = gamma
        self._rate = 2 * gamma / (nodes_per_axis - 1)
        self._lower = np.expm1(-self._rate / 2)
        self._span = np.expm1(self._rate / 2) - self._lower
        indices = np.arange(nodes_per_axis)
        self._sites = (1 - np.cos(np.pi * indices / (nodes_per_axis - 1))) / 2
        self._barycentric_weights = np.where(indices % 2 == 0, 1.0, -1.0)
        self._barycentric_weights[[0, -1]] *= 0.5
        self._nodes = 0.5 + np.log1p(self._lower + self._span * self._sites) / self._rate
        self._nodes[[0, -1]] = [0.0, 1.0]
        self._samples: FloatArray | None = None

    def _envelope(self, points: FloatArray) -> FloatArray:
        shifted = points - 0.5
        return np.exp(-self._gamma * (shifted**2 + shifted))

    def fit(self, function: GridFunction) -> None:
        """Sample a broadcastable pointwise function at the collocation sites."""
        nodes = self._nodes
        values = np.asarray(function(nodes[:, None, None], nodes[None, :, None],
                                     nodes[None, None, :]), dtype=np.float64)
        envelope = self._envelope(nodes)
        self._samples = values / (
            envelope[:, None, None] * envelope[None, :, None] * envelope[None, None, :]
        )

    def _basis(self, points: FloatArray) -> FloatArray:
        points = np.asarray(points, dtype=np.float64)
        coordinate = (np.expm1(self._rate * (points - 0.5)) - self._lower) / self._span
        differences = coordinate[:, None] - self._sites
        matches = np.abs(differences) <= 4 * np.finfo(np.float64).eps
        regular = ~matches.any(axis=1)

        basis = matches.astype(np.float64)
        terms = self._barycentric_weights / differences[regular]
        basis[regular] = terms / terms.sum(axis=1, keepdims=True)
        return basis * self._envelope(points)[:, None]

    def evaluate_grid(self, x: FloatArray, y: FloatArray, z: FloatArray) -> FloatArray:
        """Evaluate at every point of the Cartesian product of three 1D arrays."""
        if self._samples is None:
            raise RuntimeError("Fit the interpolant before evaluating it")

        return np.einsum(
            "ia,jb,kc,abc->ijk",
            self._basis(x), self._basis(y), self._basis(z), self._samples,
            optimize=True,
        )
