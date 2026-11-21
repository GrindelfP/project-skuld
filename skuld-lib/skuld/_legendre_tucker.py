"""Compact functional Tucker approximation in shifted Legendre coordinates."""

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mode_product(tensor: FloatArray, matrix: FloatArray, axis: int) -> FloatArray:
    return np.moveaxis(np.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)


@dataclass(frozen=True)
class LegendreTucker:
    core: FloatArray
    factors: tuple[FloatArray, ...]

    @classmethod
    def fit(
        cls,
        coefficients: FloatArray,
        budget: int,
        *,
        learned: bool = True,
        same_ranks: tuple[int, ...] | None = None,
    ) -> "LegendreTucker":
        """Choose ranks by retained squared L² energy under a coefficient budget."""
        core = coefficients
        factors = []
        if learned:
            for axis, size in enumerate(coefficients.shape):
                unfolding = np.moveaxis(coefficients, axis, 0).reshape(size, -1)
                factor, _, _ = np.linalg.svd(unfolding, full_matrices=False)
                core = mode_product(core, factor.T, axis)
                factors.append(factor)

        if same_ranks is None:
            energy = core**2
            for axis in range(core.ndim):
                energy = energy.cumsum(axis=axis)
            candidates = product(*(range(1, size + 1) for size in core.shape))
            ranks = max(
                (ranks for ranks in candidates if coefficient_count(coefficients.shape, ranks, learned) <= budget),
                key=lambda ranks: energy[tuple(rank - 1 for rank in ranks)],
            )
        else:
            ranks = same_ranks

        keep = tuple(slice(rank) for rank in ranks)
        return cls(
            np.ascontiguousarray(core[keep]),
            tuple(np.ascontiguousarray(factor[:, keep[axis]]) for axis, factor in enumerate(factors)),
        )

    @property
    def n_coefficients(self) -> int:
        return self.core.size + sum(factor.size for factor in self.factors)

    def integrate_weighted(self, moments: FloatArray) -> FloatArray:
        """Integrate the fitted field against separable weights for each output."""
        if self.factors:
            projected = [axis_moments @ factor for axis_moments, factor in zip(moments, self.factors, strict=True)]
        else:
            projected = [axis_moments[:, :rank] for axis_moments, rank in zip(moments, self.core.shape, strict=True)]
        return np.einsum("ci,cj,ck,ijk->c", *projected, self.core, optimize=True)


def coefficient_count(shape: tuple[int, ...], ranks: tuple[int, ...], learned: bool) -> int:
    return prod(ranks) + (sum(size * rank for size, rank in zip(shape, ranks, strict=True)) if learned else 0)
