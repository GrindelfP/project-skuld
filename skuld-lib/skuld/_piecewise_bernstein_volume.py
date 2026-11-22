"""Bounded piecewise Bernstein networks with exact polynomial-weighted readouts."""

from dataclasses import dataclass
from itertools import product
from math import comb, fsum

import numpy as np

from ._bounded_water_volume import BernsteinSurface, _bernstein_basis
from ._water_volume import FloatArray


def collocation_points(cells: tuple[int, ...], degree: int) -> tuple[FloatArray, ...]:
    """Share Chebyshev–Lobatto samples at the faces of adjacent cells."""
    local = _local_nodes(degree)
    return tuple(
        np.append(((np.arange(count)[:, None] + local[:-1]) / count).ravel(), 1.0)
        for count in cells
    )


def _local_nodes(degree: int) -> FloatArray:
    nodes = (1 - np.cos(np.linspace(0, np.pi, degree + 1))) / 2
    nodes[[0, -1]] = [0.0, 1.0]
    return nodes


def _weighted_moments(count: int, degree: int, powers: tuple[int, int]) -> FloatArray:
    """Integrate each local Bernstein unit against x**p * (1-x)**q."""
    p, q = powers
    width = 1 / count
    moments = np.zeros((count, degree + 1))
    # On a cell, x = left + width*u and 1-x = right_gap + width*(1-u).
    # Expand with positive coefficients and integrate each term via the beta integral.
    for cell in range(count):
        left = cell / count
        right_gap = (count - cell - 1) / count
        for k in range(degree + 1):
            moments[cell, k] = width * fsum(
                comb(p, i) * comb(q, j) * left**(p - i) * right_gap**(q - j)
                * width**(i + j) * comb(degree, k)
                / ((degree + i + j + 1) * comb(degree + i + j, k + i))
                for i in range(p + 1) for j in range(q + 1)
            )
    return moments


@dataclass(frozen=True)
class PiecewiseBernsteinSurface:
    """One bounded tensor-product hidden layer per cell of a regular partition."""

    coefficients: FloatArray
    cells: tuple[int, ...]

    @property
    def degree(self) -> int:
        return self.coefficients.shape[-1] - 1

    @classmethod
    def fit(
        cls, heights: FloatArray, cells: tuple[int, ...], degree: int,
    ) -> "PiecewiseBernsteinSurface":
        if degree < 1 or not cells or min(cells) < 1:
            raise ValueError("Expected positive degree and cell counts")
        if heights.shape != tuple(count * degree + 1 for count in cells):
            raise ValueError("Height array does not match the collocation grid")
        if not np.all((0 <= heights) & (heights <= 1)):
            raise ValueError("Expected collocation heights in [0, 1]")

        nodes = _local_nodes(degree)
        basis = _bernstein_basis(nodes, degree)
        dimension = len(cells)
        # Tensor indices are split into a cell index and a local collocation index.
        spatial_shape = tuple((count, degree + 1) for count in cells)
        indices = np.ix_(*[
            (degree * np.arange(count)[:, None] + np.arange(degree + 1)).ravel()
            for count in cells
        ])
        blocks = heights[indices].reshape(tuple(v for pair in spatial_shape for v in pair))
        coefficients = blocks.transpose(*range(0, 2 * dimension, 2), *range(1, 2 * dimension, 2))
        for axis in range(dimension, 2 * dimension):
            leading = np.moveaxis(coefficients, axis, 0)
            lines = leading.reshape(degree + 1, -1)
            offset = lines[degree // 2]
            # The basis sums to one, so remove the level before the solve.
            projected = np.linalg.solve(basis, lines - offset) + offset
            coefficients = np.moveaxis(projected.reshape(leading.shape), 0, axis)
        return cls(np.clip(coefficients, 0, 1), cells)

    def height(self, points: FloatArray) -> FloatArray:
        scaled = points * self.cells
        indices = np.minimum(scaled.astype(int), np.asarray(self.cells) - 1)
        local = scaled - indices
        prediction = np.empty(len(points))
        for cell in product(*(range(count) for count in self.cells)):
            mask = np.all(indices == cell, axis=1)
            if np.any(mask):
                prediction[mask] = BernsteinSurface(self.coefficients[cell]).height(local[mask])
        return prediction

    def integrate_weighted(self, powers: tuple[tuple[int, int], ...]) -> float:
        if len(powers) != len(self.cells) or any(min(pair) < 0 for pair in powers):
            raise ValueError("Expected two nonnegative integer powers per coordinate")
        moments = [
            _weighted_moments(count, self.degree, pair)
            for count, pair in zip(self.cells, powers, strict=True)
        ]
        dimension = len(self.cells)
        dimensions = list(range(2 * dimension))
        arguments = [self.coefficients, dimensions]
        for axis, moment in enumerate(moments):
            arguments.extend((moment, [axis, axis + dimension]))
        return float(np.einsum(*arguments, [], optimize=True))
