"""Bounded neural surfaces formed from products of piecewise Bernstein factors."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from heapq import heapify, heappop, heappush
from math import fsum, prod

import numpy as np

from ._bounded_water_volume import _bernstein_basis
from ._water_volume import FloatArray


@lru_cache(maxsize=4)
def _collocation(degree: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    nodes = (1 - np.cos(np.linspace(0, np.pi, degree + 1))) / 2
    nodes[[0, -1]] = [0, 1]
    checks = (nodes[:-1] + nodes[1:]) / 2
    basis = _bernstein_basis(nodes, degree)
    return nodes, checks, np.linalg.solve(basis, np.eye(len(nodes)))


def _bisect(coefficients: FloatArray) -> tuple[FloatArray, FloatArray]:
    left = [coefficients[0]]
    right = [coefficients[-1]]
    while len(coefficients) > 1:
        coefficients = (coefficients[1:] + coefficients[:-1]) / 2
        left.append(coefficients[0])
        right.append(coefficients[-1])
    return np.asarray(left), np.asarray(right[::-1])


@dataclass(frozen=True)
class BernsteinFactor:
    edges: FloatArray
    coefficients: FloatArray

    def evaluate(self, points: FloatArray) -> FloatArray:
        cells = np.searchsorted(self.edges[1:-1], points, side="right")
        local = (points - self.edges[cells]) / (self.edges[cells + 1] - self.edges[cells])
        return np.einsum("ij,ij->i", self.coefficients[cells],
                         _bernstein_basis(local, self.coefficients.shape[1] - 1))

    def integral(self) -> float:
        return fsum(np.diff(self.edges) * self.coefficients.mean(axis=1))

    def upper_bound(self) -> float:
        """Use the Bernstein convex hull and tighten it near candidate maxima."""
        queue = []
        counter = 0
        best = float(np.max(self.coefficients[:, [0, -1]]))
        for weights in self.coefficients:
            heappush(queue, (-float(np.max(weights)), counter, weights))
            counter += 1
        tolerance = 16 * np.finfo(float).eps * -queue[0][0]
        for _ in range(512):
            if -queue[0][0] <= best + tolerance:
                break
            _, _, weights = heappop(queue)
            children = _bisect(weights)
            best = max(best, float(children[0][-1]))
            for child in children:
                heappush(queue, (-float(np.max(child)), counter, child))
                counter += 1
        return -queue[0][0] + tolerance


def _fit_factor(
    oracle: Callable[[FloatArray], FloatArray], degree: int, max_splits: int, tolerance: float,
) -> tuple[BernsteinFactor, int]:
    nodes, checks, inverse = _collocation(degree)
    check_basis = _bernstein_basis(checks, degree)
    samples = 0
    counter = 0

    def fit_cell(left: float, right: float) -> tuple:
        nonlocal samples, counter
        points = left + (right - left) * np.concatenate((nodes, checks))
        values = oracle(points)
        samples += len(points)
        coefficients = np.maximum(inverse @ values[:len(nodes)], 0)
        error = float(np.max(np.abs(check_basis @ coefficients - values[len(nodes):])))
        counter += 1
        # Prioritize cells by their potential absolute contribution to volume error.
        return (-error * (right - left), counter, left, right, coefficients)

    edges = np.linspace(0, 1, 9)
    pending = [fit_cell(left, right) for left, right in zip(edges[:-1], edges[1:])]
    heapify(pending)
    scale = max(1.0, *(float(np.max(cell[4])) for cell in pending))
    complete = []
    for _ in range(max_splits):
        if not pending or -pending[0][0] <= tolerance * scale:
            break
        cell = heappop(pending)
        _, _, left, right, _ = cell
        if right - left < 2**-22:
            complete.append(cell)
            continue
        middle = (left + right) / 2
        heappush(pending, fit_cell(left, middle))
        heappush(pending, fit_cell(middle, right))

    cells = sorted(pending + complete, key=lambda cell: cell[2])
    factor = BernsteinFactor(
        np.asarray([cell[2] for cell in cells] + [1.0]),
        np.stack([cell[4] for cell in cells]),
    )
    return factor, samples


@dataclass(frozen=True)
class FiberSurface:
    """A nonnegative multiplicative network trained on one-dimensional fibers."""

    scale: float
    factors: tuple[BernsteinFactor, ...]

    @property
    def parameter_count(self) -> int:
        return 1 + sum(factor.coefficients.size + len(factor.edges) - 2 for factor in self.factors)

    def height(self, points: FloatArray) -> FloatArray:
        prediction = np.full(len(points), self.scale)
        for axis, factor in enumerate(self.factors):
            prediction *= factor.evaluate(points[:, axis])
        return prediction

    def integrate(self) -> float:
        return self.scale * prod(factor.integral() for factor in self.factors)

    @classmethod
    def fit(
        cls, oracle: Callable[[FloatArray], FloatArray], points: FloatArray, heights: FloatArray,
        *, degree: int, max_splits: int, tolerance: float,
    ) -> tuple["FiberSurface | None", int]:
        anchor_index = int(np.argmax(heights))
        anchor = points[anchor_index]
        signal = float(heights[anchor_index])
        if signal <= 0:
            return None, 0

        dimension = points.shape[1]

        def fiber(axis: int, values: FloatArray) -> FloatArray:
            queries = np.broadcast_to(anchor, (len(values), dimension)).copy()
            queries[:, axis] = values
            return oracle(queries) / signal

        # Gate based on pointwise separability on already sampled training sites.
        probe_points = points[:64]
        probe = np.full(len(probe_points), signal)
        for axis in range(dimension):
            probe *= fiber(axis, probe_points[:, axis])
        samples = dimension * len(probe_points)
        if np.max(np.abs(probe - heights[:len(probe_points)])) > 5e-12 * signal:
            return None, samples

        factors = []
        for axis in range(dimension):
            factor, used = _fit_factor(
                lambda values: fiber(axis, values), degree, max_splits, tolerance
            )
            factors.append(factor)
            samples += used
        amplitude_bound = prod(factor.upper_bound() for factor in factors)
        scale = min(signal, (1 - 16 * np.finfo(float).eps) / amplitude_bound)
        return cls(scale, tuple(factors)), samples
