"""Learn a signed water surface and measure the volume of its columns."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import fsum, prod

import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.typing import NDArray
from scipy.fft import dctn

Array = NDArray[np.float64]


def _learn_layers(coefficients: Array, relative_tolerance: float) -> tuple[Array, ...]:
    """Factor latent activations into learned, multiplicatively coupled layers."""
    shape = coefficients.shape
    remaining = coefficients
    layers = []
    previous_rank = 1
    for width in shape[:-1]:
        matrix = remaining.reshape(previous_rank * width, -1)
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        tail_norm = np.sqrt(np.cumsum(singular[::-1] ** 2))[::-1]
        rank = max(1, int(np.count_nonzero(tail_norm > singular[0] * relative_tolerance)))
        layers.append(left[:, :rank].reshape(previous_rank, width, rank))
        remaining = singular[:rank, None] * right[:rank]
        previous_rank = rank
    layers.append(remaining.reshape(previous_rank, shape[-1], 1))
    return tuple(layers)


@dataclass(frozen=True)
class _ColumnNetwork:
    layers: tuple[Array, ...]
    correction: tuple[Array, ...]
    scale: float

    @classmethod
    def learn(cls, heights: Array) -> "_ColumnNetwork":
        # Keep the inverse activation well conditioned even for signed heights.
        peak = float(np.max(np.abs(heights)))
        scale = (8 * peak if peak <= np.finfo(float).max / 8 else peak) or 1.0
        latent_targets = np.arcsinh(heights / scale)
        coefficients = dctn(latent_targets, type=2)
        for axis, size in enumerate(heights.shape):
            coefficients /= size
            constant = [slice(None)] * heights.ndim
            constant[axis] = 0
            coefficients[tuple(constant)] *= 0.5
        layers = _learn_layers(coefficients, 3e-15)
        # Fit a residual branch to restore detail discarded by the low-rank
        # layers and compensate for roundoff from their factorization.
        reconstructed = layers[0][0]
        for layer in layers[1:]:
            reconstructed = np.tensordot(reconstructed, layer, axes=1)
        correction = _learn_layers(coefficients - reconstructed[..., 0], 1e-4)
        return cls(layers, correction, scale)

    def _activate(self, coordinate: Array, layer: Array) -> Array:
        basis = chebvander(coordinate, layer.shape[1] - 1)
        weights = layer.transpose(1, 0, 2).reshape(layer.shape[1], -1)
        return (basis @ weights).reshape(len(coordinate), layer.shape[0], layer.shape[2])

    def _latent(self, points: Array, layers: tuple[Array, ...]) -> Array:
        state = np.ones((len(points), 1))
        for coordinate, layer in zip(points.T, layers, strict=True):
            activated = self._activate(coordinate, layer)
            state = np.einsum("np,npr->nr", state, activated)
        return state[:, 0]

    def depths(self, points: Array) -> Array:
        latent = self._latent(points, self.layers) + self._latent(points, self.correction)
        return self.scale * np.sinh(latent)

    def _latent_grid(self, nodes: Array, layers: tuple[Array, ...]) -> Array:
        state = np.ones((1, 1))
        for layer in layers:
            activated = self._activate(nodes, layer)
            kernel = activated.transpose(1, 0, 2).reshape(layer.shape[0], -1)
            state = (state @ kernel).reshape(-1, layer.shape[2])
        return state[:, 0]

    def volume(self, order: int) -> float:
        nodes, weights = _gauss_rule(order)
        latent = self._latent_grid(nodes, self.layers) + self._latent_grid(nodes, self.correction)
        tensor_weights = np.ones(1)
        for _ in self.layers:
            tensor_weights = np.multiply.outer(tensor_weights, weights).ravel()
        # Each pair of columns has lengths max(h, 0) and max(-h, 0).
        # Accumulate their difference without first rounding the two totals.
        return self.scale * fsum(np.sinh(latent) * tensor_weights)


@lru_cache(maxsize=16)
def _gauss_rule(order: int) -> tuple[Array, Array]:
    context = mp.mp.clone()
    context.dps = 40
    nodes, weights = context.gauss_quadrature(order, "legendre")
    nodes = np.asarray(list(nodes), dtype=float)
    weights = np.asarray(list(weights), dtype=float) / 2
    return nodes, weights


@dataclass(frozen=True)
class _Tank:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    network: _ColumnNetwork

    @property
    def widths(self) -> Array:
        return np.subtract(self.upper, self.lower)

    def volume(self, order: int) -> float:
        return prod(self.widths) * self.network.volume(order)


@dataclass(frozen=True)
class NeuralWaterMap:
    """Trained water geometry with no access to the original integrand."""

    _tanks: tuple[_Tank, ...]
    fit_order: int

    @property
    def sample_count(self) -> int:
        return len(self._tanks) * self.fit_order ** len(self._tanks[0].lower)

    @property
    def parameter_count(self) -> int:
        return sum(
            1 + sum(layer.size for layer in (*tank.network.layers, *tank.network.correction))
            for tank in self._tanks
        )

    def integrate(self, *, order: int | None = None) -> float:
        """Measure the signed volume of the learned columns."""
        if order is None:
            order = self.fit_order + 4
        if order < 1:
            raise ValueError("Readout order must be positive")
        return fsum(tank.volume(order) for tank in self._tanks)

    def depths(self, points: Array) -> Array:
        """Evaluate actual network depths at points in the original domain."""
        points = np.asarray(points, dtype=float)
        dimension = len(self._tanks[0].lower)
        if points.ndim != 2 or points.shape[1] != dimension:
            raise ValueError("Expected a batch with one coordinate per dimension")
        result = np.full(len(points), np.nan)
        for tank in self._tanks:
            inside = np.all((points >= tank.lower) & (points <= tank.upper), axis=1)
            inside &= np.isnan(result)
            if np.any(inside):
                local = (points[inside] - tank.lower) / tank.widths
                result[inside] = tank.network.depths(2 * local - 1)
        if np.any(np.isnan(result)):
            raise ValueError("Point outside the fitted domain")
        return result


def fit_water_map(
    integrand: Callable[[Array], Array],
    bounds: Sequence[tuple[float, float]],
    *,
    order: int,
    breaks: Sequence[Sequence[float]] | None = None,
) -> NeuralWaterMap:
    """Fit a water map from pointwise integrand values on open cosine grids.

    Supply axis-aligned breakpoints for known jumps or derivative discontinuities.
    Each panel gets its own network. The callback receives shape (N, dimension)
    and returns one signed height per row.
    """
    if len(bounds) == 0 or order < 2:
        raise ValueError("Expected a nonempty domain and training order of at least two")
    if breaks is None:
        breaks = [()] * len(bounds)
    panels = []
    for (lo, hi), cuts in zip(bounds, breaks, strict=True):
        edges = (lo, *cuts, hi)
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError("Domain endpoints and breakpoints must be strictly increasing and finite")
        panels.append(tuple(zip(edges[:-1], edges[1:], strict=True)))

    nodes = (1 + np.cos(np.pi * (np.arange(order) + 0.5) / order)) / 2
    local_points = np.stack(np.meshgrid(*([nodes] * len(bounds)), indexing="ij"), axis=-1)
    local_points = local_points.reshape(-1, len(bounds))
    tanks = []
    for panel in product(*panels):
        lower, upper = zip(*panel, strict=True)
        points = np.asarray(lower) + np.subtract(upper, lower) * local_points
        heights = np.asarray(integrand(points), dtype=float)
        if heights.shape != (len(points),) or not np.all(np.isfinite(heights)):
            raise ValueError("The integrand must return one finite scalar per point")
        network = _ColumnNetwork.learn(heights.reshape((order,) * len(bounds)))
        tanks.append(_Tank(lower, upper, network))
    return NeuralWaterMap(tuple(tanks), order)
