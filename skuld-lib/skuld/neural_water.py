"""Learn a signed water surface and measure the volume of its columns."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import fsum, prod

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.typing import NDArray
from scipy.fft import dctn

Array = NDArray[np.float64]


def _chebyshev_means(degrees: NDArray[np.int64]) -> Array:
    """Mean of each Chebyshev mode over a uniform axis of the tank."""
    return np.divide(
        1.0, 1.0 - degrees**2,
        out=np.zeros(degrees.shape), where=degrees % 2 == 0,
    )


@lru_cache(maxsize=16)
def _volume_moments(width: int) -> tuple[Array, Array]:
    degrees = np.arange(width)
    sums = degrees[:, None] + degrees
    differences = np.abs(degrees[:, None] - degrees)
    gram = (_chebyshev_means(sums) + _chebyshev_means(differences)) / 2
    return _chebyshev_means(degrees), np.linalg.cholesky(gram).T


def _paired_volume(left: tuple[Array, ...], right: tuple[Array, ...]) -> float:
    """Propagate the joint second moment of two neural branches."""
    joint = np.ones((1, 1))
    for a, b in zip(left, right, strict=True):
        shared_modes = min(len(a), len(b))
        a, b = a[:shared_modes], b[:shared_modes]
        weighted_b = joint @ b
        joint = a.reshape(-1, a.shape[-1]).T @ weighted_b.reshape(-1, b.shape[-1])
    return float(joint[0, 0])


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
        scale = float(np.max(np.abs(heights))) or 1.0
        normalized = (heights / scale) / 8
        # Stable inverse of z + z², the depth above the reference fill.
        latent_targets = 2 * normalized / (1 + np.sqrt(1 + 4 * normalized))
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
        return self.scale * (8 * (latent + latent**2))

    def _branch_moments(self, layers: tuple[Array, ...]) -> tuple[float, tuple[Array, ...]]:
        first = np.ones(1)
        orthogonal = []
        for layer in layers:
            width = layer.shape[1]
            means, factor = _volume_moments(width)
            modes = layer.transpose(1, 0, 2)
            averaged = (means @ modes.reshape(width, -1)).reshape(layer.shape[::2])
            first = first @ averaged
            orthogonal.append((factor @ modes.reshape(width, -1)).reshape(modes.shape))
        return float(first[0]), tuple(orthogonal)

    def volume(self) -> float:
        """Measure occupied volume relative to a flat tank at height 1/4."""
        first, layers = self._branch_moments(self.layers)
        first_correction, correction = self._branch_moments(self.correction)
        # Fill is (1/2 + z)². Cancel the reference fill algebraically and
        # propagate the remaining first and second moments through the weights.
        centered_fill = fsum((
            first, first_correction,
            _paired_volume(layers, layers),
            2 * _paired_volume(layers, correction),
            _paired_volume(correction, correction),
        ))
        return self.scale * (8 * centered_fill)


@dataclass(frozen=True)
class _Tank:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    network: _ColumnNetwork

    @property
    def widths(self) -> Array:
        return np.subtract(self.upper, self.lower)

    def volume(self) -> float:
        return prod(self.widths) * self.network.volume()


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

    def integrate(self) -> float:
        """Measure the signed volume of the learned columns."""
        return fsum(tank.volume() for tank in self._tanks)

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
