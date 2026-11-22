"""Range-certified neural surfaces with analytic water-volume readout."""

from collections import deque
from dataclasses import dataclass
from math import comb, fsum

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from ._water_volume import FloatArray, WaterVolumeNetwork, _features


@dataclass(frozen=True)
class RangeCheck:
    status: str
    cells: int = 0
    witness: FloatArray | None = None


def certify_ridge(model: WaterVolumeNetwork, max_cells: int = 1023) -> RangeCheck:
    """Enclose heights on adaptive boxes using a centered Taylor expansion.

    Sum the gradients before bounding their contribution so that cancellation
    between neurons is preserved. The second-order remainder is bounded per
    neuron using its amplitude. A float64 safety margin guards the comparison.
    """
    bias = model.coefficients[0]
    cosine, sine = np.split(model.coefficients[1:], 2)
    omega = model.frequencies
    amplitude = np.hypot(cosine, sine)
    rounding = 64 * np.finfo(float).eps * (
        abs(bias) + fsum(amplitude * (1 + np.abs(omega).sum(axis=1)))
    )
    lower, upper = model.depth_bounds()
    if rounding <= lower <= upper <= 1 - rounding:
        return RangeCheck("amplitude")

    rates = np.abs(omega)
    splitting_weight = np.sqrt(amplitude @ rates**2)
    dimension = omega.shape[1]
    pending = deque([(np.zeros(dimension), np.full(dimension, 0.5))])
    visits = 0
    unresolved = False
    while pending and visits < max_cells:
        center, half_width = pending.popleft()
        visits += 1
        phase = omega @ center
        cos_phase, sin_phase = np.cos(phase), np.sin(phase)
        value = fsum([bias, *(cosine * cos_phase + sine * sin_phase)])
        if value < -rounding or value > 1 + rounding:
            return RangeCheck("violation", visits, center + 0.5)
        gradient = (-cosine * sin_phase + sine * cos_phase) @ omega
        excursion = rates @ half_width
        radius = fsum(abs(gradient) * half_width) + 0.5 * fsum(amplitude * excursion**2)
        if rounding <= value - radius and value + radius <= 1 - rounding:
            continue

        axis = int(np.argmax(splitting_weight * half_width))
        if half_width[axis] <= 2**-22:
            unresolved = True
            continue
        half_width[axis] /= 2
        shift = np.zeros(dimension)
        shift[axis] = half_width[axis]
        pending.extend(((center - shift, half_width.copy()), (center + shift, half_width)))
    return RangeCheck("unresolved" if pending or unresolved else "subdivision", visits)


def _least_squares_objective(
    features: FloatArray, values: FloatArray, regularization: float,
):
    normal = features.T @ features / len(values)
    normal += regularization * np.eye(normal.shape[0])
    cross = features.T @ values / len(values)

    def objective(weights: FloatArray) -> tuple[float, FloatArray]:
        product = normal @ weights
        return float(weights @ (0.5 * product - cross)), product - cross

    return objective


def refit_at_violations(
    model: WaterVolumeNetwork, check: RangeCheck, points: FloatArray, heights: FloatArray,
    max_cells: int,
) -> tuple[WaterVolumeNetwork, RangeCheck]:
    """Impose local output constraints at counterexamples found by certification."""
    features = _features(points, model.frequencies)
    objective = _least_squares_objective(features, heights, 1e-12)
    weights = np.zeros_like(model.coefficients)
    weights[0] = float(np.mean(heights))
    witnesses = []
    for _ in range(4):
        if check.witness is None:
            break
        witnesses.append(check.witness)
        active = _features(np.asarray(witnesses), model.frequencies)
        result = minimize(
            objective, weights, jac=True, method="SLSQP",
            constraints=(LinearConstraint(active, 2e-8, 1 - 2e-8),),
            options={"ftol": 1e-13, "maxiter": 160},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            break
        weights = result.x
        model = WaterVolumeNetwork(model.frequencies, weights)
        check = certify_ridge(model, max_cells)
    return model, check


@dataclass(frozen=True)
class ConvexRidgeSurface:
    """A convex combination of 0, 1, and shifted trig neurons in [0,1]."""

    frequencies: FloatArray
    weights: FloatArray

    def height(self, points: FloatArray) -> FloatArray:
        trig = _features(points, self.frequencies)[:, 1:]
        return _bounded_features(trig) @ self.weights

    def integrate(self) -> float:
        cos_moments = np.sinc(self.frequencies / (2 * np.pi)).prod(axis=1)
        moments = np.concatenate((cos_moments, np.zeros_like(cos_moments)))
        return fsum(_bounded_features(moments[None, :])[0] * self.weights)

    @classmethod
    def fit(
        cls, frequencies: FloatArray, points: FloatArray, heights: FloatArray,
    ) -> "ConvexRidgeSurface":
        activation = _bounded_features(_features(points, frequencies)[:, 1:])
        initial = np.zeros(activation.shape[1])
        initial[1] = np.mean(heights)
        initial[0] = 1 - initial[1]
        result = minimize(
            _least_squares_objective(activation, heights, 0), initial,
            jac=True, method="SLSQP", bounds=[(0, 1)] * len(initial),
            constraints=(LinearConstraint(np.ones((1, len(initial))), 1, 1),),
            options={"ftol": 1e-13, "maxiter": 160},
        )
        weights = np.maximum(result.x, 0) if result.success else initial
        return cls(frequencies, weights / fsum(weights))


def _bounded_features(trig: FloatArray) -> FloatArray:
    return np.column_stack((np.zeros(len(trig)), np.ones(len(trig)), (1 + trig) / 2, (1 - trig) / 2))


def _bernstein_basis(points: FloatArray, degree: int) -> FloatArray:
    index = np.arange(degree + 1)
    return np.array([comb(degree, k) for k in index]) * points[:, None]**index * (
        1 - points[:, None]
    )**(degree - index)


@dataclass(frozen=True)
class BernsteinSurface:
    """A tensor-product hidden layer with nonnegative activations summing to one."""

    coefficients: FloatArray

    def height(self, points: FloatArray) -> FloatArray:
        degree = self.coefficients.shape[0] - 1
        dimension = self.coefficients.ndim
        axes = "abcdef"[:dimension]
        features = [_bernstein_basis(points[:, j], degree) for j in range(dimension)]
        inputs = ",".join(f"n{axis}" for axis in axes)
        return np.einsum(f"{inputs},{axes}->n", *features, self.coefficients, optimize=True)

    def integrate(self) -> float:
        return fsum(self.coefficients.flat) / self.coefficients.size

    @classmethod
    def fit(
        cls, heights: FloatArray, nodes: FloatArray,
    ) -> "BernsteinSurface":
        basis = _bernstein_basis(nodes, len(nodes) - 1)
        coefficients = heights.copy()
        for axis in range(heights.ndim):
            leading = np.moveaxis(coefficients, axis, 0)
            solved = np.linalg.solve(basis, leading.reshape(len(nodes), -1))
            coefficients = np.moveaxis(solved.reshape(leading.shape), 0, axis)
        return cls(np.clip(coefficients, 0, 1))
