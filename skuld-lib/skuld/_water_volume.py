"""Experimental neural occupancy model on the unit hypercube."""

from dataclasses import dataclass
from math import fsum

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]


class UncertifiedWaterDepthError(RuntimeError):
    """The fitted networks could not certify a water depth in [0, 1]."""


def _features(points: FloatArray, frequencies: FloatArray) -> FloatArray:
    angles = (points - 0.5) @ frequencies.T
    return np.column_stack((np.ones(len(points)), np.cos(angles), np.sin(angles)))


@dataclass(frozen=True)
class WaterVolumeNetwork:
    """A sine ridge network for water depth h(x) above the graph of f(x).

    Its occupancy at (x, z) is the hard threshold z >= 1 - h(x). Each
    sine/cosine pair represents one sine neuron with a learned phase.
    """

    frequencies: FloatArray
    coefficients: FloatArray

    def depth(self, points: FloatArray) -> FloatArray:
        return _features(points, self.frequencies) @ self.coefficients

    def occupancy(self, points: FloatArray, z: FloatArray | float) -> NDArray[np.bool_]:
        """Mark whether the queried heights are in the water above the surface."""
        return z >= 1 - self.depth(points)

    def depth_bounds(self) -> tuple[float, float]:
        """Return global bounds by summing the amplitudes of the neurons."""
        cosine, sine = np.split(self.coefficients[1:], 2)
        radius = fsum(np.hypot(cosine, sine))
        bias = self.coefficients[0]
        return float(bias - radius), float(bias + radius)

    def water_volume(self) -> float:
        lower, upper = self.depth_bounds()
        if not 0 <= lower <= upper <= 1:
            raise ValueError("The model's depth bounds must lie in [0, 1]")

        # After vertical integration of occupancy, integrate h(x). On the
        # centered cube the sine moments vanish by symmetry, while each
        # cosine moment factors into sinc(w_j / 2) over the coordinates.
        moments = np.sinc(self.frequencies / (2 * np.pi)).prod(axis=1)
        cosine = self.coefficients[1 : 1 + len(moments)]
        return fsum([self.coefficients[0], *(cosine * moments)])

    @classmethod
    def fit(
        cls,
        points: FloatArray,
        heights: FloatArray,
        *,
        width: int,
        frequency_limit: float,
        restarts: int,
        rng: np.random.Generator,
        max_nfev: int = 300,
    ) -> "WaterVolumeNetwork":
        """Learn frequencies and output weights from pointwise surface heights."""
        if points.ndim != 2 or not points.shape[1]:
            raise ValueError("Expected sample points of shape (n_samples, dimension)")
        if heights.shape != (len(points),) or not np.all((0 <= heights) & (heights <= 1)):
            raise ValueError("Expected one height in [0, 1] per sample")
        if not np.all((0 <= points) & (points <= 1)):
            raise ValueError("Sample points must lie in the unit cube")
        if width < 1 or restarts < 1 or frequency_limit <= 0 or not np.isfinite(frequency_limit):
            raise ValueError("Expected positive width, restarts and finite frequency limit")
        if max_nfev < 1:
            raise ValueError("Expected a positive optimization budget")
        dimension = points.shape[1]
        if len(points) < 1 + width * (dimension + 2):
            raise ValueError("Expected at least as many samples as fitted parameters")

        water_depth = 1 - heights

        def project(flat_frequencies: FloatArray) -> tuple[FloatArray, FloatArray]:
            features = _features(points, flat_frequencies.reshape(width, dimension))
            coefficients = np.linalg.lstsq(features, water_depth, rcond=None)[0]
            return coefficients, features @ coefficients - water_depth

        def residual(flat_frequencies: FloatArray) -> FloatArray:
            return project(flat_frequencies)[1]

        best = None
        best_rms = float("inf")
        for _ in range(restarts):
            start = rng.uniform(-frequency_limit, frequency_limit, size=width * dimension)
            fit = least_squares(
                residual,
                start,
                bounds=(-frequency_limit, frequency_limit),
                gtol=None,
                xtol=1e-13,
                ftol=1e-13,
                max_nfev=max_nfev,
            )
            coefficients, errors = project(fit.x)
            candidate = cls(fit.x.reshape(width, dimension), coefficients)
            lower, upper = candidate.depth_bounds()
            rms = float(np.sqrt(np.mean(errors**2)))
            if 0 <= lower <= upper <= 1 and rms < best_rms:
                best, best_rms = candidate, rms
                if best_rms <= 5e-15:
                    break

        if best is None:
            raise UncertifiedWaterDepthError("No fitted network certified a water depth in [0, 1]")
        return best
