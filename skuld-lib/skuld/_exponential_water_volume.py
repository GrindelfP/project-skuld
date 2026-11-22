"""Train a bounded exponential neural surface and integrate its activation series."""

from dataclasses import dataclass
from itertools import product
from math import comb, exp, fsum, lgamma, log

import mpmath as mp
import numpy as np
from numpy.polynomial.polynomial import polymul, polypow, polyvander2d
from scipy.optimize import least_squares
from scipy.signal import convolve2d, correlate2d

from ._water_volume import FloatArray

DEGREE = 2


def _spatial_features(points: FloatArray) -> FloatArray:
    xy = 2 * points[:, :2] - 1
    return polyvander2d(xy[:, 0], xy[:, 1], (DEGREE, DEGREE))


def _features(spatial: FloatArray, z: FloatArray, rates: FloatArray) -> FloatArray:
    time = np.column_stack((np.ones(len(z)), np.exp(z[:, None] * rates)))
    return (spatial[:, None, :] * time[:, :, None]).reshape(len(z), -1)


@dataclass(frozen=True)
class ExponentialWaterSurface:
    """Two shared trainable exponential rates, spatial gates and a positive output."""

    rates: FloatArray
    coefficients: FloatArray

    @classmethod
    def fit(
        cls, points: FloatArray, positive_values: FloatArray, initial_rates: FloatArray,
        *, learn_rates: bool = True,
    ) -> "ExponentialWaterSurface":
        if (points.ndim != 2 or points.shape[1] != 3
                or positive_values.shape != (len(points),)
                or not np.all((points >= 0) & (points <= 1))
                or not np.all(np.isfinite(positive_values) & (positive_values > 0))):
            raise ValueError("Expected positive point samples on the 3D unit cube")
        target = np.log(positive_values)
        spatial = _spatial_features(points)
        z = points[:, 2] - 0.5

        def project(rates: FloatArray) -> tuple[FloatArray, FloatArray]:
            features = _features(spatial, z, rates)
            coefficients = np.linalg.lstsq(features, target, rcond=None)[0]
            return coefficients, features @ coefficients - target

        rates = initial_rates
        if learn_rates:
            result = least_squares(
                lambda trial: project(trial)[1], initial_rates,
                bounds=([0.15, -8], [8, -0.15]), xtol=1e-14, ftol=1e-14,
                gtol=None, max_nfev=200,
            )
            if not result.success:
                raise RuntimeError(result.message)
            rates = result.x
        coefficients, _ = project(rates)
        return cls(rates.copy(), coefficients.reshape(3, DEGREE + 1, DEGREE + 1))

    def log_surface(self, points: FloatArray) -> FloatArray:
        return _features(_spatial_features(points), points[:, 2] - 0.5, self.rates) @ self.coefficients.ravel()

    def log_upper_bound(self) -> float:
        bounds = np.abs(self.coefficients).sum(axis=(1, 2)) * np.exp(np.r_[0, np.abs(self.rates) / 2])
        return fsum(bounds)

    def integrate_weighted(
        self, weights: tuple[tuple[int, int, int], ...], log_extent: float,
        *, series_tolerance: float = 1e-20,
    ) -> list[float]:
        """Integrate exp(log_surface) times the separable polynomial/time weights.

        On each spatial patch, expand the three outer exponential factors
        separately. Spatial monomials and the temporal exponentials then have
        closed-form moments. No height evaluations or cubature are used.
        """
        spatial_weights = tuple(dict.fromkeys((a, b) for a, b, _ in weights))
        partial = [[] for _ in weights]
        moment_cache = {}
        for cx, cy in product((0.25, 0.75), repeat=2):
            static, positive, negative = (
                _translate(poly, cx, cy) for poly in self.coefficients
            )
            static_powers = _exp_powers(static, 0, series_tolerance)
            static_sum = np.zeros_like(static_powers[-1])
            for poly in static_powers:
                static_sum[:poly.shape[0], :poly.shape[1]] += poly
            plus = _exp_powers(positive, self.rates[0], series_tolerance)
            minus = _exp_powers(negative, self.rates[1], series_tolerance)
            spatial_size = plus[-1].shape[0] + minus[-1].shape[0] - 1
            padded_plus = np.zeros((len(plus), *plus[-1].shape))
            for index, poly in enumerate(plus):
                padded_plus[index, :poly.shape[0], :poly.shape[1]] = poly

            moments = {}
            for a, b in spatial_weights:
                wx = _monomial_moments(cx, a, b + 1, spatial_size, static_sum.shape[0])
                wy = _monomial_moments(cy, b, 0, spatial_size, static_sum.shape[1])
                combined = wx @ static_sum @ wy.T
                table = np.empty((len(plus), len(minus)))
                for j, poly in enumerate(minus):
                    window = correlate2d(combined, poly, mode="valid")[:plus[-1].shape[0], :plus[-1].shape[1]]
                    table[:, j] = np.einsum("ijk,jk->i", padded_plus, window)
                moments[a, b] = table

            for index, (a, b, m) in enumerate(weights):
                key = (len(plus), len(minus), m)
                if key not in moment_cache:
                    moment_cache[key] = _time_moments(len(plus), len(minus), m, self.rates, log_extent)
                partial[index].append(fsum((moments[a, b] * moment_cache[key]).flat))
        return [fsum(values) for values in partial]


def _translate(poly: FloatArray, cx: float, cy: float) -> FloatArray:
    # Transform 2*x-1 and 2*y-1 to centered coordinates on a half-width patch.
    def matrix(center: float, degree: int) -> FloatArray:
        return np.array([
            [comb(p, k) * (2 * center - 1)**(p - k) * 0.5**k if k <= p else 0
             for p in range(degree + 1)]
            for k in range(degree + 1)
        ])
    return matrix(cx, poly.shape[0] - 1) @ poly @ matrix(cy, poly.shape[1] - 1).T


def _exp_powers(poly: FloatArray, rate: float, tolerance: float) -> list[FloatArray]:
    bound = fsum(np.abs(poly).flat) * exp(abs(rate) / 2)
    terms = [np.ones((1, 1))]
    if bound == 0:
        return terms
    # The degree-k exponential remainder relative to exp(v), |v| <= bound,
    # is at most exp(2*bound) * bound**(k+1) / (k+1)!.
    while True:
        k = len(terms)
        if 2 * bound + k * log(bound) - lgamma(k + 1) <= log(tolerance):
            return terms
        terms.append(convolve2d(terms[-1], poly) / k)


def _monomial_moments(center: float, p: int, q: int, size: int, extra: int) -> FloatArray:
    weight = polymul(polypow([center, 0.25], p), polypow([1 - center, -0.25], q))
    k = np.arange(size + extra - 1)[:, None] + np.arange(len(weight))[None, :]
    moments = (np.where(k % 2 == 0, 0.5 / (k + 1), 0.0) @ weight)
    return moments[np.arange(size)[:, None] + np.arange(extra)[None, :]]


def _time_moments(n: int, k: int, m: int, rates: FloatArray, extent: float) -> FloatArray:
    with mp.workdps(40):
        first, second = (mp.mpf(float(rate)) for rate in rates)
        length = mp.mpf(extent)
        binomial = [(-1)**j * comb(m, j) * mp.exp(-j * length / 2) for j in range(m + 1)]

        def moment(i: int, ell: int) -> float:
            slope = i * first + ell * second
            terms = []
            for j, weight in enumerate(binomial):
                half = (slope - j * length) / 2
                terms.append(weight * (mp.sinh(half) / half if half else 1))
            return float(mp.fsum(terms))

        return np.array([[moment(i, ell) for ell in range(k)] for i in range(n)])
