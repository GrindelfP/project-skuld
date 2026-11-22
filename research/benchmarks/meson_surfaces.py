"""Mesh-free point samples and known boundary weights for the meson surfaces."""

import numpy as np

from physics import D_R2, L_LOG


def known_weights(points: np.ndarray, a: int, b: int, m: int) -> np.ndarray:
    x, y, z = points.T
    return x**a * (1 - x)**(b + 1) * y**b * (-np.expm1(-L_LOG * z))**m


def meson_samples(points: np.ndarray, a: int, b: int, m: int) -> np.ndarray:
    """Evaluate a full meson integrand after simplex and log-time substitutions."""
    x, y, z = points.T
    t = np.expm1(L_LOG * z)
    d, r2 = D_R2(x, (1 - x) * y)
    return L_LOG * known_weights(points, a, b, m) * np.exp(-t * d - t / (1 + t) * r2)


def training_values(points: np.ndarray) -> np.ndarray:
    """Use only pointwise values of one channel after removing its known envelope."""
    return meson_samples(points, 0, 0, 1) / (L_LOG * known_weights(points, 0, 0, 1))
