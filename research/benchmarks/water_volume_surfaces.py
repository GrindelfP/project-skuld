"""Analytic bounded surfaces for the water-volume integration experiments."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from math import comb, factorial, prod

import mpmath as mp
import numpy as np
from numpy.typing import NDArray

from . import genz

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Surface:
    name: str
    evaluate: Callable[[FloatArray], FloatArray]
    exact: mp.mpf


def ridge_surface(
    name: str, frequencies: FloatArray, phases: FloatArray, amplitudes: FloatArray,
    offset: str, scale: str = "1",
) -> Surface:
    """Create a sine mixture and score it via vertex differences in mp arithmetic."""
    dimension = frequencies.shape[1]
    vertices = tuple(product((0, 1), repeat=dimension))
    integral = mp.mpf(offset)
    for frequency, phase, amplitude in zip(frequencies, phases, amplitudes, strict=True):
        omega = [mp.mpf(float(w)) for w in frequency]
        # The n-fold mixed primitive of sin(s) is sin(s - n*pi/2) / prod(w).
        corner_sum = mp.fsum(
            (-1)**(dimension - sum(vertex)) *
            mp.sin(mp.mpf(float(phase)) + mp.fdot(omega, vertex) - dimension * mp.pi / 2)
            for vertex in vertices
        )
        integral += mp.mpf(float(amplitude)) * corner_sum / mp.fprod(omega)

    def evaluate(points: FloatArray) -> FloatArray:
        return float(scale) * (float(offset) + np.sin(points @ frequencies.T + phases) @ amplitudes)

    return Surface(name, evaluate, mp.mpf(scale) * integral)


def surfaces(dimension: int) -> tuple[Surface, ...]:
    """Return eleven analytic fixtures, all with heights in the unit interval."""
    frequencies = np.array([
        [2.35, 3.10, -2.65, 1.75, 2.85, -1.55],
        [4.70, -1.80, 3.25, -2.15, 1.45, 2.25],
    ])[:, :dimension]
    phases = np.array([0.4, np.pi / 2 - 0.7])
    amplitudes = np.array([0.18, 0.13])
    ridge = ridge_surface("two_ridges", frequencies, phases, amplitudes, ".43")
    tiny = ridge_surface("tiny_integral", frequencies, phases, amplitudes, ".43", "1e-6")

    rng = np.random.default_rng(8305)
    # Generate all six coordinates before slicing so the 2D modes extend to 6D.
    signs = rng.choice([-1, 1], size=(6, 6))
    fast = (9 + 14 * rng.random((6, 6))) * signs
    multiscale = ridge_surface(
        "six_fast_ridges", fast[:, :dimension], rng.uniform(-3, 3, size=6),
        np.array([.095, .076, .062, .048, .037, .025]), ".48",
    )

    def polynomial(points: FloatArray) -> FloatArray:
        return np.prod(1 + points * np.roll(points, -1, axis=1), axis=1) / 2**dimension

    polynomial_exact = mp.mpf(0)
    for edges in product((0, 1), repeat=dimension):
        polynomial_exact += 1 / mp.mpf(prod(1 + edges[i] + edges[i - 1] for i in range(dimension)))

    def exponential(points: FloatArray) -> FloatArray:
        return np.exp(4 * (points.prod(axis=1) - 1))

    exponential_exact = mp.exp(-4) * mp.fsum(
        mp.mpf(4)**k / (mp.factorial(k) * (k + 1)**dimension) for k in range(96)
    )

    phases = np.array([np.pi / 2 + 0.78])
    oscillation = ridge_surface(
        "genz_oscillatory", np.linspace(3.8, 7.2, 6)[None, :dimension],
        phases, np.array([0.43]), ".5",
    )
    offsets = np.linspace(.23, .77, 6)[:dimension]
    u = [mp.mpf(float(value)) for value in offsets]
    lorentz_width = np.linspace(.07, .16, 6)[:dimension]
    gaussian_rate = np.linspace(7, 13, 6)[:dimension]
    kink_rate = np.linspace(3, 7, 6)[:dimension]
    corner_rate = np.linspace(.9, 1.9, 6)[:dimension]

    def high_precision(values: FloatArray) -> list[mp.mpf]:
        return [mp.mpf(float(value)) for value in values]

    lorentz_exact = mp.fprod(
        a * (mp.atan((1 - x) / a) + mp.atan(x / a))
        for a, x in zip(high_precision(lorentz_width), u, strict=True)
    )
    gaussian_exact = mp.fprod(
        mp.sqrt(mp.pi) / (2 * a) * (mp.erf(a * (1 - x)) + mp.erf(a * x))
        for a, x in zip(high_precision(gaussian_rate), u, strict=True)
    )
    kink_exact = mp.fprod(
        (2 - mp.exp(-a * x) - mp.exp(-a * (1 - x))) / a
        for a, x in zip(high_precision(kink_rate), u, strict=True)
    )
    a = high_precision(corner_rate)
    corner_exact = mp.fsum(
        (-1)**sum(vertex) / (1 + mp.fdot(a, vertex))
        for vertex in product((0, 1), repeat=dimension)
    ) / (factorial(dimension) * mp.fprod(a))

    threshold = mp.mpf(dimension) / 2 + mp.mpf(".17")
    below_jump = mp.fsum(
        (-1)**k * comb(dimension, k) * (threshold - k)**dimension
        for k in range(int(threshold) + 1)
    ) / factorial(dimension)

    def jump(points: FloatArray) -> FloatArray:
        return np.where(points.sum(axis=1) < float(threshold), .08, .92)

    return (
        ridge, multiscale,
        Surface("coupled_polynomial", polynomial, polynomial_exact / 2**dimension),
        Surface("coupled_exponential", exponential, exponential_exact),
        oscillation,
        Surface("genz_product_peak", lambda x: genz.product_peak_np(x, lorentz_width, offsets), lorentz_exact),
        Surface("genz_corner_peak", lambda x: genz.corner_peak_np(x, corner_rate), corner_exact),
        Surface("genz_gaussian", lambda x: genz.gaussian_np(x, gaussian_rate, offsets), gaussian_exact),
        Surface("genz_c0", lambda x: genz.c0_continuous_np(x, kink_rate, offsets), kink_exact),
        Surface("oblique_jump", jump, mp.mpf(".92") - mp.mpf(".84") * below_jump),
        tiny,
    )


def digits(estimate: float, exact: mp.mpf) -> tuple[float, float]:
    relative = abs(mp.mpf(estimate) - exact) / abs(exact)
    return float(relative), float(-mp.log10(relative))
