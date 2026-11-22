"""Refine a trained water-surface network and integrate its moments at high precision."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb

import mpmath as mp
import numpy as np

from ._neural_water_volume import NeuralWaterSurface


def _mp_values(values):
    return [mp.mpf(float(value)) for value in values]


@dataclass
class PrecisionSurface:
    """Scalar version of the anchored PyTorch layers, with multiprecision backpropagation."""

    x_weights: list
    x_biases: list
    y_weights: list
    y_biases: list
    x_mixing: list
    x_offset: list
    y_mixing: list
    y_offset: list
    rates: list
    head: list

    @classmethod
    def from_network(cls, network: NeuralWaterSurface) -> "PrecisionSurface":
        if not network.anchored:
            raise ValueError("The precision readout expects the anchored network")
        parameters = [_mp_values(p.detach().flatten().numpy()) for p in network.parameters()]
        return cls(*parameters)

    def _axis(self, coordinate, weights, biases, mixing, offset):
        value = 2 * coordinate - 1
        pre = [w * value + b for w, b in zip(weights, biases, strict=True)]
        activations = [v**2 for v in pre]
        features = [
            mp.fdot(mixing[3*i:3*i+3], activations) + offset[i] for i in range(3)
        ]
        return features, pre, activations

    def log_and_gradient(self, point) -> tuple:
        x, y, z = point
        xf, xp, xa = self._axis(x, self.x_weights, self.x_biases, self.x_mixing, self.x_offset)
        yf, yp, ya = self._axis(y, self.y_weights, self.y_biases, self.y_mixing, self.y_offset)
        base = [mp.exp(-rate / 2) for rate in self.rates]
        time = [b * mp.expm1(rate * z) for b, rate in zip(base, self.rates, strict=True)]
        time_derivative = [z * (b + f) - f / 2 for b, f in zip(base, time, strict=True)]
        spatial = [a * b for a in xf for b in yf]
        gates = [mp.fdot(self.head[9*i:9*i+9], spatial) for i in range(2)]
        mixing = [mp.fdot(time, (self.head[j], self.head[9+j])) for j in range(9)]
        gx = [mp.fdot(mixing[3*i:3*i+3], yf) for i in range(3)]
        gy = [mp.fdot(mixing[j::3], xf) for j in range(3)]
        gx_pre = [2 * xp[j] * mp.fdot(gx, self.x_mixing[j::3]) for j in range(3)]
        gy_pre = [2 * yp[j] * mp.fdot(gy, self.y_mixing[j::3]) for j in range(3)]
        gradient = [
            *[(2*x - 1) * v for v in gx_pre], *gx_pre,
            *[(2*y - 1) * v for v in gy_pre], *gy_pre,
            *[v * a for v in gx for a in xa], *gx,
            *[v * a for v in gy for a in ya], *gy,
            *[v * a for v, a in zip(gates, time_derivative, strict=True)],
            *[v * a for v in time for a in spatial],
        ]
        return mp.fdot(time, gates), gradient

    def apply(self, step) -> None:
        offset = 0
        for group in vars(self).values():
            for i in range(len(group)):
                group[i] += step[offset]
                offset += 1

    def refine(self, points, samples) -> tuple:
        """Take high-precision Gauss–Newton steps in identifiable weight directions."""
        target = [mp.log(value) for value in samples]
        errors = []
        for _ in range(5):
            predictions = [self.log_and_gradient(point) for point in points]
            residual = mp.matrix([value - ref for (value, _), ref in zip(predictions, target, strict=True)])
            errors.append(max(map(abs, residual)))
            if errors[-1] < mp.mpf("1e-48"):
                return errors[0], errors[-1]
            jacobian = mp.matrix([row for _, row in predictions])
            _, singular, right = np.linalg.svd(
                np.array(jacobian.tolist(), dtype=np.float64), full_matrices=False,
            )
            rank = np.count_nonzero(singular > singular[0] * 1e-12)
            tangent = mp.matrix(right[:rank].T.tolist())
            step, _ = mp.qr_solve(jacobian * tangent, -residual)
            self.apply(tangent * step)
        raise RuntimeError(f"High-precision network fit stalled at residual {errors[-1]}")

    def moment_polynomials(self) -> list:
        """Expand the trained network layers into three exact spatial polynomials."""
        def branch(weights, biases, mixing, offset):
            powers = [[b*b, 2*w*b, w*w] for w, b in zip(weights, biases, strict=True)]
            result = mp.matrix([mixing[3*i:3*i+3] for i in range(3)]) * mp.matrix(powers)
            for i, bias in enumerate(offset):
                result[i, 0] += bias
            return result

        x = branch(self.x_weights, self.x_biases, self.x_mixing, self.x_offset)
        y = branch(self.y_weights, self.y_biases, self.y_mixing, self.y_offset)
        moving = [
            x.T * mp.matrix([self.head[i*9+3*j:i*9+3*j+3] for j in range(3)]) * y
            for i in range(2)
        ]
        static = -sum((mp.exp(-rate/2) * poly for rate, poly in zip(self.rates, moving, strict=True)),
                      start=mp.zeros(3))
        return [static, *moving]


@dataclass(frozen=True)
class NeuralWaterGeometry:
    """The boundary, water depth and occupied volume induced by the fitted weights."""

    surface: PrecisionSurface
    log_extent: object

    def scale(self):
        polys = self.surface.moment_polynomials()
        bound = mp.fsum(mp.exp(abs(rate) / 2) * mp.fsum(abs(v) for v in poly)
                        for rate, poly in zip((0, *self.surface.rates), polys, strict=True))
        return self.log_extent * mp.exp(bound)

    def water_depth(self, point, a: int, b: int, m: int):
        x, y, z = point
        weight = x**a * (1-x)**(b+1) * y**b * (-mp.expm1(-self.log_extent*z))**m
        return 1 - self.log_extent * weight * mp.exp(self.surface.log_and_gradient(point)[0]) / self.scale()

    def water_volumes(self, weights, *, tolerance):
        scale = self.scale()
        occupied = _integrate_network(
            self.surface.moment_polynomials(), self.surface.rates, weights, self.log_extent, tolerance,
        )
        return [1 - self.log_extent * volume / scale for volume in occupied]


# Fixed-point integer contractions keep spatial and temporal moment products
# well above the requested resolution. Each temporal factor is bounded by one.
_SCALE_BITS = 224
_UNIT = 1 << _SCALE_BITS


def _fixed(values) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    integers = [int(mp.nint(v * _UNIT)) for v in array.flat]
    return np.asarray(integers, dtype=object).reshape(array.shape)


def _patch_bound(poly, cx, cy):
    def shift(center):
        return mp.matrix([[comb(p, k) * (2*center - 1)**(p-k) / 2**k if k <= p else 0
                           for p in range(3)] for k in range(3)])
    return mp.fsum(abs(value) for value in shift(cx) * poly * shift(cy).T)


def _last_power(bound, tolerance) -> int:
    if not bound:
        return 0
    k = 0
    term = mp.exp(2*bound) * bound
    while term > tolerance:
        k += 1
        term *= bound / (k + 1)
    return k


def _powers(values, last):
    sequence = [np.full_like(values, _UNIT)]
    for n in range(1, last+1):
        sequence.append(sequence[-1] * values // (_UNIT * n))
    return np.stack(sequence)


@lru_cache(maxsize=32)
def _polynomial_rule(n: int, precision: int):
    nodes, weights = mp.gauss_quadrature(n, "legendre01")
    return tuple(nodes), tuple(weights)


def _time_moments(last_plus, last_minus, m, rates, extent):
    first, second = rates
    result = []
    powers = [(-1)**k * comb(m, k) * mp.exp(-k*extent/2) for k in range(m+1)]
    for i in range(last_plus+1):
        row = []
        for j in range(last_minus+1):
            normalized = mp.exp(-(i*abs(first) + j*abs(second)) / 2)
            components = []
            for k, scale in enumerate(powers):
                half = (i*first + j*second - k*extent) / 2
                components.append(scale * (mp.sinh(half)/half if half else 1))
            row.append(normalized * mp.fsum(components))
        result.append(row)
    return _fixed(result)


def _integrate_network(polys, rates, weights, extent, tolerance):
    # Normalize each temporal exponential to <=1, avoiding large moment/weight cancellations.
    factors = [p * mp.exp(abs(r)/2) for p, r in zip(polys, (0, *rates), strict=True)]
    output = [0 for _ in weights]
    for left_x, left_y in product((mp.mpf(0), mp.mpf("0.5")), repeat=2):
        orders = tuple(_last_power(_patch_bound(p, left_x + mp.mpf(".25"), left_y + mp.mpf(".25")), tolerance)
                       for p in factors)
        # The rule contracts *finite spatial polynomial moments* exactly in real
        # arithmetic. Its order derives from the maximum degree, not a tolerance
        # check on integrand values. Only the fitted coefficients enter here.
        degree = 2 * sum(orders) + max(a + b + 1 for a, b, _ in weights)
        nodes, quadrature_weights = _polynomial_rule((degree + 2)//2, mp.mp.prec)
        x = [left_x + node/2 for node in nodes]
        y = [left_y + node/2 for node in nodes]
        vx = _fixed([[(2*v - 1)**k for k in range(3)] for v in x])
        vy = _fixed([[(2*v - 1)**k for k in range(3)] for v in y])
        static, plus, minus = (
            _powers(((vx @ _fixed(p.tolist()) @ vy.T) >> (2*_SCALE_BITS)).ravel(), order)
            for p, order in zip(factors, orders, strict=True)
        )
        constant = static.sum(axis=0)
        temporal = {m: _time_moments(*orders[1:], m, rates, extent) for m in {case[2] for case in weights}}
        spatial = {}
        for a, b, m in weights:
            if (a, b) not in spatial:
                wx = _fixed([q * v**a * (1-v)**(b+1)/2 for v, q in zip(x, quadrature_weights, strict=True)])
                wy = _fixed([q * v**b/2 for v, q in zip(y, quadrature_weights, strict=True)])
                measure = (np.outer(wx, wy).ravel() * constant) >> (2*_SCALE_BITS)
                spatial[a, b] = (((plus * measure) >> _SCALE_BITS) @ minus.T) >> _SCALE_BITS
        for index, (a, b, m) in enumerate(weights):
            output[index] += int((spatial[a, b] * temporal[m]).sum())
    return [mp.mpf(value) / (_UNIT * _UNIT) for value in output]
