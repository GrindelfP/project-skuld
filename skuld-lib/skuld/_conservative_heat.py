"""Pointwise collocation networks for conservative insulated heat flow."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as cheb


def sites(size):
    """Interior, increasingly dense near both endpoints."""
    theta = np.pi * (np.arange(size) + .5) / size
    return np.cos(theta + .06 * np.sin(2 * theta) / size)


def apply_axis(weights, field, axis):
    return np.moveaxis(np.tensordot(weights, field, axes=(1, axis)), 0, axis)


def evaluate(field, xyz):
    shape = field.shape
    basis = [cheb.chebvander(xyz[:, i], shape[i] - 1) for i in range(3)]
    return np.einsum('ni,nj,nk,ijk->n', *basis, field, optimize=True)


class InitialNetwork:
    """Chebyshev activations with three trainable multiplicative tensor cores."""

    def __init__(self, left, middle, right):
        self.left, self.middle, self.right = left, middle, right

    @classmethod
    def fit(cls, values, nodes, rank):
        size = len(nodes)
        vander = cheb.chebvander(nodes, size - 1)
        coefficients = values
        inverse = np.linalg.solve(vander, np.eye(size))
        for axis in range(3):
            coefficients = apply_axis(inverse, coefficients, axis)
        left, singular, right = np.linalg.svd(coefficients.reshape(size, -1), full_matrices=False)
        left = left[:, :rank]
        middle = (singular[:rank, None] * right[:rank]).reshape(rank * size, size)
        factors, singular, right = np.linalg.svd(middle, full_matrices=False)
        middle = (factors[:, :rank] * singular[:rank]).reshape(rank, size, rank)
        return cls(left, middle, right[:rank])

    @property
    def parameters(self):
        return self.left.size + self.middle.size + self.right.size

    def forward(self, xyz):
        basis = cheb.chebvander(xyz, self.left.shape[0] - 1)
        left = basis[:, 0] @ self.left
        right = basis[:, 2] @ self.right.T
        return np.einsum('na,ni,aib,nb->n', left, basis[:, 1], self.middle, right, optimize=True)

    def coefficients(self):
        return np.einsum('ia,ajb,bk->ijk', self.left, self.middle, self.right, optimize=True)


# Two-stage, third-order, L-stable Radau IIA method.
RADAU = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])


class HeatNetwork:
    """Shared separable time-step layers fitted to the local heat equations.

    In each axis the stage field's derivative is parameterized as
    (1-z²) times a learned polynomial, so the normal flux vanishes at z=±1.
    The next state is formed from the time-integrated divergence of this flux.
    """

    def __init__(self, stage, stage_dx, stage_dxx, duration):
        self.stage = stage
        self.stage_dx = stage_dx
        self.stage_dxx = stage_dxx
        self.duration = duration
        # The transition uses the same weighted second derivatives as the last stage equation.
        self.transition = np.eye(stage.shape[-1]) + duration * np.einsum('j,jab->ab', RADAU[-1], stage_dxx)

    @classmethod
    def fit(cls, size, duration):
        # z=2x-1, hence d/dx=2 d/dz. The first potential feature is 1.
        neumann = np.zeros((size + 2, size))
        neumann[0, 0] = 1
        for k in range(size - 1):
            feature = cheb.chebint(cheb.chebmul([.5, 0, -.5], np.eye(size)[k]))
            neumann[:len(feature), k + 1] = feature
        dx = 2 * cheb.chebder(neumann, axis=0)
        dxx = 2 * cheb.chebder(dx, axis=0)
        z = sites(size + 4)
        stage_values = cheb.chebvander(z, size + 1) @ neumann
        second_values = cheb.chebvander(z, size - 1) @ dxx
        inputs = cheb.chebvander(z, size - 1)
        system = np.kron(np.eye(2), stage_values) - duration * np.kron(RADAU, second_values)
        weights = np.linalg.lstsq(system, np.tile(inputs, (2, 1)), rcond=None)[0].reshape(2, size, size)
        return cls(neumann @ weights, dx @ weights, dxx @ weights, duration)

    def advance(self, field, axis):
        return apply_axis(self.transition, field, axis)

    def defects(self, field, xyz, axis):
        """Independent pointwise stage balance, boundary flux, and collocation residual."""
        initial = evaluate(field, xyz)
        delta = np.stack([evaluate(apply_axis(layer, field, axis), xyz) for layer in self.stage_dxx])
        stage = np.stack([evaluate(apply_axis(layer, field, axis), xyz) for layer in self.stage])
        upper, lower = xyz.copy(), xyz.copy()
        upper[:, axis], lower[:, axis] = 1., -1.
        flux = np.stack([evaluate(apply_axis(layer, field, axis), face)
                         for layer in self.stage_dx for face in (upper, lower)])
        # The stage approximation oversamples a derivative-constrained basis, so
        # its pointwise balance residual need not vanish at the first time steps.
        residual = stage - initial - self.duration * RADAU @ delta
        return np.max(np.abs(residual), axis=0), np.max(np.abs(flux), axis=0)

    def balance(self, field, later, xyz, axis):
        change = evaluate(later - field, xyz)
        divergence = [evaluate(apply_axis(stage, field, axis), xyz) for stage in self.stage_dxx]
        represented = self.duration * (RADAU[-1] @ divergence)
        return abs(change - represented)


@dataclass
class ControlReport:
    measured_max: np.ndarray
    proposed_max: np.ndarray
    prediction_error: np.ndarray


class HeatController:
    """Learn a nonlinear spatial defect map and nominate new audit sites."""

    def __init__(self, seed, width=48):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 2, (3, width))
        self.bias = rng.uniform(-2, 2, width)

    def features(self, xyz):
        return np.column_stack((np.ones(len(xyz)), np.tanh(xyz @ self.weights + self.bias)))

    def audit(self, defects, train, candidates, verification):
        labels = defects(train)
        scale = np.maximum(np.max(labels, axis=0), np.finfo(float).tiny)
        design = self.features(train)
        regularized = np.vstack((design, 1e-4 * np.eye(design.shape[1])))
        targets = np.vstack((labels / scale, np.zeros((design.shape[1], labels.shape[1]))))
        self.output = np.linalg.lstsq(regularized, targets, rcond=None)[0]
        prediction = self.features(candidates) @ self.output * scale
        indices = np.unique(np.argsort(prediction, axis=0)[-8:])
        proposed = defects(candidates[indices])
        actual = defects(verification)
        held_prediction = self.features(verification) @ self.output * scale
        measured_max = np.maximum.reduce((labels.max(0), proposed.max(0), actual.max(0)))
        return ControlReport(measured_max, proposed.max(0), abs(held_prediction - actual).max(0))
