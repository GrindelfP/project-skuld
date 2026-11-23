"""Multiprecision optimization and readout for the compact heat PINN."""
from itertools import product

import mpmath as mp
import torch

from heat_pinn import HORIZON, sample_sites


class CompactField:
    """Same sine/square network, evaluated with multiprecision derivatives."""

    def __init__(self, model):
        if not model.compact or len(model.rates) != 1:
            raise ValueError('Precision refinement requires the compact single-rate PINN')
        self.dimension = model.dimension
        self.width = model.spatial.out_features
        self.rate = mp.pi**2
        spatial = torch.cat((model.spatial.weight, model.spatial.bias[:, None]), dim=1)
        hidden = torch.cat((model.layers[0].weight, model.layers[0].bias[:, None]), dim=1)
        output = torch.cat((model.output.weight, model.output.bias[:, None]), dim=1)
        self.weights = mp.matrix([mp.mpf(v) for group in (spatial, hidden, output)
                                  for v in group.detach().flatten().tolist()])

    def evaluate(self, x, t, kind='value', axis=0, differentiate=False):
        dimension, width = self.dimension, self.width
        spatial_end = width * (dimension + 1)
        hidden_end = spatial_end + width * (width + 2)
        spatial = [self.weights[i:i + dimension + 1] for i in range(0, spatial_end, dimension + 1)]
        hidden = [self.weights[i:i + width + 2] for i in range(spatial_end, hidden_end, width + 2)]
        output = self.weights[hidden_end:]
        inputs = [2 * v - 1 for v in x] + [mp.mpf(1)]
        temporal = mp.exp(-self.rate * t)
        phase = [mp.fdot(row, inputs) for row in spatial]
        sine, cosine = [mp.sin(p) for p in phase], [mp.cos(p) for p in phase]
        feature = sine + [temporal, mp.mpf(1)]
        active_axes = range(dimension) if kind == 'pde' else ((axis,) if kind == 'flux' else ())
        space = [[2 * row[k] * c for row, c in zip(spatial, cosine, strict=True)] + [0, 0]
                 for k in active_axes]
        norm = [4 * mp.fdot(row[:-1], row[:-1]) for row in spatial] if kind == 'pde' else []
        # Store the jet components that appear in the selected residual.
        operator = ([-n * v for n, v in zip(norm, sine, strict=True)] + [self.rate * temporal, 0]
                    if kind == 'pde' else [])
        # operator is Laplacian - time derivative; the PDE residual uses its negative.
        z = [mp.fdot(row, feature) for row in hidden]
        dz = [[mp.fdot(row, derivative) for derivative in space] for row in hidden]
        op = [mp.fdot(row, operator) for row in hidden] if operator else []
        if kind == 'pde':
            activations = [-2 * (v * o + mp.fdot(g, g)) for v, o, g in zip(z, op, dz, strict=True)]
        elif kind == 'flux':
            activations = [2 * v * g[0] for v, g in zip(z, dz, strict=True)]
        else:
            activations = [v**2 for v in z]
        bias_multiplier = int(kind == 'value')
        result = mp.fdot(output[:width], activations) + bias_multiplier * output[width]
        if not differentiate:
            return result

        hidden_jac = []
        sin_grad = [mp.mpf(0)] * width
        derivative_grad = [[mp.mpf(0)] * width for _ in active_axes]
        operator_grad = [mp.mpf(0)] * width
        for j, row in enumerate(hidden):
            if kind == 'pde':
                z_adj, op_adj = -2 * output[j] * op[j], -2 * output[j] * z[j]
                derivatives_adj = [-4 * output[j] * v for v in dz[j]]
            elif kind == 'flux':
                z_adj, op_adj = 2 * output[j] * dz[j][0], 0
                derivatives_adj = [2 * output[j] * z[j]]
            else:
                z_adj, op_adj = 2 * output[j] * z[j], 0
                derivatives_adj = []
            for k, v in enumerate(feature):
                gradient = z_adj * v
                if operator:
                    gradient += op_adj * operator[k]
                gradient += mp.fdot(derivatives_adj, [g[k] for g in space])
                hidden_jac.append(gradient)
                if k < width:
                    sin_grad[k] += z_adj * row[k]
                    operator_grad[k] += op_adj * row[k]
                    for target, adjoint in zip(derivative_grad, derivatives_adj, strict=True):
                        target[k] += adjoint * row[k]

        spatial_jac = []
        for i, row in enumerate(spatial):
            for k, v in enumerate(inputs):
                ds = cosine[i] * v
                gradient = sin_grad[i] * ds
                if kind == 'pde':
                    dl = -norm[i] * ds - (8 * row[k] * sine[i] if k < dimension else 0)
                    gradient += operator_grad[i] * dl
                for axis_index, target in zip(active_axes, derivative_grad, strict=True):
                    dg = -2 * row[axis_index] * sine[i] * v
                    if axis_index == k:
                        dg += 2 * cosine[i]
                    gradient += target[i] * dg
                spatial_jac.append(gradient)
        return result, spatial_jac + hidden_jac + activations + [bias_multiplier]


def points_as_mp(tensor):
    return [tuple(map(mp.mpf, row)) for row in tensor.tolist()]


def refine(model, point_fn, seed, steps=20):
    field = CompactField(model)
    sites = sample_sites(model.dimension, 40, seed + 1)
    x_initial = points_as_mp(sites.initial)
    labels = [point_fn(x) for x in x_initial]
    x_pde, x_faces = points_as_mp(sites.interior), points_as_mp(sites.faces)
    pde_times = list(map(mp.mpf, sites.times.tolist()))
    face_times = list(map(mp.mpf, sites.face_times.tolist()))

    def system(differentiate):
        rows, jacobian = [], []
        for points, times, axes, kind, targets, scale in (
            (x_initial, [mp.mpf(0)] * len(labels), [0] * len(labels), 'value', labels, 1),
            (x_pde, pde_times, [0] * len(labels), 'pde', [0] * len(labels), 1 / mp.pi**2),
            (x_faces, face_times, sites.axes.tolist(), 'flux', [0] * len(labels), 1 / mp.pi),
        ):
            for x, t, axis, target in zip(points, times, axes, targets, strict=True):
                value = field.evaluate(x, t, kind, axis, differentiate)
                if differentiate:
                    value, derivative = value
                    jacobian.append([v * scale for v in derivative])
                rows.append((value - target) * scale)
        return mp.matrix(rows), mp.matrix(jacobian) if differentiate else None

    history = []

    def residual_log(vector):
        count = len(labels)
        return dict(initial_mse=mp.fdot(vector[:count], vector[:count]) / count,
                    pde_mse=mp.fdot(vector[count:2 * count], vector[count:2 * count]) * mp.pi**4 / count,
                    neumann_mse=mp.fdot(vector[2 * count:], vector[2 * count:]) * mp.pi**2 / count)

    for _ in range(steps):
        residual, jac = system(True)
        history.append(residual_log(residual))
        norm = mp.norm(residual)
        if norm < mp.mpf('1e-38'):
            break
        damping = max(mp.mpf('1e-48'), norm**2 * mp.mpf('.01'))
        normal = jac.T * jac
        rhs = -jac.T * residual
        update = mp.cholesky_solve(normal + damping * mp.eye(len(field.weights)), rhs)
        original = field.weights
        for shrink in (1, .5, .25, .125):
            field.weights = original + shrink * update
            if mp.norm(system(False)[0]) < norm:
                break
        else:
            field.weights = original
            break
    history.append(residual_log(system(False)[0]))
    return field, history, len(labels)


def mp_readout(field):
    anchor = (mp.mpf(1) / 3,) * field.dimension
    return field.evaluate(anchor, mp.mpf(HORIZON))


def precision_diagnostics(field, seed):
    sites = sample_sites(field.dimension, 512, seed + 10_000, balanced=True)
    anchor = (mp.mpf(1) / 3,) * field.dimension
    points = points_as_mp(sites.interior) + list(product((mp.mpf(0), mp.mpf(1)),
                                                       repeat=field.dimension)) + [anchor]
    times = (1., 2., 4., 5., HORIZON)
    at_terminal = [field.evaluate(point, mp.mpf(HORIZON)) for point in points]
    final_drift = max(abs(field.evaluate(point, mp.mpf(5)) - value)
                      for point, value in zip(points, at_terminal, strict=True))
    drift = max(abs(field.evaluate(point, mp.mpf(t)) - value)
                for point, value in zip(points, at_terminal, strict=True) for t in (2., 4.))
    pde = [field.evaluate(x, mp.mpf(t), 'pde')
           for x, t in zip(points_as_mp(sites.interior), sites.times.tolist(), strict=True)]
    flux = [field.evaluate(x, mp.mpf(t), 'flux', axis)
            for x, t, axis in zip(points_as_mp(sites.faces), sites.face_times.tolist(),
                                  sites.axes.tolist(), strict=True)]
    generator = torch.Generator().manual_seed(seed + 20_000)
    output_noise = torch.randn(field.width, generator=generator, dtype=torch.float64).tolist()
    backup = field.weights.copy()
    for i, noise in enumerate(output_noise):
        field.weights[len(field.weights) - field.width - 1 + i] += mp.mpf('.01') * noise
    perturbed = mp_readout(field)
    field.weights = backup
    return dict(plateau_spread=max(at_terminal) - min(at_terminal),
                plateau_time_drift=max(drift, final_drift), final_time_drift=final_drift,
                heldout_pde_rms=mp.sqrt(mp.fdot(pde, pde) / len(pde)),
                heldout_neumann_rms=mp.sqrt(mp.fdot(flux, flux) / len(flux)),
                heldout_neumann_max=max(map(abs, flux)),
                weight_perturbation_delta=abs(perturbed - at_terminal[-1]), perturbed_plateau=perturbed,
                anchor_time_values={str(t): field.evaluate(anchor, mp.mpf(t)) for t in times})
