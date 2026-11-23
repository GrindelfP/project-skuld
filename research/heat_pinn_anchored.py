"""Initial-value-anchored PINN with a time-dependent reflecting coordinate map."""
from dataclasses import dataclass
import math
import time

import torch
from torch import nn

from heat_pinn import Config, DTYPE, GAP, HORIZON, HeatField, plateau, residuals, sample_sites


@dataclass(frozen=True)
class AnchoredConfig(Config):
    initial_width: int = 96
    initial_adam: int = 1800
    initial_lbfgs: int = 250
    reflected: bool = True
    switch_rate: float = 40.
    initial_feature: bool = False
    scaled_pde: bool = False


def coordinates(x, t, reflected):
    """Return Phi, Phi_x, Phi_xx, Phi_t independently for each spatial axis."""
    if not reflected:
        return x, torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x)
    # The reflection layer shrinks as sqrt(t) at the initial/face corner.
    time = t[:, None]
    tau = .03
    positive_time = time.clamp_min(1e-24)
    length = .18 * (positive_time / (positive_time + tau)).sqrt()
    lower = torch.exp(-x / length)
    upper = torch.exp(-(1 - x) / length)
    endpoint = torch.exp(-1 / length)
    denom = 1 + endpoint
    difference = (lower - upper) / denom
    warped = x + length * difference
    dx = 1 - (lower + upper) / denom
    dxx = difference / length
    dl = difference + (lower * x - upper * (1 - x) - difference * endpoint) / (length * denom)
    dt = dl * length * tau / (2 * positive_time * (positive_time + tau))
    zero = torch.zeros_like(x)
    return (torch.where(time > 0, warped, x), torch.where(time > 0, dx, torch.ones_like(x)),
            torch.where(time > 0, dxx, zero), torch.where(time > 0, dt, zero))


class TanhField(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(inputs, width, dtype=DTYPE),
                                     nn.Linear(width, width, dtype=DTYPE)])
        self.output = nn.Linear(width, 1, dtype=DTYPE)

    def forward(self, features, jet=None):
        if jet is None:
            for layer in self.layers:
                features = layer(features).tanh()
            return self.output(features)[:, 0]
        for layer in self.layers:
            jet = HeatField.activate(HeatField.affine(layer, jet), 'tanh')
        value, gradient, lap = HeatField.affine(self.output, jet)
        return value[:, 0], gradient[:, :, 0], lap[:, 0]


class AnchoredField(nn.Module):
    def __init__(self, dimension, config):
        super().__init__()
        self.dimension = dimension
        self.reflected = config.reflected
        self.switch_rate = config.switch_rate
        self.initial_feature = config.initial_feature
        self.register_buffer('rates', GAP * torch.tensor(config.rates, dtype=DTYPE))
        self.initial = TanhField(dimension, config.initial_width)
        self.dynamic = TanhField(dimension + len(self.rates) + int(self.initial_feature), config.width)

    @property
    def output(self):
        return self.dynamic.output

    def inputs(self, x, t):
        position, dx, dxx, dt = coordinates(x, t, self.reflected)
        space = 2 * position - 1
        fast_time = t[:, None] * self.rates
        # Rational time inputs remain sensitive at late times.
        temporal = 1 / (1 + fast_time)
        features = torch.cat((space, temporal), dim=1)
        decay = torch.exp(-self.switch_rate * t)
        switch = -torch.expm1(-self.switch_rate * t)
        space_jac = torch.cat((torch.diag_embed(2 * dx), (2 * dt)[:, None]), dim=1)
        time_jac = torch.cat((x.new_zeros(len(x), self.dimension, len(self.rates)),
                              (-self.rates * temporal.square())[:, None]), dim=1)
        spatial_jet = (space, space_jac, 2 * dxx)
        combined = (features, torch.cat((space_jac, time_jac), dim=2),
                    torch.cat((2 * dxx, torch.zeros_like(temporal)), dim=1))
        initial_value, initial_grad, initial_lap = self.initial(space, spatial_jet)
        if self.initial_feature:
            combined = tuple(torch.cat((part, initial[..., None]), dim=-1) for part, initial in
                             zip(combined, (initial_value, initial_grad, initial_lap), strict=True))
        return combined + (decay, switch, initial_value, initial_grad, initial_lap)

    def propagate(self, inputs):
        features, gradient, lap, decay, switch, initial_value, initial_grad, initial_lap = inputs
        combined = features, gradient, lap
        later_value, later_grad, later_lap = self.dynamic(features, combined)
        dt_correction = self.switch_rate * decay * (later_value - initial_value)
        gradient = initial_grad * decay[:, None] + later_grad * switch[:, None]
        gradient = torch.cat((gradient[:, :-1], gradient[:, -1:] + dt_correction[:, None]), dim=1)
        return (initial_value * decay + later_value * switch, gradient,
                initial_lap * decay + later_lap * switch)

    def forward(self, x, t, derivatives=False):
        if derivatives:
            return self.propagate(self.inputs(x, t))
        position = coordinates(x, t, self.reflected)[0]
        space = 2 * position - 1
        temporal = 1 / (1 + t[:, None] * self.rates)
        features = torch.cat((space, temporal), dim=1)
        decay = torch.exp(-self.switch_rate * t)
        switch = -torch.expm1(-self.switch_rate * t)
        initial = self.initial(space)
        if self.initial_feature:
            features = torch.cat((features, initial[:, None]), dim=1)
        return initial * decay + self.dynamic(features) * switch


def fit_anchored(point_fn, dimension, config, seed):
    torch.manual_seed(seed)
    model = AnchoredField(dimension, config)
    initial_plateau = plateau(model)
    sites = sample_sites(dimension, config.sites, seed + 1, config.balanced_time)
    x = sites.initial
    with torch.no_grad():
        scale = point_fn(torch.full((1, dimension), .5, dtype=DTYPE)).item() if config.normalize else 1.
        if not math.isfinite(scale) or abs(scale) < 1e-12:
            scale = 1.
        labels = point_fn(x) / scale
        if labels.shape != (config.sites,) or not labels.isfinite().all():
            raise ValueError('Expected a finite scalar initial value at each site')
    start = time.perf_counter()
    history = []
    batch_size = min(config.batch, config.sites)

    @torch.no_grad()
    def record(stage, evaluation):
        parts = residuals(model, sites, labels, config.late_weight)
        initial, pde, flux = parts[:3]
        history.append(dict(stage=stage, evaluation=evaluation,
                            initial_mse=initial.square().mean().item(),
                            pde_mse=pde.square().mean().item(),
                            neumann_mse=flux.square().mean().item(),
                            late_gradient_mse=(parts[3].square().mean().item() if config.late_weight else 0.)))

    def train(parameters, adam_steps, lbfgs_steps, stage, loss):
        optimizer = torch.optim.Adam(parameters, lr=.002)
        for epoch in range(1, adam_steps + 1):
            indices = torch.randperm(config.sites)[:batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss(indices).backward()
            optimizer.step()
            if epoch % config.log_every == 0 or epoch == adam_steps:
                record(stage, epoch)

        optimizer = torch.optim.LBFGS(parameters, max_iter=lbfgs_steps,
                                      history_size=25, line_search_fn='strong_wolfe',
                                      tolerance_grad=1e-11, tolerance_change=1e-14)
        evaluations = 0

        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            objective = loss(slice(None))
            objective.backward()
            evaluations += 1
            if evaluations % config.log_every == 0:
                record(stage, adam_steps + evaluations)
            return objective

        if lbfgs_steps:
            optimizer.step(closure)
        record(stage, adam_steps + evaluations + 1)

    def initial_loss(indices):
        return (model.initial(2 * x[indices] - 1) - labels[indices]).square().mean()

    record('initialization', 0)
    train(list(model.initial.parameters()), config.initial_adam, config.initial_lbfgs,
          'initial_fit', initial_loss)
    model.initial.requires_grad_(False)
    with torch.no_grad():
        interior = model.inputs(sites.interior, sites.times)
        boundary = model.inputs(sites.faces, sites.face_times)
        late = (model.inputs(sites.interior, torch.full_like(sites.times, HORIZON))
                if config.late_weight else None)
        pde_scale = .15 + .85 * sites.times / (sites.times + .03) if config.scaled_pde else None

    def pde_loss(indices):
        _, gradient, lap = model.propagate(tuple(value[indices] for value in interior))
        pde = gradient[:, -1] - lap
        if pde_scale is not None:
            pde = pde * pde_scale[indices]
        _, gradient, _ = model.propagate(tuple(value[indices] for value in boundary))
        flux = gradient.gather(1, sites.axes[indices, None])
        loss = (config.pde_weight * pde.square().mean() / GAP**2
                + config.neumann_weight * flux.square().mean() / math.pi**2)
        if late is not None:
            _, gradient, _ = model.propagate(tuple(value[indices] for value in late))
            loss = loss + config.late_weight * gradient[:, :-1].square().mean()
        return loss

    train(list(model.dynamic.parameters()), config.adam, config.lbfgs, 'dynamics', pde_loss)
    if not all(math.isfinite(row[key]) for row in history for key in
               ('initial_mse', 'pde_mse', 'neumann_mse')):
        raise RuntimeError('Nonfinite training loss')
    return model, scale, initial_plateau, history, time.perf_counter() - start
