"""Train an insulated heat field from pointwise initial conditions."""
from dataclasses import dataclass
import math
import time

from scipy.optimize import least_squares
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, jacfwd

HORIZON = 6.
ANCHOR = 1. / 3.
DTYPE = torch.float64
GAP = math.pi**2


@dataclass(frozen=True)
class Config:
    width: int = 48
    sites: int = 1024
    batch: int = 128
    adam: int = 600
    lbfgs: int = 180
    trust: int = 0
    compact: bool = False
    rates: tuple = (1., 8., 64.)
    pde_weight: float = .2
    neumann_weight: float = 1.
    late_weight: float = 0.
    balanced_time: bool = False
    normalize: bool = True
    log_every: int = 25


class HeatField(nn.Module):
    """Trainable space-time field with an ordinary scalar readout."""

    def __init__(self, dimension, config):
        super().__init__()
        self.dimension = dimension
        self.compact = config.compact
        self.register_buffer('rates', GAP * torch.tensor(config.rates, dtype=DTYPE))
        width = config.width
        if self.compact:
            self.spatial = nn.Linear(dimension, width, dtype=DTYPE)
            self.layers = nn.ModuleList([nn.Linear(width + len(self.rates), width, dtype=DTYPE)])
            with torch.no_grad():
                self.spatial.weight.mul_(4)
        else:
            self.layers = nn.ModuleList([
                nn.Linear(dimension + len(self.rates), width, dtype=DTYPE),
                nn.Linear(width, width, dtype=DTYPE),
            ])
        self.output = nn.Linear(width, 1, dtype=DTYPE)

    @staticmethod
    def affine(layer, jet):
        value, gradient, laplacian = jet
        return layer(value), F.linear(gradient, layer.weight), F.linear(laplacian, layer.weight)

    @staticmethod
    def activate(jet, activation):
        value, gradient, laplacian = jet
        if activation == 'sine':
            result = value.sin()
            first, second = value.cos(), -result
        elif activation == 'square':
            result = value.square()
            first, second = 2 * value, 2.
        else:
            result = value.tanh()
            first = 1 - result.square()
            second = -2 * result * first
        spatial_squared = gradient[:, :-1].square().sum(dim=1)
        return result, gradient * first[:, None], laplacian * first + second * spatial_squared

    def forward(self, x, t, derivatives=False):
        spatial = 2 * x - 1
        temporal = torch.exp(-t[:, None] * self.rates)
        if not derivatives:
            if self.compact:
                spatial = self.spatial(spatial).sin()
            value = torch.cat((spatial, temporal), dim=1)
            for layer in self.layers:
                value = layer(value)
                value = value.square() if self.compact else value.tanh()
            return self.output(value)[:, 0]

        count, dimension = x.shape
        space_jac = 2 * torch.eye(dimension + 1, dtype=x.dtype, device=x.device)[:, :dimension]
        spatial_jet = (spatial, space_jac.expand(count, -1, -1), torch.zeros_like(spatial))
        if self.compact:
            spatial_jet = self.activate(self.affine(self.spatial, spatial_jet), 'sine')
        time_jac = torch.cat((x.new_zeros(count, dimension, len(self.rates)),
                              (-temporal * self.rates)[:, None]), dim=1)
        jet = (torch.cat((spatial_jet[0], temporal), dim=1),
               torch.cat((spatial_jet[1], time_jac), dim=2),
               torch.cat((spatial_jet[2], torch.zeros_like(temporal)), dim=1))
        for layer in self.layers:
            jet = self.activate(self.affine(layer, jet), 'square' if self.compact else 'tanh')
        value, gradient, laplacian = self.affine(self.output, jet)
        return value[:, 0], gradient[:, :, 0], laplacian[:, 0]


@dataclass(frozen=True)
class Sites:
    initial: torch.Tensor
    interior: torch.Tensor
    times: torch.Tensor
    faces: torch.Tensor
    face_times: torch.Tensor
    axes: torch.Tensor


def sample_time(q, balanced=False):
    if balanced:
        early = 1e-5 * torch.expm1(3 * q * math.log1p(.04 / 1e-5))
        middle = .04 + .46 * (3 * q - 1)
        late = .5 + (HORIZON - .5) * (3 * q - 2)
        return torch.where(q < 1 / 3, early, torch.where(q < 2 / 3, middle, late))
    # Log-uniform through the initial layer, plus a uniform late-time tail.
    return torch.where(q < .75, 1e-5 * torch.expm1(q / .75 * math.log1p(.5 / 1e-5)),
                       .5 + (HORIZON - .5) * (q - .75) / .25)


def sample_sites(dimension, count, seed, balanced=False):
    sampler = torch.quasirandom.SobolEngine(dimension + 1, scramble=True, seed=seed)
    sample = sampler.draw(3 * count, dtype=DTYPE)
    interior = sample[count:2 * count]
    boundary = sample[2 * count:]
    axes = torch.arange(count) % dimension
    sides = (torch.arange(count) // dimension) % 2
    faces = boundary[:, :dimension].clone()
    faces[torch.arange(count), axes] = sides.to(DTYPE)
    # Permute time independently of face selection.
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    if balanced:
        faces = boundary[permutation, :dimension].clone()
        faces[torch.arange(count), axes] = sides.to(DTYPE)
    return Sites(sample[:count, :dimension], interior[:, :dimension],
                 sample_time(interior[:, -1], balanced), faces,
                 sample_time(boundary[permutation, -1], balanced), axes)


def residuals(field, sites, labels, late_weight=0.):
    dx = sites.interior.shape[1]
    initial = field(sites.initial, labels.new_zeros(len(labels))) - labels
    _, grad, lap = field(torch.cat((sites.interior, sites.faces)),
                         torch.cat((sites.times, sites.face_times)), True)
    interior_grad, face_grad = grad.split((len(sites.times), len(sites.face_times)))
    pde = interior_grad[:, dx] - lap[:len(sites.times)]
    boundary = face_grad.gather(1, sites.axes[:, None]).squeeze(1)
    if late_weight:
        _, late_grad, _ = field(sites.interior, torch.full_like(sites.times, HORIZON), True)
        return initial, pde, boundary, late_grad[:, :dx].flatten()
    return initial, pde, boundary


def weighted_residual(parts, config):
    weights = (1., config.pde_weight / GAP**2, config.neumann_weight / math.pi**2,
               config.late_weight)
    return torch.cat([part * math.sqrt(weight / part.numel())
                      for part, weight in zip(parts, weights)])


def plateau(model):
    with torch.no_grad():
        return model(torch.full((1, model.dimension), ANCHOR, dtype=DTYPE),
                     torch.tensor([HORIZON], dtype=DTYPE)).item()


def fit(point_fn, dimension, config, seed):
    torch.manual_seed(seed)
    model = HeatField(dimension, config)
    initial_plateau = plateau(model)
    sites = sample_sites(dimension, config.sites, seed + 1, config.balanced_time)
    with torch.no_grad():
        scale = (point_fn(torch.full((1, dimension), .5, dtype=DTYPE)).item()
                 if config.normalize else 1.)
        if not math.isfinite(scale) or abs(scale) < 1e-12:
            scale = 1.
        point_values = point_fn(sites.initial)
        if point_values.shape != (config.sites,) or not point_values.isfinite().all():
            raise ValueError('Initial condition must provide one finite label per site')
        labels = point_values / scale
    history = []
    started = time.perf_counter()

    def log(stage, evaluation, parts):
        with torch.no_grad():
            history.append(dict(stage=stage, evaluation=evaluation,
                                initial_mse=parts[0].square().mean().item(),
                                pde_mse=parts[1].square().mean().item(),
                                neumann_mse=parts[2].square().mean().item(),
                                late_gradient_mse=(parts[3].square().mean().item() if len(parts) == 4 else 0.)))

    with torch.no_grad():
        log('initialization', 0, residuals(model, sites, labels, config.late_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    batch_size = min(config.batch, config.sites)
    for epoch in range(1, config.adam + 1):
        index = torch.randperm(config.sites)[:batch_size]
        batch = Sites(*(getattr(sites, name)[index] for name in Sites.__dataclass_fields__))
        parts = residuals(model, batch, labels[index], config.late_weight)
        optimizer.zero_grad(set_to_none=True)
        weighted_residual(parts, config).square().sum().backward()
        optimizer.step()
        if epoch % config.log_every == 0 or epoch == config.adam:
            with torch.no_grad():
                log('adam', epoch, residuals(model, sites, labels, config.late_weight))

    if config.trust:
        named = dict(model.named_parameters())
        shapes = [parameter.shape for parameter in named.values()]
        sizes = [parameter.numel() for parameter in named.values()]

        def unpack(vector):
            return {name: part.reshape(shape) for name, part, shape in
                    zip(named, vector.split(sizes), shapes, strict=True)}

        def evaluate(vector):
            parameters = unpack(vector)
            return residuals(lambda *args: functional_call(model, parameters, args),
                             sites, labels, config.late_weight)

        def vector_residual(vector):
            return weighted_residual(evaluate(vector), config)

        jacobian = jacfwd(vector_residual)
        calls = 0

        def objective(vector):
            nonlocal calls
            with torch.no_grad():
                parts = evaluate(torch.from_numpy(vector))
                calls += 1
                if calls == 1 or calls % config.log_every == 0:
                    log('trust_region', calls, parts)
                return weighted_residual(parts, config).numpy()

        def jac(vector):
            with torch.no_grad():
                return jacobian(torch.from_numpy(vector)).numpy()

        start = torch.cat([parameter.detach().flatten() for parameter in named.values()]).numpy()
        solution = least_squares(objective, start, jac=jac, method='trf',
                                 max_nfev=config.trust, ftol=3e-15, xtol=3e-15, gtol=3e-15)
        with torch.no_grad():
            for name, value in unpack(torch.from_numpy(solution.x)).items():
                named[name].copy_(value)
            log('trust_region_final', calls, residuals(model, sites, labels, config.late_weight))

    if config.lbfgs:
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=config.lbfgs,
                                      history_size=25, line_search_fn='strong_wolfe',
                                      tolerance_grad=1e-12, tolerance_change=1e-14)
        calls = 0

        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            parts = residuals(model, sites, labels, config.late_weight)
            loss = weighted_residual(parts, config).square().sum()
            loss.backward()
            calls += 1
            if calls == 1 or calls % config.log_every == 0:
                log('lbfgs', calls, parts)
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            log('lbfgs_final', calls, residuals(model, sites, labels, config.late_weight))

    if not all(math.isfinite(row[key]) for row in history for key in
               ('initial_mse', 'pde_mse', 'neumann_mse')):
        raise RuntimeError('Nonfinite training loss')
    return model, scale, initial_plateau, history, time.perf_counter() - started


def diagnostics(model, seed):
    sites = sample_sites(model.dimension, 512, seed + 10_000, balanced=True)
    anchor = torch.full((1, model.dimension), ANCHOR, dtype=DTYPE)
    corners = torch.cartesian_prod(*[torch.tensor([0., 1.], dtype=DTYPE)] * model.dimension)
    x = torch.cat((sites.interior, corners.reshape(-1, model.dimension), anchor))
    with torch.no_grad():
        times = (1., 2., 4., 5., HORIZON)
        fields = {t: model(x, x.new_full((len(x),), t)) for t in times}
        _, gradient, lap = model(sites.interior, sites.times, True)
        _, flux, _ = model(sites.faces, sites.face_times, True)
        pde = gradient[:, -1] - lap
        normal = flux.gather(1, sites.axes[:, None])
        terminal = fields[HORIZON]
        before = plateau(model)
        generator = torch.Generator().manual_seed(seed + 20_000)
        weight = model.output.weight
        backup = weight.clone()
        weight.add_(.01 * torch.randn(weight.shape, generator=generator, dtype=DTYPE))
        perturbed = plateau(model)
        weight.copy_(backup)
    return dict(plateau_spread=(terminal.max() - terminal.min()).item(),
                plateau_time_drift=max((fields[t] - terminal).abs().max().item() for t in (2., 4., 5.)),
                final_time_drift=(fields[5.] - terminal).abs().max().item(),
                heldout_pde_rms=pde.square().mean().sqrt().item(),
                heldout_neumann_rms=normal.square().mean().sqrt().item(),
                heldout_neumann_max=normal.abs().max().item(),
                weight_perturbation_delta=abs(perturbed - before), perturbed_plateau=perturbed,
                anchor_time_values={str(t): field[-1].item() for t, field in fields.items()})
