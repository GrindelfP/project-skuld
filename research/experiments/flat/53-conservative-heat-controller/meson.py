"""Fit and audit an insulated heat PINN for the finite-cutoff meson integrals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

import torch

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH), str(RESEARCH.parent / 'skuld-lib')]

from physics import PARAM_SETS, T_MAX, integrand_transformed
from skuld._conservative_heat import HeatController, HeatNetwork, InitialNetwork, gradient
from utils.paths import get_mirror_path


def references():
    path, = (RESEARCH / 'results' / 'flat').glob('*-precision-water-volume/integrals.csv')
    with path.open(newline='') as source:
        rows = list(csv.DictReader(source))
    expected = [(f'I_{i + 1}', *params, T_MAX) for i, params in enumerate(PARAM_SETS)]
    observed = [(row['integral'], *(int(row[k]) for k in 'abmn'), float(row['t_max'])) for row in rows]
    if observed != expected:
        raise ValueError('Saved reference does not match the finite-cutoff benchmark')
    return [float(row['reference']) for row in rows]


HORIZON = 6.0
N_DEFECTS = 6


def spacetime(count, generator):
    points = torch.rand(count, 4, generator=generator)
    # Resolve the initial transient while covering the whole horizon.
    time = points[:, 3]
    points[:, 3] = HORIZON * torch.expm1(5 * time) / math.expm1(5)
    return points


def with_time(space, time):
    return torch.cat((space, torch.full_like(space[:, :1], time)), dim=1)


def train(model, loss, steps, polish, learning_rate):
    before = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for step in range(steps):
        optimizer.zero_grad()
        value = loss(step)
        value.backward()
        optimizer.step()
    if polish:
        optimizer = torch.optim.LBFGS(
            model.parameters(), max_iter=polish, line_search_fn='strong_wolfe',
            tolerance_grad=1e-12, tolerance_change=1e-15,
        )

        def closure():
            optimizer.zero_grad()
            value = loss(None)
            value.backward()
            return value

        optimizer.step(closure)
    after = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    return torch.linalg.vector_norm(after - before).item()


def readout(model, time):
    point = with_time(torch.full((1, 3), 1 / 3), time).requires_grad_()
    return model(point).item()


def defects(model, points, estimate, *, source=0.0, plateau_bias=0.0):
    space = points[:, :3]
    points = points.detach().requires_grad_()
    field, derivatives = model.field(points)
    flux_time = torch.stack([derivative[:, 3] for derivative in derivatives], dim=-1)
    field_gradient = gradient(field, points)
    pde = (flux_time - field_gradient[:, :3]).abs().amax(dim=1)
    divergence = sum(gradient(flux_time[:, i], points)[:, i] for i in range(3))
    laplacian = sum(gradient(field_gradient[:, i], points)[:, i] for i in range(3))
    heat_residual = (field_gradient[:, 3] - laplacian).abs()
    # An injected uniform source changes u_s without changing the flux.
    balance = (field_gradient[:, 3] + source - divergence).abs()
    face_defects = []
    for axis in range(3):
        for side in (0., 1.):
            face = points.detach().clone()
            face[:, axis] = side
            face.requires_grad_()
            normal_gradient = gradient(model(face), face)[:, axis]
            normal_flux = model.flux(face)[:, axis]
            face_defects.extend((normal_gradient.abs(), normal_flux.abs()))
    face = torch.stack(face_defects).amax(dim=0)
    final = model(with_time(space, HORIZON).requires_grad_())
    previous = model(with_time(space, HORIZON - 1).requires_grad_())
    final = final + plateau_bias * (space[:, 0] - 1 / 3)
    amplitude = max(abs(estimate), torch.finfo(points.dtype).tiny)
    return torch.stack((pde, balance, face, (final - estimate).abs() / amplitude,
                        (final - previous).abs() / amplitude, heat_residual), dim=1).detach()


def audit(model, estimate, options, generator):
    controller = HeatController(32, N_DEFECTS)
    frozen_controller = deepcopy(controller)
    tolerances = torch.tensor([options.pde_tolerance, options.conservation_tolerance,
                               options.conservation_tolerance, options.plateau_tolerance,
                               options.plateau_tolerance, options.pde_tolerance])
    points = [spacetime(count, generator) for count in (192, 1024, 256)]
    training, pool, holdout = points
    targets = torch.log1p(defects(model, training, estimate) / tolerances)

    def loss(_):
        return (controller(training) - targets).square().mean()

    update = train(controller, loss, 300, 30, 0.005)
    with torch.no_grad():
        proposed = pool[controller(pool).topk(8, dim=0).indices.unique()]
        frozen = pool[frozen_controller(pool).topk(8, dim=0).indices.unique()]
    measured = defects(model, holdout, estimate)
    proposed_defects = defects(model, proposed, estimate)
    frozen_defects = defects(model, frozen, estimate)
    maxima = torch.cat((torch.expm1(targets) * tolerances, measured, proposed_defects)).amax(dim=0)
    injected = defects(model, holdout, estimate, source=1e-3, plateau_bias=1e-2)
    names = ('flux_pde_residual', 'local_heat_balance', 'normal_face_defect',
             'plateau_spread', 'plateau_drift', 'heat_pde_residual')
    return dict(
        controller_update=update,
        controller_proposed_sites=len(proposed),
        controller_accepted=bool((maxima <= tolerances).all()),
        controller_injected_balance_detected=bool(injected[:, 1].max() > tolerances[1]),
        injected_plateau_drift=injected[:, 4].max().item(),
        **{name: value.item() for name, value in zip(names, maxima, strict=True)},
        **{f'proposed_{name}': value.item()
           for name, value in zip(names, proposed_defects.amax(dim=0), strict=True)},
        **{f'frozen_controller_{name}': value.item()
           for name, value in zip(names, frozen_defects.amax(dim=0), strict=True)},
    )


def fit_case(index, options):
    begin = perf_counter()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    generator = torch.Generator().manual_seed(options.seed)
    params = PARAM_SETS[index]
    sites = torch.rand(options.initial_sites, 3, generator=generator)
    values = integrand_transformed(sites, *params)
    scale = values.abs().max().item()
    values = values / scale
    initial = InitialNetwork(options.initial_width)

    def initial_loss(step):
        batch = slice(None) if step is None else torch.randint(len(sites), (options.batch,), generator=generator)
        return (initial(sites[batch]) - values[batch]).square().mean()

    initial_update = train(initial, initial_loss, options.initial_steps, options.initial_polish, 0.003)
    holdout = torch.rand(4096, 3, generator=generator)
    with torch.no_grad():
        holdout_error = initial(holdout) - integrand_transformed(holdout, *params) / scale
        disabled = scale * initial(torch.full((1, 3), 1 / 3)).item()
    model = HeatNetwork(initial, options.width)
    fixed = spacetime(options.pde_sites, generator)

    def heat_loss(step):
        batch = fixed if step is None else spacetime(options.batch, generator)
        residual = model.residual(batch.detach().requires_grad_())
        late = with_time(batch[:, :3], HORIZON).requires_grad_()
        late_gradient = gradient(model(late), late)[:, :3]
        return residual.square().mean() + options.late_weight * late_gradient.square().mean()

    heat_update = train(model, heat_loss, options.heat_steps, options.heat_polish, 0.002)
    estimate = readout(model, HORIZON)
    check_points = torch.rand(256, 3, generator=generator)
    anchored = model(with_time(check_points, 0).requires_grad_()).detach()
    audit_report = audit(model, estimate, options, generator)
    return dict(
        integral=f'I_{index + 1}', **dict(zip('abmn', params, strict=True)), t_max=T_MAX,
        estimate=scale * estimate, disabled_heat_estimate=disabled,
        initial_update=initial_update, heat_update=heat_update,
        initial_weights=sum(p.numel() for p in initial.parameters()),
        heat_weights=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_train_sites=len(sites), initial_holdout_sites=len(holdout),
        initial_holdout_max_over_peak=holdout_error.abs().max().item(),
        initial_holdout_rmse_over_peak=holdout_error.square().mean().sqrt().item(),
        initial_anchor_error=(anchored - initial(check_points)).abs().max().item(),
        **audit_report,
        wall_seconds=perf_counter() - begin,
    )


def self_check(options):
    """Check the derivative and controller paths on an exact transient."""
    class InitialMode(torch.nn.Module):
        def forward(self, xyz):
            return 2 + torch.cos(math.pi * xyz[:, 0]) * torch.cos(2 * math.pi * xyz[:, 1])

    class ExactHeat(HeatNetwork):
        def flux(self, points):
            x, y, z, time = points.unbind(dim=1)
            angle_x, angle_y = math.pi * x, 2 * math.pi * y
            time_factor = torch.expm1(-5 * math.pi**2 * time) / (5 * math.pi)
            return time_factor[:, None] * torch.stack((
                torch.sin(angle_x) * torch.cos(angle_y), 2 * torch.cos(angle_x) * torch.sin(angle_y),
                0 * z,
            ), dim=1)

    generator = torch.Generator().manual_seed(options.seed)
    torch.manual_seed(options.seed)
    model = ExactHeat(InitialMode(), 1)
    points = spacetime(256, generator)
    field = model(points.clone().requires_grad_())
    initial = model.initial(points[:, :3])
    expected = 2 + (initial - 2) * torch.exp(-5 * math.pi**2 * points[:, 3])
    discrepancy = (field - expected).detach().abs().max().item()
    report = audit(model, readout(model, HORIZON), options, generator)
    if discrepancy > 1e-13 or not report['controller_accepted']:
        raise RuntimeError(f'Exact transient rejected: {discrepancy=}, {report=}')
    if not report['controller_injected_balance_detected'] or report['injected_plateau_drift'] < 1e-3:
        raise RuntimeError(f'Injected defect was not detected: {report=}')
    print(json.dumps(dict(exact_transient_max_error=discrepancy, **report), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial-width', type=int, default=48)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--initial-sites', type=int, default=4096)
    parser.add_argument('--pde-sites', type=int, default=768)
    parser.add_argument('--batch', type=int, default=192)
    parser.add_argument('--initial-steps', type=int, default=1200)
    parser.add_argument('--initial-polish', type=int, default=180)
    parser.add_argument('--heat-steps', type=int, default=1000)
    parser.add_argument('--heat-polish', type=int, default=250)
    parser.add_argument('--late-weight', type=float, default=10.)
    parser.add_argument('--plateau-tolerance', type=float, default=1e-7)
    parser.add_argument('--conservation-tolerance', type=float, default=1e-10)
    parser.add_argument('--pde-tolerance', type=float, default=0.02)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    if min(options.initial_width, options.width, options.initial_sites, options.pde_sites,
           options.batch, options.workers) < 1:
        parser.error('widths, sample counts, and worker count must be positive')
    if min(options.initial_steps, options.heat_steps, options.initial_polish, options.heat_polish) < 0:
        parser.error('optimizer step counts must be nonnegative')
    if min(options.pde_tolerance, options.conservation_tolerance, options.plateau_tolerance) <= 0:
        parser.error('controller tolerances must be positive')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if options.self_check:
        self_check(options)
        return

    order = (0, 5) if options.smoke else tuple(range(len(PARAM_SETS)))
    rows = []

    def record(row):
        index = int(row['integral'].split('_')[1])
        print(f'I_{index} estimate={row["estimate"]:.12g}, plateau={row["plateau_spread"]:.3g}, '
              f'initial max={row["initial_holdout_max_over_peak"]:.3g}, time={row["wall_seconds"]:.1f}s',
              flush=True)
        rows.append(row)

    if options.workers == 1:
        for index in order:
            record(fit_case(index, options))
    else:
        with ProcessPoolExecutor(max_workers=options.workers, mp_context=get_context('spawn')) as executor:
            for row in executor.map(fit_case, order, [options] * len(order)):
                record(row)

    # References are opened after every estimate and audit.
    reference_values = references()
    for index, row in zip(order, rows, strict=True):
        reference = reference_values[index]
        error = abs(row['estimate'] - reference)
        relative = error / abs(reference)
        row.update(reference=reference, absolute_error=error, relative_error=relative,
                   correct_digits=-math.log10(relative) if relative else float('inf'),
                   disabled_heat_relative_error=abs(row['disabled_heat_estimate'] - reference) / abs(reference))
    digits = [row['correct_digits'] for row in rows]
    summary = dict(min_digits=min(digits), mean_digits=sum(digits) / len(digits), max_digits=max(digits),
                   scores_at_least_7_digits=sum(digit >= 7 for digit in digits),
                   accepted_at_least_7_digits=sum(row['controller_accepted'] and row['correct_digits'] >= 7 for row in rows),
                   controller_accepted=sum(row['controller_accepted'] for row in rows), cases=len(rows))
    directory = get_mirror_path(__file__, 'results')
    if options.smoke:
        directory = directory / 'smoke'
        directory.mkdir(exist_ok=True)
    with (directory / 'integrals.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with (directory / 'run.json').open('w') as output:
        json.dump(dict(config=vars(options), diffusion_horizon=HORIZON, **summary), output, indent=2)
        output.write('\n')
    print(summary)


if __name__ == '__main__':
    main()
