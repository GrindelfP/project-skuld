"""Fit and audit an insulated heat PINN for the finite-cutoff meson integrals."""
import argparse
import csv
import json
import math
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
N_DEFECTS = 5


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


def defects(model, points, estimate, inject=0.0):
    space = points[:, :3]
    points = points.detach().requires_grad_()
    field, derivatives = model.field(points)
    flux_time = torch.stack([derivative[:, 3] for derivative in derivatives], dim=-1)
    field_gradient = gradient(field, points)
    pde = (flux_time - field_gradient[:, :3]).abs().amax(dim=1)
    divergence = sum(gradient(flux_time[:, i], points)[:, i] for i in range(3))
    # An injected uniform source changes u_s without changing the flux.
    balance = (field_gradient[:, 3] + inject - divergence).abs()
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
    amplitude = max(abs(estimate), torch.finfo(points.dtype).tiny)
    return torch.stack((pde, balance, face, (final - estimate).abs() / amplitude,
                        (final - previous).abs() / amplitude), dim=1).detach()


def audit(model, estimate, options, generator):
    controller = HeatController(32, N_DEFECTS)
    tolerances = torch.tensor([options.pde_tolerance, options.conservation_tolerance,
                               options.conservation_tolerance, options.plateau_tolerance,
                               options.plateau_tolerance])
    points = [spacetime(count, generator) for count in (192, 1024, 256)]
    training, pool, holdout = points
    targets = torch.log1p(defects(model, training, estimate) / tolerances)

    def loss(_):
        return (controller(training) - targets).square().mean()

    update = train(controller, loss, 300, 30, 0.005)
    with torch.no_grad():
        proposed = pool[controller(pool).topk(8, dim=0).indices.unique()]
        frozen = pool[HeatController(32, N_DEFECTS)(pool).topk(8, dim=0).indices.unique()]
    measured = defects(model, holdout, estimate)
    proposed_defects = defects(model, proposed, estimate)
    frozen_defects = defects(model, frozen, estimate)
    maxima = torch.cat((torch.expm1(targets) * tolerances, measured, proposed_defects)).amax(dim=0)
    injected = defects(model, proposed, estimate, inject=1e-3)
    names = ('flux_pde_residual', 'local_heat_balance', 'normal_face_defect',
             'plateau_spread', 'plateau_drift')
    return dict(
        controller_update=update,
        controller_proposed_sites=len(proposed),
        controller_accepted=bool((maxima <= tolerances).all()),
        controller_injected_balance_detected=bool(injected[:, 1].max() > tolerances[1]),
        **{name: value.item() for name, value in zip(names, maxima, strict=True)},
        **{f'proposed_{name}': value.item()
           for name, value in zip(names, proposed_defects.amax(dim=0), strict=True)},
        **{f'frozen_controller_{name}': value.item()
           for name, value in zip(names, frozen_defects.amax(dim=0), strict=True)},
    )


def fit_case(index, options):
    begin = perf_counter()
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
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    order = (0, 5) if options.smoke else tuple(range(len(PARAM_SETS)))
    rows = []
    for index in order:
        row = fit_case(index, options)
        print(f'I_{index + 1} estimate={row["estimate"]:.12g}, plateau={row["plateau_spread"]:.3g}, '
              f'initial max={row["initial_holdout_max_over_peak"]:.3g}, time={row["wall_seconds"]:.1f}s',
              flush=True)
        rows.append(row)

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
