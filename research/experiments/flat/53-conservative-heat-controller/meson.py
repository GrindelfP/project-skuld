"""Fit, diffuse, and audit the finite-cutoff meson integrands."""
import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.stats import qmc
import torch

RESEARCH = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RESEARCH), str(RESEARCH.parent / 'skuld-lib')]

from physics import PARAM_SETS, T_MAX, integrand_transformed
from skuld._conservative_heat import HeatController, HeatNetwork, InitialNetwork, evaluate, sites
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


def mesh(z):
    return np.stack(np.meshgrid(z, z, z, indexing='ij'), axis=-1).reshape(-1, 3)


def sample(params, xyz):
    points = torch.from_numpy((xyz + 1) / 2)
    return integrand_transformed(points, *params).numpy()


def sobol(seed, power):
    return 2 * qmc.Sobol(3, scramble=True, seed=seed).random_base2(power) - 1


def fit_case(params, nodes, layers, options):
    begin = perf_counter()
    n = len(nodes)
    initial = InitialNetwork.fit(sample(params, mesh(nodes)).reshape((n,) * 3), nodes, options.rank)
    holdout = sobol(639, 12)
    expected = sample(params, holdout)
    initial_error = np.max(abs(initial.forward(holdout) - expected))
    scale = np.max(abs(expected))
    field = initial.coefficients()
    fit_seconds = perf_counter() - begin
    controller = HeatController(seed=687)
    train, candidates, verification = sobol(734, 7), sobol(832, 10), sobol(934, 8)
    readout_point = np.full((1, 3), -1 / 3)
    disabled_heat_estimate = initial.forward(readout_point)[0]
    audits = []
    first_plateau_step = None
    for step in range(1, options.steps + 1):
        previous = field
        # Symmetric axis splitting. The continuous axis heat generators commute.
        for axis in (0, 1, 2, 1, 0):
            layer = layers[0 if axis == 2 else 1]
            before, field = field, layer.advance(field, axis)
            if step in (1, options.steps - 1, options.steps):
                def defects(points):
                    pde, flux = layer.defects(before, points, axis)
                    balance = layer.balance(before, field, points, axis)
                    return np.column_stack((pde, flux, balance)) / scale

                report = controller.audit(defects, train, candidates, verification)
                audits.append((step, axis, report))
        if step < min(7, options.steps):
            continue

        estimate = evaluate(field, readout_point)[0]

        def plateau(points):
            final = evaluate(field, points)
            center = abs(final - estimate)
            drift = abs(final - evaluate(previous, points))
            return np.column_stack((center, drift)) / max(abs(estimate), np.finfo(float).tiny)

        report = controller.audit(plateau, train, candidates, verification)
        if first_plateau_step is None and report.measured_max.max() <= options.plateau_tolerance:
            # Keep evolving until the requested physical horizon.
            first_plateau_step = step
    last_report = report
    pde, flux, balance = np.max([row[2].measured_max for row in audits], axis=0)
    accepted = bool(max(flux, balance) <= options.conservation_tolerance
                    and last_report.measured_max.max() <= options.plateau_tolerance)
    return dict(estimate=estimate, initial_holdout_max_over_peak=initial_error / scale,
                disabled_heat_estimate=disabled_heat_estimate, controller_accepted=accepted,
                initial_weights=initial.parameters, initial_train_sites=n**3,
                initial_holdout_sites=len(holdout), initial_fit_seconds=fit_seconds,
                first_accepted_plateau_step=first_plateau_step, plateau_max_from_readout=last_report.measured_max[0],
                plateau_max_step_drift=last_report.measured_max[1], controller_candidate_max_from_readout=last_report.proposed_max[0],
                controller_candidate_max_drift=last_report.proposed_max[1],
                controller_plateau_prediction_max_error=last_report.prediction_error.max(),
                stage_residual_over_peak=pde, face_flux_over_peak=flux, local_balance_over_peak=balance,
                wall_seconds=perf_counter() - begin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--rank', type=int, default=14)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--plateau-tolerance', type=float, default=1e-8)
    parser.add_argument('--conservation-tolerance', type=float, default=1e-10)
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    if not 2 <= options.rank <= options.size or options.steps < 3:
        parser.error('expected 2 <= rank <= size and at least three time steps')
    if not (options.plateau_tolerance > 0 and options.conservation_tolerance > 0):
        parser.error('controller tolerances must be positive')

    torch.set_num_threads(1)
    nodes = sites(options.size)
    h = 6 / options.steps
    layers = (HeatNetwork.fit(options.size, h), HeatNetwork.fit(options.size, h / 2))
    order = (0, 5) if options.smoke else tuple(range(len(PARAM_SETS)))
    rows = []
    for index in order:
        params = PARAM_SETS[index]
        fit = fit_case(params, nodes, layers, options)
        print(f'I_{index + 1} estimate={fit["estimate"]:.12g}, plateau={fit["plateau_max_from_readout"]:.3g}', flush=True)
        rows.append(dict(integral=f'I_{index + 1}', a=params[0], b=params[1], m=params[2], n=params[3], t_max=T_MAX, **fit))

    # Opening the scoring values is deferred until all three networks have finished.
    reference_values = references()
    for index, row in zip(order, rows, strict=True):
        reference = reference_values[index]
        error = abs(row['estimate'] - reference)
        relative = error / abs(reference)
        digits = float(-np.log10(relative)) if relative else float('inf')
        disabled_relative = abs(row['disabled_heat_estimate'] - reference) / abs(reference)
        row.update(reference=reference, absolute_error=error, relative_error=relative, correct_digits=digits,
                   disabled_heat_relative_error=disabled_relative)
    digits = [row['correct_digits'] for row in rows]
    summary = dict(min_digits=min(digits), mean_digits=float(np.mean(digits)), max_digits=max(digits),
                   passed_7_digits=sum(row['controller_accepted'] and row['correct_digits'] >= 7 for row in rows),
                   controller_accepted=sum(row['controller_accepted'] for row in rows),
                   cases=len(rows))
    directory = get_mirror_path(__file__, 'results')
    if options.smoke:
        directory = directory / 'smoke'
        directory.mkdir(exist_ok=True)
    with (directory / 'integrals.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with (directory / 'run.json').open('w') as output:
        json.dump(dict(config=vars(options), diffusion_horizon=6, **summary), output, indent=2)
        output.write('\n')
    print(summary)


if __name__ == '__main__':
    main()
