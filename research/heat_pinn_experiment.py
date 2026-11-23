"""Run and score heat PINNs. Reference callbacks run after training and readout."""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
import platform
import statistics
import time
from typing import Callable

import mpmath as mp
import torch

from heat_pinn import ANCHOR, HORIZON, diagnostics, fit, plateau
from utils.paths import get_mirror_path


@dataclass(frozen=True)
class Case:
    name: str
    dimension: int
    point_fn: Callable[[torch.Tensor], torch.Tensor]
    reference: Callable[[], tuple[mp.mpf, mp.mpf]]
    precise_point_fn: Callable[[tuple[mp.mpf, ...]], mp.mpf] | None = None


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_loss(path, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    criteria = (('initial_mse', 'Начальное условие'),
                ('pde_mse', 'Уравнение теплопроводности'),
                ('neumann_mse', 'Условие Неймана'))
    figure, panels = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for panel, (key, label) in zip(panels, criteria, strict=True):
        for run, history in histories.items():
            offsets = {'initialization': 0, 'adam': 0}
            adam_steps = max((row['evaluation'] for row in history if row['stage'] == 'adam'), default=0)
            for stage in ('trust_region', 'trust_region_final', 'lbfgs', 'lbfgs_final'):
                offsets[stage] = adam_steps
            offsets['initial_fit'] = 0
            offsets['dynamics'] = max((row['evaluation'] for row in history
                                       if row['stage'] == 'initial_fit'), default=0)
            offsets['multiprecision'] = max(row['evaluation'] + offsets[row['stage']]
                                             for row in history if row['stage'] != 'multiprecision')
            x = [row['evaluation'] + offsets[row['stage']] for row in history]
            panel.semilogy(x, [max(row[key], 1e-76) for row in history], linewidth=1, label=run)
        panel.set(title=label, xlabel='Шаг Adam / вызов невязки')
        panel.grid(alpha=.2)
    panels[0].set_ylabel('Средний квадрат невязки')
    if len(histories) <= 10:
        handles, labels = panels[-1].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncol=min(len(labels), 5), fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def statistics_row(label, rows):
    values = [row['I_1_digits'] for row in rows]
    return dict(group=label, count=len(values), min_digits=min(values),
                mean_digits=statistics.mean(values), max_digits=max(values),
                passes_1e_7=sum(row['passes_1e_7'] for row in rows),
                passes_1e_20=sum(row['passes_1e_20'] for row in rows))


def run(script, cases, config, seeds, *, baseline=None, width_by_dimension=None, trainer=fit):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--case', action='append', help='Run a named case, optionally restricted by --dimension')
    parser.add_argument('--dimension', type=int, action='append')
    parser.add_argument('--seed', type=int, action='append')
    if baseline:
        parser.add_argument('--baseline', action='store_true', help='Use the original training protocol')
    args = parser.parse_args()
    selected = [case for case in cases if (not args.case or case.name in args.case)
                and (not args.dimension or case.dimension in args.dimension)]
    if not selected:
        parser.error('No matching cases')
    seeds = args.seed or seeds
    use_baseline = bool(baseline and args.baseline)
    if use_baseline:
        config = baseline
    torch.set_num_threads(1)
    if args.smoke:
        config = replace(config, sites=min(config.sites, 32), adam=2, lbfgs=min(config.lbfgs, 2),
                         trust=min(config.trust, 3))
        if hasattr(config, 'initial_adam'):
            config = replace(config, initial_adam=2, initial_lbfgs=min(config.initial_lbfgs, 2))
    out = get_mirror_path(script, 'results')
    if use_baseline:
        out = out / 'baseline'
        out.mkdir(exist_ok=True)
    if args.smoke or args.case or args.dimension or args.seed:
        suffix = 'smoke' if args.smoke else 'subset'
        out = out / suffix
        out.mkdir(exist_ok=True)
    metadata = dict(config=asdict(config), seeds=seeds, time=HORIZON, readout_coordinate=ANCHOR,
                    dtype='float64', device='cpu', platform=platform.platform(),
                    torch_version=torch.__version__, reference_arithmetic_digits=65,
                    width_by_dimension=width_by_dimension)
    (out / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows, history_rows, histories = [], [], {}
    with mp.workdps(65):
        for case in selected:
            case_config = config
            if width_by_dimension and case.dimension in width_by_dimension:
                case_config = replace(config, width=width_by_dimension[case.dimension])
            for seed in seeds:
                model, scale, untrained, history, seconds = trainer(case.point_fn, case.dimension, case_config, seed)
                raw_estimate = plateau(model)
                check = diagnostics(model, seed)
                precision_sites = 0
                # Refine only converged binary64 fields. The gate uses training loss,
                # never an integral or an exact heat solution.
                if (case.precise_point_fn and not args.smoke and
                        max(history[-1][key] for key in ('initial_mse', 'pde_mse', 'neumann_mse')) < 1e-12):
                    from heat_pinn_precision import mp_readout, precision_diagnostics, refine
                    started = time.perf_counter()
                    field, precise_history, precision_sites = refine(model, case.precise_point_fn, seed)
                    raw_estimate = mp_readout(field)
                    check = precision_diagnostics(field, seed)
                    history.extend(dict(stage='multiprecision', evaluation=i,
                                        **{key: float(value) for key, value in losses.items()},
                                        late_gradient_mse=0.)
                                   for i, losses in enumerate(precise_history, 1))
                    seconds += time.perf_counter() - started
                reference, uncertainty = case.reference()
                if not reference or uncertainty >= abs(reference) * mp.mpf('1e-22'):
                    raise ValueError('Scoring reference lacks the required precision')
                estimate = mp.mpf(raw_estimate) * scale
                absolute = abs(estimate - reference)
                relative = absolute / abs(reference)
                stabilized = max(mp.mpf(0), -mp.log10(relative + mp.mpf('1e-30')))
                random_error = abs(mp.mpf(untrained) * scale - reference) / abs(reference)
                spatial_relative = mp.mpf(check['plateau_spread']) * abs(scale) / abs(reference)
                temporal_relative = mp.mpf(check['plateau_time_drift']) * abs(scale) / abs(reference)
                final_drift_relative = mp.mpf(check['final_time_drift']) * abs(scale) / abs(reference)
                perturbation_relative = mp.mpf(check['weight_perturbation_delta']) * abs(scale) / abs(reference)
                row = dict(case=case.name, dimension=case.dimension, seed=seed,
                           estimate=mp.nstr(estimate, 30), reference=mp.nstr(reference, 32),
                           reference_abs_uncertainty=mp.nstr(uncertainty, 8),
                           Err_abs=mp.nstr(absolute, 16), Err_rel=mp.nstr(relative, 16),
                           Cd=int(mp.floor(stabilized)), I_1_digits=float(stabilized),
                           passes_1e_7=(relative + uncertainty / abs(reference) <= mp.mpf('1e-7')),
                           passes_1e_20=bool(precision_sites and
                                            relative + uncertainty / abs(reference) <= mp.mpf('1e-20')),
                           untrained_value=mp.nstr(mp.mpf(untrained) * scale, 20),
                           untrained_relative_error=float(random_error),
                           untrained_differs=bool(random_error > mp.mpf('1e-7')),
                           perturbed_value=mp.nstr(mp.mpf(check['perturbed_plateau']) * scale, 20),
                           perturbation_relative_change=float(perturbation_relative),
                           perturbation_changes_answer=bool(perturbation_relative > mp.mpf('1e-12')),
                           plateau_spatial_relative_range=float(spatial_relative),
                           plateau_temporal_relative_drift=float(temporal_relative),
                           plateau_temporal_5_to_6_relative_drift=float(final_drift_relative),
                           plateau_verified_at_1e_7=bool(max(spatial_relative, temporal_relative) <= mp.mpf('1e-7')),
                           plateau_verified_at_1e_20=bool(
                               precision_sites and max(spatial_relative, final_drift_relative) <= mp.mpf('1e-20')),
                           heldout_pde_rms=check['heldout_pde_rms'],
                           heldout_neumann_rms=check['heldout_neumann_rms'],
                           heldout_neumann_max=check['heldout_neumann_max'],
                           anchor_time_values=json.dumps({t: mp.nstr(mp.mpf(value) * scale, 38)
                                                          for t, value in check['anchor_time_values'].items()}),
                           raw_network_value=mp.nstr(raw_estimate, 38), scale=scale,
                           parameter_count=sum(p.numel() for p in model.parameters()),
                           initial_condition_sites=config.sites, precision_initial_sites=precision_sites,
                           training_seconds=round(seconds, 4))
                if not math.isfinite(row['I_1_digits']) or not all(
                        math.isfinite(float(row[key])) for key in ('Err_abs', 'Err_rel', 'estimate')):
                    raise RuntimeError(f'Nonfinite estimate for {case.name}')
                rows.append(row)
                run_name = f'{case.name} d={case.dimension} s={seed}'
                histories[run_name] = history
                history_rows.extend(dict(case=case.name, dimension=case.dimension, seed=seed, **h)
                                    for h in history)
                write_csv(out / 'integrals.csv', rows)
                write_csv(out / 'loss.csv', history_rows)
                print(f'{run_name}: error={float(relative):.4g} space={float(spatial_relative):.4g} '
                      f'time={float(temporal_relative):.4g} ({seconds:.1f}s)', flush=True)
    summary = [statistics_row('all', rows)]
    for name in dict.fromkeys(row['case'] for row in rows):
        summary.append(statistics_row(name, [row for row in rows if row['case'] == name]))
    for dimension in dict.fromkeys(row['dimension'] for row in rows):
        summary.append(statistics_row(f'{dimension}D', [row for row in rows if row['dimension'] == dimension]))
    write_csv(out / 'summary.csv', summary)
    plot_loss(out / 'convergence.png', histories)
    print(summary[0], flush=True)
