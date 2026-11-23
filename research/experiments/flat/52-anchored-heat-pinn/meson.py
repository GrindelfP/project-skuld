"""Experimental anchored PINN for the project's finite-cutoff meson benchmark."""
import csv
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
import sys

import mpmath as mp

RESEARCH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RESEARCH))
from heat_pinn_anchored import AnchoredConfig, fit_anchored
from heat_pinn_experiment import Case, run
from physics import PARAM_SETS, T_MAX, integrand_transformed


@lru_cache(maxsize=1)
def saved_rows():
    path, = (RESEARCH / 'results' / 'flat').glob('*-precision-water-volume/integrals.csv')
    with path.open(newline='') as source:
        return list(csv.DictReader(source))


def reference(index):
    row = saved_rows()[index]
    params = tuple(int(row[key]) for key in ('a', 'b', 'm', 'n'))
    if params != PARAM_SETS[index] or float(row['t_max']) != T_MAX:
        raise ValueError('Saved reference does not match the finite-cutoff benchmark')
    value = mp.mpf(row['reference'])
    uncertainty = abs(value) * 100 * mp.mpf(row['reference_relative_change'])
    return value, uncertainty


if __name__ == '__main__':
    cases = [Case(f'I_{index + 1}', 3, partial(integrand_transformed, a=a, b=b, m=m, n=n),
                  partial(reference, index))
             for index, (a, b, m, n) in enumerate(PARAM_SETS)]
    config = AnchoredConfig(width=64, sites=4096, batch=192, adam=1500, lbfgs=450,
                            initial_adam=2000, initial_lbfgs=325,
                            rates=(.125, 1., 8., 64.), pde_weight=1.,
                            neumann_weight=2., log_every=50, balanced_time=True,
                            initial_feature=True, scaled_pde=True, late_weight=1., switch_rate=100.)
    baseline = replace(config, reflected=False, initial_feature=False, scaled_pde=False,
                       late_weight=0., switch_rate=40.)
    run(__file__, cases, config, (42,), baseline=baseline, trainer=fit_anchored)
