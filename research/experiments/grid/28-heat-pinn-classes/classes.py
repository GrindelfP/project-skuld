"""Characterize heat PINNs on five Genz families from dimension two to six."""
from functools import partial
from dataclasses import replace
from pathlib import Path
import math
import sys

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks import genz
from heat_pinn import Config
from heat_pinn_experiment import Case, run


FIXTURES = (
    ('gaussian', genz.gaussian, (4., .5)),
    ('oscillatory', genz.oscillatory, (2., .5)),
    ('product_peak', genz.product_peak, (.25, .5)),
    ('corner_peak', genz.corner_peak, (1., 1.)),
    ('c0_continuous', genz.c0_continuous, (3., .5)),
)


def exact(name, dimension):
    # Project Genz formulas evaluated without binary64 cancellation.
    if name == 'gaussian':
        value = (mp.sqrt(mp.pi) * mp.erf(2) / 4)**dimension
    elif name == 'oscillatory':
        value = -mp.cos(dimension) * mp.sin(1)**dimension
    elif name == 'product_peak':
        value = (mp.atan(2) / 2)**dimension
    elif name == 'corner_peak':
        value = mp.fsum((-1)**k * mp.mpf(math.comb(dimension, k)) / (1 + k)
                        for k in range(dimension + 1)) / math.factorial(dimension)
    else:
        value = (2 * (1 - mp.exp(-mp.mpf(3) / 2)) / 3)**dimension
    return value, mp.mpf('1e-62')


if __name__ == '__main__':
    cases = [Case(name, dimension, partial(function, **({'a': shape, 'r': other}
                   if name in {'oscillatory', 'corner_peak'} else {'a': shape, 'u_off': other})),
                  partial(exact, name, dimension))
             for dimension in range(2, 7) for name, function, (shape, other) in FIXTURES]
    baseline = Config(width=40, sites=768, adam=700, lbfgs=180)
    improved = replace(baseline, sites=1536, adam=1100, lbfgs=300, balanced_time=True,
                       pde_weight=2., neumann_weight=2., late_weight=2., rates=(1., 4., 16., 64.))
    run(__file__, cases, improved, (42,), baseline=baseline)
