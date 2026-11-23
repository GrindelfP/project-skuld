"""Train a separate insulated heat PINN for each finite-cutoff meson integral."""
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
import sys

import mpmath as mp

RESEARCH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RESEARCH))
from heat_pinn import Config
from heat_pinn_experiment import Case, run
from benchmarks.meson_reference import physical_references
from physics import PARAM_SETS, integrand_transformed


def point_values(params, x):
    return integrand_transformed(x, *params)


@lru_cache(maxsize=1)
def reference_table():
    # Reuse the project's independent physical-coordinate scoring implementation.
    low, high = physical_references(36), physical_references(44)
    return [(value, 100 * abs(value - previous)) for previous, value in zip(low, high, strict=True)]


def reference(index):
    return reference_table()[index]


if __name__ == '__main__':
    cases = [Case(f'I_{index + 1}', 3, partial(point_values, params), partial(reference, index))
             for index, params in enumerate(PARAM_SETS)]
    baseline = Config(width=64, sites=2048, adam=1200, lbfgs=350,
                      rates=(1., 4., 16., 64.), pde_weight=.25)
    improved = replace(baseline, sites=3072, adam=1500, lbfgs=380, balanced_time=True,
                       rates=(1., 4., 16., 64., 256.), pde_weight=2.,
                       neumann_weight=2., late_weight=2.)
    run(__file__, cases, improved, (42,), baseline=baseline)
