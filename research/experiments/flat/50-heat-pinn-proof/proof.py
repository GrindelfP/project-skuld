"""Train on smooth low-dimensional initial conditions with exact references."""
import math
from pathlib import Path
import sys

import mpmath as mp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from heat_pinn import Config
from heat_pinn_experiment import Case, run


def cosine_line(x):
    return .75 + .625 * torch.cos(math.pi * x[:, 0])


def cosine_plane(x):
    return 1.25 + .375 * torch.cos(math.pi * x[:, 0]) - .5 * torch.cos(math.pi * x[:, 1])


def cosine_diagonal(x):
    return .875 + .5 * torch.cos(math.pi * (x[:, 0] + x[:, 1]))


def constant_on_cube(x):
    return torch.full_like(x[:, 0], .625)


def precise_line(x):
    return mp.mpf(3) / 4 + mp.mpf(5) / 8 * mp.cos(mp.pi * x[0])


def precise_plane(x):
    return mp.mpf(5) / 4 + mp.mpf(3) / 8 * mp.cos(mp.pi * x[0]) - mp.cos(mp.pi * x[1]) / 2


def precise_constant(_x):
    return mp.mpf(5) / 8


if __name__ == '__main__':
    cases = [
        Case('cosine_line', 1, cosine_line, lambda: (mp.mpf(3) / 4, mp.mpf(0)), precise_line),
        Case('cosine_plane', 2, cosine_plane, lambda: (mp.mpf(5) / 4, mp.mpf(0)), precise_plane),
        Case('cosine_diagonal', 2, cosine_diagonal,
             lambda: (mp.mpf(7) / 8 - 2 / mp.pi**2, mp.mpf('1e-63'))),
        Case('constant', 3, constant_on_cube, lambda: (mp.mpf(5) / 8, mp.mpf(0)), precise_constant),
    ]
    run(__file__, cases, Config(width=6, sites=128, adam=450, lbfgs=0, trust=140,
                                compact=True, rates=(1.,), pde_weight=1., normalize=False,
                                log_every=20), (42, 137), width_by_dimension={1: 4})
