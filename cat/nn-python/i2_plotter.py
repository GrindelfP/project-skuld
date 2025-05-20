##########################################################################
###                    MESONS FUNCTIONS (I2) PLOTTER                   ###
##########################################################################

from utils.plotly_plots import *
from meson_integrals.functions_i2 import funcI2_wrapper

##########################################################################
###                             PARAMETERS                             ###
##########################################################################
u: float = 0.5
c: float = 10.5
uu: list[float] = [u, u]
cc: list[float] = [c, c]
num_points: int = 500
unit_cube_range: tuple[int, int] = (0, 1)
x_ranges: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0)
]
y_ranges: list[tuple[float, float]] = [
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0)
]
A = [0, 1, 0, 1]
B = [0, 0, 1, 1]
M = [2, 2, 2, 2]
N = [3, 3, 3, 3]


#########################################################################
###                              PROGRAM                              ###
#########################################################################
def plot_i2(
        A_: list[int],
        B_: list[int],
        M_: list[int],
        N_: list[int],
        x_ranges_: list[tuple[float, float]],
        y_ranges_: list[tuple[float, float]]
) -> None:
    for a_, b_, m_, n_, x_range_, y_range_ in zip(A_, B_, M_, N_, x_ranges_, y_ranges_):
        plot_surface(
            func=funcI2_wrapper,
            x_range=x_range_,
            y_range=y_range_,
            num_points=num_points,
            title=f'I[{a_}, {b_}, {m_}, {n_}]',
            x_label='α',
            y_label='t',
            z_label='f(α, t)',
            a=a_,
            b=b_,
            m=m_,
            n=n_
        ).show()


if __name__ == '__main__':
    plot_i2(A, B, M, N, x_ranges, y_ranges)
