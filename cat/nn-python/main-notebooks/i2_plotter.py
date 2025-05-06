########################################
   ####### FUNCTIONS PLOTTER #######
########################################

from utils.plotly_plots import *
from meson_integrals.functions_i2 import funcI2_wrapper


############ PARAMETERS ############
u: float = 0.5
c: float = 10.5
uu: list[float] = [u, u]
cc: list[float] = [c, c]
num_points: int = 500
unit_cube_range: tuple[int, int] = (0, 1)
x_range: tuple[int, int] = (0, 1)
y_range: tuple[int, int] = (0, 30)
A = [0, 1, 0, 1]
B = [0, 0, 1, 1]
M = [2, 2, 2, 2]
N = [3, 3, 3, 3]

############ PROGRAM ############
def plot_i2():

    for a_, b_, m_, n_ in zip(A, B, M, N):
        plot_surface(
            func=funcI2_wrapper,
            x_range=x_range,
            y_range=y_range,
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

def main():
    plot_i2()

if __name__ == '__main__':
    main()
