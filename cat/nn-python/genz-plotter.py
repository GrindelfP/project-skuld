########################################
####### FUNCTIONS PLOTTER #######
########################################

from genz.genz_functions import *
from utils.plotly_plots import *

############ PARAMETERS ############
u: float = 0.5
c: float = 10.5
uu: list[float] = [u, u]
cc: list[float] = [c, c]
num_points: int = 500
unit_cube_range: tuple[int, int] = (0, 1)


############ PROGRAM ############
def plot_genz():
    ### 1D CASE
    plot_line(
        func=gauss_1d,
        x_range=unit_cube_range,
        num_points=500,
        title=f'DISCONTINUOUS: u = {u}, c = {c}.',
        x_label='X',
        y_label='Y',
        u=u,
        c=c
    ).show()

    ### 2D CASE
    plot_surface(
        func=gauss_2d,
        x_range=unit_cube_range,
        y_range=unit_cube_range,
        num_points=num_points,
        title=f'DISCONTINUOUS: u = {uu}, c = {cc}.',
        x_label='X',
        y_label='Y',
        z_label='Z',
        u=uu,
        c=cc
    ).show()


def main():
    plot_genz()


if __name__ == '__main__':
    main()
