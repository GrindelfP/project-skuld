########################################
   ####### FUNCTIONS PLOTTER #######
########################################

from genz_functions import *
from plotly_plots import *


############ PARAMETERS ############
u: float = 10
c: float = 10
uu: list[float] = [u, u]
cc: list[float] = [c, c]
num_points: int = 500
unit_cube_range: tuple[int, int] = (0, 1)


############ PROGRAM ############
def plot_genz():
        ### 1D CASE
        plot_line(
            func=corn_peek_1d,
            x_range=unit_cube_range,
            num_points=500,
            title='',
            x_label='X',
            y_label='Y',
            u=u,
            c=c
        ).show()

        ### 2D CASE
        plot_surface(
            func=corn_peek_2d,
            x_range=unit_cube_range,
            y_range=unit_cube_range,
            num_points=num_points,
            title='',
            x_label='X',
            y_label='Y',
            z_label='Z',
            u=uu,
            c=cc,
        ).show()

def main():
    plot_genz()

if __name__ == '__main__':
    main()
