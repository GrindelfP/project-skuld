from genz_functions import *
from plotly_plots import *


########################################
   ############ PROGRAM ############
########################################

def main():
    plot_surface(
        func=gauss_2d,
        x_range=(-5, 5),
        y_range=(-5, 5),
        num_points=500,
        title='',
        x_label='X',
        y_label='Y',
        z_label='Z',
        u=[1, 1],
        c=[1, 1]
    ).show()
    plot_line(
        func=gauss_1d,
        x_range=(-10, 10),
        num_points=500,
        title='',
        x_label='X',
        y_label='Y',
        u=1,
        c=1
    ).show()


if __name__ == '__main__':
    main()
