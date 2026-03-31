import numpy as np
from scipy.interpolate import griddata
from skuld import *
from scipy import integrate
import matplotlib.pyplot as plt

path_f: str = "test-i3.txt"
path_mm: str = "test-i3-15.txt"
rounds: int = 1

def main(aa, bb, mm, nn):
    a, b, m, n = aa, bb, mm, nn
    la = 1.0
    m1 = 0.3 / la
    m2 = 0.3 / la
    m3 = 0.3 / la
    p1p1 = -(0.14 / la) ** 2
    p2p2 = -(0.14 / la) ** 2
    la = 2.4
    m1 = 1.7 / la
    m2 = 1.7 / la
    PP = -(3.09688 / la) ** 2
    
    BB = 100
    k = -1

    def funcI2(alp1, t):
        b_1 = -m1 / (m1 + m2)
        b_2 = m2 / (m1 + m2)
        RR = (alp1 ** 2.0 * (b_1 * b_1) + (1.0 - alp1) ** 2.0 * (b_2 * b_2) + 2.0 * alp1 * (1.0 - alp1) * (b_1 * b_2)) * PP
        DD = alp1 * ((b_1 * b_1) * PP + m1 * m1) + (1.0 - alp1) * ((b_2 * b_2) * PP + m2 * m2) - RR
        z0 = t * DD + t / (1.0 + t) * RR
        Fz0 = np.exp(-2.0 * z0)
        f = pow(alp1, a) * pow((1.0 - alp1), b) * pow(t, m) / pow((1.0 + t), n) * Fz0

        return f


    def plot_integrand_slices():
        x0, t0 = 0.5, 1.0


        # Slice: F vs y (x=x0, t=t0)
        x_vals = np.linspace(0, 1, 200)
        F_y = np.array([funcI2(x, t0) for x in x_vals])
        #F_y = np.array([integrand_xyz(x0, y, t0) for y in y_vals])

        # Slice: F vs t (x=x0, y=y0)
        t_vals = np.linspace(0, 100, 200)
        F_t = np.array([funcI2(x0, t) for t in t_vals])
        #F_t = np.array([integrand_xyz(x0, y0, t) for t in t_vals])

        fig, axs = plt.subplots(1, 2, figsize=(15, 4))

        axs[0].plot(x_vals, F_y, 'g-')
        axs[0].set_xlabel(r'$\alpha$')
        axs[0].set_ylabel(r'f($\alpha$, t)')
        axs[0].grid(True)

        axs[1].plot(t_vals, F_t, 'r-')
        axs[1].set_xlabel('t')
        axs[1].set_ylabel(r'f($\alpha$, t)')
        axs[1].grid(True)

        plt.suptitle(f'(a={a}, b={b}, m={m}, n={n})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    plot_integrand_slices()


if __name__ == "__main__":
    main(1, 0, 1, 2)
                