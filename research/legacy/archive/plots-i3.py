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
    PP = -(0.7 / la) ** 2
    
    BB = 100
    k = -1

    def integrand(alp1, alp2, t):
        RR = (alp1**2) * p1p1 + (alp2**2) * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
        DD = (alp1 * (p1p1 + m1**2) + alp2 * (p2p2 + m2**2) + (1.0 - alp1 - alp2) * (m3**2) - RR)
        z0 = t * DD + t / (1.0 + t) * RR
        Fz0 = np.exp(k * z0)
        
        return (alp1**a) * (alp2**b) * (t**m) / ((1.0 + t)**n) * Fz0
    
    def integrand_xyz(x, y, t):
        alp1 = x
        alp2 = y * (1.0 - x)
        RR = (alp1**2) * p1p1 + (alp2**2) * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
        DD = (alp1 * (p1p1 + m1**2) + alp2 * (p2p2 + m2**2) + (1.0 - alp1 - alp2) * m3**2 - RR)
        z0 = t * DD + t / (1.0 + t) * RR
        Fz0 = np.exp(k*z0)
        jacobian = (1.0 - x)
        alp1_power = alp1 ** a
        alp2_power = alp2 ** b
        
        return alp1_power * alp2_power * jacobian * (t ** m) / ((1.0 + t) ** n) * Fz0


    def plot_integrand_slices():
        x0, y0, t0 = 0.5, 0.5, 1.0

        # Slice: F vs x (y=y0, t=t0)
        x_vals = np.linspace(0, 1, 200)
        F_x = np.array([integrand(x, y0, t0) for x in x_vals])
        #F_x = np.array([integrand(x, y0, t0) for x in x_vals])

        # Slice: F vs y (x=x0, t=t0)
        y_vals = np.linspace(0, 1, 200)
        F_y = np.array([integrand(x0, y, t0) for y in y_vals])
        #F_y = np.array([integrand_xyz(x0, y, t0) for y in y_vals])

        # Slice: F vs t (x=x0, y=y0)
        t_vals = np.linspace(0, 100, 200)
        F_t = np.array([integrand_xyz(x0, y0, t) for t in t_vals])
        #F_t = np.array([integrand_xyz(x0, y0, t) for t in t_vals])

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].plot(x_vals, F_x, 'b-')
        axs[0].set_xlabel(r'$\alpha_1$')
        axs[0].set_ylabel(r'$f(\alpha_1$, $\alpha_2$, t₀)')
        axs[0].grid(True)

        axs[1].plot(y_vals, F_y, 'g-')
        axs[1].set_xlabel(r'$\alpha_2$')
        axs[1].set_ylabel(r'$f(\alpha_1$, $\alpha_2$, t₀)')
        axs[1].grid(True)

        axs[2].plot(t_vals, F_t, 'r-')
        axs[2].set_xlabel('t')
        axs[2].set_ylabel(r'$f(\alpha_1$, $\alpha_2$, t₀)')
        axs[2].grid(True)

        plt.suptitle(f'(a={a}, b={b}, m={m}, n={n})')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    plot_integrand_slices()


if __name__ == "__main__":
    main(1, 1, 1, 2)
                