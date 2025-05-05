#########################################
   ####### MESON TASK INTEGRALS #######
#########################################
import numpy as np

#### CONSTANTS ####
la = 2.4
m1 = 1.7 / la
m2 = 1.7 / la
PP = -(3.09688 / la) ** 2

#### PARAMETERS ####
a = 0
b = 0
m = 2
n = 1

#### BOUNDARIES ####
a1 = 0.0 # t lower
b1 = 50.0000000 # t upper
a2 = 0.0000000 # α lower
b2 = 1.0 # α upper

#### INTEGRAND ####
def funcI2(t, alp1):

    global m1, m2, PP, a, b, m, n

    b_1 = -m1 / (m1 + m2)
    b_2 = m2 / (m1 + m2)
    RR = (alp1**2.0 * (b_1 * b_1) + (1.0 - alp1)**2.0 * (b_2 * b_2) + 2.0 * alp1 * (1.0 - alp1) * (b_1 * b_2)) * PP
    DD = alp1 * ((b_1 * b_1) * PP + m1 * m1) + (1.0 - alp1) * ((b_2 * b_2) * PP + m2 * m2) - RR
    z0 = t * DD + t / (1.0 + t) * RR
    Fz0 = np.exp(-2.0 * z0)
    f = (alp1**a) * (1.0 - alp1)**b * t**m / (1.0 + t)**n * Fz0

    return f


def funcI2_wrapper(X):

    t = X[:, 0]
    alp = X[:, 1]

    return funcI2(t, alp)
