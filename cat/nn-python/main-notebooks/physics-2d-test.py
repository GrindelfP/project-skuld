import plotly.graph_objects as go
from torch import Tensor

import skuld
import plots
import integrate
import numpy as np
from scipy.integrate import quad

####################################################################
##########               QUAD INTEGRATION                 ##########
####################################################################

la = 2.4
m1 = 1.7 / la
m2 = 1.7 / la
PP = -(3.09688 / la) ** 2
alp1 = 0.0
m, n, k, l = 0, 0, 0, 0

def funcI2(t):
    global m1, m2, PP, la, alp1, m, n, k, l

    b1 = -m1 / (m1 + m2)
    b2 = m2 / (m1 + m2)

    RR = (alp1**2.0 * (b1 * b1) + (1.0 - alp1)**2.0 * (b2 * b2) +
          2.0 * alp1 * (1.0 - alp1) * (b1 * b2)) * PP

    DD = alp1 * ((b1 * b1) * PP + m1 * m1) + \
         (1.0 - alp1) * ((b2 * b2) * PP + m2 * m2) - RR

    z0 = t * DD + t / (1.0 + t) * RR

    Fz0 = np.exp(-2.0 * z0)

    return (alp1**m) * (1.0 - alp1)**n * t**k / (1.0 + t)**l * Fz0

def funcI21(alp1a):
    global alp1, m, n, k, l

    alp1 = alp1a

    aa = 0.0
    bb = 100.0

    result, error = quad(funcI2, aa, bb, epsabs=0.001, epsrel=0.0001)

    return result

def funcI22(mm, nn, kk, ll):
    global m, n, k, l

    m = mm
    n = nn
    k = kk
    l = ll

    aa = 0.0
    bb = 1.0

    result, error = quad(funcI21, aa, bb, epsabs=0.001, epsrel=0.0001)

    return result

def run_quad_integration() -> None:
    print(f"funcI22(0,0,1,2)={funcI22(0, 0, 1, 2)}")
    print(f"funcI22(0,1,1,2)={funcI22(0, 1, 1, 2)}")
    print(f"funcI22(1,0,1,2)={funcI22(1, 0, 1, 2)}")
    print(f"funcI22(1,1,1,2)={funcI22(1, 1, 1, 2)}")
    print(f"funcI22(0,0,2,1)={funcI22(0, 0, 2, 1)}")
    print(f"funcI22(0,1,2,1)={funcI22(0, 1, 2, 1)}")
    print(f"funcI22(1,0,2,1)={funcI22(1, 0, 2, 1)}")
    print(f"funcI22(1,1,2,1)={funcI22(1, 1, 2, 1)}")
    print(f"funcI22(0,0,2,3)={funcI22(0, 0, 2, 3)}")
    print(f"funcI22(0,1,2,3)={funcI22(0, 1, 2, 3)}")
    print(f"funcI22(1,0,2,3)={funcI22(1, 0, 2, 3)}")
    print(f"funcI22(1,1,2,3)={funcI22(1, 1, 2, 3)}")
    print(f"funcI22(0,0,3,2)={funcI22(0, 0, 3, 2)}")
    print(f"funcI22(0,1,3,2)={funcI22(0, 1, 3, 2)}")
    print(f"funcI22(1,0,3,2)={funcI22(1, 0, 3, 2)}")
    print(f"funcI22(1,1,3,2)={funcI22(1, 1, 3, 2)}")



####################################################################
##########                NNI INTEGRATION                 ##########
####################################################################


########### INTEGRAND ###########

def integrand(t, alp1):

    global m1, m2, PP, a, b, m, n

    b1 = -m1 / (m1 + m2)
    b2 = m2 / (m1 + m2)

    RR = (alp1**2.0 * (b1 * b1) + (1.0 - alp1)**2.0 * (b2 * b2) +
          2.0 * alp1 * (1.0 - alp1) * (b1 * b2)) * PP

    DD = alp1 * ((b1 * b1) * PP + m1 * m1) + (1.0 - alp1) * ((b2 * b2) * PP + m2 * m2) - RR

    z0 = t * DD + t / (1.0 + t) * RR

    Fz0 = np.exp(-2.0 * z0)

    f = (alp1**a) * (1.0 - alp1)**b * t**m / (1.0 + t)**n * Fz0

    return f


def integrand_wrapper(X):
    T = X[:, 0]
    A = X[:, 1]
    return integrand(T, A)

########### HYPERPARAMS ###########
input_size = 2
hidden_size = 30
learning_rate = 0.001
num_epochs = 5000
num_samples = 200
distribution = 'sud' # 'grid'


def prepare_dataset(params: list[float], boundaries: list[float]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    #### INTEGRATION PARAMS ####
    a = params[0]
    b = params[1]
    m = params[2]
    n = params[3]
    a1 = boundaries[0] # t lower
    b1 = boundaries[1]  # t upper
    a2 = boundaries[2]  # α lower
    b2 = boundaries[3]  # α upper

    if distribution == 'grid':
        X_init, y_init = skuld.generate_data(integrand_wrapper, [a1, a2], [b1, b2], num_samples, input_size)
    else:
        X_init, y_init = skuld.generate_data_uniform(integrand_wrapper, [a1, a2], [b1, b2], num_samples ** 2, input_size)
    plots.plot_2d_function_heatmap(X_init, y_init, 'f(t, alpha)')
    X, y = skuld.scale_data(X_init, y_init, n_dim=2)

    return skuld.split_data(X=X, y=y, test_size=0.1, shuffle=True)


########### TEST FUNCTION ###########
def test_physics_nni(params: list[float], boundaries: list[float]) -> None:
    #### INTEGRATION PARAMS ####
    a = params[0]
    b = params[1]
    m = params[2]
    n = params[3]
    a1 = boundaries[0] # t lower
    b1 = boundaries[1]  # t upper
    a2 = boundaries[2]  # α lower
    b2 = boundaries[3]  # α upper

    #### DATASET PREP ####
    _ = prepare_dataset(params, boundaries)



if __name__ == "__main__":
    run_quad_integration()
    test_physics_nni([0.0, 0.0, 2.0, 1.0], [0.0, 50.0, 0.0, 1.0])
