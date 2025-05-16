#########################################
####### MESON TASK INTEGRALS #######
#########################################
from typing import TypeAlias, Annotated, Any

import numpy as np
from torch import Tensor

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]

#### CONSTANTS ####
la = 2.4
m1 = 1.7 / la
m2 = 1.7 / la
PP = -(3.09688 / la) ** 2

#### BOUNDARIES ####
a1 = 0.0  # t lower
b1 = 50.0000000  # t upper
a2 = 0.0000000  # α lower
b2 = 1.0  # α upper


#### INTEGRAND ####
def funcI2(t, alp1, a, b, m, n):
    global m1, m2, PP

    b_1 = -m1 / (m1 + m2)
    b_2 = m2 / (m1 + m2)
    RR = (alp1 ** 2.0 * (b_1 * b_1) + (1.0 - alp1) ** 2.0 * (b_2 * b_2) + 2.0 * alp1 * (1.0 - alp1) * (b_1 * b_2)) * PP
    DD = alp1 * ((b_1 * b_1) * PP + m1 * m1) + (1.0 - alp1) * ((b_2 * b_2) * PP + m2 * m2) - RR
    z0 = t * DD + t / (1.0 + t) * RR
    Fz0 = np.exp(-2.0 * z0)
    f = pow(alp1, a) * pow((1.0 - alp1), b) * pow(t, m) / pow((1.0 + t), n) * Fz0

    return f


def funcI2_wrapper(X: Matrix, **func_params: Any) -> Vector:
    a = func_params["a"]
    b = func_params["b"]
    m = func_params["m"]
    n = func_params["n"]

    t: Vector = X[:, 1]
    alp: Vector = X[:, 0]

    return funcI2(t, alp, a, b, m, n)
