import math
import torch
from torch import Tensor
from global_definitions import Vector, Matrix


########################################
############ GENZ FUNCTIONS ############
########################################

########### 1D FUNCTIONS ###########
def osc_1d(X: Vector, u: float , c: float) -> Tensor:
    return torch.cos(torch.Tensor(2 * math.pi * u + X * c))

def prod_peak_1d(X: Vector, u: float , c: float) -> Tensor:
    return 1 / (c**(-2) + (X - u)**2)

def corn_peek_1d(X: Vector, u: float , c: float) -> Tensor:
    base: Vector = torch.Tensor(1 + c * X)
    return torch.pow(base, -2)

def gauss_1d(X: Vector, u: float , c: float) -> Tensor:
    return torch.exp(torch.Tensor(-c**2 * (X - u)**2))

def cont_1d(X: Vector, u: float , c: float) -> Tensor:
    return torch.exp(torch.Tensor(-c * abs(X - u)))

def disco_1d(X: Vector, u: float , c: float) -> Tensor:
    y_list = []

    for x in X:
        if x > u:
            y_list.append(0)
        else:
            y_list.append(math.exp(c * x))

    return torch.tensor(y_list)


########### 2D FUNCTIONS ###########
def osc_2d(X: Matrix, u: list[float], c: list[float]) -> Tensor:
    sum_ = 0
    for i in range(2):
        sum_ += c[i] * X[:, i]

    return torch.cos(2 * math.pi * u[0] + sum_)

def prod_peak_2d(X: Matrix, u: list[float], c: list[float]) -> Tensor:
    prod_ = 1
    for i in range(2):
        prod_ *= (c[i] ** (-2) + (X[:, i] - u[i])**2) ** (-1)
    return prod_

def corn_peek_2d(X: Matrix, u: list[float], c: list[float]) -> Tensor:
    sum_ = 0
    for i in range(2):
        sum_ += c[i] * X[:, i]
    return (1 + sum_) ** (-3)

def gauss_2d(X, u, c):
    sum_ = 0
    for i in range(2):
        sum_ += (c[i] ** 2) * ((X[:, i] - u[i]) ** 2)
    return torch.exp(-sum_)

def cont_2d(X: Matrix, u: list[float], c: list[float]) -> Tensor:
    sum_ = 0
    for i in range(2):
        sum_ += c[i]  * abs(X[:, i] - u[i])
    return torch.exp(-sum_)


def disco_2d(X: Matrix, u: list[float], c: list[float]) -> Tensor:
    result = torch.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i, 0].item()

        if x > u[0]:
            result[i] = 0.0
        else:
            result[i] = math.exp(c[0] * x)

    return result
