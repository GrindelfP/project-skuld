##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################
from typing import Tuple, TypeAlias, Annotated, Callable, Any

from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Linear, Sigmoid
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss

from mpmath import polylog
from sklearn.model_selection import train_test_split

import torch
import mpmath
import time
import itertools
import torch.nn as nn
import torch.optim as optim
import numpy as np

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]

##########################################################################
###                           DATA SCALERS                             ###
##########################################################################
def scale_data(
        X_init: Matrix,
        y_init: Vector,
        frange: tuple[int, int] = (0, 1),
) -> tuple[Matrix, Vector]:
    """
        Scales function dataset to the specific range frange.

        :param X_init: initial variables
        :param y_init: initial function values
        :param frange: range of scaled values (default is [0, 1])

        :returns: tuple of scaled X-s and y-s to range frange.
    """
    if not isinstance(X_init, torch.Tensor) or not isinstance(
            y_init, torch.Tensor):
        raise TypeError("Input X_init and y_init must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor (m, n_dim).")
    if y_init.ndim != 2 or y_init.shape[1] != 1:
        raise ValueError("y_init must be a 2D tensor (m, 1).")
    if X_init.shape[0] != y_init.shape[0]:
        raise ValueError("X_init and y_init must have the same number "
                         "of rows (m).")
    min_val_x = torch.min(X_init, dim=0).values
    max_val_x = torch.max(X_init, dim=0).values
    min_val_y = torch.min(y_init)
    max_val_y = torch.max(y_init)
    scaled_X = ((frange[1] - frange[0]) * (X_init - min_val_x) /
                (max_val_x - min_val_x) + frange[0])
    scaled_y = ((frange[1] - frange[0]) * (y_init - min_val_y) /
                (max_val_y - min_val_y) + frange[0])
    return scaled_X, scaled_y


def descale_result(
        nni_scaled: float,
        X_init: Matrix,
        y_init: Vector,
        frange: tuple[int, int] = (0, 1),
        n_dim: int = 1
) -> float:
    """
        Restores true value to the scaled integral.

        :param nni_scaled: scaled integral value
        :param X_init:     initial variables
        :param y_init:     initial function values
        :param frange:     range of scaled values (default is [0, 1])
        :param n_dim:      number of function dimensions (default is 1)

        :returns: descaled integral value.
    """
    if not isinstance(X_init, torch.Tensor) or not isinstance(
            y_init, torch.Tensor):
        raise TypeError("Inputs must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor.")
    if y_init.ndim != 1 and y_init.ndim != 2:
        raise ValueError("y_init must be a 2D tensor with shape "
                         "(num_of_points,1).")
    if y_init.ndim == 2 and y_init.shape[1] != 1:
        raise ValueError("y_init must be a 2D tensor with shape "
                         "(num_of_points,1).")

    x_min = torch.min(X_init, dim=0).values
    x_max = torch.max(X_init, dim=0).values
    f_min = torch.min(y_init)
    f_max = torch.max(y_init)
    frange_size = frange[1] - frange[0]
    VS = torch.prod(x_max - x_min)
    VSS = frange_size ** n_dim

    return (nni_scaled * (VS * (f_max - f_min) / (VSS * frange_size)) +
            (f_min - (f_max - f_min) / frange_size * frange[0]) * VS).item()
