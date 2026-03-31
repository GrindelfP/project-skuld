##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################
import itertools
import math
from typing import TypeAlias, Annotated, Callable, Any

import numpy as np
import torch
from torch import Tensor

# noinspection PyProtectedMember

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]


##########################################################################
###                          DATA GENERATORS                           ###
##########################################################################
def generate_data_unfirm_grid(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
        Generates data in the form of a 2D tensor of variables for the function and neural
        network input as well as the function values for the generated tensor of variables.

        :param func:      function to provide values for the variables.
        :param lower:     lower bounds of variable values.
        :param upper:     upper bounds of variable values.
        :param n_samples: number of points of data to generate per dimension (default value 100).
        :param n_dim:     number of dimensions of the function func (default value 1).
        :param func_params: Additional arguments to pass to the function.

        :returns: dataset of variables X and function values y
    """
    if n_dim == 1:
        X: Matrix = torch.linspace(lower[0], upper[0], n_samples).view(n_samples, 1)
        y: Vector = func(X, **func_params).view(n_samples, 1)
    else:
        ranges = [torch.linspace(lower[n], upper[n], n_samples).tolist()
                  for n in range(n_dim)]
        combinations: list = list(itertools.product(*ranges))
        X: Matrix = torch.tensor(combinations, dtype=torch.float32)
        y: Vector = func(X, **func_params).view(-1, 1)
        
    return X, y


def generate_data_uniform(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        **func_args: Any
) -> tuple[Matrix, Vector]:
    """
        Generates data UNIFORMLY DISTRIBUTED in the form of a 2D tensor of variables
        for the function and neural network input as well as the function values for
        the generated tensor of variables.

        :param func:      function to provide values for the variables.
        :param lower:     lower bounds of variable values.
        :param upper:     upper bounds of variable values.
        :param n_samples: number of points of data to generate per dimension (default value 100).
        :param n_dim:     number of dimensions of the function func (default value 1).
        :param func_args: Additional arguments to pass to the function.

        :returns: dataset of variables X and function values y
    """
    if n_dim == 1:
        X: Matrix = torch.rand(n_samples, 1) * (upper[0] - lower[0]) + lower[0]
        y: Vector = func(X, **func_args).view(n_samples, 1)
    else:
        X: Matrix = torch.rand(n_samples, n_dim)
        for i in range(n_dim):
            X[:, i] = X[:, i] * (upper[i] - lower[i]) + lower[i]
        y: Vector = func(X, **func_args).view(n_samples, 1)
    return X, y


def generate_data_logarithmic(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        epsilon: float = 1e-6,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    ranges = []
    for i in range(n_dim):
        start_val = lower[i] if lower[i] > 1e-9 else epsilon
        log_start = np.log10(start_val)
        log_end = np.log10(upper[i])
        
        dim_range = torch.logspace(log_start, log_end, n_samples)

        if lower[i] == 0:
            dim_range[0] = 0.0
            
        ranges.append(dim_range.tolist())

    if n_dim == 1:
        X: Matrix = torch.tensor(ranges[0], dtype=torch.float32).view(-1, 1)
    else:
        combinations: list = list(itertools.product(*ranges))
        X: Matrix = torch.tensor(combinations, dtype=torch.float32)

    y: Vector = func(X, **func_params).view(-1, 1)
    return X, y


def generate_data_mixed(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        split_ratio: float = 0.2,
        density_ratio: float = 0.8,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    n_dense = int(n_samples * density_ratio)
    n_sparse = n_samples - n_dense
    if n_dense < 2 or n_sparse < 1:
        raise ValueError("n_samples is too small for the given density_ratio")

    ranges = []
    for i in range(n_dim):
        l, u = lower[i], upper[i]
        split_val = l + (u - l) * split_ratio
        x_dense = torch.linspace(l, split_val, n_dense)
        x_sparse = torch.linspace(split_val, u, n_sparse + 1)[1:]
        ranges.append(torch.cat([x_dense, x_sparse]).tolist())

    if n_dim == 1:
        X: Matrix = torch.tensor(ranges[0], dtype=torch.float32).view(-1, 1)
    else:
        combinations: list = list(itertools.product(*ranges))
        X: Matrix = torch.tensor(combinations, dtype=torch.float32)

    y: Vector = func(X, **func_params).view(-1, 1)
    return X, y


def generate_data_exponential(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 1000,
        n_dim: int = 1,
        decay_rate: float = 5.0,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    
    torch.tensor(lower, dtype=torch.float32)
    torch.tensor(upper, dtype=torch.float32)
    U = torch.rand(n_samples, n_dim)
    X_parts = []
    for i in range(n_dim):
        R = upper[i] - lower[i]
        max_val = 1.0 - torch.exp(torch.tensor(-decay_rate * R))
        xi = - (1.0 / decay_rate) * torch.log(1.0 - U[:, i] * max_val)
        X_parts.append(xi + lower[i])
        
    X = torch.stack(X_parts, dim=1)
    y: Vector = func(X, **func_params).view(-1, 1)
    
    return X, y


def generate_data(
        dis_type: str,
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        decay_rate: float = 5.0, ### EXP DISTRIBUTION REQ
        split_ratio: float = 0.2,   ### MIX DISTRIBUTION REQ
        density_ratio: float = 0.8, ### MIX DISTRIBUTION REQ
        epsilon: float = 1e-6, ### LOG DISTRIBUTION REQ
        **func_params: Any
) -> tuple[Matrix, Vector]:

    if dis_type == "exp" or dis_type == "std":
        set_size = n_samples
    else:
        root: float = n_samples ** (1/n_dim)
        set_size = math.ceil(root)

    if dis_type == "exp":              
        return generate_data_exponential(
            func=func,
            lower=lower,
            upper=upper,
            n_samples=set_size,
            n_dim=n_dim,
            decay_rate=decay_rate,
            **func_params
        )                       
    elif dis_type == "log":
        return generate_data_logarithmic(
            func=func,
            lower=lower,
            upper=upper,
            n_samples=set_size,
            n_dim=n_dim,
            epsilon=epsilon,
            **func_params
        )                       
    elif dis_type == "mix":        
        return generate_data_mixed(
            func=func,
            lower=lower,
            upper=upper,
            n_samples=set_size,
            n_dim=n_dim,
            split_ratio=split_ratio,
            density_ratio=density_ratio,
            **func_params
        )                           
    elif dis_type == "std":
        return generate_data_uniform(
            func=func,
            lower=lower,
            upper=upper,
            n_samples=set_size,
            n_dim=n_dim,
            **func_params
        )                       
    elif dis_type == "grd":
        return generate_data_unfirm_grid(
            func=func,
            lower=lower,
            upper=upper,
            n_samples=set_size,
            n_dim=n_dim,
            **func_params
        )
    else:
        raise ValueError(f"Unknown distribution type: {dis_type}")
