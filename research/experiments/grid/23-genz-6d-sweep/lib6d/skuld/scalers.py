##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################

from typing import TypeAlias, Annotated
from torch import Tensor
import torch

Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]


def scale_data(
        X_init: Matrix,
        y_init: Vector,
        frange: tuple[int, int] = (0, 1),
) -> tuple[Matrix, Vector]:
    """
    Scales dataset to frange. Works for any number of input columns —
    integration variables AND parameter columns are all scaled uniformly.
    The split between them is handled only in descale_result.
    """
    if not isinstance(X_init, torch.Tensor) or not isinstance(y_init, torch.Tensor):
        raise TypeError("Input X_init and y_init must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor (m, n_dim).")
    if y_init.ndim != 2 or y_init.shape[1] != 1:
        raise ValueError("y_init must be a 2D tensor (m, 1).")
    if X_init.shape[0] != y_init.shape[0]:
        raise ValueError("X_init and y_init must have the same number of rows.")

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
        n_dim: int = 1,
        n_int_dims: int | None = None,   # NEW: number of integration variables
) -> float:
    """
    Restores true value to the scaled integral.

    :param n_int_dims: how many of the n_dim columns are actual integration
                       variables. The remaining columns are parameters and must
                       NOT contribute to VS (the integration volume). Defaults
                       to n_dim for backward compatibility (old behaviour).
    """
    if not isinstance(X_init, torch.Tensor) or not isinstance(y_init, torch.Tensor):
        raise TypeError("Inputs must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor.")
    if y_init.ndim == 2 and y_init.shape[1] != 1:
        raise ValueError("y_init must have shape (m, 1).")

    # Backward-compatible default: treat all columns as integration dims
    if n_int_dims is None:
        n_int_dims = n_dim

    x_min = torch.min(X_init, dim=0).values   # shape: (n_dim,)
    x_max = torch.max(X_init, dim=0).values

    f_min = torch.min(y_init)
    f_max = torch.max(y_init)
    frange_size = frange[1] - frange[0]

    # Integration volume: only over the first n_int_dims columns
    VS  = torch.prod(x_max[:n_int_dims] - x_min[:n_int_dims])
    VSS = frange_size ** n_int_dims            # scaled volume (unit cube in frange)

    return (nni_scaled * (VS * (f_max - f_min) / (VSS * frange_size)) +
            (f_min - (f_max - f_min) / frange_size * frange[0]) * VS).item()
