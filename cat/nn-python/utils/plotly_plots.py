from typing import Tuple, Any, TypeAlias, Annotated
import numpy as np
import torch
from torch import Tensor
from typing_extensions import Callable
import plotly.graph_objects as go

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]

#########################################################################
###                        PLOTTING FUNCTIONS                         ###
#########################################################################
def plot_surface(
    func: Callable[[Matrix, ...], Tensor],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    num_points: int = 50,
    title: str = "3D Plot",
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
    **func_kwargs,
) -> go.Figure:
    """
    Generates a 3D surface utils of a given 2D function.

    Args:
        func: The 2D function to utils. It should take a tensor of shape (N, 2)
              as the first argument, representing the (x, y) coordinates, and
              return a tensor of shape (N,) representing the function values.
        x_range: A tuple (min_x, max_x) defining the range of x values.
        y_range: A tuple (min_y, max_y) defining the range of y values.
        num_points: The number of points to use along each axis for creating the meshgrid.
        title: The title of the utils.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        z_label: The label for the z-axis.
        **func_kwargs: Additional keyword arguments to pass to the plotting function.

    Returns:
        A Plotly Figure object containing the 3D surface utils.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    Z = func(points, **func_kwargs).numpy().reshape(X.shape)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        )
    )
    return fig


def plot_line(
    func: Callable[[np.ndarray, ...], np.ndarray],
    x_range: Tuple[float, float],
    num_points: int = 50,
    title: str = "1D Plot",
    x_label: str = "X",
    y_label: str = "Y",
    **func_kwargs: Any,
) -> go.Figure:
    """
    Creates a 1D function utils using Plotly.

    Args:
        func: A callable that takes a NumPy array (representing x values)
              and optional keyword arguments, and returns a NumPy array
              (representing the corresponding y values).
        x_range: A tuple containing the start and end values for the x-axis.
        num_points: The number of points to sample within the x-range.
        title: The title of the utils.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        **func_kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        A Plotly Figure object.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x, **func_kwargs)

    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines')])

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )

    return fig
