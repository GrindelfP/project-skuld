#########################################################
  ##### SCIPY.INTEGRATE-BASED INTEGRATION METHODS #####
#########################################################

from scipy.integrate import quad, nquad
import numpy as np
import torch


def integrate_1d_quad(func, a, b, *params):
    """
    Integrates a function, handling cases with and without additional parameters.

    :param func: The function to integrate.
    :param a: Lower bound of integration.
    :param b: Upper bound of integration.
    :param *params: Optional additional parameters to pass to the function.
    
    :returns: The integration result and error.
    """
    if params:
        def wrapped_func(x):
            return func(x, *params)
        result = quad(wrapped_func, a, b)
    else:
        result = quad(func, a, b)
    return result
    

def integrate_2d_nquad(func, lower, upper, *params):
    """
    Performs 2D integration using scipy.integrate.nquad, handling cases with and without parameters.

    :param func: The 2D function to integrate.
    :param lower: List of lower bounds [lower_x, lower_y].
    :param upper: List of upper bounds [upper_x, upper_y].
    :param *params: Optional additional parameters to pass to the function.
    
    :returns: The integration result.
    """
    if params:
        def func_for_nquad(x, y):
            return func([x, y], *params)
    else:
        def func_for_nquad(x, y):
            return func([x, y])

    return nquad(func_for_nquad, [[lower[0], upper[0]], [lower[1], upper[1]]])[0]
    

def integrate_2d_trapz(func, lower, upper, num_points=100, *params):
    x = np.linspace(lower[0], upper[0], num_points)
    y = np.linspace(lower[1], upper[1], num_points)
    xv, yv = np.meshgrid(x, y)
    points = np.stack((xv.flatten(), yv.flatten()), axis=-1)
    z = func(torch.tensor(points), *params).numpy().reshape(num_points, num_points)
    
    return np.trapz(np.trapz(z, x), y)

