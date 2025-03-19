import scipy.integrate
import numpy as np
import torch


def integrate_2d_nquad(func, lower, upper):
    def func_for_nquad(x, y):
        return func([x, y])
    
    return scipy.integrate.nquad(func_for_nquad, [[lower[0], upper[0]], [lower[1], upper[1]]])[0]


def integrate_2d_trapz(func, lower, upper, num_points=100):
    x = np.linspace(lower[0], upper[0], num_points)
    y = np.linspace(lower[1], upper[1], num_points)
    xv, yv = np.meshgrid(x, y)
    points = np.stack((xv.flatten(), yv.flatten()), axis=-1)
    z = func(torch.tensor(points)).numpy().reshape(num_points, num_points)
    
    return np.trapz(np.trapz(z, x), y)

