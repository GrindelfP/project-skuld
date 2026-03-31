import time
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import erf

from skuld import *
from skuld.model import set_global_device

EPOCHS: Final[int] = 500


def genz_oscillatory_torch(X, c=torch.tensor([1.0, 1.5, 2.0]), d=0.1):
    c = c.to(X.device)
    phase = 2 * np.pi * d + torch.matmul(X, c)
    return torch.cos(phase).unsqueeze(1)


def genz_product_peak_torch(X, c=0.5, d=torch.tensor([0.2, 0.4, 0.6])):
    d = d.to(X.device)
    inv_c2 = 1.0 / (c * c)
    diffs = X - d
    denom = inv_c2 + diffs * diffs
    return torch.prod(1.0 / denom, dim=1, keepdim=True)


def genz_corner_peak_torch(X, c=torch.tensor([1.0, 1.0, 1.0])):
    c = c.to(X.device)
    s = 1.0 + torch.matmul(X, c)
    return (s ** -4).unsqueeze(1)


def genz_gaussian_torch(X, _c=1.0, d=0.5):
    diff = X - d
    return torch.exp(-torch.sum(diff * diff, dim=1, keepdim=True))


def genz_c0_torch(X, c=1.0, d=0.5):
    diff = torch.abs(X - d)
    return torch.exp(-c * torch.sum(diff, dim=1, keepdim=True))


def genz_oscillatory(x, c=np.array([1.0, 1.5, 2.0]), d=0.1):
    return np.cos(2 * np.pi * d + np.dot(c, x.T)).reshape(-1, 1)


def I_oscillatory(c=np.array([1.0, 1.5, 2.0]), d=0.1):
    term = np.sin(c[0]) * np.sin(c[1]) * np.sin(c[2]) / (c[0] * c[1] * c[2])
    return term * np.cos(2 * np.pi * d)


def genz_product_peak(x, c=0.5, d=np.array([0.2, 0.4, 0.6])):
    inv_c2 = 1.0 / (c * c)
    diffs = x - d
    denom = inv_c2 + diffs * diffs
    return np.prod(1.0 / denom, axis=1, keepdims=True)


def I_product_peak(c=0.5, d=np.array([0.2, 0.4, 0.6])):
    terms = c * (np.arctan(c * (1 - d)) + np.arctan(c * d))
    return np.prod(terms)


def genz_corner_peak(x, c=np.array([1.0, 1.0, 1.0])):
    s = 1.0 + np.dot(x, c)
    return (s ** -4).reshape(-1, 1)


def I_corner_peak():
    return 21.0 / 128.0


def genz_gaussian(x, _c=1.0, d=0.5):
    diff = x - d
    return np.exp(-np.sum(diff * diff, axis=1, keepdims=True))


def I_gaussian(_c=1.0, _d=0.5):
    val_1d = np.sqrt(np.pi) * erf(0.5)
    return val_1d ** 3


def genz_c0(x, c=1.0, d=0.5):
    diff = np.abs(x - d)
    return np.exp(-c * np.sum(diff, axis=1, keepdims=True))


def I_c0(_c=1.0, _d=0.5):
    val_1d = 2.0 * (1.0 - np.exp(-0.5))
    return val_1d ** 3


def main(device: str) -> None:
    genz_functions = [
        (genz_oscillatory_torch, I_oscillatory(), "Oscillatory"),
        (genz_product_peak_torch, I_product_peak(), "Product Peak"),
        (genz_corner_peak_torch, I_corner_peak(), "Corner Peak"),
        (genz_gaussian_torch, I_gaussian(), "Gaussian"),
        (genz_c0_torch, I_c0(), "C⁰ (Exponential)"),
    ]

    # sample_sizes_k = np.arange(10, 51, 10)
    # sample_sizes = [k ** 3 for k in sample_sizes_k]
    sample_sizes = [10000]
    np.random.seed(2025)
    all_errors = []

    for func, I_exact, name in genz_functions:
        print(f"Testing: {name}")
        errors = []

        for N in sample_sizes:
            print(f"  N = {N} ({int(round(N ** (1 / 3)))}³)")
            X, y = generate_data(dis_type="std", func=func, lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0], n_samples=N,
                                 n_dim=3)
            X_scaled, y_scaled = scale_data(X_init=X, y_init=y, frange=(0, 1))
            x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)

            model = init_model(input_size=3, hidden_size=1024)
            set_global_device(device)
            model.compile_default(learning_rate=0.05)
            model.fit(x_train=x_train, y_train=y_train, epochs=EPOCHS, verbose=True)

            nni_scaled = NeuralNumericalIntegration.integrate(model=model, n_dims=3, unit_cube=True, alphas=[],
                                                              betas=[])
            nni_result = descale_result(
                nni_scaled, X_init=X, y_init=y, frange=(0, 1), n_dim=3
            )
            error = abs(nni_result - I_exact)
            errors.append(error)
            print(f"    I_NNI = {nni_result:.6e}, I_exact = {I_exact:.6e}, |err| = {error:.3e}")

        all_errors.append(errors)
        print()


if __name__ == "__main__":
    start1: float
    start2: float
    end1: float
    end2: float

    start1 = time.time()
    main("mps")
    end1 = time.time()

    start2 = time.time()
    main("cpu")
    end2 = time.time()

    print(f"MPS integration test took {end1 - start1:.2f} seconds")
    print(f"CPU integration test took {end2 - start2:.2f} seconds")
