##############################################################################
#  GAUSSIAN-based Neural Numerical Integration
#
#  Gaussian activation: exp(-(omega_0 * linear(x))^2)
#  - Localized bump, positive, C^inf
#  - Antiderivative is erf(x) — bounded, smooth
#  - Derivatives involve Hermite polynomials
#  - Classic RBF network activation
#
#  The "next bump" after sech in the localization hierarchy.
##############################################################################
import math
from typing import Callable

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  GAUSSIAN Architecture
##############################################################################

class GaussianLayer(nn.Module):
    """Single GAUSSIAN linear + exp(-x^2) layer."""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, fan_in: int):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.omega_0 * self.linear(x)
        return torch.exp(-z * z)


class GaussianPrimitiveNet(nn.Module):
    """
    GAUSSIAN-based primitive network N(s, u) ~ F(s; u).

    Architecture: GAUSSIAN hidden layers + linear output.
    Learnable output scale on the final layer.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.omega_0 = omega_0

        layers = []
        in_dim = n_params + n_int_vars
        for i, h in enumerate(hidden_sizes):
            layers.append(GaussianLayer(in_dim, h,
                                        omega_0=omega_0,
                                        is_first=(i == 0)))
            in_dim = h

        # Final linear layer — no activation, learnable output scale
        final = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.output_scale


##############################################################################
#  GaussianIntegrator: train + evaluate
##############################################################################

class GaussianIntegrator(MaitreIntegrator):
    """
    GAUSSIAN for numerical integration (Maitre et al. approach).

    Trains a GaussianPrimitiveNet to approximate the antiderivative,
    then evaluates the integral via an alternating-sign corner sum over
    the unit hypercube [0,1]^n_int_vars.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = GaussianPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            output_scale=output_scale,
        )

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 5000,
              n_per_param: int = 512,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 500,
              weight_decay: float = 0.0) -> tuple:
        """
        Train the GAUSSIAN to approximate the antiderivative.

        Returns: (loss_history, norm_cache)
        """
        return self._fit(integrand_fn, param_sets, n_epochs, n_per_param,
                         lr, device, verbose_every, weight_decay=weight_decay)
