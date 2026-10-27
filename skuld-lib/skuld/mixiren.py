##############################################################################
#  MIXIREN-based Neural Numerical Integration
#
#  Mixed-activation architecture: [sech, sin, sech]
#
#  Hypothesis: the middle sin layer adds a smooth, non-local basis that
#  sech alone cannot provide. Since the integrand's antiderivative is not
#  oscillatory, the sin layer acts as a global function combiner rather
#  than an oscillator — its low frequency (omega_0=1) lets it capture
#  broad structure while the sech layers handle local features.
#
#  Design (grilled 2026-10-01):
#    Layer 0: sech(omega_0=30, learnable)  — local feature extraction
#    Layer 1: sin(omega_0=1, fixed)        — smooth non-local basis
#    Layer 2: sech(omega_0=30, learnable)  — local feature extraction
#    Output: linear with output_scale = 1/900
#
#  Output scale logic: mixed partial passes through 3 layers.
#    sech(30) contributes 30, sin(1) contributes 1, sech(30) contributes 30.
#    Total: 30 * 1 * 30 = 900.  output_scale = 1/900.
##############################################################################
import math
from typing import Callable

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  MIXIREN Architecture
##############################################################################

class MixirenLayer(nn.Module):
    """Single MIXIREN linear + activation layer.

    activation: 'sech' or 'sin'
    learnable_omega: if True, omega_0 is an nn.Parameter (learnable);
                     if False, omega_0 is a fixed float.
    """

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float, activation: str,
                 is_first: bool = False, learnable_omega: bool = True):
        super().__init__()
        self.activation = activation
        self.is_first = is_first
        self.learnable_omega = learnable_omega

        if learnable_omega:
            self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))
        else:
            self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features, omega_0)

    def _init_weights(self, fan_in: int, omega_0: float):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == 'sech':
            return 1.0 / torch.cosh(self.omega_0 * self.linear(x))
        elif self.activation == 'sin':
            return torch.sin(self.omega_0 * self.linear(x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class MixirenPrimitiveNet(nn.Module):
    """
    MIXIREN-based primitive network N(s, u) ~ F(s; u).

    Architecture: [sech, sin, sech] hidden layers + linear output.
    sech layers have learnable omega_0, sin layer has fixed omega_0.
    Learnable output scale on the final layer.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0_sech: float = 30.0,
                 omega_0_sin: float = 1.0,
                 output_scale: float = 1.0 / 900):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 128]

        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.omega_0_sech = omega_0_sech
        self.omega_0_sin = omega_0_sin

        layers = []
        in_dim = n_params + n_int_vars
        for i, h in enumerate(hidden_sizes):
            if i == 1:
                # Middle layer: sin with fixed omega_0
                layers.append(MixirenLayer(in_dim, h,
                                            omega_0=omega_0_sin,
                                            activation='sin',
                                            is_first=(i == 0),
                                            learnable_omega=False))
            else:
                # Other layers: sech with learnable omega_0
                layers.append(MixirenLayer(in_dim, h,
                                            omega_0=omega_0_sech,
                                            activation='sech',
                                            is_first=(i == 0),
                                            learnable_omega=True))
            in_dim = h

        # Final linear layer — no activation, learnable output scale
        final = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega_0_sech
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.output_scale


##############################################################################
#  MixirenIntegrator: train + evaluate
##############################################################################

class MixirenIntegrator(MaitreIntegrator):
    """
    MIXIREN for numerical integration (Maitre et al. approach).

    Trains a MixirenPrimitiveNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0_sech: float = 30.0,
                 omega_0_sin: float = 1.0,
                 output_scale: float = 1.0 / 900):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = MixirenPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0_sech=omega_0_sech,
            omega_0_sin=omega_0_sin,
            output_scale=output_scale,
        )

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 8000,
              n_per_param: int = 1024,
              lr: float = 5e-4,
              device: torch.device = None,
              verbose_every: int = 500,
              weight_decay: float = 0.0) -> tuple:
        """
        Train the MIXIREN to approximate the antiderivative.

        Returns: (loss_history, norm_cache)
        """
        return self._fit(integrand_fn, param_sets, n_epochs, n_per_param,
                         lr, device, verbose_every, weight_decay=weight_decay)
