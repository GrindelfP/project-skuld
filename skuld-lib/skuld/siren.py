##############################################################################
#  SIREN-based Neural Numerical Integration
#
#  Sitzmann et al. 2020: "Implicit Neural Representations with Periodic
#  Activation Functions" — https://arxiv.org/abs/2006.09661
#
#  This module implements the Maitre et al. approach: a SIREN is trained
#  to approximate the antiderivative F such that ∂³F/∂u₁∂u₂∂u₃ ≈ f̃,
#  then the integral is evaluated by an alternating-sign corner sum
#  over the unit hypercube.
##############################################################################
import math

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  SIREN Architecture
##############################################################################

class SirenLayer(nn.Module):
    """Single SIREN linear + sin layer."""

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
        return torch.sin(self.omega_0 * self.linear(x))


class SirenPrimitiveNet(nn.Module):
    """
    SIREN-based primitive network N(s, u) ≈ F(s; u).

    Approximates the antiderivative such that
        ∂ⁿN/∂u₁...∂uₙ  ≈  f̃(s; u).

    Architecture: SIREN hidden layers + linear output (no final sin).
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
            layers.append(SirenLayer(in_dim, h,
                                     omega_0=omega_0,
                                     is_first=(i == 0)))
            in_dim = h

        # Final linear layer — no sin activation (output is a scalar primitive)
        final = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
            # Scale output so that derivative values are O(1) initially.
            # Each differentiation pulls down a factor ω₀.
            final.weight.data *= output_scale
            final.bias.data *= output_scale
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


##############################################################################
#  SirenIntegrator: train + evaluate
##############################################################################

class SirenIntegrator(MaitreIntegrator):
    """
    Application of SIREN for numerical integration (Maitre et al. approach).

    Trains a SirenPrimitiveNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = SirenPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            output_scale=output_scale,
        )
