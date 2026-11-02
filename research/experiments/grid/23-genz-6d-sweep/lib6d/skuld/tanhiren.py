##############################################################################
#  TANHIREN-based Neural Numerical Integration
#
#  Same architecture as SIREN (Sitzmann et al. 2020) but with hyperbolic
#  tangent activation: tanh(omega_0 * linear(x)).
#
#  tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) — bounded (-1,1), monotonic, odd.
#  Bounded like sin/sech, but saturating (S-shaped) rather than periodic.
#  Tests the "integration ladder" hypothesis: sin (wave) -> sech (bump) ->
#  tanh (soft step) -> gd (step).
##############################################################################
import math

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  TANHIREN Architecture
##############################################################################

class TanhrenLayer(nn.Module):
    """Single TANHIREN linear + tanh layer."""

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
        return torch.tanh(self.omega_0 * self.linear(x))


class TanhrenPrimitiveNet(nn.Module):
    """
    TANHIREN-based primitive network N(s, u) ~ F(s; u).

    Architecture: TANHIREN hidden layers + linear output (no final tanh).
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
            layers.append(TanhrenLayer(in_dim, h,
                                     omega_0=omega_0,
                                     is_first=(i == 0)))
            in_dim = h

        # Final linear layer — no tanh activation (output is a scalar primitive)
        final = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
            final.weight.data *= output_scale
            final.bias.data *= output_scale
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


##############################################################################
#  TanhrenIntegrator: train + evaluate
##############################################################################

class TanhrenIntegrator(MaitreIntegrator):
    """
    Application of TANHIREN for numerical integration (Maitre et al. approach).

    Trains a TanhrenPrimitiveNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over
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
        self.net = TanhrenPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            output_scale=output_scale,
        )
