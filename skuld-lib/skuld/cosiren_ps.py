##############################################################################
#  COSIREN-PS: COSIREN with Learnable Phase Shift
#
#  Same as COSIREN but each layer has a learnable phase parameter phi:
#      cos(omega_0 * linear(x) + phi)
#  phi is initialized at 0 (pure COSIREN) and learned during training.
##############################################################################
import math

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  COSIREN-PS Architecture
##############################################################################

class CosirenPSLayer(nn.Module):
    """COSIREN linear + cos layer with learnable phase shift."""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.phi = nn.Parameter(torch.zeros(1))
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
        return torch.cos(self.omega_0 * self.linear(x) + self.phi)


class CosirenPSPrimitiveNet(nn.Module):
    """
    COSIREN-PS primitive network N(s, u) ≈ F(s; u).

    Architecture: COSIREN-PS hidden layers + linear output.
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
            layers.append(CosirenPSLayer(in_dim, h,
                                         omega_0=omega_0,
                                         is_first=(i == 0)))
            in_dim = h

        # Final linear layer — no activation
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
#  CosirenPSIntegrator: train + evaluate
##############################################################################

class CosirenPSIntegrator(MaitreIntegrator):
    """
    COSIREN with learnable phase shift for numerical integration.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = CosirenPSPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            output_scale=output_scale,
        )
