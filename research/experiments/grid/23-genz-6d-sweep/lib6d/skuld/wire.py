##############################################################################
#  WIRE-based Neural Numerical Integration
#
#  Saragadam et al. 2023: "Wire: Wavelet Implicit Neural Representations"
#  https://arxiv.org/abs/2301.05187
#
#  WIRE uses complex Gabor / Morlet wavelet activations:
#      output = [cos(z)*env, sin(z)*env]
#      z   = omega_0 * (W*x + b)
#      env = exp(-z^2 / (2*sigma_0^2))
#
#  The Gaussian envelope provides local support, making WIRE well-suited
#  for integrands with localized structure (e.g., exponential decay).
##############################################################################
import math

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  WIRE Architecture
##############################################################################

class WireLayer(nn.Module):
    """Single Wire (complex Gabor / Morlet wavelet) layer.

    Output dimension is 2 * out_features because we return
    [Re(Gabor), Im(Gabor)] as real tensors.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 omega_0: float = 10.0,
                 sigma_0: float = 10.0,
                 is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.is_first = is_first
        self.out_features = out_features

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
        envelope = torch.exp(-z ** 2 / (2.0 * self.sigma_0 ** 2))
        real = torch.cos(z) * envelope
        imag = torch.sin(z) * envelope
        return torch.cat([real, imag], dim=-1)


class WireResidualBlock(nn.Module):
    """Two Wire layers with a linear residual skip connection."""

    def __init__(self, features: int, omega_0: float, sigma_0: float):
        super().__init__()
        half = features // 2
        self.layer1 = WireLayer(features, half, omega_0=omega_0, sigma_0=sigma_0)
        self.layer2 = WireLayer(features, half, omega_0=omega_0, sigma_0=sigma_0)
        self.skip = nn.Linear(features, features, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.skip.weight) if features == features else \
                nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(x) + self.layer2(x) + self.skip(x)


class WirePrimitiveNet(nn.Module):
    """Wire-based primitive network N(s; u) ≈ F(s; u).

    Approximates the antiderivative such that
        d^3 N / (du1 du2 du3) ≈ f_tilde(s; u).
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 entry_width: int = 64,
                 n_blocks: int = 3,
                 omega_0: float = 10.0,
                 sigma_0: float = 10.0,
                 output_scale: float = 1.0):
        super().__init__()
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0

        in_dim = n_params + n_int_vars
        mid_dim = 2 * entry_width

        self.entry = WireLayer(in_dim, entry_width,
                               omega_0=omega_0, sigma_0=sigma_0,
                               is_first=True)

        self.blocks = nn.ModuleList([
            WireResidualBlock(mid_dim, omega_0=omega_0, sigma_0=sigma_0)
            for _ in range(n_blocks)
        ])

        self.out = nn.Linear(mid_dim, 1)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.out.weight)
            self.out.weight.data *= output_scale
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.entry(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


##############################################################################
#  WireIntegrator: train + evaluate
##############################################################################

class WireIntegrator(MaitreIntegrator):
    """Application of WIRE for numerical integration (Maitre et al. approach).

    Trains a WirePrimitiveNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 entry_width: int = 64,
                 n_blocks: int = 3,
                 omega_0: float = 10.0,
                 sigma_0: float = 10.0,
                 output_scale: float = 1.0):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = WirePrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            entry_width=entry_width,
            n_blocks=n_blocks,
            omega_0=omega_0,
            sigma_0=sigma_0,
            output_scale=output_scale,
        )

    def _make_optimizer(self, lr: float, weight_decay: float):
        # WIRE always uses AdamW with weight_decay=1e-5.
        return torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)

    def _make_scheduler(self, optimizer, n_epochs: int, lr: float):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, n_epochs // 4), T_mult=1, eta_min=lr / 50
        )

    def _scheduler_step(self, scheduler, epoch: int) -> None:
        scheduler.step(epoch)
