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
import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


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
#  Mixed Partial Derivative via Autograd
##############################################################################

def mixed_partial_3(net: WirePrimitiveNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """Compute d^3 N / (du1 du2 du3) by sequential autograd."""
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)

    inp = torch.cat([s, u], dim=1)
    N_out = net(inp).squeeze(-1)

    ones_N = torch.ones_like(N_out)

    g1 = torch.autograd.grad(
        N_out, u, grad_outputs=ones_N,
        create_graph=True, retain_graph=True,
    )[0][:, 0]

    g12 = torch.autograd.grad(
        g1, u, grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]

    g123 = torch.autograd.grad(
        g12, u, grad_outputs=torch.ones_like(g12),
        create_graph=True,
    )[0][:, 2]

    return g123


##############################################################################
#  WireIntegrator: train + evaluate
##############################################################################

class WireIntegrator:
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

    @property
    def n_weights(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def make_batch(self,
                   integrand_fn: Callable,
                   param_sets: list,
                   n_per_param: int,
                   device: torch.device,
                   norm_cache: dict = None) -> tuple:
        """Generate a training batch."""
        if norm_cache is None:
            norm_cache = {}

        all_xu, all_f = [], []

        for params in param_sets:
            params_key = tuple(params)
            if params_key not in norm_cache:
                norm_cache[params_key] = self._center_value(integrand_fn, self.n_int_vars, *params)
            fc = norm_cache[params_key]

            u = torch.rand(n_per_param, self.n_int_vars)
            f = integrand_fn(u, *params) / fc

            s = torch.tensor(params, dtype=u.dtype).expand(n_per_param, -1)
            xu = torch.cat([s, u], dim=1)

            all_xu.append(xu)
            all_f.append(f)

        batch_xu = torch.cat(all_xu, dim=0).to(device)
        f_tilde = torch.cat(all_f, dim=0).to(device)
        return batch_xu, f_tilde, norm_cache

    @staticmethod
    def _center_value(integrand_fn, n_int_vars: int, *params) -> float:
        """Value of integrand at u=(0.5, ..., 0.5) — used for normalization."""
        uc = torch.full((1, n_int_vars), 0.5)
        fc = integrand_fn(uc, *params).item()
        return fc if abs(fc) > 1e-30 else 1.0

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 5000,
              n_per_param: int = 512,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 500) -> tuple:
        """Train the WIRE network to approximate the antiderivative.

        Returns: (loss_history, norm_cache)
        """
        if device is None:
            device = torch.device("cpu")

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, n_epochs // 4), T_mult=1, eta_min=lr / 50
        )
        loss_fn = nn.MSELoss()
        self.net.to(device)
        self.net.train()

        history = []
        norm_cache = {}
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            batch_xu, f_tilde, norm_cache = self.make_batch(
                integrand_fn, param_sets, n_per_param, device, norm_cache
            )

            dN = mixed_partial_3(self.net, batch_xu)
            loss = loss_fn(dN, f_tilde)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step(epoch)

            lv = loss.item()
            history.append(lv)

            if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:5d}/{n_epochs}  "
                      f"loss={lv:.4e}  lr={lr_now:.2e}  "
                      f"({elapsed:.1f}s)")

        return history, norm_cache

    def integrate(self,
                  params: tuple,
                  norm_cache: dict = None,
                  device: torch.device = None) -> float:
        """Evaluate the integral for a given parameter set via corner sum."""
        if device is None:
            device = torch.device("cpu")

        self.net.eval()
        self.net.to(device)

        if norm_cache and tuple(params) in norm_cache:
            fc = norm_cache[tuple(params)]
        else:
            fc = 1.0

        s_row = torch.tensor([params], dtype=torch.float32).to(device)
        I_tilde = 0.0

        with torch.no_grad():
            for corner in itertools.product([0.0, 1.0], repeat=self.n_int_vars):
                sign = (-1) ** (self.n_int_vars - sum(corner))
                u_t = torch.tensor([corner], dtype=torch.float32).to(device)
                inp = torch.cat([s_row, u_t], dim=1)
                I_tilde += sign * self.net(inp).item()

        return float(I_tilde * fc)
