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
import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


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
#  Mixed Partial Derivative via Autograd
##############################################################################

def mixed_partial_3(net: SirenPrimitiveNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    ∂³N/∂u₁∂u₂∂u₃ computed by sequential autograd.

    batch: (N, n_params + 3)
        - [:, :n_params]  — s (not differentiated)
        - [:, n_params:]  — u₁,u₂,u₃ (differentiated)
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)

    inp = torch.cat([s, u], dim=1)
    N_out = net(inp).squeeze(-1)

    ones_N = torch.ones_like(N_out)

    # ∂N/∂u₁
    g1 = torch.autograd.grad(
        N_out, u,
        grad_outputs=ones_N,
        create_graph=True, retain_graph=True,
    )[0][:, 0]

    # ∂²N/∂u₁∂u₂
    g12 = torch.autograd.grad(
        g1, u,
        grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]

    # ∂³N/∂u₁∂u₂∂u₃
    g123 = torch.autograd.grad(
        g12, u,
        grad_outputs=torch.ones_like(g12),
        create_graph=True,
    )[0][:, 2]

    return g123


##############################################################################
#  SirenIntegrator: train + evaluate
##############################################################################

class SirenIntegrator:
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

    @property
    def n_weights(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def make_batch(self,
                   integrand_fn: Callable,
                   param_sets: list,
                   n_per_param: int,
                   device: torch.device,
                   norm_cache: dict = None) -> tuple:
        """
        Generate a training batch.

        integrand_fn: callable(u1, u2, u3, *params) -> torch.Tensor
            The integrand in the unit cube (already Jacobian-adjusted).
        param_sets: list of tuples [(a, b, m, n), ...]
        n_per_param: number of random samples per parameter set
        device: torch device
        norm_cache: dict for caching normalization constants

        Returns: (batch_xu, f_tilde, norm_cache)
        """
        if norm_cache is None:
            norm_cache = {}

        all_xu, all_f = [], []

        for params in param_sets:
            # Compute normalization constant (center value)
            params_key = tuple(params)
            if params_key not in norm_cache:
                norm_cache[params_key] = self._center_value(integrand_fn, self.n_int_vars, *params)
            fc = norm_cache[params_key]

            # Generate random samples in unit cube
            u = torch.rand(n_per_param, self.n_int_vars)

            # Evaluate integrand
            f = integrand_fn(u, *params) / fc

            # Build input: [params, u]
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
        """
        Train the SIREN to approximate the antiderivative.

        Returns: (loss_history, norm_cache)
        """
        if device is None:
            device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr / 10
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
            scheduler.step()

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
        """
        Evaluate the integral for a given parameter set via corner sum.

        params: tuple (a, b, m, n) — the fixed parameters
        norm_cache: dict with normalization constants from training
        device: torch device

        Returns: the integral value (normalized, needs descaling)
        """
        if device is None:
            device = torch.device("cpu")

        self.net.eval()
        self.net.to(device)

        if norm_cache and tuple(params) in norm_cache:
            fc = norm_cache[tuple(params)]
        else:
            # This shouldn't happen if you trained first, but handle it
            fc = 1.0

        s_row = torch.tensor([params], dtype=torch.float32).to(device)
        I_tilde = 0.0

        with torch.no_grad():
            for corner in itertools.product([0.0, 1.0], repeat=self.n_int_vars):
                # Alternating sign: (-1)^(n_zeros)
                sign = (-1) ** (self.n_int_vars - sum(corner))
                u_t = torch.tensor([corner], dtype=torch.float32).to(device)
                inp = torch.cat([s_row, u_t], dim=1)
                I_tilde += sign * self.net(inp).item()

        return float(I_tilde * fc)
