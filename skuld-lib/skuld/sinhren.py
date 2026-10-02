##############################################################################
#  SINHREN-based Neural Numerical Integration
#
#  sinh activation: sinh(omega_0 * linear(x)) — with omega_0 scaling.
#  sinh is unbounded (unlike sin), monotonic, and odd.
#  sinh(0) = 0, sinh'(x) = cosh(x), sinh''(x) = sinh(x), sinh'''(x) = cosh(x)
#
#  omega_0 scales the pre-activation to control sinh growth.
#  Init bounds are divided by omega_0 to keep pre-activations small.
#  Gradient clipping is lowered to 1.0 to handle exploding cosh gradients.
##############################################################################
import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


##############################################################################
#  SINHREN Architecture
##############################################################################

class SinhrenLayer(nn.Module):
    """Single SINHREN linear + sinh layer with omega_0 scaling."""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 1.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
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
        return torch.sinh(self.omega_0 * self.linear(x))


class SinhrenPrimitiveNet(nn.Module):
    """
    SINHREN-based primitive network N(s, u) ~ F(s; u).

    Architecture: SINHREN hidden layers + linear output.
    Learnable output scale on the final layer.
    omega_0 scales the pre-activation to control sinh growth.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0: float = 1.0,
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
            layers.append(SinhrenLayer(in_dim, h,
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
#  Mixed Partial Derivative via Autograd
##############################################################################

def mixed_partial_3(net: SinhrenPrimitiveNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """d3N/du1du2du3 computed by sequential autograd."""
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)

    inp = torch.cat([s, u], dim=1)
    N_out = net(inp).squeeze(-1)

    ones_N = torch.ones_like(N_out)

    g1 = torch.autograd.grad(
        N_out, u,
        grad_outputs=ones_N,
        create_graph=True, retain_graph=True,
    )[0][:, 0]

    g12 = torch.autograd.grad(
        g1, u,
        grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]

    g123 = torch.autograd.grad(
        g12, u,
        grad_outputs=torch.ones_like(g12),
        create_graph=True,
    )[0][:, 2]

    return g123


##############################################################################
#  SinhrenIntegrator: train + evaluate
##############################################################################

class SinhrenIntegrator:
    """
    SINHREN for numerical integration (Maitre et al. approach).

    Trains a SinhrenPrimitiveNet to approximate the antiderivative,
    then evaluates the integral via an alternating-sign corner sum over
    the unit hypercube [0,1]^n_int_vars.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0: float = 1.0,
                 output_scale: float = 1.0):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = SinhrenPrimitiveNet(
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
        """Value of integrand at u=(0.5, ..., 0.5)."""
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
              verbose_every: int = 500,
              weight_decay: float = 0.0) -> tuple:
        """Train the SINHREN to approximate the antiderivative."""
        if device is None:
            device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
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
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
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
        """Evaluate the integral via corner sum."""
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
