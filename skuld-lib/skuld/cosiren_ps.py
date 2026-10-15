##############################################################################
#  COSIREN-PS: COSIREN with Learnable Phase Shift
#
#  Same as COSIREN but each layer has a learnable phase parameter phi:
#      cos(omega_0 * linear(x) + phi)
#  phi is initialized at 0 (pure COSIREN) and learned during training.
##############################################################################
import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


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
#  Mixed Partial Derivative via Autograd
##############################################################################

def mixed_partial_n(net: CosirenPSPrimitiveNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    d^n N / du_1 ... du_n, computed by sequential autograd, where n is the
    number of integration variables.

    This replaces a former `mixed_partial_3`, which took exactly three
    successive autograd passes over columns 0, 1, 2 regardless of the actual
    dimension. That was silently wrong for any n != 3: the network was trained
    to match a third-order partial derivative and then evaluated with an
    n-dimensional corner sum. No exception was raised, the loss decreased
    happily, and the result was confidently wrong.

    The loop below is the same computation for arbitrary n; at n = 3 it is
    identical to the previous implementation.
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)

    dim = u.shape[1]
    declared = getattr(net, "n_int_vars", dim)
    if dim != declared:
        raise ValueError(
            f"batch has {dim} integration variables but net declares {declared}"
        )
    if dim < 1:
        raise ValueError(f"need at least 1 integration variable, got {dim}")

    g = net(torch.cat([s, u], dim=1)).squeeze(-1)

    for i in range(dim):
        # retain_graph MUST stay True even on the final pass. In
        # torch.autograd.grad an omitted retain_graph defaults to the value of
        # create_graph, so the previous three-call implementation — which
        # omitted it on the last call — was in fact retaining the graph, and
        # loss.backward() depends on that. Passing False here frees a graph the
        # backward pass still needs and raises "backward through the graph a
        # second time".
        g = torch.autograd.grad(
            g, u,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0][:, i]

    return g


# Backwards-compatible alias: the name predates n-dimensional support.
mixed_partial_3 = mixed_partial_n


##############################################################################
#  CosirenPSIntegrator: train + evaluate
##############################################################################

class CosirenPSIntegrator:
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
              verbose_every: int = 500) -> tuple:
        """Train the COSIREN-PS to approximate the antiderivative."""
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
