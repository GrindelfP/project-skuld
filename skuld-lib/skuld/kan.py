##############################################################################
#  KAN-based Neural Numerical Integration
#
#  Liu et al. 2024: "KAN: Kolmogorov-Arnold Networks"
#  https://arxiv.org/abs/2404.19756
#
#  KAN replaces fixed activation functions with learnable univariate
#  B-spline functions on each edge:
#      phi_{i->j}(x_i) = w_ij * SiLU(x_i) + c_{ij} . B(x_i)
#
#  B-spline derivatives are piecewise polynomials of lower degree,
#  making autograd through three differentiations numerically stable.
##############################################################################
import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


##############################################################################
#  KAN Architecture
##############################################################################

class BSplineBasis(nn.Module):
    """Precomputes and evaluates a uniform B-spline basis over [-1, 1]."""

    def __init__(self, G: int = 8, k: int = 3):
        super().__init__()
        self.G = G
        self.k = k
        self.n_basis = G + k

        h = 2.0 / G
        inner = torch.linspace(-1.0, 1.0, G + 1)
        t = torch.cat([
            torch.full((k,), -1.0),
            inner,
            torch.full((k,), 1.0)
        ])
        self.register_buffer("t", t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions at x.

        x: (..., in_features) values in [-1, 1]
        returns: (..., in_features, n_basis)
        """
        t = self.t
        k = self.k

        x_exp = x.unsqueeze(-1)
        t_exp = t.unsqueeze(0)
        n_knots = t.shape[0]

        left = t[:-1].view(1, -1)
        right = t[1:].view(1, -1)
        B = ((x_exp >= left) & (x_exp < right)).to(x.dtype)
        last_mask = (x_exp == t[-1])
        B[..., -1] = B[..., -1] + last_mask.squeeze(-1).to(x.dtype)

        for d in range(1, k + 1):
            n_b = n_knots - 1 - d
            j = torch.arange(n_b, device=x.device)
            tj = t[j]
            tjd = t[j + d]
            tj1 = t[j + 1]
            tjd1 = t[j + d + 1]

            denom1 = (tjd - tj).clamp(min=1e-8)
            denom2 = (tjd1 - tj1).clamp(min=1e-8)

            alpha1 = (x_exp - tj) / denom1
            alpha2 = (tjd1 - x_exp) / denom2

            B_prev = B[..., :n_b]
            B_next = B[..., 1:n_b + 1]

            B = alpha1 * B_prev + alpha2 * B_next

        return B


class KANLayer(nn.Module):
    """One KAN layer using learned univariate B-spline functions on each edge."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 G: int = 8,
                 k: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        n_basis = G + k

        self.basis = BSplineBasis(G=G, k=k)

        self.w = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        self.c = nn.Parameter(
            torch.zeros(out_features, in_features, n_basis)
        )
        nn.init.normal_(self.c, std=0.1 / math.sqrt(in_features * n_basis))

        self.norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_clamped = torch.tanh(x)

        silu_x = F.silu(x_clamped)
        residual = torch.einsum('oi,ni->no', self.w, silu_x)

        B = self.basis(x_clamped)
        spline = torch.einsum('oig,nig->no', self.c, B)

        return residual + spline


class KANPrimitiveNet(nn.Module):
    """KAN-based primitive network N(s; u) ≈ F(s; u).

    Approximates the antiderivative such that
        d^3 N / (du1 du2 du3) ≈ f_tilde(s; u).
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 width: int = 32,
                 depth: int = 2,
                 G: int = 8,
                 k: int = 3,
                 output_scale: float = 1.0):
        super().__init__()
        self.n_params = n_params
        self.n_int_vars = n_int_vars

        in_dim = n_params + n_int_vars

        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers.append(KANLayer(d_in, width, G=G, k=k))
            d_in = width
        layers.append(KANLayer(d_in, 1, G=G, k=k))

        self.layers = nn.ModuleList(layers)
        self._output_scale = output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h * self._output_scale


##############################################################################
#  Mixed Partial Derivative via Autograd
##############################################################################

def mixed_partial_3(net: KANPrimitiveNet,
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
#  KANIntegrator: train + evaluate
##############################################################################

class KANIntegrator:
    """Application of KAN for numerical integration (Maitre et al. approach).

    Trains a KANPrimitiveNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 width: int = 32,
                 depth: int = 2,
                 G: int = 8,
                 k: int = 3,
                 output_scale: float = 1.0):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = KANPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            width=width,
            depth=depth,
            G=G,
            k=k,
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
        """Train the KAN network to approximate the antiderivative.

        Returns: (loss_history, norm_cache)
        """
        if device is None:
            device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=n_epochs,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e3,
        )
        loss_fn = nn.HuberLoss(delta=0.5)
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
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=3.0)
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
