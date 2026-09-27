##############################################################################
#  BrokNET: Mixture-of-Experts Neural Numerical Integration
#
#  BrokNet is an ensemble of expert subnetworks (SindriNet), each trained
#  on a single parameter set. A deterministic router dispatches inputs
#  to the correct expert based on the discrete parameter label.
#
#  Named after Brok and Sindri, the Huldra dwarves from God of War.
##############################################################################
import itertools
import math
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .siren import SirenLayer


##############################################################################
#  SindriNet — Expert Subnetwork
##############################################################################

class SindriNet(nn.Module):
    """Expert subnetwork S_i(u1, u2, u3) — SIREN-based antiderivative approximator
    for a single fixed parameter set (a_i, b_i, m_i, n_i).

    Input:  integration variables only  (n_int_vars = 3 dimensions)
    Output: scalar F_i(u) such that d^3 F_i / du1 du2 du3  ≈  f_tilde_i(u)
    """

    def __init__(self,
                 n_int_vars: int = 3,
                 hidden_sizes: List[int] = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.n_int_vars = n_int_vars
        self.omega_0 = omega_0

        layers = []
        in_dim = n_int_vars
        for i, h in enumerate(hidden_sizes):
            layers.append(SirenLayer(in_dim, h,
                                     omega_0=omega_0,
                                     is_first=(i == 0)))
            in_dim = h

        final = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
            final.weight.data *= output_scale
            final.bias.data *= output_scale
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


##############################################################################
#  BrokNet — Mixture-of-Experts Wrapper
##############################################################################

class BrokNet(nn.Module):
    """Brok — Mixture-of-Experts SIREN for parametric numerical integration.

    Accepts full input (a, b, m, n, u1, u2, u3), routes by the discrete key
    (a, b, m, n) to the corresponding SindriNet expert, which then operates
    only on (u1, u2, u3).

    The router is a plain dict — no learnable gating, no softmax.
    """

    def __init__(self,
                 param_sets: List[Tuple],
                 n_int_vars: int = 3,
                 hidden_sizes: List[int] = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.param_sets = param_sets
        self.n_int_vars = n_int_vars
        self.n_params = 4

        self.experts = nn.ModuleList([
            SindriNet(n_int_vars=n_int_vars,
                      hidden_sizes=hidden_sizes,
                      omega_0=omega_0,
                      output_scale=output_scale)
            for _ in param_sets
        ])

        self.router: Dict[Tuple, int] = {
            tuple(float(v) for v in ps): i
            for i, ps in enumerate(param_sets)
        }

    def expert_index(self, a: float, b: float, m: float, n: float) -> int:
        key = (float(a), float(b), float(m), float(n))
        if key not in self.router:
            raise KeyError(
                f"Parameter set {key} was not registered at construction time. "
                f"Available sets: {list(self.router.keys())}"
            )
        return self.router[key]

    def forward_expert(self, expert_idx: int, u: torch.Tensor) -> torch.Tensor:
        return self.experts[expert_idx](u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = tuple(float(v) for v in x[0, :4].tolist())
        idx = self.router[key]
        u = x[:, 4:]
        return self.experts[idx](u)


##############################################################################
#  Mixed Third Partial Derivative via Autograd
##############################################################################

def mixed_partial_3_expert(expert: SindriNet,
                            u: torch.Tensor) -> torch.Tensor:
    """Compute d^3 S_i / du1 du2 du3 for a single SindriNet expert."""
    u_d = u.detach().requires_grad_(True)
    G_out = expert(u_d).squeeze(-1)
    ones = torch.ones_like(G_out)

    g1 = torch.autograd.grad(
        G_out, u_d, grad_outputs=ones,
        create_graph=True, retain_graph=True,
    )[0][:, 0]

    g12 = torch.autograd.grad(
        g1, u_d, grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]

    g123 = torch.autograd.grad(
        g12, u_d, grad_outputs=torch.ones_like(g12),
        create_graph=True,
    )[0][:, 2]

    return g123


##############################################################################
#  BrokNetIntegrator: train + evaluate
##############################################################################

class BrokNetIntegrator:
    """Application of BrokNet for numerical integration (Maitre et al. approach).

    Trains a BrokNet (mixture of SindriNet experts) to approximate the
    antiderivative for each parameter set, then evaluates the integral
    via an alternating-sign corner sum over the unit hypercube.
    """

    def __init__(self,
                 param_sets: List[Tuple],
                 n_int_vars: int = 3,
                 hidden_sizes: List[int] = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0):
        self.param_sets = param_sets
        self.n_int_vars = n_int_vars
        self.n_params = 4
        self.net = BrokNet(
            param_sets=param_sets,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            output_scale=output_scale,
        )

    @property
    def n_weights(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    @property
    def n_experts(self) -> int:
        return len(self.param_sets)

    def _center_value(self, integrand_fn, a, b, m, n) -> float:
        uc = torch.full((1, self.n_int_vars), 0.5)
        fc = integrand_fn(uc, a, b, m, n).item()
        return fc if abs(fc) > 1e-30 else 1.0

    def _make_expert_batch(self, integrand_fn, a, b, m, n, n_samples, device, norm_cache):
        key = (float(a), float(b), float(m), float(n))
        if key not in norm_cache:
            norm_cache[key] = self._center_value(integrand_fn, a, b, m, n)
        fc = norm_cache[key]

        u = torch.rand(n_samples, self.n_int_vars)
        f = integrand_fn(u, a, b, m, n) / fc

        return u.to(device), f.to(device)

    def train(self,
              integrand_fn: Callable,
              n_epochs: int = 8000,
              n_per_expert: int = 1024,
              lr: float = 5e-4,
              device: torch.device = None,
              verbose_every: int = 500) -> tuple:
        """Train BrokNet experts.

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

        n_experts = len(self.param_sets)
        history = []
        norm_cache: Dict = {}
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            expert_losses_tensors = []

            for i, (a, b, m, n) in enumerate(self.param_sets):
                u, f_tilde = self._make_expert_batch(
                    integrand_fn, a, b, m, n, n_per_expert, device, norm_cache
                )
                dG = mixed_partial_3_expert(self.net.experts[i], u)
                loss_i = loss_fn(dG, f_tilde)
                expert_losses_tensors.append(loss_i)

            total_loss = torch.stack(expert_losses_tensors).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            lv = total_loss.item()
            history.append(lv)

            if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:5d}/{n_epochs}  "
                      f"loss={lv:.4e}  lr={lr_now:.2e}  "
                      f"({elapsed:.1f}s)")

        return history, norm_cache

    def integrate(self,
                  a: float, b: float, m: float, n: float,
                  norm_cache: dict = None,
                  device: torch.device = None) -> float:
        """Evaluate the integral for a given parameter set via corner sum."""
        if device is None:
            device = torch.device("cpu")

        self.net.eval()
        self.net.to(device)

        key = (float(a), float(b), float(m), float(n))
        if norm_cache and key in norm_cache:
            fc = norm_cache[key]
        else:
            fc = 1.0

        expert_idx = self.net.expert_index(a, b, m, n)
        expert = self.net.experts[expert_idx]

        I_tilde = 0.0
        with torch.no_grad():
            for corner in itertools.product([0.0, 1.0], repeat=self.n_int_vars):
                sign = (-1) ** (self.n_int_vars - sum(corner))
                u_t = torch.tensor([corner], dtype=torch.float32).to(device)
                I_tilde += sign * expert(u_t).item()

        return float(I_tilde * fc)
