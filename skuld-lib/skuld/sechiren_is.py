##############################################################################
#  SECHIREN-IS: SECHIREN with Importance Sampling + Float64 Evaluation
#
#  Two orthogonal improvements over the SECHIREN baseline:
#
#  1. Importance sampling for u3: the integrand varies exponentially in u3
#     through t = exp(L_LOG * u3) - 1, creating sharp variation near u3=1.
#     Uniform sampling wastes most points where the integrand is flat.
#     We use a mixture distribution: (1-beta) uniform + beta concentrated
#     near u3=1 via u3 = u^(1/alpha).
#
#  2. Float64 corner-sum evaluation: the alternating-sign corner sum
#     I = N(000) - N(001) - N(010) + N(011) - ... can suffer catastrophic
#     cancellation in float32. Converting the network to float64 for
#     evaluation is cheap and can recover lost digits.
##############################################################################
import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from skuld.sechiren import SechirenPrimitiveNet, mixed_partial_3


class SechirenISIntegrator:
    """
    SECHIREN with importance sampling and float64 corner-sum evaluation.

    Trains a SechirenPrimitiveNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.

    Importance sampling: samples u3 from a mixture of uniform and
    concentrated distributions to focus training on regions where the
    integrand varies rapidly.

    Float64 evaluation: converts the network to float64 for the corner-sum
    evaluation to avoid catastrophic cancellation.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 omega_0: float = 30.0,
                 output_scale: float = 1.0,
                 is_beta: float = 0.5,
                 is_alpha: float = 2.0,
                 float64_eval: bool = True):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.is_beta = is_beta
        self.is_alpha = is_alpha
        self.float64_eval = float64_eval
        self.net = SechirenPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            output_scale=output_scale,
        )

    @property
    def n_weights(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def _sample_u3(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        """Sample u3 from a mixture of uniform and concentrated distributions.

        With probability (1-beta), sample uniformly from [0,1].
        With probability beta, sample from u3 = u^(1/alpha) which
        concentrates points near u3=1 for alpha > 1.
        """
        u = torch.rand(n, 1, dtype=dtype)
        # Concentrated: u3 = u^(1/alpha), CDF = u3^alpha, density = alpha*u3^(alpha-1)
        u3_conc = u ** (1.0 / self.is_alpha)
        # Mixture
        mask = torch.rand(n, 1, dtype=dtype) < self.is_beta
        u3 = torch.where(mask, u3_conc, u)
        return u3

    def make_batch(self,
                   integrand_fn: Callable,
                   param_sets: list,
                   n_per_param: int,
                   device: torch.device,
                   norm_cache: dict = None) -> tuple:
        """Generate a training batch with importance sampling for u3."""
        if norm_cache is None:
            norm_cache = {}

        all_xu, all_f = [], []

        for params in param_sets:
            params_key = tuple(params)
            if params_key not in norm_cache:
                norm_cache[params_key] = self._center_value(integrand_fn, self.n_int_vars, *params)
            fc = norm_cache[params_key]

            # Sample u1, u2 uniformly, u3 with importance sampling
            u12 = torch.rand(n_per_param, 2)
            u3 = self._sample_u3(n_per_param, u12.dtype)
            u = torch.cat([u12, u3], dim=1)
            f = integrand_fn(u, *params) / fc

            s = torch.tensor(params, dtype=u.dtype).expand(u.shape[0], -1)
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
        """Train the SECHIREN-IS to approximate the antiderivative."""
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
                integrand_fn, param_sets, n_per_param, device, norm_cache,
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
        """Evaluate the integral via corner sum, optionally in float64."""
        if device is None:
            device = torch.device("cpu")

        self.net.eval()
        self.net.to(device)

        if norm_cache and tuple(params) in norm_cache:
            fc = norm_cache[tuple(params)]
        else:
            fc = 1.0

        # Optionally convert to float64 for corner-sum evaluation
        # MPS doesn't support float64, so fall back to float32 there
        use_f64 = self.float64_eval and device.type != "mps"
        if use_f64:
            self.net.double()

        dtype = torch.float64 if use_f64 else torch.float32
        s_row = torch.tensor([params], dtype=dtype).to(device)
        I_tilde = 0.0

        with torch.no_grad():
            for corner in itertools.product([0.0, 1.0], repeat=self.n_int_vars):
                sign = (-1) ** (self.n_int_vars - sum(corner))
                u_t = torch.tensor([corner], dtype=dtype).to(device)
                inp = torch.cat([s_row, u_t], dim=1)
                I_tilde += sign * self.net(inp).item()

        # Convert back to float32 if needed
        if use_f64:
            self.net.float()

        return float(I_tilde * fc)
