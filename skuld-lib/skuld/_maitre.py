"""Shared machinery for the Maitre et al. antiderivative integrators."""
import itertools
import time
from typing import Callable

import torch
import torch.nn as nn


def _nth_mixed_partial(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Differentiate g ([N], built from u) successively w.r.t. each column of u ([N, dim])."""
    for i in range(u.shape[1]):
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


def mixed_partial_n(net: torch.nn.Module,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    d^n N / du_1 ... du_n by sequential autograd, where n is the number of
    integration variables (the columns of `batch` after the first
    `net.n_params` parameter columns).

    This replaced a former `mixed_partial_3` hardcoded to three columns.
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

    return _nth_mixed_partial(g, u)


# Backwards-compatible alias: the name predates n-dimensional support.
mixed_partial_3 = mixed_partial_n


class MaitreIntegrator:
    """
    Base for integrators that train a primitive net N(s, u) so that its mixed
    partial matches the integrand, then read the integral off an
    alternating-sign corner sum over [0,1]^n_int_vars. Subclasses set
    self.n_params, self.n_int_vars and self.net in __init__ and may override
    the hooks below.
    """

    _clip_norm = 5.0  # max grad norm for clip_grad_norm_

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

        integrand_fn: callable(u, *params) -> torch.Tensor of shape [N],
            where u has shape [N, n_int_vars].
            The integrand in the unit cube (already Jacobian-adjusted).
        param_sets: list of parameter tuples
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
            u = self._sample_u(n_per_param)

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

    def _sample_u(self, n: int) -> torch.Tensor:
        return torch.rand(n, self.n_int_vars)

    @staticmethod
    def _center_value(integrand_fn, n_int_vars: int, *params) -> float:
        """Value of integrand at u=(0.5, ..., 0.5) — used for normalization."""
        uc = torch.full((1, n_int_vars), 0.5)
        fc = integrand_fn(uc, *params).item()
        return fc if abs(fc) > 1e-30 else 1.0

    def _make_optimizer(self, lr: float, weight_decay: float):
        return torch.optim.Adam(self.net.parameters(), lr=lr,
                                weight_decay=weight_decay)

    def _make_scheduler(self, optimizer, n_epochs: int, lr: float):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr / 10
        )

    def _scheduler_step(self, scheduler, epoch: int) -> None:
        scheduler.step()

    def _make_loss(self):
        return nn.MSELoss()

    def _fit(self, integrand_fn, param_sets, n_epochs, n_per_param, lr,
             device, verbose_every, weight_decay: float = 0.0) -> tuple:
        """Shared training loop. Returns (loss_history, norm_cache)."""
        if device is None:
            device = torch.device("cpu")

        optimizer = self._make_optimizer(lr, weight_decay)
        scheduler = self._make_scheduler(optimizer, n_epochs, lr)
        loss_fn = self._make_loss()
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
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           max_norm=self._clip_norm)
            optimizer.step()
            self._scheduler_step(scheduler, epoch)

            lv = loss.item()
            history.append(lv)

            if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:5d}/{n_epochs}  "
                      f"loss={lv:.4e}  lr={lr_now:.2e}  "
                      f"({elapsed:.1f}s)")

        return history, norm_cache

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 5000,
              n_per_param: int = 512,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 500) -> tuple:
        """
        Train the network to approximate the antiderivative.

        Returns: (loss_history, norm_cache)
        """
        return self._fit(integrand_fn, param_sets, n_epochs, n_per_param,
                         lr, device, verbose_every)

    def integrate(self,
                  params: tuple,
                  norm_cache: dict = None,
                  device: torch.device = None) -> float:
        """
        Evaluate the integral for a given parameter set via corner sum.

        params: tuple — the fixed parameters
        norm_cache: dict with normalization constants from training
        device: torch device

        Returns: the integral estimate, already multiplied back by the
            normalisation constant stored in norm_cache
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
