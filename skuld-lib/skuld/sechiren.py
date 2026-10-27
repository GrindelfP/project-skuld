##############################################################################
#  SECHIREN-based Neural Numerical Integration
#
#  Same architecture as SIREN (Sitzmann et al. 2020) but with hyperbolic
#  secant activation: sech(omega_0 * linear(x)).
#
#  sech(x) = 1/cosh(x) — a localized bump function, not periodic.
#  This makes SECHIREN more like an RBF network than a SIREN.
#
#  v2: learnable omega_0 per layer + learnable output scale.
##############################################################################
import itertools
import math
import time
from typing import Callable

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  SECHIREN Architecture
##############################################################################

class SechirenLayer(nn.Module):
    """Single SECHIREN linear + sech layer with learnable omega_0."""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))
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
        return 1.0 / torch.cosh(self.omega_0 * self.linear(x))


class SechirenPrimitiveNet(nn.Module):
    """
    SECHIREN-based primitive network N(s, u) ≈ F(s; u).

    Architecture: SECHIREN hidden layers + linear output (no final sech).
    Learnable output scale on the final layer.
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
            layers.append(SechirenLayer(in_dim, h,
                                        omega_0=omega_0,
                                        is_first=(i == 0)))
            in_dim = h

        # Final linear layer — no sech activation, learnable output scale
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
#  SechirenIntegrator: train + evaluate
##############################################################################

class SechirenIntegrator(MaitreIntegrator):
    """
    SECHIREN for numerical integration (Maitre et al. approach).

    Trains a SechirenPrimitiveNet to approximate the antiderivative, then
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
        self.net = SechirenPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
            omega_0=omega_0,
            output_scale=output_scale,
        )

    def make_batch(self,
                   integrand_fn: Callable,
                   param_sets: list,
                   n_per_param: int,
                   device: torch.device,
                   norm_cache: dict = None,
                   n_corner_per_corner: int = 0,
                   corner_fraction: float = 0.1) -> tuple:
        """Generate a training batch.

        If n_corner_per_corner > 0, adds corner-focused training points.
        For each corner of the unit hypercube, samples n_corner_per_corner
        points in [0, corner_fraction] (or [1-corner_fraction, 1]) near
        that corner. This targets the corner-sum evaluation points.
        """
        if norm_cache is None:
            norm_cache = {}

        all_xu, all_f = [], []

        for params in param_sets:
            params_key = tuple(params)
            if params_key not in norm_cache:
                norm_cache[params_key] = self._center_value(integrand_fn, self.n_int_vars, *params)
            fc = norm_cache[params_key]

            # Uniform random points
            u_uniform = torch.rand(n_per_param, self.n_int_vars)
            f_uniform = integrand_fn(u_uniform, *params) / fc

            parts_u = [u_uniform]
            parts_f = [f_uniform]

            # Corner-focused points
            if n_corner_per_corner > 0:
                for corner in itertools.product([0.0, 1.0], repeat=self.n_int_vars):
                    corner_t = torch.tensor(corner, dtype=u_uniform.dtype)
                    # Sample in [0, corner_fraction] or [1-corner_fraction, 1]
                    noise = torch.rand(n_corner_per_corner, self.n_int_vars)
                    noise = torch.where(corner_t == 0, noise, -noise)
                    u_corner = corner_t.unsqueeze(0) + noise * corner_fraction
                    u_corner = torch.clamp(u_corner, 0.0, 1.0)
                    f_corner = integrand_fn(u_corner, *params) / fc

                    parts_u.append(u_corner)
                    parts_f.append(f_corner)

            u = torch.cat(parts_u, dim=0)
            f = torch.cat(parts_f, dim=0)

            s = torch.tensor(params, dtype=u.dtype).expand(u.shape[0], -1)
            xu = torch.cat([s, u], dim=1)

            all_xu.append(xu)
            all_f.append(f)

        batch_xu = torch.cat(all_xu, dim=0).to(device)
        f_tilde = torch.cat(all_f, dim=0).to(device)
        return batch_xu, f_tilde, norm_cache

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 5000,
              n_per_param: int = 512,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 500,
              weight_decay: float = 0.0,
              n_corner_per_corner: int = 0,
              corner_fraction: float = 0.1) -> tuple:
        """Train the SECHIREN to approximate the antiderivative.

        If n_corner_per_corner > 0, adds corner-focused training points
        near the corners of the unit hypercube.

        Returns:
            tuple: (history, norm_cache, epoch_data) where epoch_data is a list
                of per-epoch dicts with keys: epoch, loss, clean_time,
                data_gen_time, total_time, cumulative_time.
        """
        if device is None:
            device = torch.device("cpu")

        optimizer = self._make_optimizer(lr, weight_decay)
        scheduler = self._make_scheduler(optimizer, n_epochs, lr)
        loss_fn = self._make_loss()
        self.net.to(device)
        self.net.train()

        history = []
        epoch_data = []
        norm_cache = {}
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            # ── data generation ───────────────────────────────────────────
            t_data_start = time.time()
            batch_xu, f_tilde, norm_cache = self.make_batch(
                integrand_fn, param_sets, n_per_param, device, norm_cache,
                n_corner_per_corner=n_corner_per_corner,
                corner_fraction=corner_fraction,
            )
            data_gen_time = time.time() - t_data_start

            # ── forward-backward pass (clean) ────────────────────────────
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_train_start = time.time()

            dN = mixed_partial_3(self.net, batch_xu)
            loss = loss_fn(dN, f_tilde)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           max_norm=self._clip_norm)
            optimizer.step()
            self._scheduler_step(scheduler, epoch)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            clean_time = time.time() - t_train_start

            total_time = data_gen_time + clean_time
            lv = loss.item()
            history.append(lv)
            epoch_data.append({
                'epoch': epoch,
                'loss': lv,
                'clean_time': clean_time,
                'data_gen_time': data_gen_time,
                'total_time': total_time,
                'cumulative_time': time.time() - t0,
            })

            if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:5d}/{n_epochs}  "
                      f"loss={lv:.4e}  lr={lr_now:.2e}  "
                      f"({elapsed:.1f}s)")

        return history, norm_cache, epoch_data
