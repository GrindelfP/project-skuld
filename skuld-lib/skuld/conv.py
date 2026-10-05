##############################################################################
#  Conv-based Neural Numerical Integration
#
#  Pure convolutional architecture (no fully-connected layers) for the
#  Maître et al. (2022) antiderivative approximation method.
#  Input: 1×7 image [a, b, m, n, u1, u2, u3]  (params first, matching the
#  rest of the library).
#  4 Conv2D layers (32→64→128→256, k=3, same, GELU, residual),
#  1×1 conv output head → scalar N(u, θ).
#  Trained so that the third mixed partial ∂³N/∂u₁∂u₂∂u₃ matches f, which
#  is what makes the corner sum telescope to the integral (see mixed_partial_3).
##############################################################################

import itertools
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


##############################################################################
#  ConvAntiderivativeNet: pure CNN, no fully-connected layers
##############################################################################


class ConvBlock(nn.Module):
    """Conv2d → GELU → residual skip.

    If in_ch != out_ch, a 1×1 projection shortcut is used so the skip
    connection dimensions match.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, pad: str = "same"):
        super().__init__()
        if pad == "same":
            p = k // 2
        else:
            p = 0
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, stride=1)
        self.act = nn.GELU()
        # Projection shortcut when dimensions change
        self.use_proj = in_ch != out_ch
        if self.use_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.act(out)
        # Residual: match dimensions if needed
        if self.use_proj:
            x = self.proj(x)
        return x + out


class ConvAntiderivativeNet(nn.Module):
    """
    Pure CNN antiderivative network N(u, θ) ≈ F(u) integrated over u ∈ [0,1]^3.

    Input: 1×7 image  [a, b, m, n, u1, u2, u3]  (last dim = 7 features)
    Architecture: 4 ConvBlock(32→64→128→256), then Conv2d(256→1, k=1).
    No fully-connected layers anywhere.
    """

    def __init__(
        self,
        n_params: int = 4,
        n_int_vars: int = 3,
        hidden_sizes: list = None,
    ):
        super().__init__()
        self.n_params = n_params
        self.n_int_vars = n_int_vars

        if hidden_sizes is None:
            hidden_sizes = [32, 64, 128, 256]

        # Conv blocks — channels double each layer
        blocks = []
        in_ch = 1  # the image is always 1 channel
        for h in hidden_sizes:
            blocks.append(ConvBlock(in_ch, h))
            in_ch = h
        self.blocks = nn.ModuleList(blocks)

        # Output head: 1×1 conv, last block's channels → scalar per input point
        self.head = nn.Conv2d(hidden_sizes[-1], 1, kernel_size=1)

        # Initialise head weights after construction so init is device-agnostic
        self._initialize_head(hidden_sizes[-1])

    def _initialize_head(self, fan_in: int):
        """Init head weights (called after module construction)."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / fan_in)
            self.head.weight.uniform_(-bound, bound)
            if self.head.bias is not None:
                self.head.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: tensor of shape (batch, 7) — the 7 features [a,b,m,n,u1,u2,u3].

        Returns:
            tensor of shape (batch,) — scalar antiderivative value N(u,θ) per sample.
        """
        # Reshape (batch, 7) → (batch, 1, 1, 7)  [1×7 image, batch dim first]
        x = x.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 7)

        # Pass through conv blocks — each preserves 1×7 spatial size
        for block in self.blocks:
            x = block(x)  # (B, 256, 1, 7)

        # Global average pooling over the 1×7 spatial grid → (B, 256, 1, 1)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # 1×1 conv output head → (B, 1, 1, 1)
        x = self.head(x)

        # Flatten to (B,) — one scalar per sample, safe when B == 1
        return x.reshape(x.shape[0])


##############################################################################
#  Third Mixed Partial Derivative via Autograd
##############################################################################

def mixed_partial_3(net: ConvAntiderivativeNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """∂³N/∂u₁∂u₂∂u₃ computed by sequential autograd.

    This is the quantity that must match the integrand f. A corner sum of N
    over the unit hypercube telescopes to the integral of this mixed partial,
    which is why the loss is defined on it rather than on the first-order
    partials ∂N/∂uᵢ — those three conditions are mutually inconsistent for a
    generic integrand and have no solution.
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)

    inp = torch.cat([s, u], dim=1)
    N_out = net(inp)

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
        create_graph=True, retain_graph=True,
    )[0][:, 2]

    return g123


##############################################################################
#  ConvIntegrator: train + evaluate  (Maitre et al. approach)
##############################################################################


class ConvIntegrator:
    """
    CNN for numerical integration (Maitre et al. approach).

    Trains a ConvAntiderivativeNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.
    """

    def __init__(
        self,
        n_params: int = 4,
        n_int_vars: int = 3,
        hidden_sizes: list = None,
    ):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = ConvAntiderivativeNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            hidden_sizes=hidden_sizes,
        )

    @property
    def n_weights(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def make_batch(
        self,
        integrand_fn: Callable,
        param_sets: list,
        n_per_param: int,
        device: torch.device,
        norm_cache: dict = None,
    ) -> tuple:
        """Generate a training batch: uniform in [0,1]^3 × discrete params."""

        if norm_cache is None:
            norm_cache = {}

        all_xu, all_f = [], []

        for params in param_sets:
            params_key = tuple(params)
            if params_key not in norm_cache:
                norm_cache[params_key] = self._center_value(
                    integrand_fn, self.n_int_vars, *params
                )
            fc = norm_cache[params_key]

            # Uniform random points in [0,1]^3
            u_uniform = torch.rand(n_per_param, self.n_int_vars)
            f_uniform = integrand_fn(u_uniform, *params) / fc

            s = torch.tensor(params, dtype=u_uniform.dtype).expand(
                u_uniform.shape[0], -1
            )
            xu = torch.cat([s, u_uniform], dim=1)  # (n_per_param, 7)

            all_xu.append(xu)
            all_f.append(f_uniform)

        batch_xu = torch.cat(all_xu, dim=0).to(device)
        f_tilde = torch.cat(all_f, dim=0).to(device)
        return batch_xu, f_tilde, norm_cache

    @staticmethod
    def _center_value(integrand_fn, n_int_vars: int, *params) -> float:
        """Value of integrand at u=(0.5, ..., 0.5)."""
        uc = torch.full((1, n_int_vars), 0.5)
        fc = integrand_fn(uc, *params).item()
        return fc if abs(fc) > 1e-30 else 1.0

    def train(
        self,
        integrand_fn: Callable,
        param_sets: list,
        n_epochs: int = 5000,
        n_per_param: int = 100_000,
        lr: float = 1e-3,
        device: torch.device = None,
        verbose_every: int = 500,
        weight_decay: float = 0.0,
    ) -> tuple:
        """Train the CNN to approximate the antiderivative."""

        if device is None:
            device = torch.device("cpu")

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr / 10
        )
        self.net.to(device)
        self.net.train()

        history = []
        norm_cache = {}
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            batch_xu, f_tilde, norm_cache = self.make_batch(
                integrand_fn, param_sets, n_per_param, device, norm_cache,
            )

            # Third mixed partial ∂³N/∂u₁∂u₂∂u₃ ≈ f, via autograd end-to-end.
            # batch_xu shape: (total, 7) → first n_params cols are params, rest are u
            dN = mixed_partial_3(self.net, batch_xu)
            loss = ((dN - f_tilde) ** 2).mean()

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

    def integrate(self, params: tuple, norm_cache: dict = None, device: torch.device = None) -> float:
        """Evaluate the integral via corner sum over [0,1]^3 for fixed params."""

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
                inp = torch.cat([s_row, u_t], dim=1)  # (1, 7)
                I_tilde += sign * self.net(inp).item()

        return float(I_tilde * fc)