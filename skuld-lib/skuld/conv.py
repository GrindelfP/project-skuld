##############################################################################
#  Conv-based Neural Numerical Integration
#
#  Pure convolutional architecture (no fully-connected layers) for the
#  Maitre et al. (2022) antiderivative approximation method.
#
#  Input: 1×7 image [a, b, m, n, u1, u2, u3]  (params first, matching the
#  rest of the library).
#
#  4 Conv2D layers (k=3, same padding, GELU, residual), global average
#  pooling, 1×1 conv output head → scalar N(u, θ).
#
#  Trained so that the third mixed partial ∂³N/∂u₁∂u₂∂u₃ matches f, which
#  is what makes the corner sum telescope to the integral.
#
#  Evaluation: network is transferred to CPU and converted to float64 for
#  the corner-sum evaluation, since CUDA/MPS are limited to float32.
##############################################################################

import math
from typing import Callable

import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  ConvBlock: Conv2d → GELU → residual skip
##############################################################################

class ConvBlock(nn.Module):
    """Conv2d → GELU → residual skip.

    If in_ch != out_ch, a 1×1 projection shortcut is used so the skip
    connection dimensions match.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2)
        self.act = nn.GELU()
        self.use_proj = in_ch != out_ch
        if self.use_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv(x))
        if self.use_proj:
            x = self.proj(x)
        return x + out


##############################################################################
#  ConvAntiderivativeNet: pure CNN, no fully-connected layers
##############################################################################

class ConvAntiderivativeNet(nn.Module):
    """
    Pure CNN antiderivative network N(u, θ) ≈ F(u) integrated over u ∈ [0,1]^3.

    Input: tensor of shape (batch, 7) — [a, b, m, n, u1, u2, u3].
    Output: tensor of shape (batch,) — scalar antiderivative value per sample.

    Architecture: 4 ConvBlock(32→64→128→256), global average pool,
    1×1 conv head → scalar. No fully-connected layers anywhere.
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

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, 7) → (batch, 1, 1, 7) for conv processing.

        This method is separate so that a future version can change the
        input representation (e.g., separate flows for params and
        integration variables) without touching the conv blocks.
        """
        return x.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: tensor of shape (batch, 7) — [a, b, m, n, u1, u2, u3].

        Returns:
            tensor of shape (batch,) — scalar antiderivative value N(u,θ).
        """
        # Reshape to 1×7 image
        x = self._process_input(x)  # (B, 1, 1, 7)

        # Pass through conv blocks — each preserves 1×7 spatial size
        for block in self.blocks:
            x = block(x)  # (B, C, 1, 7)

        # Global average pooling over the 1×7 spatial grid → (B, C, 1, 1)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # 1×1 conv output head → (B, 1, 1, 1)
        x = self.head(x)

        # Flatten to (B,) — one scalar per sample
        return x.reshape(x.shape[0])


##############################################################################
#  ConvIntegrator: train + evaluate  (Maitre et al. approach)
##############################################################################

class ConvIntegrator(MaitreIntegrator):
    """
    CNN for numerical integration (Maitre et al. approach).

    Trains a ConvAntiderivativeNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.

    Training is done in float32 on CUDA (or CPU fallback). Evaluation is
    done in float64 on CPU — the network is transferred from the training
    device and converted to float64 for the corner-sum evaluation.
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

    def _make_loss(self):
        return lambda dN, f_tilde: ((dN - f_tilde) ** 2).mean()

    def train(
        self,
        integrand_fn: Callable,
        param_sets: list,
        n_epochs: int = 5000,
        n_per_param: int = 512,
        lr: float = 1e-3,
        device: torch.device = None,
        verbose_every: int = 500,
        weight_decay: float = 0.0,
    ) -> tuple:
        """Train the CNN to approximate the antiderivative.

        Training is done in float32 on the specified device (CUDA preferred,
        CPU fallback). The network remains on the training device after
        training; use integrate() for float64 evaluation on CPU.
        """
        return self._fit(integrand_fn, param_sets, n_epochs, n_per_param,
                         lr, device, verbose_every, weight_decay=weight_decay)

    def integrate(self, params: tuple, norm_cache: dict = None, device: torch.device = None) -> float:
        """Evaluate the integral via corner sum over [0,1]^3 for fixed params.

        The network is transferred to CPU and converted to float64 for this
        evaluation, since CUDA/MPS are limited to float32 and the corner-sum
        is where precision matters most. This is a single forward pass per
        corner (8 corners for 3D), so the CPU transfer is cheap.

        `device` is accepted for API parity with the other integrators but
        deliberately ignored: evaluation always runs on CPU in float64.
        After evaluation the network is left on CPU in float32; train()
        moves it back to the training device.
        """

        # Always evaluate on CPU in float64 for maximum precision
        cpu_device = torch.device("cpu")

        self.net.eval()
        self.net.to(cpu_device)
        self.net.double()  # convert to float64

        fc = self._norm_constant(params, norm_cache)

        I_tilde = self._corner_sum(params, cpu_device, torch.float64)

        # Convert back to float32 for consistency with the rest of the library
        self.net.float()

        return float(I_tilde * fc)
