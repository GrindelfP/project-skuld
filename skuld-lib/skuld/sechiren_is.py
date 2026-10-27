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
from typing import Callable

import torch

from .sechiren import SechirenPrimitiveNet
from ._maitre import MaitreIntegrator
from ._maitre import mixed_partial_3  # noqa: F401


class SechirenISIntegrator(MaitreIntegrator):
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

    def _sample_u(self, n: int) -> torch.Tensor:
        # Sample u1, u2 uniformly, u3 with importance sampling.
        # Hardcoded to 3 integration variables (known limitation).
        u12 = torch.rand(n, 2)
        u3 = self._sample_u3(n, u12.dtype)
        return torch.cat([u12, u3], dim=1)

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
        return self._fit(integrand_fn, param_sets, n_epochs, n_per_param,
                         lr, device, verbose_every, weight_decay=weight_decay)

    def integrate(self,
                  params: tuple,
                  norm_cache: dict = None,
                  device: torch.device = None) -> float:
        """Evaluate the integral via corner sum, optionally in float64."""
        if device is None:
            device = torch.device("cpu")

        self.net.eval()
        self.net.to(device)

        fc = self._norm_constant(params, norm_cache)

        # Optionally convert to float64 for corner-sum evaluation
        # MPS doesn't support float64, so fall back to float32 there
        use_f64 = self.float64_eval and device.type != "mps"
        if use_f64:
            self.net.double()

        dtype = torch.float64 if use_f64 else torch.float32
        I_tilde = self._corner_sum(params, device, dtype)

        # Convert back to float32 if needed
        if use_f64:
            self.net.float()

        return float(I_tilde * fc)
