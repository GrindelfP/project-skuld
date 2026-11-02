##############################################################################
#  AenTROPICAL Neural Numerical Integration
#
#  Tropical max-plus antiderivative with log-sum-exp smoothing and
#  temperature annealing (tau: 1.0 -> 0.01).
#
#  Architecture: N(u, θ) = LSE_tau( max_k(⟨w_k(θ), u⟩ + b_k(θ)) )
#  where LSE_tau is the log-sum-exp smoothing of the max operation.
#
#  Trained with corner-sum loss: direct optimization of
#  |Σ_v (-1)^|v| N(v, θ) - I_exact|²
#
#  Reference: Maître et al. (2022) + tropical geometry
##############################################################################
import itertools
import math
import time
from typing import Callable

import torch
import torch.nn as nn


##############################################################################
#  AenTrop Architecture
##############################################################################

class AenTropPrimitiveNet(nn.Module):
    """
    Tropical max-plus primitive network with log-sum-exp smoothing.

    The antiderivative is approximated as:
        N(u, θ) = τ * log(Σ_k exp((⟨w_k(θ), u⟩ + b_k(θ)) / τ))

    As τ → 0, this converges to max_k(⟨w_k(θ), u⟩ + b_k(θ)).
    The temperature τ is annealed during training (1.0 -> 0.01).
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 n_pieces: int = 12):
        super().__init__()
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.n_pieces = n_pieces

        # Parameter layer: θ → {w_k, b_k} for k=1..n_pieces
        # Output shape: [batch, n_pieces, n_int_vars + 1]
        # where last dimension is [w_1, ..., w_n_int_vars, b]
        self.param_layer = nn.Linear(n_params, n_pieces * (n_int_vars + 1))

        with torch.no_grad():
            bound = 1.0 / math.sqrt(n_params)
            self.param_layer.weight.uniform_(-bound, bound)
            self.param_layer.bias.uniform_(-bound, bound)

        # Precompute corners and signs for corner-sum evaluation
        # Sign convention: (-1)^|v| where |v| = sum of corner coordinates
        self.register_buffer('corners',
            torch.tensor(list(itertools.product([0.0, 1.0], repeat=n_int_vars)),
                         dtype=torch.float32))
        self.register_buffer('signs',
            torch.tensor([(-1.0) ** sum(c) for c in itertools.product([0.0, 1.0], repeat=n_int_vars)],
                         dtype=torch.float32))

    def get_weights(self, theta: torch.Tensor) -> tuple:
        """Extract piecewise weights and biases from parameter vector."""
        out = self.param_layer(theta)
        out = out.view(-1, self.n_pieces, self.n_int_vars + 1)
        w = out[:, :, :self.n_int_vars]
        b = out[:, :, self.n_int_vars]
        return w, b

    def forward(self, u: torch.Tensor, theta: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Evaluate N(u, θ) with log-sum-exp smoothing at temperature tau.

        Args:
            u: [N, n_int_vars] integration variables
            theta: [N, n_params] parameter vectors
            tau: float, temperature for log-sum-exp smoothing

        Returns:
            [N] antiderivative values
        """
        w, b = self.get_weights(theta)           # [N, K, n_int_vars], [N, K]
        u_expanded = u.unsqueeze(1)              # [N, 1, n_int_vars]

        # Linear pieces: ⟨w_k, u⟩ + b_k
        linear = (w * u_expanded).sum(dim=2) + b  # [N, K]

        # Log-sum-exp smoothing
        max_linear = linear.max(dim=1, keepdim=True).values  # [N, 1]
        exp_shifted = torch.exp((linear - max_linear) / tau)  # [N, K]
        lse = max_linear + tau * torch.log(exp_shifted.sum(dim=1, keepdim=True))  # [N, 1]

        return lse.squeeze(1)  # [N]

    def corner_sum(self, theta: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Evaluate Σ_v (-1)^|v| N(v, θ) at all corners of the unit hypercube.

        Args:
            theta: [N, n_params] parameter vectors
            tau: float, temperature for log-sum-exp smoothing

        Returns:
            [N] integral estimates
        """
        N = theta.shape[0]
        device = theta.device
        dtype = theta.dtype

        corner_sum = torch.zeros(N, device=device, dtype=dtype)

        for corner, sign in zip(self.corners, self.signs):
            u_corner = corner.unsqueeze(0).expand(N, -1)  # [N, n_int_vars]
            n_corner = self.forward(u_corner, theta, tau)  # [N]
            corner_sum += sign * n_corner

        return corner_sum

    @property
    def n_weights(self) -> int:
        return sum(p.numel() for p in self.parameters())


##############################################################################
#  AenTropIntegrator: train + evaluate
##############################################################################

class AenTropIntegrator:
    """
    AenTrop for numerical integration (Maître et al. approach with corner-sum loss).

    Trains a tropical max-plus network to approximate the antiderivative,
    then evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.

    The key difference from smooth architectures:
    - Loss is computed directly on the corner-sum (integral estimate), NOT on
      the mixed partial derivative
    - Temperature tau is annealed from tau_start to tau_end during training
    - This avoids the train/eval mismatch that limits smooth architectures
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 n_pieces: int = 12,
                 tau_start: float = 1.0,
                 tau_end: float = 0.01):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.n_pieces = n_pieces
        self.tau_start = tau_start
        self.tau_end = tau_end

        self.net = AenTropPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            n_pieces=n_pieces,
        )

    @property
    def n_weights(self) -> int:
        return self.net.n_weights

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              targets: dict,
              n_epochs: int = 16000,
              n_per_param: int = 1024,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 0) -> list:
        """
        Train the AenTrop network using corner-sum loss.

        Args:
            integrand_fn: Callable(u, *params) -> torch.Tensor of shape [N]
                          (not used in corner-sum loss, kept for API compatibility)
            param_sets: List of parameter tuples
            targets: Dict mapping param_tuple -> exact integral value
            n_epochs: Number of training epochs
            n_per_param: Number of random samples per parameter set per epoch
                         (kept for API compatibility, not used in corner-sum loss)
            lr: Learning rate
            device: torch device
            verbose_every: Print progress every N epochs (0 = silent)

        Returns:
            List of per-epoch losses
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        loss_fn = nn.MSELoss()

        self.net.to(device)
        self.net.train()

        history = []
        t0 = time.time()

        for epoch in range(n_epochs):
            # Anneal temperature
            tau = self.tau_start * (self.tau_end / self.tau_start) ** (epoch / n_epochs)

            total_loss = 0.0
            n_batches = 0

            for params_tuple in param_sets:
                theta = torch.tensor(params_tuple, dtype=torch.float32,
                                      device=device).unsqueeze(0)  # [1, n_params]

                # Corner-sum prediction
                cs = self.net.corner_sum(theta, tau)  # [1]

                target = torch.tensor([targets[params_tuple]], dtype=torch.float32,
                                       device=device)  # [1] - match shape
                loss = loss_fn(cs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / n_batches
            history.append(avg_loss)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch + 1:>6d}/{n_epochs}  "
                      f"tau={tau:.4f}  loss = {avg_loss:.6e}  ({elapsed:.1f}s)")

        return history

    def integrate(self,
                  params: tuple,
                  device: torch.device = None,
                  tau: float = None) -> float:
        """
        Evaluate the integral via corner sum.

        Args:
            params: Parameter tuple (a, b, m, n)
            device: torch device
            tau: Temperature for evaluation (default: tau_end)

        Returns:
            Integral estimate
        """
        if device is None:
            device = torch.device("cpu")
        if tau is None:
            tau = self.tau_end

        self.net.eval()
        self.net.to(device)

        theta = torch.tensor([params], dtype=torch.float32, device=device)

        with torch.no_grad():
            result = self.net.corner_sum(theta, tau).item()

        return float(result)