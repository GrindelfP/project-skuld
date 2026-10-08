##############################################################################
#  PCN — Polynomial Chaos Network for Neural Numerical Integration
#
#  Not a neural network at all. Represents the antiderivative N(u, theta) as
#  a sum of tensor-product Legendre polynomials with learnable coefficients:
#
#      N(u, theta) = sum_k c_k(theta) * P_k(u)
#
#  where P_k are tensor-product Legendre polynomials and c_k(theta) is a
#  linear map from the physical parameters theta.
#
#  The corner-sum evaluation is exact and differentiable — no approximation
#  error in evaluation. The mixed-partial derivative is also exact.
#
#  This is the wildest architecture in the project: no hidden layers, no
#  activations, no omega_0. Just polynomials.
##############################################################################
import itertools
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


##############################################################################
#  PCN Architecture
##############################################################################

class PolynomialChaosNet(nn.Module):
    """
    Polynomial Chaos Network: antiderivative as a sum of tensor-product
    Legendre polynomials with learnable coefficients.

    N(u, theta) = sum_k c_k(theta) * P_k(u)

    where:
      - P_k(u) = P_i(2*u1-1) * P_j(2*u2-1) * P_k(2*u3-1) for multi-index (i,j,k)
      - c_k(theta) = W @ theta + b  (linear coefficient map)
      - Legendre polynomials P_n are orthogonal on [-1, 1]

    The corner-sum of each basis function is precomputed as a constant,
    making evaluation exact and cheap.
    """

    def __init__(self, n_params: int = 4, n_int_vars: int = 3, degree: int = 6):
        super().__init__()
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.degree = degree
        self.n_basis = (degree + 1) ** n_int_vars

        # Coefficient layer: params -> polynomial coefficients
        # Scaled down because high-degree Legendre derivatives are large
        self.coeff_layer = nn.Linear(n_params, self.n_basis)
        with torch.no_grad():
            bound = 0.01 / math.sqrt(self.n_basis)
            self.coeff_layer.weight.uniform_(-bound, bound)
            self.coeff_layer.bias.uniform_(-bound, bound)

        # Precompute corner-sum of each basis function (constant)
        corner_sum = self._compute_corner_sum_basis()
        self.register_buffer('corner_sum_basis', corner_sum)

    # ----------------------------------------------------------------------
    #  Legendre polynomial evaluation (stable recurrence)
    # ----------------------------------------------------------------------

    def _legendre_eval(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Evaluate Legendre polynomials P_0(x) ... P_degree(x) at x.

        Uses the stable recurrence:
            P_0(x) = 1
            P_1(x) = x
            P_n(x) = ((2n-1)*x*P_{n-1}(x) - (n-1)*P_{n-2}(x)) / n

        Args:
            x: (batch,) tensor of points in [-1, 1]
            degree: maximum polynomial degree

        Returns:
            (batch, degree+1) tensor of Legendre polynomial values
        """
        batch = x.shape[0]
        P = torch.zeros(batch, degree + 1, device=x.device, dtype=x.dtype)
        P[:, 0] = 1.0
        if degree >= 1:
            P[:, 1] = x
        for n in range(2, degree + 1):
            P[:, n] = ((2 * n - 1) * x * P[:, n - 1] - (n - 1) * P[:, n - 2]) / n
        return P

    def _legendre_deriv(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Evaluate Legendre polynomial derivatives P_0'(x) ... P_degree'(x) at x.

        Uses the recurrence:
            P_0'(x) = 0
            P_1'(x) = 1
            P_n'(x) = ((2n-1)*(P_{n-1}(x) + x*P_{n-1}'(x)) - (n-1)*P_{n-2}'(x)) / n

        Args:
            x: (batch,) tensor of points in [-1, 1]
            degree: maximum polynomial degree

        Returns:
            (batch, degree+1) tensor of Legendre polynomial derivatives
        """
        batch = x.shape[0]
        P = self._legendre_eval(x, degree)
        dP = torch.zeros(batch, degree + 1, device=x.device, dtype=x.dtype)
        dP[:, 0] = 0.0
        if degree >= 1:
            dP[:, 1] = 1.0
        for n in range(2, degree + 1):
            dP[:, n] = ((2 * n - 1) * (P[:, n - 1] + x * dP[:, n - 1])
                        - (n - 1) * dP[:, n - 2]) / n
        return dP

    # ----------------------------------------------------------------------
    #  Tensor-product basis evaluation
    # ----------------------------------------------------------------------

    def _eval_basis(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate tensor-product Legendre basis at u.

        Args:
            u: (batch, n_int_vars) tensor in [0, 1]^n_int_vars

        Returns:
            (batch, n_basis) tensor of basis function values
        """
        batch = u.shape[0]
        x = 2.0 * u - 1.0  # Map [0, 1] -> [-1, 1]

        # Evaluate Legendre polynomials for each variable
        # P_vars[i]: (batch, degree+1)
        P_vars = [self._legendre_eval(x[:, i], self.degree)
                  for i in range(self.n_int_vars)]

        # Tensor product: iteratively build the full basis
        # Start with the first variable: (batch, degree+1)
        basis = P_vars[0]
        for i in range(1, self.n_int_vars):
            # Outer product: (batch, n_basis_so_far, degree+1)
            basis = basis.unsqueeze(2) * P_vars[i].unsqueeze(1)
            basis = basis.reshape(batch, -1)

        return basis

    def _eval_basis_deriv(self, u: torch.Tensor, var_idx: int) -> torch.Tensor:
        """
        Evaluate derivative of tensor-product basis w.r.t. integration variable var_idx.

        d/d u_var [P_i(2*u1-1) * P_j(2*u2-1) * P_k(2*u3-1)]
        = 2 * P_i(2*u1-1) * ... * P_var'(2*u_var-1) * ... * P_k(2*u3-1)

        Args:
            u: (batch, n_int_vars) tensor in [0, 1]^n_int_vars
            var_idx: which integration variable to differentiate w.r.t.

        Returns:
            (batch, n_basis) tensor of basis function derivatives
        """
        batch = u.shape[0]
        x = 2.0 * u - 1.0  # Map [0, 1] -> [-1, 1]

        # Evaluate Legendre polynomials and derivatives for each variable
        P_vars = []
        dP_vars = []
        for i in range(self.n_int_vars):
            P_vars.append(self._legendre_eval(x[:, i], self.degree))
            dP_vars.append(self._legendre_deriv(x[:, i], self.degree))

        # Use the derivative for the target variable, regular for others
        # The factor of 2 comes from the chain rule: d/du = 2 * d/dx where x = 2u-1
        basis = P_vars[0]
        if var_idx == 0:
            basis = 2.0 * dP_vars[0]
        for i in range(1, self.n_int_vars):
            if i == var_idx:
                component = 2.0 * dP_vars[i]
            else:
                component = P_vars[i]
            basis = basis.unsqueeze(2) * component.unsqueeze(1)
            basis = basis.reshape(batch, -1)

        return basis

    # ----------------------------------------------------------------------
    #  Corner-sum precomputation
    # ----------------------------------------------------------------------

    def _compute_corner_sum_basis(self) -> torch.Tensor:
        """
        Compute the corner-sum of each tensor-product Legendre basis function.

        The corner-sum for the Maitre method is:
            corner_sum(P) = sum_{corners} (-1)^(sum of corner coords) * P(corner)

        For a tensor-product basis function P_i(x1) * P_j(x2) * P_k(x3):
            corner_sum = (P_i(-1) - P_i(1)) * (P_j(-1) - P_j(1)) * (P_k(-1) - P_k(1))

        Since P_n(-1) = (-1)^n and P_n(1) = 1:
            P_n(-1) - P_n(1) = (-1)^n - 1 = { 0 if n even, -2 if n odd }

        So corner_sum(P_i*P_j*P_k) = 0 unless i, j, k are all odd,
        in which case it's (-2)^3 = -8.

        Returns:
            (n_basis,) tensor of corner-sum values
        """
        # Generate all corners of [0, 1]^n_int_vars
        corners = list(itertools.product([0.0, 1.0], repeat=self.n_int_vars))

        corner_sum = torch.zeros(self.n_basis)
        for corner in corners:
            # Sign: (-1)^(sum of corner coordinates)
            sign = (-1.0) ** sum(corner)
            # Evaluate basis at this corner
            u = torch.tensor(corner, dtype=torch.float32).unsqueeze(0)
            basis_val = self._eval_basis(u).squeeze(0)
            corner_sum += sign * basis_val

        return corner_sum

    # ----------------------------------------------------------------------
    #  Forward pass
    # ----------------------------------------------------------------------

    def forward(self, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Evaluate N(u, params) = sum_k c_k(params) * P_k(u).

        Args:
            u: (batch, n_int_vars) tensor in [0, 1]^n_int_vars
            params: (batch, n_params) tensor of physical parameters

        Returns:
            (batch,) tensor of antiderivative values
        """
        coeffs = self.coeff_layer(params)    # (batch, n_basis)
        basis_vals = self._eval_basis(u)     # (batch, n_basis)
        return (coeffs * basis_vals).sum(dim=1)

    def corner_sum(self, params: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the corner-sum of N for given params.

        corner_sum(N) = sum_k c_k(params) * corner_sum(P_k)

        This is exact — no approximation error.

        Args:
            params: (batch, n_params) tensor of physical parameters

        Returns:
            (batch,) tensor of corner-sum values
        """
        coeffs = self.coeff_layer(params)    # (batch, n_basis)
        return (coeffs * self.corner_sum_basis.unsqueeze(0)).sum(dim=1)

    def mixed_partial(self, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the third mixed partial derivative d3N/du1 du2 du3 at (u, params).

        For the Maitre method, the antiderivative N must satisfy:
            d3N/du1 du2 du3 = f(u1, u2, u3)

        For a tensor-product Legendre basis:
            d3/du1 du2 du3 [P_i(2u1-1) * P_j(2u2-1) * P_k(2u3-1)]
            = 2^3 * P_i'(2u1-1) * P_j'(2u2-1) * P_k'(2u3-1)

        The factor of 2^3 = 8 comes from the chain rule (each variable
        contributes a factor of 2 from the mapping x = 2u - 1).

        Args:
            u: (batch, n_int_vars) tensor in [0, 1]^n_int_vars
            params: (batch, n_params) tensor of physical parameters

        Returns:
            (batch,) tensor of third mixed partial derivatives
        """
        coeffs = self.coeff_layer(params)    # (batch, n_basis)
        batch = u.shape[0]
        x = 2.0 * u - 1.0  # Map [0, 1] -> [-1, 1]

        # Evaluate Legendre derivatives for all variables
        dP_vars = [self._legendre_deriv(x[:, i], self.degree)
                   for i in range(self.n_int_vars)]

        # Tensor product of derivatives: 2^n_int_vars * prod_i P_i'(2u_i - 1)
        basis_deriv = dP_vars[0]
        for i in range(1, self.n_int_vars):
            basis_deriv = basis_deriv.unsqueeze(2) * dP_vars[i].unsqueeze(1)
            basis_deriv = basis_deriv.reshape(batch, -1)

        # Chain rule factor: 2^n_int_vars
        chain_factor = 2.0 ** self.n_int_vars

        return chain_factor * (coeffs * basis_deriv).sum(dim=1)

    # ----------------------------------------------------------------------
    #  Training
    # ----------------------------------------------------------------------

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 8000,
              n_per_param: int = 512,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 1000) -> list:
        """
        Train the PCN using mixed-partial loss: |dN/du - f(u, params)|^2.

        Args:
            integrand_fn: function f(u, params) -> integrand values
            param_sets: list of parameter tuples (a, b, m, n)
            n_epochs: number of training epochs
            n_per_param: number of sample points per parameter set per epoch
            lr: learning rate
            device: torch device
            verbose_every: print loss every this many epochs

        Returns:
            list of loss values (one per epoch)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        max_grad_norm = 1.0

        history = []
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            for params_tuple in param_sets:
                a, b, m, n = params_tuple
                params = torch.tensor(params_tuple, dtype=torch.float32,
                                       device=device).unsqueeze(0)
                params_batch = params.expand(n_per_param, -1)

                # Sample random points in [0, 1]^n_int_vars
                u = torch.rand(n_per_param, self.n_int_vars, device=device)

                # Compute mixed partial derivative
                dN_du = self.mixed_partial(u, params_batch)

                # Evaluate integrand (physics.py signature: u, a, b, m, n)
                f = integrand_fn(u, a, b, m, n)

                # Mixed-partial loss
                loss = ((dN_du - f) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / n_batches
            history.append(avg_loss)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  Epoch {epoch + 1:>6d}/{n_epochs}  "
                      f"loss = {avg_loss:.6e}")

        return history

    # ----------------------------------------------------------------------
    #  Integration (corner-sum evaluation)
    # ----------------------------------------------------------------------

    def integrate(self, params_tuple: tuple, device: torch.device = None) -> float:
        """
        Evaluate the integral using corner-sum for a single parameter set.

        Args:
            params_tuple: (a, b, m, n) parameter tuple
            device: torch device

        Returns:
            float: the integral value
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        params = torch.tensor(params_tuple, dtype=torch.float32,
                               device=device).unsqueeze(0)
        with torch.no_grad():
            result = self.corner_sum(params).item()
        return result

    @property
    def n_weights(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


##############################################################################
#  PCN Integrator — common API wrapper
##############################################################################

class PCNIntegrator:
    """
    PCN Integrator with the same API as other skuld-lib integrators.

    Usage:
        integrator = PCNIntegrator(n_params=4, n_int_vars=3, degree=6)
        history = integrator.train(integrand_fn, param_sets, n_epochs, ...)
        result = integrator.integrate(params)
    """

    def __init__(self, n_params: int = 4, n_int_vars: int = 3, degree: int = 6):
        self.net = PolynomialChaosNet(n_params, n_int_vars, degree)
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.degree = degree

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 8000,
              n_per_param: int = 512,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 1000) -> list:
        return self.net.train(integrand_fn, param_sets, n_epochs,
                              n_per_param, lr, device, verbose_every)

    def integrate(self, params_tuple: tuple, device: torch.device = None) -> float:
        return self.net.integrate(params_tuple, device)

    @property
    def n_weights(self) -> int:
        return self.net.n_weights
