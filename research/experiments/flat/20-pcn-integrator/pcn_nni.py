"""
pcn_nni.py — standalone flat test of the Polynomial Chaos Network (PCN).

PCN is NOT a neural network. It represents the antiderivative N(u, theta) as
a sum of tensor-product Legendre polynomials with learnable coefficients:

    N(u, theta) = sum_k c_k(theta) * P_k(u)

where P_k are tensor-product Legendre polynomials and c_k(theta) is a linear
map from the physical parameters theta. The corner-sum evaluation is exact.

Trained so that the third mixed partial d3N/du1 du2 du3 matches f, which is
what makes the corner sum telescope to the integral (Maitre et al. 2022).

This is a fully standalone script — no skuld-lib dependency.

Usage:
    python pcn_nni.py
    python pcn_nni.py --epochs 8000 --seed 42 --degree 6
    python pcn_nni.py --device cpu --n-per-param 512
"""
import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch
import torch.nn as nn

from physics import integrand_transformed, reference_scipy, PARAM_SETS


##############################################################################
#  PCN — Polynomial Chaos Network (standalone implementation)
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

    def _legendre_eval(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """Evaluate Legendre polynomials P_0(x) ... P_degree(x) at x."""
        batch = x.shape[0]
        P = torch.zeros(batch, degree + 1, device=x.device, dtype=x.dtype)
        P[:, 0] = 1.0
        if degree >= 1:
            P[:, 1] = x
        for n in range(2, degree + 1):
            P[:, n] = ((2 * n - 1) * x * P[:, n - 1] - (n - 1) * P[:, n - 2]) / n
        return P

    def _legendre_deriv(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """Evaluate Legendre polynomial derivatives P_0'(x) ... P_degree'(x) at x."""
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

    def _eval_basis(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate tensor-product Legendre basis at u."""
        batch = u.shape[0]
        x = 2.0 * u - 1.0  # Map [0, 1] -> [-1, 1]

        P_vars = [self._legendre_eval(x[:, i], self.degree)
                  for i in range(self.n_int_vars)]

        basis = P_vars[0]
        for i in range(1, self.n_int_vars):
            basis = basis.unsqueeze(2) * P_vars[i].unsqueeze(1)
            basis = basis.reshape(batch, -1)

        return basis

    def _compute_corner_sum_basis(self) -> torch.Tensor:
        """Compute the corner-sum of each tensor-product Legendre basis function."""
        corners = list(itertools.product([0.0, 1.0], repeat=self.n_int_vars))

        corner_sum = torch.zeros(self.n_basis)
        for corner in corners:
            sign = (-1.0) ** sum(corner)
            u = torch.tensor(corner, dtype=torch.float32).unsqueeze(0)
            basis_val = self._eval_basis(u).squeeze(0)
            corner_sum += sign * basis_val

        return corner_sum

    def forward(self, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Evaluate N(u, params) = sum_k c_k(params) * P_k(u)."""
        coeffs = self.coeff_layer(params)
        basis_vals = self._eval_basis(u)
        return (coeffs * basis_vals).sum(dim=1)

    def corner_sum(self, params: torch.Tensor) -> torch.Tensor:
        """Evaluate the corner-sum of N for given params."""
        coeffs = self.coeff_layer(params)
        return (coeffs * self.corner_sum_basis.unsqueeze(0)).sum(dim=1)

    def mixed_partial(self, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the third mixed partial derivative d3N/du1 du2 du3 at (u, params).
        """
        coeffs = self.coeff_layer(params)
        batch = u.shape[0]
        x = 2.0 * u - 1.0

        dP_vars = [self._legendre_deriv(x[:, i], self.degree)
                   for i in range(self.n_int_vars)]

        basis_deriv = dP_vars[0]
        for i in range(1, self.n_int_vars):
            basis_deriv = basis_deriv.unsqueeze(2) * dP_vars[i].unsqueeze(1)
            basis_deriv = basis_deriv.reshape(batch, -1)

        chain_factor = 2.0 ** self.n_int_vars

        return chain_factor * (coeffs * basis_deriv).sum(dim=1)

    def train(self,
              integrand_fn: Callable,
              param_sets: list,
              n_epochs: int = 8000,
              n_per_param: int = 512,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 1000) -> list:
        """Train the PCN using mixed-partial loss: |d3N/du1 du2 du3 - f(u, params)|^2."""
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

                u = torch.rand(n_per_param, self.n_int_vars, device=device)

                dN_du = self.mixed_partial(u, params_batch)
                f = integrand_fn(u, a, b, m, n)

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

    def integrate(self, params_tuple: tuple, device: torch.device = None) -> float:
        """Evaluate the integral using corner-sum for a single parameter set."""
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
#  Flat test
##############################################################################

def digits_of(abs_err: float) -> int:
    return max(0, -math.floor(math.log10(abs_err + 1e-30)))


def main():
    parser = argparse.ArgumentParser(description="PCN Integrator single test")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--n-per-param", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--degree", type=int, default=6,
                        help="Legendre polynomial degree per variable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="training device (auto-selects cuda -> cpu if omitted)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    n_basis = (args.degree + 1) ** 3

    print(f"{'=' * 78}")
    print("  PCN INTEGRATOR — single test (standalone)")
    print(f"  device={device}  seed={args.seed}")
    print(f"  degree={args.degree}  n_basis={n_basis}  "
          f"epochs={args.epochs}  npp={args.n_per_param}  lr={args.lr}")
    print(f"{'=' * 78}\n")

    net = PolynomialChaosNet(
        n_params=4,
        n_int_vars=3,
        degree=args.degree,
    )
    print(f"  Architecture : tensor-product Legendre polynomials, degree {args.degree}")
    print(f"  Basis funcs  : {n_basis}")
    print(f"  Parameters   : {net.n_weights:,}\n")

    print("Computing scipy reference integrals ...")
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    print("Done.\n")

    t0 = time.time()
    history = net.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=args.epochs,
        n_per_param=args.n_per_param,
        lr=args.lr,
        device=device,
        verbose_every=max(1, args.epochs // 10),
    )
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s "
          f"({elapsed / max(1, args.epochs) * 1000:.1f} ms/epoch)")
    print(f"  Final loss: {history[-1]:.4e}")
    print(f"  Min loss:   {min(history):.4e}\n")

    print(f"{'=' * 78}")
    print("  RESULTS")
    print(f"{'=' * 78}")
    print(f"  {'I':>3} {'(a,b,m,n)':^14} {'NNI':>18} {'ref':>18} "
          f"{'abs_err':>12} {'rel_err':>12} {'digits':>7}")
    print("-" * 78)

    rows = []
    for i, params in enumerate(PARAM_SETS, 1):
        nni_val = net.integrate(params, device=device)
        ref_val, _ = refs[params]
        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)
        d = digits_of(abs_err)
        rows.append((i, nni_val, ref_val, abs_err, rel_err, d))
        print(f"  I{i:<3} {str(tuple(int(x) for x in params)):^14} "
              f"{nni_val:>18.10e} {ref_val:>18.10e} "
              f"{abs_err:>12.4e} {rel_err:>12.4e} {d:>7}")

    digits = [r[5] for r in rows]
    rel_errs = [r[4] for r in rows]
    print("-" * 78)
    print(f"  min_digits  = {min(digits)}")
    print(f"  mean_digits = {sum(digits) / len(digits):.1f}")
    print(f"  max_digits  = {max(digits)}")
    print(f"  mean_rel_err = {sum(rel_errs) / len(rel_errs):.4e}")
    print(f"  max_rel_err  = {max(rel_errs):.4e}")
    print(f"  (SECHIREN baseline for reference: mean 4.9 digits)")
    print(f"{'=' * 78}\n")

    results_dir = get_mirror_path(__file__, "results")
    results_file = os.path.join(results_dir, "pcn_nni.csv")
    header = "I,nni_val,ref_val,abs_err,rel_err,digits"
    with open(results_file, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, nni_val, ref_val, abs_err, rel_err, d in rows:
            fh.write(f"I{i},{nni_val:.10e},{ref_val:.10e},"
                     f"{abs_err:.6e},{rel_err:.6e},{d}\n")

    summary_file = os.path.join(results_dir, "pcn_nni_summary.csv")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("architecture,degree,n_basis,epochs,n_per_param,lr,seed,device,"
                 "n_weights,final_loss,min_loss,elapsed_s,"
                 "min_digits,mean_digits,max_digits,mean_rel_err,max_rel_err\n")
        fh.write(f"PCN,{args.degree},{n_basis},{args.epochs},{args.n_per_param},"
                 f"{args.lr},{args.seed},{device},{net.n_weights},"
                 f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},"
                 f"{min(digits)},{sum(digits) / len(digits):.2f},{max(digits)},"
                 f"{sum(rel_errs) / len(rel_errs):.6e},{max(rel_errs):.6e}\n")

    print(f"  Results saved -> {results_file}")
    print(f"  Summary saved -> {summary_file}\n")


if __name__ == "__main__":
    main()
