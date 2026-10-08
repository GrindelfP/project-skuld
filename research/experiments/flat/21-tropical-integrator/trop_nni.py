"""
trop_nni.py — standalone flat test of the Tropical Antiderivative (TROP).

TROP is NOT a neural network. It uses tropical (max-plus) algebra:

    N(u, theta) = max_k( <w_k(theta), u> + b_k(theta) )

where w_k and b_k are linear functions of the physical parameters theta.
The antiderivative is piecewise linear and convex in u.

The corner-sum evaluation is exact:
    corner_sum(N) = sum_{corners} (-1)^(sum of corner coords) * N(corner)

Training uses corner-sum loss (NOT mixed-partial, because the derivative of
a piecewise linear function is zero almost everywhere):
    L = |corner_sum(N) - target|^2

This is the wildest architecture in the project: non-smooth, piecewise linear,
convex. No hidden layers, no activations, no omega_0. Just max and linear.

Usage:
    python trop_nni.py
    python trop_nni.py --epochs 8000 --device cpu
    python trop_nni.py --pieces 8 --lr 1e-3
"""
import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch

from physics import integrand_transformed, reference_scipy, PARAM_SETS


##############################################################################
#  TROP — Tropical Antiderivative
##############################################################################

class TropicalAntiderivative:
    """
    Tropical (max-plus) antiderivative: N(u) = max_k(<w_k, u> + b_k).

    The parameters are {w_k, b_k} for k = 1, ..., K, where each w_k is
    a vector of size n_int_vars and each b_k is a scalar.

    The corner-sum is exact and differentiable (in the subgradient sense).
    """

    def __init__(self, n_params: int = 4, n_int_vars: int = 3, n_pieces: int = 8):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.n_pieces = n_pieces

        # Parameter layer: theta -> {w_k, b_k}
        # Output size: n_pieces * (n_int_vars + 1)
        self.param_layer = torch.nn.Linear(n_params, n_pieces * (n_int_vars + 1))

        # Precompute corners and signs for corner-sum
        self.corners = list(itertools.product([0.0, 1.0], repeat=n_int_vars))
        self.signs = [(-1.0) ** sum(c) for c in self.corners]

    def get_weights(self, theta: torch.Tensor) -> tuple:
        """
        Get {w_k, b_k} from theta.

        Args:
            theta: (batch, n_params) tensor

        Returns:
            w: (batch, n_pieces, n_int_vars) tensor
            b: (batch, n_pieces) tensor
        """
        out = self.param_layer(theta)  # (batch, n_pieces * (n_int_vars + 1))
        out = out.view(-1, self.n_pieces, self.n_int_vars + 1)
        w = out[:, :, :self.n_int_vars]  # (batch, n_pieces, n_int_vars)
        b = out[:, :, self.n_int_vars]   # (batch, n_pieces)
        return w, b

    def forward(self, u: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate N(u, theta) = max_k(<w_k, u> + b_k).

        Args:
            u: (batch, n_int_vars) tensor in [0, 1]^n_int_vars
            theta: (batch, n_params) tensor

        Returns:
            (batch,) tensor of antiderivative values
        """
        w, b = self.get_weights(theta)  # w: (batch, n_pieces, n_int_vars), b: (batch, n_pieces)

        # Compute <w_k, u> + b_k for each piece
        # u: (batch, n_int_vars) -> (batch, 1, n_int_vars)
        u_expanded = u.unsqueeze(1)  # (batch, 1, n_int_vars)
        linear = (w * u_expanded).sum(dim=2) + b  # (batch, n_pieces)

        # Max over pieces
        return linear.max(dim=1).values  # (batch,)

    def corner_sum(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the corner-sum of N for given theta.

        corner_sum(N) = sum_{corners} (-1)^(sum of corner coords) * N(corner)

        This is exact — no approximation error.

        Args:
            theta: (batch, n_params) tensor

        Returns:
            (batch,) tensor of corner-sum values
        """
        w, b = self.get_weights(theta)  # w: (batch, n_pieces, n_int_vars), b: (batch, n_pieces)

        # Evaluate N at each corner
        # corners: list of tuples, each of size n_int_vars
        corner_sum = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
        for corner, sign in zip(self.corners, self.signs):
            u_corner = torch.tensor(corner, device=theta.device, dtype=theta.dtype)
            u_corner = u_corner.unsqueeze(0).expand(theta.shape[0], -1)  # (batch, n_int_vars)
            n_corner = self.forward(u_corner, theta)  # (batch,)
            corner_sum += sign * n_corner

        return corner_sum

    def train(self,
              param_sets: list,
              targets: dict,
              n_epochs: int = 8000,
              lr: float = 1e-3,
              device: torch.device = None,
              verbose_every: int = 1000) -> list:
        """
        Train TROP using corner-sum loss: |corner_sum(N) - target|^2.

        Args:
            param_sets: list of parameter tuples (a, b, m, n)
            targets: dict mapping parameter tuple -> target integral value
            n_epochs: number of training epochs
            lr: learning rate
            device: torch device
            verbose_every: print loss every this many epochs

        Returns:
            list of loss values (one per epoch)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.param_layer.to(device)
        optimizer = torch.optim.Adam(self.param_layer.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        history = []
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            for params_tuple in param_sets:
                theta = torch.tensor(params_tuple, dtype=torch.float32,
                                      device=device).unsqueeze(0)

                # Corner-sum loss
                cs = self.corner_sum(theta)  # (1,)
                target = torch.tensor(targets[params_tuple], dtype=torch.float32,
                                       device=device)
                loss = ((cs - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
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

        self.param_layer.to(device)
        theta = torch.tensor(params_tuple, dtype=torch.float32,
                              device=device).unsqueeze(0)
        with torch.no_grad():
            result = self.corner_sum(theta).item()
        return result

    @property
    def n_weights(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.param_layer.parameters())


##############################################################################
#  Flat test
##############################################################################

def digits_of(abs_err: float) -> int:
    return max(0, -math.floor(math.log10(abs_err + 1e-30)))


def main():
    parser = argparse.ArgumentParser(description="TROP Integrator single test")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pieces", type=int, default=8,
                        help="number of linear pieces (tropical max terms)")
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

    print(f"{'=' * 78}")
    print("  TROP INTEGRATOR — single test")
    print(f"  device={device}  seed={args.seed}")
    print(f"  pieces={args.pieces}  epochs={args.epochs}  lr={args.lr}")
    print(f"{'=' * 78}\n")

    trop = TropicalAntiderivative(
        n_params=4,
        n_int_vars=3,
        n_pieces=args.pieces,
    )
    print(f"  Architecture : tropical max-plus, {args.pieces} linear pieces")
    print(f"  Parameters   : {trop.n_weights:,}")
    print(f"  Training     : corner-sum loss (non-smooth)\n")

    print("Computing scipy reference integrals ...")
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    targets = {p: refs[p][0] for p in PARAM_SETS}
    print("Done.\n")

    t0 = time.time()
    history = trop.train(
        param_sets=PARAM_SETS,
        targets=targets,
        n_epochs=args.epochs,
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
        nni_val = trop.integrate(params, device=device)
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
    results_file = os.path.join(results_dir, "trop_nni.csv")
    header = "I,nni_val,ref_val,abs_err,rel_err,digits"
    with open(results_file, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, nni_val, ref_val, abs_err, rel_err, d in rows:
            fh.write(f"I{i},{nni_val:.10e},{ref_val:.10e},"
                     f"{abs_err:.6e},{rel_err:.6e},{d}\n")

    summary_file = os.path.join(results_dir, "trop_nni_summary.csv")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("architecture,pieces,epochs,lr,seed,device,"
                 "n_weights,final_loss,min_loss,elapsed_s,"
                 "min_digits,mean_digits,max_digits,mean_rel_err,max_rel_err\n")
        fh.write(f"TROP,{args.pieces},{args.epochs},{args.lr},{args.seed},"
                 f"{device},{trop.n_weights},"
                 f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},"
                 f"{min(digits)},{sum(digits) / len(digits):.2f},{max(digits)},"
                 f"{sum(rel_errs) / len(rel_errs):.6e},{max(rel_errs):.6e}\n")

    print(f"  Results saved -> {results_file}")
    print(f"  Summary saved -> {summary_file}\n")


if __name__ == "__main__":
    main()
