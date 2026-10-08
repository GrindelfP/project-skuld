"""
pcn_nni.py — single flat test of the Polynomial Chaos Network (PCN).

PCN is NOT a neural network. It represents the antiderivative N(u, theta) as
a sum of tensor-product Legendre polynomials with learnable coefficients:

    N(u, theta) = sum_k c_k(theta) * P_k(u)

where P_k are tensor-product Legendre polynomials and c_k(theta) is a linear
map from the physical parameters theta. The corner-sum evaluation is exact.

Trained so that the mixed partial dN/du matches f, which is what makes the
corner sum telescope to the integral (Maitre et al. 2022).

Hypothesis under test: does a spectral/polynomial parameterisation of the
antiderivative break the ~5-digit ceiling that every neural architecture hits?
Compare against SECHIREN (mean 4.875 digits) on the identical 8 parameter sets.

Usage:
    python pcn_nni.py
    python pcn_nni.py --epochs 8000 --seed 42 --degree 6
    python pcn_nni.py --device cpu --n-per-param 512
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from utils.paths import get_mirror_path

import numpy as np
import torch

from skuld.pcn import PCNIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS


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
    print("  PCN INTEGRATOR — single test")
    print(f"  device={device}  seed={args.seed}")
    print(f"  degree={args.degree}  n_basis={n_basis}  "
          f"epochs={args.epochs}  npp={args.n_per_param}  lr={args.lr}")
    print(f"{'=' * 78}\n")

    integrator = PCNIntegrator(
        n_params=4,
        n_int_vars=3,
        degree=args.degree,
    )
    print(f"  Architecture : tensor-product Legendre polynomials, degree {args.degree}")
    print(f"  Basis funcs  : {n_basis}")
    print(f"  Parameters   : {integrator.n_weights:,}\n")

    print("Computing scipy reference integrals ...")
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    print("Done.\n")

    t0 = time.time()
    history = integrator.train(
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
        nni_val = integrator.integrate(params, device=device)
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
                 f"{args.lr},{args.seed},{device},{integrator.n_weights},"
                 f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},"
                 f"{min(digits)},{sum(digits) / len(digits):.2f},{max(digits)},"
                 f"{sum(rel_errs) / len(rel_errs):.6e},{max(rel_errs):.6e}\n")

    print(f"  Results saved -> {results_file}")
    print(f"  Summary saved -> {summary_file}\n")


if __name__ == "__main__":
    main()
