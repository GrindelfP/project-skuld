"""
conv_nni.py — single flat test of the ConvIntegrator (pure-CNN antiderivative).

CONV: no fully-connected layers anywhere. Input is a 1x7 image
[a,b,m,n,u1,u2,u3]; 4 residual ConvBlock(32->64->128->256) with k=3, same
padding and GELU; global average pool; 1x1 conv head -> scalar N(u, theta).

Trained so that the third mixed partial d3N/du1 du2 du3 matches f, which is
what makes the corner sum telescope to the integral (Maître et al. 2022).

Hypothesis under test: does a purely convolutional parameterisation of the
antiderivative break the ~4-digit ceiling that every MLP-based architecture
hits? Compare against the SECHIREN baseline (mean 4.875 digits,
grid/13-sechiren-sweep-m2) on the identical 8 parameter sets.

Usage:
    python conv_nni.py
    python conv_nni.py --epochs 4000 --seed 42 --hidden 32,64,128,256
    python conv_nni.py --device cpu --n-per-param 512
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

from skuld.conv import ConvIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS


def digits_of(abs_err: float) -> int:
    return max(0, -math.floor(math.log10(abs_err + 1e-30)))


def main():
    parser = argparse.ArgumentParser(description="ConvIntegrator single test")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--n-per-param", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="32,64,128,256",
                        help="comma-separated channel widths, one per conv block")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    hidden = [int(h) for h in args.hidden.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )

    print(f"{'=' * 78}")
    print("  CONV INTEGRATOR — single test")
    print(f"  device={device}  seed={args.seed}")
    print(f"  hidden={hidden}  epochs={args.epochs}  npp={args.n_per_param}  lr={args.lr}")
    print(f"{'=' * 78}\n")

    integrator = ConvIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=hidden,
    )
    print(f"  Architecture : pure CNN  {hidden}  (no fully-connected layers)")
    print(f"  Parameters   : {integrator.n_weights:,}\n")

    print("Computing scipy reference integrals ...")
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    print("Done.\n")

    t0 = time.time()
    history, norm_cache = integrator.train(
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
        nni_val = integrator.integrate(params, norm_cache=norm_cache, device=device)
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
    results_file = os.path.join(results_dir, "conv_nni.csv")
    header = "I,nni_val,ref_val,abs_err,rel_err,digits"
    with open(results_file, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, nni_val, ref_val, abs_err, rel_err, d in rows:
            fh.write(f"I{i},{nni_val:.10e},{ref_val:.10e},"
                     f"{abs_err:.6e},{rel_err:.6e},{d}\n")

    summary_file = os.path.join(results_dir, "conv_nni_summary.csv")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("architecture,hidden,epochs,n_per_param,lr,seed,device,"
                 "n_weights,final_loss,min_loss,elapsed_s,"
                 "min_digits,mean_digits,max_digits,mean_rel_err,max_rel_err\n")
        fh.write(f"CONV,'{args.hidden}',{args.epochs},{args.n_per_param},"
                 f"{args.lr},{args.seed},{device},{integrator.n_weights},"
                 f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},"
                 f"{min(digits)},{sum(digits) / len(digits):.2f},{max(digits)},"
                 f"{sum(rel_errs) / len(rel_errs):.6e},{max(rel_errs):.6e}\n")

    print(f"  Results saved -> {results_file}")
    print(f"  Summary saved -> {summary_file}\n")


if __name__ == "__main__":
    main()