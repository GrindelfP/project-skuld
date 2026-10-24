"""
aentrop_nni.py — AenTrop Integrator single test (golden lands champion).

Architecture: Tropical max-plus antiderivative with log-sum-exp smoothing.
Best config: 12 pieces, 16000 epochs, lr=1e-3, tau=1.0->0.01 (mean 6.25 digits)

Usage:
    python aentrop_nni.py
    python aentrop_nni.py --epochs 16000 --pieces 12 --device cpu
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

from skuld.aentrop import AenTropIntegrator


##############################################################################
#  Flat test
##############################################################################

def digits_of(abs_err: float) -> int:
    return max(0, -math.floor(math.log10(abs_err + 1e-30)))


def main():
    parser = argparse.ArgumentParser(description="AenTrop Integrator single test")
    parser.add_argument("--epochs", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pieces", type=int, default=12)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.01)
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
    print("  AenTROP INTEGRATOR — single test (golden lands champion)")
    print(f"  device={device}  seed={args.seed}")
    print(f"  pieces={args.pieces}  epochs={args.epochs}  lr={args.lr}")
    print(f"  tau: {args.tau_start} -> {args.tau_end}")
    print(f"{'=' * 78}\n")

    aentrop = AenTropIntegrator(
        n_params=4,
        n_int_vars=3,
        n_pieces=args.pieces,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
    )
    print(f"  Architecture : tropical max-plus (log-sum-exp smoothed), {args.pieces} pieces")
    print(f"  Parameters   : {aentrop.n_weights:,}")
    print(f"  Training     : corner-sum loss with temperature annealing\n")

    print("Computing scipy reference integrals ...")
    refs = {p: reference_scipy(*p) for p in PARAM_SETS}
    targets = {p: refs[p][0] for p in PARAM_SETS}
    print("Done.\n")

    t0 = time.time()
    history = aentrop.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        targets=targets,
        n_epochs=args.epochs,
        n_per_param=1024,  # not used in corner-sum loss, kept for API compat
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
        nni_val = aentrop.integrate(params, device=device, tau=args.tau_end)
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
    print(f"  mean_digits = {sum(digits) / len(digits):.2f}")
    print(f"  max_digits  = {max(digits)}")
    print(f"  mean_rel_err = {sum(rel_errs) / len(rel_errs):.4e}")
    print(f"  max_rel_err  = {max(rel_errs):.4e}")
    print(f"  (TROP v2 baseline: mean 6.1 digits)")
    print(f"  (TROP v5 standalone: mean 6.25 digits)")
    print(f"{'=' * 78}\n")

    results_dir = get_mirror_path(__file__, "results")
    results_file = os.path.join(results_dir, "aentrop_nni.csv")
    header = "I,nni_val,ref_val,abs_err,rel_err,digits"
    with open(results_file, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, nni_val, ref_val, abs_err, rel_err, d in rows:
            fh.write(f"I{i},{nni_val:.10e},{ref_val:.10e},"
                     f"{abs_err:.6e},{rel_err:.6e},{d}\n")

    summary_file = os.path.join(results_dir, "aentrop_nni_summary.csv")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("architecture,pieces,epochs,lr,tau_start,tau_end,seed,device,"
                 "n_weights,final_loss,min_loss,elapsed_s,"
                 "min_digits,mean_digits,max_digits,mean_rel_err,max_rel_err\n")
        fh.write(f"AenTrop,{args.pieces},{args.epochs},{args.lr},"
                 f"{args.tau_start},{args.tau_end},{args.seed},{device},{aentrop.n_weights},"
                 f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},"
                 f"{min(digits)},{sum(digits) / len(digits):.2f},{max(digits)},"
                 f"{sum(rel_errs) / len(rel_errs):.6e},{max(rel_errs):.6e}\n")

    print(f"  Results saved -> {results_file}")
    print(f"  Summary saved -> {summary_file}\n")


if __name__ == "__main__":
    main()