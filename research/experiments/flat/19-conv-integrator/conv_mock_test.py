"""
conv_mock_test.py — quick proof-of-concept test of the ConvIntegrator.

Small network, few epochs, all 8 parameter sets. Verifies the code runs
and produces reasonable output before committing to a full flat test.

Usage:
    python conv_mock_test.py
    python conv_mock_test.py --epochs 500 --device cpu
    python conv_mock_test.py --epochs 1000 --device cuda
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
    parser = argparse.ArgumentParser(description="ConvIntegrator mock test")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-per-param", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="16,32,64,128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="training device (MPS is not supported for CONV; "
                             "auto-selects cuda -> cpu if omitted)")
    args = parser.parse_args()

    hidden = [int(h) for h in args.hidden.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"{'=' * 78}")
    print("  CONV INTEGRATOR — MOCK TEST")
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
    results_file = os.path.join(results_dir, "conv_mock_test.csv")
    header = "I,nni_val,ref_val,abs_err,rel_err,digits"
    with open(results_file, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, nni_val, ref_val, abs_err, rel_err, d in rows:
            fh.write(f"I{i},{nni_val:.10e},{ref_val:.10e},"
                     f"{abs_err:.6e},{rel_err:.6e},{d}\n")

    summary_file = os.path.join(results_dir, "conv_mock_test_summary.csv")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("architecture,hidden,epochs,n_per_param,lr,seed,device,"
                 "n_weights,final_loss,min_loss,elapsed_s,"
                 "min_digits,mean_digits,max_digits,mean_rel_err,max_rel_err\n")
        fh.write(f"CONV_MOCK,'{args.hidden}',{args.epochs},{args.n_per_param},"
                 f"{args.lr},{args.seed},{device},{integrator.n_weights},"
                 f"{history[-1]:.6e},{min(history):.6e},{elapsed:.2f},"
                 f"{min(digits)},{sum(digits) / len(digits):.2f},{max(digits)},"
                 f"{sum(rel_errs) / len(rel_errs):.6e},{max(rel_errs):.6e}\n")

    print(f"  Results saved -> {results_file}")
    print(f"  Summary saved -> {summary_file}\n")


if __name__ == "__main__":
    main()
