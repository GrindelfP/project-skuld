"""
sechiren_is_nni.py — single test of SECHIREN-IS (importance sampling + float64).

Tests the two improvements over the SECHIREN baseline:
  1. Importance sampling for u3: mixture of uniform + concentrated near u3=1
  2. Float64 corner-sum evaluation

Config: best m2 config + both improvements.
  hidden_sizes=[64,64,64], omega_0=20, lr=1e-4, epochs=4000, npp=1024
  is_beta=0.5, is_alpha=2.0, float64_eval=True

Usage:
    python sechiren_is_nni.py
    python sechiren_is_nni.py --epochs 4000 --seed 42
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../skuld-lib'))

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch

from skuld.sechiren_is import SechirenISIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS


def main():
    parser = argparse.ArgumentParser(description="SECHIREN-IS single test")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--is-beta", type=float, default=0.5)
    parser.add_argument("--is-alpha", type=float, default=2.0)
    parser.add_argument("--float64", action="store_true", default=True)
    parser.add_argument("--no-float64", action="store_true")
    args = parser.parse_args()

    float64_eval = not args.no_float64

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    print(f"\n{'=' * 72}")
    print(f"  SECHIREN-IS SINGLE TEST")
    print(f"  device={device}, seed={args.seed}")
    print(f"  hidden=[64,64,64], omega_0=20, lr=1e-4, epochs={args.epochs}, npp=1024")
    print(f"  is_beta={args.is_beta}, is_alpha={args.is_alpha}, float64_eval={float64_eval}")
    print(f"{'=' * 72}\n")

    omega_0 = 20.0
    output_scale = 1.0 / (omega_0 ** 3)

    integrator = SechirenISIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=[64, 64, 64],
        omega_0=omega_0,
        output_scale=output_scale,
        is_beta=args.is_beta,
        is_alpha=args.is_alpha,
        float64_eval=float64_eval,
    )
    print(f"  n_weights = {integrator.n_weights}\n")

    # Reference values
    print("Computing scipy reference integrals...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    # Train
    t0 = time.time()
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=args.epochs,
        n_per_param=1024,
        lr=1e-4,
        device=device,
        verbose_every=500,
    )
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s")
    print(f"  Final loss: {history[-1]:.4e}")
    print(f"  Min loss:   {min(history):.4e}\n")

    # Evaluate
    print(f"{'=' * 72}")
    print(f"  RESULTS")
    print(f"{'=' * 72}")
    header = (f"  {'I':>3} {'NNI':>18} {'ref':>18} {'abs_err':>12} "
              f"{'rel_err':>12} {'digits':>7}")
    print(header)
    print("-" * 72)

    abs_errs, rel_errs, digits = [], [], []
    for i, (a, b, m, n) in enumerate(PARAM_SETS, 1):
        nni_val = integrator.integrate((a, b, m, n), norm_cache=norm_cache, device=device)
        ref_val, _ = refs[(a, b, m, n)]
        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)
        d = max(0, -math.floor(math.log10(abs_err + 1e-30)))
        abs_errs.append(abs_err)
        rel_errs.append(rel_err)
        digits.append(d)
        print(f"  I{i:<3} {nni_val:>18.10e} {ref_val:>18.10e} "
              f"{abs_err:>12.4e} {rel_err:>12.4e} {d:>7}")

    print("-" * 72)
    print(f"  min_dig  = {min(digits)}")
    print(f"  mean_dig = {sum(digits) / len(digits):.1f}")
    print(f"  max_dig  = {max(digits)}")
    print(f"  mean_rel_err = {sum(rel_errs) / len(rel_errs):.4e}")
    print(f"  max_rel_err  = {max(rel_errs):.4e}")
    print(f"{'=' * 72}\n")

    # Save results
    results_dir = get_mirror_path(__file__, "results")
    results_file = os.path.join(results_dir, "sechiren_is_nni.csv")
    with open(results_file, "w", encoding="utf-8") as fh:
        fh.write("I,nni_val,ref_val,abs_err,rel_err,digits\n")
        for i, (a, b, m, n) in enumerate(PARAM_SETS, 1):
            nni_val = integrator.integrate((a, b, m, n), norm_cache=norm_cache, device=device)
            ref_val, _ = refs[(a, b, m, n)]
            abs_err = abs(nni_val - ref_val)
            rel_err = abs_err / (abs(ref_val) + 1e-30)
            d = max(0, -math.floor(math.log10(abs_err + 1e-30)))
            fh.write(f"I{i},{nni_val},{ref_val},{abs_err},{rel_err},{d}\n")
    print(f"  Results saved -> {results_file}\n")


if __name__ == "__main__":
    main()
