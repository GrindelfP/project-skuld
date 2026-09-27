"""
test_siren_flat.py — quick flat test of SirenIntegrator.

Trains a small SIREN for a few epochs and evaluates one integral.
Proves the new lib structure works end-to-end.

Usage:
    python test_siren_flat.py
"""
import os
import sys
import math
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from skuld.siren import SirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    print("=" * 60)
    print("  QUICK FLAT TEST: SirenIntegrator")
    print("=" * 60)

    integrator = SirenIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=[64, 64, 64],
        omega_0=30.0,
        output_scale=1.0 / (30.0 ** 3),
    )
    print(f"  Network: {integrator.n_weights:,} parameters")

    t0 = time.time()
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS[:2],
        n_epochs=50,
        n_per_param=256,
        lr=5e-4,
        device=device,
        verbose_every=10,
    )
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.1f}s")

    # Test one integral
    params = (0, 0, 1, 2)
    nni_val = integrator.integrate(params, norm_cache=norm_cache, device=device)
    ref_val, ref_err = reference_scipy(*params)
    abs_err = abs(nni_val - ref_val)
    rel_err = abs_err / (abs(ref_val) + 1e-30)

    print(f"\n  Result for params {params}:")
    print(f"    NNI:   {nni_val:.8e}")
    print(f"    Ref:   {ref_val:.8e}")
    print(f"    Error: {abs_err:.3e} (rel: {rel_err:.3e})")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/flat/5-siren-integrator')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'test_siren_flat.out')
    with open(out_path, 'w') as f:
        f.write("QUICK FLAT TEST: SirenIntegrator\n")
        f.write("=" * 60 + "\n")
        f.write(f"Network: {integrator.n_weights:,} parameters\n")
        f.write(f"Epochs: 50\n")
        f.write(f"N per param: 256\n")
        f.write(f"Training time: {elapsed:.1f}s\n")
        f.write(f"Final loss: {history[-1]:.6e}\n")
        f.write(f"Min loss: {min(history):.6e}\n\n")
        f.write(f"Result for params {params}:\n")
        f.write(f"  NNI:   {nni_val:.8e}\n")
        f.write(f"  Ref:   {ref_val:.8e}\n")
        f.write(f"  Error: {abs_err:.3e} (rel: {rel_err:.3e})\n")

    print(f"\n  Results saved to: {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
