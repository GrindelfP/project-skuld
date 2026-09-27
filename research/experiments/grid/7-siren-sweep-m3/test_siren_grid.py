"""
test_siren_grid.py — quick grid test of SirenIntegrator.

Runs a tiny hyperparameter sweep (2x2 grid) to prove the grid test structure works.

Usage:
    python test_siren_grid.py
"""
import os
import sys
import csv
import time
import itertools

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from skuld.siren import SirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    print("=" * 60)
    print("  QUICK GRID TEST: SirenIntegrator")
    print("=" * 60)

    # Tiny 2x2 grid
    hidden_sizes_list = [[64, 64, 64], [128, 128, 128]]
    omega_0_list = [30.0, 45.0]

    results = []
    for hidden, omega_0 in itertools.product(hidden_sizes_list, omega_0_list):
        print(f"\n  Testing: hidden={hidden}, omega_0={omega_0}")
        t0 = time.time()

        integrator = SirenIntegrator(
            n_params=4,
            n_int_vars=3,
            hidden_sizes=hidden,
            omega_0=omega_0,
            output_scale=1.0 / (omega_0 ** 3),
        )

        history, norm_cache = integrator.train(
            integrand_fn=integrand_transformed,
            param_sets=PARAM_SETS[:2],
            n_epochs=30,
            n_per_param=256,
            lr=5e-4,
            device=device,
            verbose_every=0,
        )

        # Evaluate
        nni_val = integrator.integrate((0, 0, 1, 2), norm_cache=norm_cache, device=device)
        ref_val, _ = reference_scipy(0, 0, 1, 2)
        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s, Loss: {history[-1]:.4e}, RelErr: {rel_err:.3e}")

        results.append({
            'hidden_sizes': str(hidden),
            'omega_0': omega_0,
            'n_weights': integrator.n_weights,
            'final_loss': history[-1],
            'min_loss': min(history),
            'nni_result': nni_val,
            'ref_result': ref_val,
            'abs_error': abs_err,
            'rel_error': rel_err,
            'elapsed_s': elapsed,
        })

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/grid/7-siren-sweep-m3')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'test_siren_grid.out')
    with open(out_path, 'w') as f:
        f.write("QUICK GRID TEST: SirenIntegrator\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ran {len(results)} configs\n\n")
        for r in results:
            f.write(f"hidden={r['hidden_sizes']}, omega_0={r['omega_0']}\n")
            f.write(f"  weights: {r['n_weights']:,}\n")
            f.write(f"  final_loss: {r['final_loss']:.6e}\n")
            f.write(f"  min_loss: {r['min_loss']:.6e}\n")
            f.write(f"  NNI: {r['nni_result']:.8e}\n")
            f.write(f"  Ref: {r['ref_result']:.8e}\n")
            f.write(f"  Error: {r['abs_error']:.3e} (rel: {r['rel_error']:.3e})\n")
            f.write(f"  Time: {r['elapsed_s']:.1f}s\n\n")

    print(f"\n  Results saved to: {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
