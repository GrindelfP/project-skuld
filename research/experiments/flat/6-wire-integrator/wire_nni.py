"""
wire_one.py — single flat test of the WireIntegrator.

Usage:
    python wire_one.py
"""
import math
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from skuld.wire import WireIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS, FLOATING_POINT_PRECISION


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    torch.set_default_dtype(FLOATING_POINT_PRECISION)

    # Hyperparameters
    ENTRY_WIDTH = 64
    N_BLOCKS = 3
    OMEGA_0 = 10.0
    SIGMA_0 = 10.0
    N_EPOCHS = 8000
    N_PER_PARAM = 1024
    LR = 5e-4
    OUTPUT_SCALE = 1.0 / (OMEGA_0 ** 3)

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    print(f"\n  Run started : {run_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device      : {device}")

    integrator = WireIntegrator(
        n_params=4,
        n_int_vars=3,
        entry_width=ENTRY_WIDTH,
        n_blocks=N_BLOCKS,
        omega_0=OMEGA_0,
        sigma_0=SIGMA_0,
        output_scale=OUTPUT_SCALE,
    )
    print(f"\n  Architecture : Wire (Morlet wavelet) + Residual blocks")
    print(f"  Entry layer  : 7 -> {2 * ENTRY_WIDTH}  (WireLayer, is_first=True)")
    print(f"  Residual blocks: {N_BLOCKS} x WireResidualBlock ({2 * ENTRY_WIDTH} channels each)")
    print(f"  Output layer : {2 * ENTRY_WIDTH} -> 1  (linear)")
    print(f"  omega_0      : {OMEGA_0}   sigma_0: {SIGMA_0}")
    print(f"  Parameters   : {integrator.n_weights:,}")
    print(f"  Output scale : {OUTPUT_SCALE:.3e}")

    # Train
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=N_EPOCHS,
        n_per_param=N_PER_PARAM,
        lr=LR,
        device=device,
        verbose_every=500,
    )

    # Reference values
    print("\nComputing reference values (scipy) ...\n")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
        print(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  {r:.6e}  +/-  {e:.1e}")

    # Comparison table
    SEP = '=' * 84
    print(f"\n{SEP}")
    print(f"  {'(a,b,m,n)':^12}  {'Wire-NNI':^14}  {'scipy':^14}  "
          f"{'|Delta|':^11}  {'|Delta|/ref':^9}  {'Digits':^6}")
    print(SEP)

    for (a, b, m, n) in PARAM_SETS:
        nni_val = integrator.integrate((a, b, m, n),
                                        norm_cache=norm_cache, device=device)
        ref_val, ref_err = refs[(a, b, m, n)]
        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)
        correct_digits = max(0, -math.floor(math.log10(abs_err + 1e-30)))

        print(f"   ({int(a)},{int(b)},{int(m)},{int(n)})     "
              f"{nni_val:>14.6e}  {ref_val:>14.6e}  "
              f"{abs_err:>11.3e}  {rel_err:>9.3e}  "
              f"{int(correct_digits):^6}")

    print(SEP)
    print(f"\n  Final loss   : {history[-1]:.4e}")
    print(f"  Minimum loss : {min(history):.4e}")
    print(f"  Loss @ epoch 1: {history[0]:.4e}")
    print(f"\n  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Save log to mirroring results directory
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/flat/6-wire-integrator')
    os.makedirs(results_dir, exist_ok=True)
    log_filename = os.path.join(results_dir, f"results_wire_{timestamp}.out")
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(f"Run: {run_start}\n")
            fh.write(f"Device: {device}\n")
            fh.write(f"Architecture: Wire + Residual blocks\n")
            fh.write(f"Parameters: {integrator.n_weights:,}\n")
            fh.write(f"Epochs: {N_EPOCHS}\n")
            fh.write(f"Final loss: {history[-1]:.4e}\n")
            fh.write(f"Min loss: {min(history):.4e}\n")
            fh.write("\nResults:\n")
            for (a, b, m, n) in PARAM_SETS:
                nni_val = integrator.integrate((a, b, m, n),
                                                norm_cache=norm_cache, device=device)
                ref_val, _ = refs[(a, b, m, n)]
                abs_err = abs(nni_val - ref_val)
                rel_err = abs_err / (abs(ref_val) + 1e-30)
                correct_digits = max(0, -math.floor(math.log10(abs_err + 1e-30)))
                fh.write(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  "
                         f"NNI={nni_val:.6e}  ref={ref_val:.6e}  "
                         f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}  "
                         f"digits={int(correct_digits)}\n")
        print(f"\n  Log saved -> {log_filename}")
    except OSError as exc:
        print(f"\n  [WARNING] Could not save log: {exc}")


if __name__ == "__main__":
    main()
