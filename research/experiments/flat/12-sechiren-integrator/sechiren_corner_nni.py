"""
sechiren_corner_nni.py — SECHIREN with corner-focused training data.

Adds training points near the corners of the unit hypercube to improve
the corner-sum evaluation. Same config as sechiren_nni.py otherwise.

Usage:
    python sechiren_corner_nni.py [seed]
"""
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from skuld.sechiren import SechirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS, FLOATING_POINT_PRECISION

##############################################################################
#  MAIN
##############################################################################

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    torch.set_default_dtype(FLOATING_POINT_PRECISION)

    # Hyperparameters (same as sechiren_nni.py)
    HIDDEN = [128, 128, 128]
    OMEGA_0 = 30.0
    N_EPOCHS = 8000
    N_PER_PARAM = 1024
    LR = 5e-4
    OUTPUT_SCALE = 1.0 / (OMEGA_0 ** 3)

    # Corner-focused training
    N_CORNER_PER_CORNER = 128
    CORNER_FRACTION = 0.1

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    print(f"\n  Run started : {run_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Seed        : {seed}")
    print(f"  Device      : {device}")

    integrator = SechirenIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=HIDDEN,
        omega_0=OMEGA_0,
        output_scale=OUTPUT_SCALE,
    )
    print(f"\n  Architecture : SECHIREN-CORNER  {HIDDEN}  omega_0={OMEGA_0}")
    print(f"  Parameters   : {integrator.n_weights:,}")
    print(f"  Output scale : {OUTPUT_SCALE:.3e}")
    print(f"  Corner data  : {N_CORNER_PER_CORNER}/corner x 8 corners = {N_CORNER_PER_CORNER * 8} extra pts/param")
    print(f"  Corner frac  : {CORNER_FRACTION}")

    # Train
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=N_EPOCHS,
        n_per_param=N_PER_PARAM,
        lr=LR,
        device=device,
        verbose_every=500,
        n_corner_per_corner=N_CORNER_PER_CORNER,
        corner_fraction=CORNER_FRACTION,
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
    print(f"  {'(a,b,m,n)':^12}  {'NNI':^14}  {'scipy':^14}  "
          f"{'|delta|':^11}  {'|delta|/ref':^9}  {'Digits':^6}")
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
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/flat/12-sechiren-integrator')
    os.makedirs(results_dir, exist_ok=True)
    log_filename = os.path.join(results_dir, f"results_sechiren_corner_seed{seed}_{timestamp}.out")
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(f"Run: {run_start}\n")
            fh.write(f"Seed: {seed}\n")
            fh.write(f"Device: {device}\n")
            fh.write(f"Architecture: SECHIREN-CORNER {HIDDEN} omega_0={OMEGA_0}\n")
            fh.write(f"Parameters: {integrator.n_weights:,}\n")
            fh.write(f"Output scale: {OUTPUT_SCALE:.3e}\n")
            fh.write(f"Epochs: {N_EPOCHS}\n")
            fh.write(f"N per param: {N_PER_PARAM}\n")
            fh.write(f"LR: {LR}\n")
            fh.write(f"N corner per corner: {N_CORNER_PER_CORNER}\n")
            fh.write(f"Corner fraction: {CORNER_FRACTION}\n")
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
