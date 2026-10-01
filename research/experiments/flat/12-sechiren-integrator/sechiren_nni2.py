"""
sechiren_nni2.py — parameterized SECHIREN test.

Usage:
    python sechiren_nni2.py <omega_0> <n_epochs> [seed]
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
    omega_0 = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    torch.set_default_dtype(FLOATING_POINT_PRECISION)

    HIDDEN = [128, 128, 128]
    N_PER_PARAM = 1024
    LR = 5e-4
    OUTPUT_SCALE = 1.0 / (omega_0 ** 3)

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    print(f"\n  Run started : {run_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Seed        : {seed}")
    print(f"  Device      : {device}")

    integrator = SechirenIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=HIDDEN,
        omega_0=omega_0,
        output_scale=OUTPUT_SCALE,
    )
    print(f"\n  Architecture : SECHIREN  {HIDDEN}  omega_0={omega_0}")
    print(f"  Parameters   : {integrator.n_weights:,}")
    print(f"  Output scale : {OUTPUT_SCALE:.3e}")
    print(f"  Epochs       : {n_epochs}")

    # Train
    history, norm_cache = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=n_epochs,
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

    # Save log
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/flat/12-sechiren-integrator')
    os.makedirs(results_dir, exist_ok=True)
    log_filename = os.path.join(results_dir, f"results_sechiren_o{omega_0}_e{n_epochs}_seed{seed}_{timestamp}.out")
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(f"Run: {run_start}\n")
            fh.write(f"Seed: {seed}\n")
            fh.write(f"Device: {device}\n")
            fh.write(f"Architecture: SECHIREN {HIDDEN} omega_0={omega_0}\n")
            fh.write(f"Parameters: {integrator.n_weights:,}\n")
            fh.write(f"Output scale: {OUTPUT_SCALE:.3e}\n")
            fh.write(f"Epochs: {n_epochs}\n")
            fh.write(f"N per param: {N_PER_PARAM}\n")
            fh.write(f"LR: {LR}\n")
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
