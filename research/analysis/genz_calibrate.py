"""
genz_calibrate.py — find the Genz difficulty tier that matches the meson task.

The canonical Genz1984 parameters (a = 5) put every architecture at ~3 correct
digits. The meson Feynman integrand, our target benchmark, reaches ~5. Those are
not comparable numbers: a benchmark that is uniformly brutal compresses the
field and stops discriminating between architectures.

Difficulty in these families is governed almost entirely by the shape parameter
`a`, which sets both the oscillation frequency (oscillatory) and the peak width
(product peak, Gaussian, C^0). Genz's own guidance is that `a` dominates
difficulty while the offset `u` does not.

This sweeps `a` for one representative architecture and reports digits per
function, so the tier is chosen from measurement rather than taste.

Run:
    source ~/.jasher/bin/activate
    python research/genz_calibrate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skuld-lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.genz import GENZ_FUNCTIONS, correct_digits  # noqa: E402
from skuld.sechiren import SechirenIntegrator  # noqa: E402

A_VALUES = (5.0, 3.0, 2.0, 1.0)
N_EPOCHS = 3000
MESON_TARGET = 5.0


def run(a: float, n: int, device) -> dict:
    """Train one architecture on all five functions at shape parameter `a`."""
    out = {}
    for gfn in GENZ_FUNCTIONS:
        if gfn.name == "corner_peak":
            params = (a, 1.0)          # corner peak's a is the *scale*, not a width
        else:
            params = (a, 0.5)

        integrator = SechirenIntegrator(
            n_params=2, n_int_vars=n,
            hidden_sizes=[80, 80, 80], omega_0=15.0, output_scale=1.0,
        )
        kwargs = dict(
            integrand_fn=gfn.integrand,
            param_sets=[params],
            device=device,
            n_epochs=N_EPOCHS,
            n_per_param=1024,
            lr=2e-4,
        )
        try:
            kwargs["verbose_every"] = 0
        except Exception:
            pass

        res = integrator.train(**kwargs)
        history, norm_cache = res[0], res[1]

        exact = gfn.exact(n, params)
        est = integrator.integrate(params, norm_cache=norm_cache, device=device)
        out[gfn.name] = correct_digits(est, exact)
    return out


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(42)
    n = 3

    names = [f.name for f in GENZ_FUNCTIONS]
    print("Genz difficulty calibration — SECHIREN, n=3, seed 42")
    print(f"{N_EPOCHS} epochs per cell; meson reference target ~{MESON_TARGET} digits\n")
    print(f"{'a':>5}" + "".join(f"{x[:11]:>13}" for x in names) + f"{'MEAN':>8}")
    print("-" * 78)

    rows = []
    for a in A_VALUES:
        t0 = time.time()
        r = run(a, n, device)
        vals = [r[k] for k in names]
        mean = sum(vals) / len(vals)
        rows.append((a, mean, r))
        print(f"{a:>5.1f}" + "".join(f"{v:>13.2f}" for v in vals) + f"{mean:>8.2f}"
              f"   ({time.time() - t0:.0f}s)")

    print("\n" + "=" * 78)
    closest = min(rows, key=lambda r: abs(r[1] - MESON_TARGET))
    print(f"  meson reference: ~{MESON_TARGET} mean digits")
    print(f"  closest tier: a = {closest[0]:.1f}  ->  {closest[1]:.2f} mean digits")
    if closest[1] < MESON_TARGET - 1.0:
        print("  (still harder than meson; consider a < %.1f)" % closest[0])
    elif closest[1] > MESON_TARGET + 1.0:
        print("  (easier than meson; consider a > %.1f)" % closest[0])


if __name__ == "__main__":
    main()