"""
genz_runner.py — shared driver for the Genz characterisation benchmark.

One architecture is trained on each of the five Genz functions at a fixed,
hand-picked configuration, and scored against that function's closed-form
exact integral. This is a *characterisation* benchmark: synthetic functions
whose answers are known analytically, used to measure method quality
independently of the meson Feynman integral in `physics.py`, which is the
target benchmark. Neither substitutes for the other.

Scoring
-------
Accuracy is reported as **relative** correct digits, ``-log10(|I - I_exact| /
|I_exact|)``, never absolute: exact values in this function set span 4e-2 to
1, and an absolute metric would rank by the magnitude of the answer rather
than the quality of the method.

Each architecture/function cell is a single number — there is one fixed
parameter set per function, so there is nothing to average over. min/mean/max
are taken **across the five functions**, which is the axis that actually
varies here.

Two caveats worth carrying into any report:

- oscillatory's exact value is -4.76e-3 at a=5, r=0.5 — nearly zero by
  accident of parameters. Relative digits are correspondingly harsh there, and
  that is a property of the function, not of any architecture.
- corner_peak's exact value is 1/(n!(n+1)) = 1/24. It is a relative-accuracy
  trap: absolute error can look excellent while relative error is poor.
"""

from __future__ import annotations

import csv
import inspect
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genz import GENZ_FUNCTIONS, correct_digits  # noqa: E402

N_INT_VARS = 3


def quadrature_reference(n: int = N_INT_VARS, rtol: float = 1e-12) -> list[dict]:
    """
    Reference row: what a tuned classical method achieves on the same functions.

    Uses scipy.integrate.cubature (adaptive Genz-Malik). This is NOT the ground
    truth and does not define correctness — the closed forms in `genz.py` do.
    It is a calibration yardstick, so a low digit count from a network can be
    read as "weak method" vs "brutally hard function".

    Reports the *achieved* relative error rather than a digit count: cubature
    converges to ~1e-15 here, where "digits" stops being a meaningful number
    before it stops being interesting. The parity bar for this benchmark is a
    neural antiderivative matching ~1e-10..1e-13.
    """
    try:
        from scipy.integrate import cubature
    except ImportError:
        return []

    import numpy as np

    rows = []
    for gfn in GENZ_FUNCTIONS:
        exact = gfn.exact(n)
        res = cubature(
            lambda Z: gfn.numpy_fn(Z, *gfn.params),
            np.zeros(n),
            np.ones(n),
            rtol=rtol,
        )
        estimate = float(res.estimate)
        rel = abs(estimate - exact) / abs(exact)
        rows.append({
            "function": gfn.name,
            "exact": f"{exact:.17g}",
            "estimate": f"{estimate:.17g}",
            "rel_err": f"{rel:.3e}",
            "digits": f"{correct_digits(estimate, exact):.2f}",
            "status": res.status,
        })
    return rows


def print_quadrature_reference() -> None:
    rows = quadrature_reference()
    if not rows:
        return
    print("  reference (scipy cubature, adaptive Genz-Malik) — calibration, not truth:")
    for r in rows:
        print(f"    {r['function']:<15} rel_err={r['rel_err']}  {r['status']}")


# ---------------------------------------------------------------------------
# architecture registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchSpec:
    """How to build one integrator and train it."""

    name: str
    factory: Callable
    train_kwargs: dict
    train_returns_3: bool = False
    params: dict = field(default_factory=dict)
    provenance: str = ""
    single_seed: bool = False


def _siren() -> "ArchSpec":
    from skuld.siren import SirenIntegrator

    return ArchSpec(
        name="SIREN",
        factory=lambda: SirenIntegrator(
            n_params=2, n_int_vars=N_INT_VARS,
            hidden_sizes=[96, 96, 96, 96], omega_0=30.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=5000, n_per_param=1024, lr=2e-4),
        provenance="best single run (grid/7 m3); no seed column in that sweep",
        single_seed=True,
    )


def _sechiren() -> "ArchSpec":
    from skuld.sechiren import SechirenIntegrator

    return ArchSpec(
        name="SECHIREN",
        factory=lambda: SechirenIntegrator(
            n_params=2, n_int_vars=N_INT_VARS,
            hidden_sizes=[80, 80, 80], omega_0=15.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=2048, lr=2e-4),
        train_returns_3=True,
        provenance="seed-averaged best config (grid/18, 5 seeds)",
    )


def _wire() -> "ArchSpec":
    from skuld.wire import WireIntegrator

    return ArchSpec(
        name="WIRE",
        factory=lambda: WireIntegrator(
            n_params=2, n_int_vars=N_INT_VARS,
            entry_width=64, n_blocks=5, omega_0=12.0, sigma_0=3.0,
            output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=7000, n_per_param=1024, lr=5e-4),
        provenance="best single run (grid/7 m3); no seed column in that sweep",
        single_seed=True,
    )


def _tanhiren() -> "ArchSpec":
    from skuld.tanhiren import TanhrenIntegrator

    return ArchSpec(
        name="TANHIREN",
        factory=lambda: TanhrenIntegrator(
            n_params=2, n_int_vars=N_INT_VARS,
            hidden_sizes=[128, 128, 128], omega_0=5.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        provenance="seed-averaged best config (grid/10, 3 seeds)",
    )


def _mixiren() -> "ArchSpec":
    from skuld.mixiren import MixirenIntegrator

    return ArchSpec(
        name="MIXIREN",
        factory=lambda: MixirenIntegrator(
            n_params=2, n_int_vars=N_INT_VARS,
            hidden_sizes=[128, 128, 128], omega_0_sech=30.0,
            omega_0_sin=1.0, output_scale=1.0 / 900.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        provenance="flat run at fixed settings",
    )


def _cosiren_ps() -> "ArchSpec":
    from skuld.cosiren_ps import CosirenPSIntegrator

    return ArchSpec(
        name="COSIREN-PS",
        factory=lambda: CosirenPSIntegrator(
            n_params=2, n_int_vars=N_INT_VARS,
            hidden_sizes=[128, 128, 128], omega_0=30.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        provenance="flat run at fixed settings",
    )


def _gaussian() -> "ArchSpec":
    from skuld.gaussian import GaussianIntegrator

    return ArchSpec(
        name="GAUSSIAN",
        factory=lambda: GaussianIntegrator(
            n_params=2, n_int_vars=N_INT_VARS,
            hidden_sizes=[128, 128, 128], omega_0=30.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        provenance="flat run at fixed settings",
    )


BUILDERS = {
    "siren": _siren,
    "sechiren": _sechiren,
    "wire": _wire,
    "tanhiren": _tanhiren,
    "mixiren": _mixiren,
    "cosiren_ps": _cosiren_ps,
    "gaussian": _gaussian,
}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _supports(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def train_one(spec: ArchSpec, genz_fn, device, verbose_every: int = 1000):
    """Train `spec` on a single Genz function. Returns (integrator, cache, history, seconds)."""
    integrator = spec.factory()
    kwargs = dict(spec.train_kwargs)
    kwargs.setdefault("verbose_every", verbose_every)

    # Not every integrator exposes verbose_every; drop it rather than crash.
    if not _supports(integrator.train, "verbose_every"):
        kwargs.pop("verbose_every", None)

    t0 = time.time()
    out = integrator.train(
        integrand_fn=genz_fn.integrand,
        param_sets=[tuple(genz_fn.params)],
        device=device,
        **kwargs,
    )
    elapsed = time.time() - t0

    # SECHIREN is the only integrator returning a 3-tuple (history, cache, epoch_data).
    history, norm_cache = out[0], out[1]
    return integrator, norm_cache, history, elapsed


def run_architecture(
    arch_key: str,
    results_dir: Path,
    device=None,
    smoke: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Train one architecture on all five Genz functions; write a CSV; return rows."""
    spec = BUILDERS[arch_key]()
    if smoke:
        spec = ArchSpec(
            name=spec.name,
            factory=spec.factory,
            train_kwargs={**spec.train_kwargs, "n_epochs": 50},
            train_returns_3=spec.train_returns_3,
            provenance=spec.provenance,
            single_seed=spec.single_seed,
        )

    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    torch.manual_seed(seed)

    print("=" * 78)
    print(f"{spec.name} on Genz functions — n_int_vars={N_INT_VARS}, device={device}")
    print(f"  config: {spec.train_kwargs}   [{spec.provenance}]")
    print("=" * 78)

    rows = []
    for gfn in GENZ_FUNCTIONS:
        integrator, norm_cache, history, elapsed = train_one(spec, gfn, device)

        exact = gfn.exact(N_INT_VARS)
        estimate = integrator.integrate(
            tuple(gfn.params), norm_cache=norm_cache, device=device
        )
        digits = correct_digits(estimate, exact)
        rel = abs(estimate - exact) / abs(exact)

        n_weights = getattr(integrator, "n_weights", None)
        print(
            f"  {gfn.name:<15} exact={exact:>14.8e}  nni={estimate:>14.8e}  "
            f"rel={rel:>9.2e}  digits={digits:>6.2f}  "
            f"loss={history[-1]:.3e}  {elapsed:6.1f}s"
        )

        rows.append({
            "architecture": spec.name,
            "function": gfn.name,
            "params": str(gfn.params),
            "separable": gfn.separable,
            "n_epochs": spec.train_kwargs["n_epochs"],
            "n_per_param": spec.train_kwargs["n_per_param"],
            "lr": spec.train_kwargs["lr"],
            "n_weights": n_weights if n_weights is not None else "",
            "exact": f"{exact:.17g}",
            "estimate": f"{estimate:.17g}",
            "rel_err": f"{rel:.6e}",
            "digits": f"{digits:.4f}",
            "final_loss": f"{history[-1]:.6e}",
            "elapsed_s": f"{elapsed:.2f}",
            "device": str(device),
        })

    digits = [float(r["digits"]) for r in rows]
    print(f"\n  across functions: min={min(digits):.2f} "
          f"mean={sum(digits) / len(digits):.2f} max={max(digits):.2f}")
    print_quadrature_reference()

    results_dir.mkdir(parents=True, exist_ok=True)
    # A smoke run must never be mistakable for a real one: an identical filename
    # in the results mirror would read as "this experiment ran".
    suffix = "_smoke" if smoke else ""
    out_path = results_dir / f"{spec.name.lower().replace('-', '_')}_genz{suffix}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {out_path}")

    return rows