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

from .genz import GENZ_FUNCTIONS, correct_digits  # noqa: E402

N_INT_VARS = 3


def quadrature_reference(n: int = N_INT_VARS, rtol: float = 1e-12,
                         difficulty: str = "canonical",
                         max_seconds: float = 10.0) -> list[dict]:
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

    Bounded by `max_seconds` per function. Adaptive cubature becomes intractable
    above n~4: at n=6 every one of these five functions ran past 20 s without
    finishing. An unbounded call here silently turns a training benchmark into a
    quadrature benchmark — which is exactly what it did once, hanging what was
    believed to be a 50-epoch network for over an hour.
    """
    try:
        from scipy.integrate import cubature
    except ImportError:
        return []

    import signal

    import numpy as np

    class _Timeout(Exception):
        pass

    def _alarm(_sig, _frm):
        raise _Timeout()

    prev = None
    try:
        prev = signal.signal(signal.SIGALRM, _alarm)
    except (AttributeError, ValueError):
        prev = None  # not on the main thread; fall back to unbounded

    rows = []
    for gfn in GENZ_FUNCTIONS:
        params = gfn.params_for(difficulty)
        exact = gfn.exact(n, params)
        try:
            if prev is not None:
                signal.alarm(int(max_seconds))
            res = cubature(
                lambda Z, _g=gfn, _p=params: _g.numpy_fn(Z, *_p),
                np.zeros(n),
                np.ones(n),
                rtol=rtol,
            )
        except _Timeout:
            rows.append({
                "function": gfn.name,
                "exact": f"{exact:.17g}",
                "estimate": "",
                "rel_err": "",
                "digits": "",
                "status": f"timeout>{max_seconds}s",
            })
            continue
        finally:
            if prev is not None:
                signal.alarm(0)

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
    if prev is not None:
        signal.signal(signal.SIGALRM, prev)
    return rows


def print_quadrature_reference(n: int = N_INT_VARS,
                               difficulty: str = "canonical") -> None:
    rows = quadrature_reference(n=n, difficulty=difficulty)
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
    factory: Callable            # factory(n_int_vars) -> integrator
    train_kwargs: dict           # at n=3
    train_returns_3: bool = False
    provenance: str = ""
    single_seed: bool = False
    # Budget for higher dimensions. An n-th derivative chain costs ~17-21x a
    # third-order one (measured), so the 6-D runs get a reduced budget rather
    # than a silently unaffordable one.
    train_kwargs_6d: dict | None = None

    def kwargs_for(self, n: int) -> dict:
        if n > 3 and self.train_kwargs_6d is not None:
            return dict(self.train_kwargs_6d)
        return dict(self.train_kwargs)


def _siren() -> "ArchSpec":
    from skuld.siren import SirenIntegrator

    return ArchSpec(
        name="SIREN",
        factory=lambda n: SirenIntegrator(
            n_params=2, n_int_vars=n,
            hidden_sizes=[96, 96, 96, 96], omega_0=30.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=5000, n_per_param=1024, lr=2e-4),
        train_kwargs_6d=dict(n_epochs=2000, n_per_param=256, lr=2e-4),
        provenance="best single run (grid/7 m3); no seed column in that sweep",
        single_seed=True,
    )


def _sechiren() -> "ArchSpec":
    from skuld.sechiren import SechirenIntegrator

    return ArchSpec(
        name="SECHIREN",
        factory=lambda n: SechirenIntegrator(
            n_params=2, n_int_vars=n,
            hidden_sizes=[80, 80, 80], omega_0=15.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=2048, lr=2e-4),
        train_kwargs_6d=dict(n_epochs=2000, n_per_param=256, lr=2e-4),
        train_returns_3=True,
        provenance="seed-averaged best config (grid/18, 5 seeds)",
    )


def _wire() -> "ArchSpec":
    from skuld.wire import WireIntegrator

    return ArchSpec(
        name="WIRE",
        factory=lambda n: WireIntegrator(
            n_params=2, n_int_vars=n,
            entry_width=64, n_blocks=3, omega_0=12.0, sigma_0=3.0,
            output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=7000, n_per_param=1024, lr=5e-4),
        train_kwargs_6d=dict(n_epochs=1500, n_per_param=256, lr=4e-4),
        provenance="best single run (grid/7 m3), n_blocks 5->3 for the 6-D budget",
        single_seed=True,
    )


def _tanhiren() -> "ArchSpec":
    from skuld.tanhiren import TanhrenIntegrator

    return ArchSpec(
        name="TANHIREN",
        factory=lambda n: TanhrenIntegrator(
            n_params=2, n_int_vars=n,
            hidden_sizes=[128, 128, 128], omega_0=5.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        train_kwargs_6d=dict(n_epochs=2000, n_per_param=256, lr=5e-4),
        provenance="seed-averaged best config (grid/10, 3 seeds)",
    )


def _mixiren() -> "ArchSpec":
    from skuld.mixiren import MixirenIntegrator

    return ArchSpec(
        name="MIXIREN",
        factory=lambda n: MixirenIntegrator(
            n_params=2, n_int_vars=n,
            hidden_sizes=[128, 128, 128], omega_0_sech=30.0,
            omega_0_sin=1.0, output_scale=1.0 / 900.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        train_kwargs_6d=dict(n_epochs=2000, n_per_param=256, lr=5e-4),
        provenance="flat run at fixed settings",
    )


def _cosiren_ps() -> "ArchSpec":
    from skuld.cosiren_ps import CosirenPSIntegrator

    return ArchSpec(
        name="COSIREN-PS",
        factory=lambda n: CosirenPSIntegrator(
            n_params=2, n_int_vars=n,
            hidden_sizes=[128, 128, 128], omega_0=30.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        train_kwargs_6d=dict(n_epochs=2000, n_per_param=256, lr=5e-4),
        provenance="flat run at fixed settings",
    )


def _gaussian() -> "ArchSpec":
    from skuld.gaussian import GaussianIntegrator

    return ArchSpec(
        name="GAUSSIAN",
        factory=lambda n: GaussianIntegrator(
            n_params=2, n_int_vars=n,
            hidden_sizes=[128, 128, 128], omega_0=30.0, output_scale=1.0,
        ),
        train_kwargs=dict(n_epochs=8000, n_per_param=1024, lr=5e-4),
        train_kwargs_6d=dict(n_epochs=2000, n_per_param=256, lr=5e-4),
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


def train_one(spec: ArchSpec, genz_fn, device, n: int,
              verbose_every: int = 1000, params: tuple | None = None,
              train_kwargs: dict | None = None):
    """Train `spec` on a single Genz function. Returns (integrator, cache, history, seconds)."""
    integrator = spec.factory(n)
    kwargs = dict(train_kwargs if train_kwargs is not None else spec.kwargs_for(n))
    kwargs.setdefault("verbose_every", verbose_every)

    # Not every integrator exposes verbose_every; drop it rather than crash.
    if not _supports(integrator.train, "verbose_every"):
        kwargs.pop("verbose_every", None)

    t0 = time.time()
    out = integrator.train(
        integrand_fn=genz_fn.integrand,
        param_sets=[tuple(params if params is not None else genz_fn.params)],
        device=device,
        **kwargs,
    )
    elapsed = time.time() - t0

    # SECHIREN is the only integrator returning a 3-tuple (history, cache, epoch_data).
    history, norm_cache = out[0], out[1]
    return integrator, norm_cache, history, elapsed


def run_architecture(
    arch_key: str,
    results_dir: Path | None = None,
    device=None,
    smoke: bool = False,
    seed: int = 42,
    n: int = N_INT_VARS,
    difficulty: str = "canonical",
    write: bool = True,
    epochs_6d_override: int | None = None,
) -> list[dict]:
    """
    Train one architecture on all five Genz functions; return the result rows.

    `write=False` suppresses the per-architecture CSV so a benchmark script can
    aggregate several architectures into one wide table. `results_dir` is then
    unused.
    """
    spec = BUILDERS[arch_key]()
    train_kwargs = spec.kwargs_for(n)
    if n > 3 and epochs_6d_override is not None:
        train_kwargs = {**train_kwargs, "n_epochs": epochs_6d_override}
    if smoke:
        train_kwargs = {**train_kwargs, "n_epochs": 50}

    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    torch.manual_seed(seed)

    print("=" * 78)
    print(f"{spec.name} on Genz functions — n={n}, difficulty={difficulty}, "
          f"device={device}")
    print(f"  config: {train_kwargs}   [{spec.provenance}]")
    print("=" * 78)

    rows = []
    for gfn in GENZ_FUNCTIONS:
        params = gfn.params_for(difficulty)
        integrator, norm_cache, history, elapsed = train_one(
            spec, gfn, device, n, params=params, train_kwargs=train_kwargs
        )

        exact = gfn.exact(n, params)
        estimate = integrator.integrate(
            tuple(params), norm_cache=norm_cache, device=device
        )

        abs_err = abs(estimate - exact)
        rel_err = abs_err / abs(exact) if exact != 0 else float("inf")
        digits = correct_digits(estimate, exact)

        n_weights = getattr(integrator, "n_weights", None)
        print(
            f"  {gfn.name:<15} exact={exact:>14.8e}  nni={estimate:>14.8e}  "
            f"abs={abs_err:>9.2e}  rel={rel_err:>9.2e}  digits={digits:>6.2f}  "
            f"loss={history[-1]:.3e}  {elapsed:6.1f}s"
        )

        rows.append({
            "architecture": spec.name,
            "function": gfn.name,
            "n_int_vars": n,
            "difficulty": difficulty,
            "params": str(params),
            "separable": gfn.separable,
            "n_epochs": train_kwargs["n_epochs"],
            "n_per_param": train_kwargs["n_per_param"],
            "lr": train_kwargs["lr"],
            "n_weights": n_weights if n_weights is not None else "",
            "exact": f"{exact:.17g}",
            "estimate": f"{estimate:.17g}",
            "abs_err": f"{abs_err:.6e}",
            "rel_err": f"{rel_err:.6e}",
            "digits": f"{digits:.4f}",
            "final_loss": f"{history[-1]:.6e}",
            "elapsed_s": f"{elapsed:.2f}",
            "device": str(device),
        })

    digits = [float(r["digits"]) for r in rows]
    print(f"\n  across functions: min={min(digits):.2f} "
          f"mean={sum(digits) / len(digits):.2f} max={max(digits):.2f}")
    print(f"  quadrature reference (bounded):")
    for r in quadrature_reference(n=n, difficulty=difficulty):
        print(f"    {r['function']:<15} rel_err={r['rel_err'] or '-':>10}  {r['status']}")

    if not write:
        return rows

    results_dir.mkdir(parents=True, exist_ok=True)
    # A smoke run must never be mistakable for a real one: an identical filename
    # in the results mirror would read as "this experiment ran".
    suffix = "_smoke" if smoke else ""
    out_path = results_dir / (
        f"{spec.name.lower().replace('-', '_')}_genz_n{n}_{difficulty}{suffix}.csv"
    )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {out_path}")

    return rows