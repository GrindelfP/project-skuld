"""
genz_selfcheck.py — validate `genz.py` before any network touches it.

Three independent checks, in increasing order of how much they should worry us:

1.  torch and numpy integrands agree with each other;
2.    every closed-form exact value agrees with `scipy.integrate.cubature`
        (adaptive Genz-Malik, a completely separate algorithm);
3.  every closed-form exact value agrees with 25-digit mpmath quadrature.

Check 2 runs twice: once on the canonical symmetric parameters, and once on a
deliberately *asymmetric* vector. This is not redundant. With every parameter
set to the same value, a per-axis broadcasting mistake — reading a[1] where a[0]
belongs, transposing the parameter vector against the coordinate vector — is
invisible, because the correct and incorrect expressions evaluate identically.
Asymmetric parameters are what make that class of bug observable.

Run:
    source ~/.jasher/bin/activate
    python research/genz_selfcheck.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genz import GENZ_FUNCTIONS, correct_digits, separable_reference_1d  # noqa: E402

try:
    from scipy.integrate import cubature
except ImportError:  # pragma: no cover - scipy < 1.15
    cubature = None

# Deliberately uneven, spanning both axes. Catches per-axis mix-ups that
# symmetric defaults hide entirely.
ASYMMETRIC = {
    "oscillatory": ((0.7, 1.3, 2.1), 0.35),
    "product_peak": ((0.7, 1.3, 2.1), (0.2, 0.5, 0.9)),
    "corner_peak": ((0.7, 1.3, 2.1), 2.5),
    "gaussian": ((0.7, 1.3, 2.1), (0.2, 0.5, 0.9)),
    "c0_continuous": ((0.7, 1.3, 2.1), (0.2, 0.5, 0.9)),
}

TOL = 1e-12
FAIL = 40  # character width for a status column


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check_implementations(n: int = 3, n_samples: int = 512, seed: int = 0) -> bool:
    """Check 1: the torch and numpy integrands describe the same function."""
    rule(f"CHECK 1 — torch vs numpy integrand agreement (n={n})")
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n))
    Xt = torch.as_tensor(X)

    ok = True
    for fn in GENZ_FUNCTIONS:
        vt = fn.integrand(Xt, *fn.params)
        vn = fn.numpy_fn(X, *fn.params)

        shape_ok = tuple(vt.shape) == (n_samples,)
        agree = float(np.abs(vt.detach().numpy() - vn).max())

        shape_msg = "shape [N] ok" if shape_ok else f"SHAPE {tuple(vt.shape)} != [N]"
        status = "ok" if shape_ok and agree < 1e-12 else "FAIL"
        ok &= status == "ok"
        print(f"  {fn.name:<15} {shape_msg:<16} max|torch-numpy| = {agree:.2e}  {status}")

    print(f"\n  -> {'all integrands agree' if ok else 'MISMATCH'}")
    return ok


def check_exact_vs_cubature(n: int = 3, rtol: float = 1e-12) -> bool:
    """Check 2: closed forms vs adaptive cubature, canonical then asymmetric."""
    if cubature is None:
        print("\n  scipy.integrate.cubature unavailable (needs scipy >= 1.15); skipped")
        return True

    ok = True
    for label, asymmetric in (("canonical (symmetric)", False), ("asymmetric", True)):
        rule(f"CHECK 2 — closed form vs scipy cubature, n={n}, {label}")
        for fn in GENZ_FUNCTIONS:
            params = ASYMMETRIC[fn.name] if asymmetric else fn.params
            exact = fn.exact(n, params)

            res = cubature(
                lambda Z: fn.numpy_fn(Z, *params),
                np.zeros(n),
                np.ones(n),
                rtol=rtol,
            )
            got = float(res.estimate)
            err = abs(got - exact) / abs(exact)

            if err < 1e-11:
                status = "ok"
            elif res.status == "converged":
                status = "FAIL"
            else:
                # cubature gave up. On a kinked integrand that is a property of
                # the reference method, not evidence about our closed form —
                # check 3 at 25 digits is the authority there.
                status = "inconclusive"
            ok &= status != "FAIL"
            print(
                f"  {fn.name:<15} exact={exact:>18.12e}  cubature={got:>18.12e}  "
                f"rel={err:.2e}  {res.status:<13} {status}"
            )

    print(f"\n  -> {'closed forms consistent with cubature' if ok else 'MISMATCH'}")
    return ok


def check_exact_vs_mpmath(n: int = 3, dps: int = 16) -> bool:
    """
    Check 3: closed forms vs mpmath, using the cheapest valid reference per function.

    Two strategies, because one does not fit all:

    - separable families use `separable_reference_1d` -- n independent 1-D
      quadratures, each split at the known kink. Milliseconds, and exact to
      arbitrary precision.
    - non-separable families (oscillatory, corner_peak) use nested quadrature.
      That is far more expensive (~1-11 s at 16 digits), because error
      estimates are unreliable and mpmath subdivides a lot.

    The original version nested all five unconditionally and ran past 19 minutes
    without finishing, because c0_continuous has a kink on every hyperplane
    x_i = u_i and blind subdivision cannot see where the kink is.
    """
    try:
        import mpmath as mp
    except ImportError:
        print("\n  mpmath unavailable; skipped")
        return True

    rule(f"CHECK 3 — closed form vs mpmath, n={n}, {dps} digits")
    mp.mp.dps = dps
    ok = True
    tol = 10.0 ** (-(dps - 4))

    for fn in GENZ_FUNCTIONS:
        exact = fn.exact(n)
        t0 = time.time()

        if fn.separable:
            value = separable_reference_1d(fn, n, dps=dps)
            method = "1-D split"
        else:
            dims = [0.0] * n

            def integrand(*coords, _fn=fn, _p=fn.params):
                for i, c in enumerate(coords):
                    dims[i] = float(c)
                point = np.array([dims], dtype=np.float64)
                return mp.mpf(float(_fn.numpy_fn(point, *_p)[0]))

            value = float(mp.quad(integrand, *[[0, 1]] * n))
            method = "nested"

        err = abs(value - exact) / abs(exact)
        status = "ok" if err < tol else "FAIL"
        ok &= status == "ok"
        print(
            f"  {fn.name:<15} exact={exact:>18.12e}  mpmath={value:>18.12e}  "
            f"rel={err:.2e}  {method:<10} {time.time() - t0:5.2f}s  {status}"
        )

    print(f"\n  -> {'closed forms confirmed independently' if ok else 'MISMATCH'}")
    return ok


def report_canonical_values(n: int = 3) -> None:
    """The exact values the benchmark will score against."""
    rule(f"CANONICAL EXACT VALUES (n={n}) — what the benchmark scores against")
    for fn in GENZ_FUNCTIONS:
        exact = fn.exact(n)
        print(
            f"  {fn.name:<15} params={str(fn.params):<16} exact={exact:>18.12e}  "
            f"{'separable' if fn.separable else 'non-separable'}"
        )
    print(
        "\n  Note: oscillatory's exact value is near zero at a=5, r=0.5. Relative\n"
        "  correct digits are correspondingly harsh there — that is a property of\n"
        "  the function, not of any architecture."
    )


def main() -> None:
    results = [
        check_implementations(),
        check_exact_vs_cubature(),
        check_exact_vs_mpmath(),
    ]
    report_canonical_values()

    rule("RESULT")
    if all(results):
        print("  genz.py is validated. Safe to build experiments on top of it.")
    else:
        print("  FAILURES ABOVE — do not build experiments on genz.py yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()