"""
check_mixed_partial.py — verify `mixed_partial_n` against finite differences.

The antiderivative integrators learn an antiderivative N and score it by
requiring that its n-th mixed partial derivative equal the integrand f. The
previous implementation took exactly three autograd passes over columns 0, 1, 2
regardless of the true dimension, which silently produced wrong answers for any
other n — the loss went down, the corner sum came out confident and wrong.

An autograd-based check cannot catch an autograd-based bug, so this validates
the loop against central finite differences in float64. The finite-difference
routine shares no code path with autograd.

Sanity of the finite-difference routine itself is established first, against
polynomials whose mixed partials are known in closed form. Without that check
a bug in the checker is indistinguishable from a bug in the library.

Run:
    source ~/.jasher/bin/activate
    python research/check_mixed_partial.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skuld-lib"))

from skuld.siren import SirenPrimitiveNet, mixed_partial_3  # noqa: E402


def fd_mixed(fn, n: int, h: float, u0: torch.Tensor) -> float:
    """
    d^n f / du_1 ... du_n by central differences over all 2**n sign corners.

        (1 / h^n) * sum_s (prod_i s_i) f(u + h s)

    Each coordinate is perturbed independently, and each corner carries the
    product of its signs. Dropping either piece turns this into a plain average
    of f over the corners, which is not a derivative at all.
    """
    total = 0.0
    for signs in itertools.product((-1, 1), repeat=n):
        step = torch.tensor([s * h for s in signs], dtype=torch.float64)
        weight = 1.0
        for s in signs:
            weight *= s
        total += weight * float(fn(u0 + step))
    return total / (2.0**n) / (h**n)


def optimal_h(n: int) -> float:
    """
    Step size balancing truncation (O(h^2)) against roundoff (O(eps/h^n)).

    Central differences of order n divide by h^n, so a step tuned for n=3
    destroys an n=6 result: at h=1e-4 the denominator is 1e-24 and float64
    roundoff swamps the answer entirely (it returns exactly 0.0).
    """
    return float(10.0 ** (-16.0 / (n + 2)))


def fd_steps(n: int) -> tuple[float, ...]:
    """
    Two step sizes for cross-checking convergence.

    The textbook optimum eps^(1/(n+2)) assumes the integrand is O(1) and free of
    cancellation. The 6th mixed partial of a SIREN here is ~1e-5 while the
    network output is O(1), so 64 corner values cancel away roughly ten digits
    before the division. In practice a much coarser step converges and a finer
    one is pure noise. Requiring agreement between two coarse steps is a more
    honest convergence test than any single step.
    """
    if n <= 3:
        return (1e-2, 6e-3)
    return (5e-2, 3e-2)


def check_fd_routine() -> bool:
    """Validate the finite-difference routine against known polynomials."""
    print("  finite-difference routine self-test:")
    ok = True

    # These polynomials are smooth and O(1) with no cancellation, so they
    # validate the routine at the fine, textbook step size.
    # f = prod_j x_j^3  =>  d^n f / du_1..du_n = 3^n * prod_j u_j^2
    for n in (3, 6):
        def f(x, _n=n):
            return torch.prod(x[:_n] ** 3)

        u0 = torch.full((n,), 0.5, dtype=torch.float64)
        expected = float(3**n * 0.5 ** (2 * n))
        got = fd_mixed(f, n, optimal_h(n), u0)
        good = abs(got - expected) / abs(expected) < 1e-3
        ok &= good
        print(
            f"    n={n}  prod x^3: expected={expected:+.8e}  fd={got:+.8e}  "
            f"h={optimal_h(n):.1e}  {'ok' if good else 'FAIL'}"
        )

    # f = exp(sum u_j)  =>  every mixed partial equals f itself
    for n in (3, 6):
        def f(x, _n=n):
            return torch.exp(x[:_n].sum())

        u0 = torch.full((n,), 0.4, dtype=torch.float64)
        expected = float(torch.exp(u0.sum()))
        got = fd_mixed(f, n, optimal_h(n), u0)
        good = abs(got - expected) / abs(expected) < 1e-3
        ok &= good
        print(
            f"    n={n}  exp(sum):  expected={expected:+.8e}  fd={got:+.8e}  "
            f"h={optimal_h(n):.1e}  {'ok' if good else 'FAIL'}"
        )

    return ok


class _ProductNet:
    """N(u) = prod_j u_j, whose n-th mixed partial is exactly 1 everywhere."""

    def __init__(self, n: int):
        self.n_params = 0
        self.n_int_vars = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        v = x[0, 0]
        for j in range(1, self.n_int_vars):
            v = v * x[0, j]
        return v.reshape(1, 1)


class _CubicNet:
    """N(u) = (u1+u2+u3)^3 / 6, whose third mixed partial is exactly 1."""

    def __init__(self):
        self.n_params = 0
        self.n_int_vars = 3

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        s = x[0, 0] + x[0, 1] + x[0, 2]
        return (s**3 / 6).reshape(1, 1)


def check_known_answers() -> bool:
    """
    Exact verification against networks whose mixed partial is known by hand.

    This is the decisive test: no quadrature, no tolerances, no scale ambiguity.
    A fixed third-order implementation cannot pass it at n=6, because it
    differentiates only three coordinates and the product of six is not affine
    in the first three.
    """
    print("  known-answer networks (exact):")
    ok = True

    for n in (3, 6):
        net = _ProductNet(n)
        x = torch.full((1, n), 0.37, dtype=torch.float64)
        got = float(mixed_partial_3(net, x)[0])
        good = abs(got - 1.0) < 1e-12
        ok &= good
        print(
            f"    n={n}  prod(u): d^n N/du^n = 1 exactly  ->  got {got!r}  "
            f"{'ok' if good else 'FAIL'}"
        )

    net = _CubicNet()
    x = torch.full((1, 3), 0.4, dtype=torch.float64)
    got = float(mixed_partial_3(net, x)[0])
    good = abs(got - 1.0) < 1e-12
    ok &= good
    print(
        f"    n=3  (sum u)^3/6: d^3 N = 1 exactly  ->  got {got!r}  "
        f"{'ok' if good else 'FAIL'}"
    )

    return ok


def check_autograd() -> bool:
    """
    Compare the library's mixed_partial_3 against finite differences of the
    SAME function, using a scale-relative tolerance.

    Both sides must differentiate the network itself. Comparing the network's
    autograd derivative against finite differences of an unrelated hand-written
    function compares two different quantities.

    The tolerance is relative to the largest derivative magnitude seen across
    sample points, not to each point individually: a SIREN's third derivative
    can pass near a zero, where any relative comparison becomes meaningless
    even when both methods are accurate.
    """
    torch.manual_seed(0)
    ok = True

    for n in (3, 6):
        net = SirenPrimitiveNet(
            n_params=0, n_int_vars=n, hidden_sizes=[16, 16], omega_0=1.0
        )
        net.double()

        pairs = []
        for k in range(6):
            torch.manual_seed(100 + k)
            base = torch.rand(1, n, dtype=torch.float64) * 0.6 + 0.2
            ad = float(mixed_partial_3(net, base.clone())[0].detach())
            # fd_mixed evaluates fn(u0 + step): the callback receives the FULL
            # point, so it must not add the base a second time.
            estimates = [
                fd_mixed(
                    lambda point, _net=net: float(_net(point).squeeze()),
                    n,
                    h,
                    base,
                )
                for h in fd_steps(n)
            ]
            pairs.append((ad, estimates))

        scale = max(
            max([abs(a)] + [abs(e) for e in est]) for a, est in pairs
        ) or 1.0
        worst = max(
            abs(a - e) for a, est in pairs for e in est
        ) / scale
        good = worst < 1e-2
        ok &= good
        print(
            f"    n={n}  6 points x {len(fd_steps(n))} step sizes, "
            f"max |autograd - fd| / scale = {worst:.2e}  "
            f"(h={fd_steps(n)})  {'ok' if good else 'FAIL'}"
        )

    return ok


def check_dimension_guard() -> bool:
    """A net must refuse a batch whose width contradicts its declared dimension."""
    net3 = SirenPrimitiveNet(n_params=0, n_int_vars=3, hidden_sizes=[8], omega_0=1.0)
    try:
        mixed_partial_3(net3, torch.rand(4, 6, dtype=torch.float64))
    except ValueError as exc:
        print(f"    3-D net fed a 6-variable batch -> ValueError: {exc}")
        return True
    print("    3-D net fed a 6-variable batch -> NO ERROR (bad)")
    return False


def main() -> None:
    torch.set_default_dtype(torch.float64)

    print("=" * 74)
    print("VERIFY mixed_partial_n — the n-th mixed partial of the antiderivative")
    print("=" * 74)

    print("\n[1] finite-difference routine self-test")
    fd_ok = check_fd_routine()

    print("\n[2] known-answer networks (exact)")
    ka_ok = check_known_answers()

    print("\n[3] library implementation vs finite differences")
    ad_ok = check_autograd()

    print("\n[4] dimension guard")
    guard_ok = check_dimension_guard()

    print("\n" + "=" * 74)
    if fd_ok and ka_ok and ad_ok and guard_ok:
        print("  PASS — mixed_partial_n verified for n=3 and n=6.")
    else:
        print("  FAIL — do not use this for n != 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()