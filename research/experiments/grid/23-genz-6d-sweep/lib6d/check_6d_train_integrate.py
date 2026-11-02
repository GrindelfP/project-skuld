#!/usr/bin/env python3
"""Short CPU validation for the grid/23 isolated six-dimensional package.

Run from any directory:

    python research/experiments/grid/23-genz-6d-sweep/lib6d/check_6d_train_integrate.py

The train/integrate checks use f(u) = 1 on [0, 1]^6, whose closed-form
integral is one.  That value is used only after training for reporting; it is
never supplied to an integrator or its training method.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn


LIB6D = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB6D))

import skuld  # noqa: E402
from skuld._maitre import MaitreIntegrator, mixed_partial_n  # noqa: E402
from skuld.cosiren_ps import CosirenPSIntegrator  # noqa: E402
from skuld.gaussian import GaussianIntegrator  # noqa: E402
from skuld.mixiren import MixirenIntegrator  # noqa: E402
from skuld.sechiren import SechirenIntegrator  # noqa: E402
from skuld.siren import SirenIntegrator  # noqa: E402
from skuld.tanhiren import TanhrenIntegrator  # noqa: E402
from skuld.wire import WireIntegrator  # noqa: E402


N_INT_VARS = 6
PARAMS = (0.0, 0.0)
EXACT = 1.0
DEVICE = torch.device("cpu")


class ProductPrimitive(nn.Module):
    """N(s, u) = product(u_i), with mixed sixth derivative and integral one."""

    n_params = 2
    n_int_vars = N_INT_VARS

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[:, self.n_params:].prod(dim=1, keepdim=True)


def unit_integrand(u: torch.Tensor, *_params: float) -> torch.Tensor:
    """f(u) = 1, returned in the required [N] shape."""
    return torch.ones(u.shape[0], dtype=u.dtype, device=u.device)


def verify_analytic_six_dimensional_path() -> None:
    primitive = ProductPrimitive()
    batch = torch.cat([
        torch.zeros(5, primitive.n_params),
        torch.rand(5, primitive.n_int_vars),
    ], dim=1)
    derivative = mixed_partial_n(primitive, batch)
    assert torch.allclose(derivative, torch.ones_like(derivative)), derivative

    integrator = MaitreIntegrator()
    integrator.n_params = primitive.n_params
    integrator.n_int_vars = primitive.n_int_vars
    integrator.net = primitive
    estimate = integrator.integrate(PARAMS, {PARAMS: 1.0}, DEVICE)
    assert math.isclose(estimate, EXACT, rel_tol=0.0, abs_tol=1e-6), estimate


def builders() -> dict[str, object]:
    common = dict(n_params=2, n_int_vars=N_INT_VARS)
    return {
        "SIREN": SirenIntegrator(**common, hidden_sizes=[8, 8], omega_0=5.0),
        "SECHIREN": SechirenIntegrator(**common, hidden_sizes=[8, 8], omega_0=2.0),
        "WIRE": WireIntegrator(**common, entry_width=8, n_blocks=1,
                               omega_0=2.0, sigma_0=2.0),
        "TANHIREN": TanhrenIntegrator(**common, hidden_sizes=[8, 8], omega_0=2.0),
        "MIXIREN": MixirenIntegrator(**common, hidden_sizes=[8, 8],
                                     omega_0_sech=2.0, omega_0_sin=1.0,
                                     output_scale=1.0),
        "COSIREN-PS": CosirenPSIntegrator(**common, hidden_sizes=[8, 8], omega_0=5.0),
        "GAUSSIAN": GaussianIntegrator(**common, hidden_sizes=[8, 8], omega_0=2.0),
    }


def main() -> None:
    expected_root = LIB6D / "skuld"
    assert Path(skuld.__file__).resolve().is_relative_to(expected_root), skuld.__file__
    print(f"skuld: {skuld.__file__}")

    verify_analytic_six_dimensional_path()
    print("analytic sixth mixed derivative and 64-corner sum: PASS")

    torch.manual_seed(42)
    for name, integrator in builders().items():
        output = integrator.train(
            integrand_fn=unit_integrand,
            param_sets=[PARAMS],
            n_epochs=2,
            n_per_param=8,
            lr=1e-3,
            device=DEVICE,
            verbose_every=0,
        )
        history, norm_cache = output[:2]
        estimate = integrator.integrate(PARAMS, norm_cache, DEVICE)
        assert len(history) == 2
        assert math.isfinite(history[-1]) and math.isfinite(estimate)
        print(f"{name:<10} train->integrate: PASS  "
              f"loss={history[-1]:.3e}  estimate={estimate:.6e}  exact={EXACT:.1f}")


if __name__ == "__main__":
    main()
