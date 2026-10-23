"""
mock_test.py — Quick smoke test for AenTrop grid sweep.

Validates all code paths work: construction, forward, corner_sum, train, integrate.
Runs 2 epochs on CPU with dummy integrand.
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from skuld.aentrop import AenTropIntegrator


def dummy_integrand(u: torch.Tensor, a: float, b: float, m: float, n: float) -> torch.Tensor:
    """Simple polynomial integrand: u1^a * u2^b * u3^m."""
    return u[:, 0] ** a * u[:, 1] ** b * u[:, 2] ** m


def exact_integral(a: float, b: float, m: float, n: float) -> float:
    """Exact integral of u1^a * u2^b * u3^m over [0,1]^3 = 1/((a+1)(b+1)(m+1))."""
    return 1.0 / ((a + 1) * (b + 1) * (m + 1))


def main():
    print("=" * 60)
    print("  AenTrop SWEEP MOCK TEST — smoke validation")
    print("=" * 60)

    device = torch.device("cpu")
    torch.manual_seed(42)

    # Test config
    param_sets = [(0, 0, 1, 2), (1, 1, 1, 1)]
    targets = {p: exact_integral(*p) for p in param_sets}

    print(f"\n  Device: {device}")
    print(f"  Param sets: {param_sets}")
    print(f"  Targets: {targets}")

    # 1. Construction
    print("\n  [1/5] Construction ...")
    aentrop = AenTropIntegrator(
        n_params=4,
        n_int_vars=3,
        n_pieces=12,
        tau_start=1.0,
        tau_end=0.01,
    )
    print(f"      OK — {aentrop.n_weights:,} parameters")

    # 2. Forward pass
    print("\n  [2/5] Forward pass ...")
    aentrop.net.eval()
    u = torch.rand(4, 3)
    theta = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
    with torch.no_grad():
        out = aentrop.net.forward(u, theta, tau=0.01)
    print(f"      Input shape: {u.shape}")
    print(f"      Output shape: {out.shape}")
    assert out.shape == (4,), f"Expected [4], got {out.shape}"
    print("      OK")

    # 3. Corner sum
    print("\n  [3/5] Corner sum ...")
    with torch.no_grad():
        cs = aentrop.net.corner_sum(theta, tau=0.01)
    print(f"      Corner sum shape: {cs.shape}")
    assert cs.shape == (1,), f"Expected [1], got {cs.shape}"
    print(f"      Value: {cs.item():.6e}")
    print("      OK")

    # 4. Training (2 epochs)
    print("\n  [4/5] Training (2 epochs) ...")
    history = aentrop.train(
        integrand_fn=dummy_integrand,
        param_sets=param_sets,
        targets=targets,
        n_epochs=2,
        n_per_param=16,
        lr=1e-3,
        device=device,
        verbose_every=1,
    )
    print(f"      Loss history: {[f'{h:.4e}' for h in history]}")
    assert len(history) == 2
    # Loss may increase due to aggressive tau annealing (1.0 -> 0.1 in 2 epochs)
    # Just check it's finite
    assert all(h > 0 and h < 1e6 for h in history), "Loss should be finite"
    print("      OK")

    # 5. Integration
    print("\n  [5/5] Integration ...")
    for params in param_sets:
        val = aentrop.integrate(params, device=device, tau=0.01)
        exact = targets[params]
        rel_err = abs(val - exact) / (abs(exact) + 1e-30)
        print(f"      {params}: NNI={val:.6e}, exact={exact:.6e}, rel_err={rel_err:.4e}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()