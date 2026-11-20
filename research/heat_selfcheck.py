"""Analytic heat/AD checks. Run in jasher before trusting pilot results."""
from pathlib import Path
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skuld-lib"))
from skuld.diffusion_integrator import DiffusionIntegrator
from heat_runner import cosine_solution, synthetic


class ExactHeat(DiffusionIntegrator):
    def forward(self, t, x):
        return cosine_solution(t, x)


def main():
    torch.manual_seed(123)
    for d in (1, 2, 3):
        model = ExactHeat(d).double()
        x = torch.rand(64, d, dtype=torch.float64)
        t = torch.rand(64, 1, dtype=torch.float64)
        assert model.residual(t, x).abs().max().item() < 1e-12
        for axis in range(d):
            for side in (0., 1.):
                face = x.clone()
                face[:, axis] = side
                face.requires_grad_(True)
                gradient = torch.autograd.grad(model(t, face).sum(), face)[0]
                assert gradient[:, axis].abs().max().item() < 1e-12
        assert synthetic(x).shape == (64,)
        assert torch.all(synthetic(x) >= 0)
        # Midpoint integration independently checks all analytic synthetic references.
        n = 100
        axis = (torch.arange(n, dtype=torch.float64) + .5) / n
        mesh = torch.meshgrid(*([axis] * d), indexing="ij")
        points = torch.stack([a.reshape(-1) for a in mesh], dim=1)
        reference = (2-4/math.e, 2-4/math.e+(4-10/math.e)/5, 12/5-5/math.e)[d-1]
        assert abs(synthetic(points).mean().item() - reference) < 1e-4
        bad = DiffusionIntegrator(d)
        try:
            bad.fit_initial(lambda points: torch.ones(len(points), 1), steps=1, batch_size=2)
        except ValueError as error:
            assert "shape" in str(error)
        else:
            raise AssertionError("Column labels must be rejected")
        print(f"{d}D: analytic PDE, zero flux, shape and integral checks passed")


if __name__ == "__main__":
    main()
