"""Analytic heat/AD checks. Run in jasher before trusting pilot results."""
from pathlib import Path
import math
import importlib.util

import torch

def main():
    torch.manual_seed(123)
    paths = sorted((Path(__file__).resolve().parent / "experiments" / "flat").glob("*-heat-*/heat_nni.py"))
    assert len(paths) == 3
    for d, path in enumerate(paths, start=1):
        spec = importlib.util.spec_from_file_location(f"heat_pilot_{d}", path)
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        synthetic = pilot.synthetic
        class ExactHeat(pilot.DiffusionIntegrator):
            def forward(self, t, x):
                return pilot.cosine_solution(t, x)

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
        bad = pilot.DiffusionIntegrator(d)
        try:
            bad.fit_initial(lambda points: torch.ones(len(points), 1), steps=1, batch_size=2)
        except ValueError as error:
            assert "shape" in str(error)
        else:
            raise AssertionError("Column labels must be rejected")
        print(f"{d}D: analytic PDE, zero flux, shape and integral checks passed")


if __name__ == "__main__":
    main()
