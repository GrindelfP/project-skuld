"""Experimental insulated-heat PINN. Integral labels are never used in fitting."""
import math
import time

import torch
from torch import nn


class DiffusionIntegrator(nn.Module):
    """Fit U_t = Laplacian(U), U(0,x)=f(x), with hard zero normal flux.

    The terminal point value is a biased experimental integral estimate. A small
    collocation loss or flat terminal field does not certify conservation.
    """

    def __init__(self, dimension, width=32, horizon=1.0):
        super().__init__()
        if dimension < 1 or horizon <= 0:
            raise ValueError("Positive dimension and horizon required")
        self.dimension, self.horizon = dimension, horizon
        self.net = nn.Sequential(nn.Linear(dimension + 1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, t, x):
        features = torch.cat((t / self.horizon, torch.cos(math.pi * x)), dim=1)
        return self.net(features).squeeze(-1)

    def residual(self, t, x):
        t = t.detach().requires_grad_(True)
        x = x.detach().requires_grad_(True)
        value = self(t, x)
        dt, dx = torch.autograd.grad(value.sum(), (t, x), create_graph=True)
        lap = torch.zeros_like(value)
        for axis in range(self.dimension):
            second = torch.autograd.grad(dx[:, axis].sum(), x, create_graph=True)[0]
            lap = lap + second[:, axis]
        return dt[:, 0] - lap

    def fit_initial(self, point_fn, steps=512, batch_size=128, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        device = next(self.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        history = []
        observed_sum = 0.
        self.train()
        for step in range(steps):
            x0 = torch.rand(batch_size, self.dimension, device=device)
            with torch.no_grad():
                y0 = point_fn(x0)
            if y0.shape != (batch_size,):
                raise ValueError("Initial labels must have shape [N]")
            observed_sum += y0.sum().item()
            initial_loss = (self(torch.zeros(batch_size, 1, device=device), x0)
                            - y0).square().mean()
            x = torch.rand(batch_size, self.dimension, device=device)
            t = self.horizon * torch.rand(batch_size, 1, device=device)
            # Half of collocations resolve the fast initial transient.
            t[:batch_size // 2] = self.horizon * torch.rand(batch_size // 2, 1, device=device).square()
            pde_loss = self.residual(t, x).square().mean()
            loss = initial_loss + pde_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 64 == 0 or step + 1 == steps:
                history.append(dict(step=step + 1, initial_mse=initial_loss.item(),
                                    pde_mse=pde_loss.item()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return history, time.perf_counter() - start, observed_sum / (steps * batch_size)

    def integrate(self):
        device = next(self.parameters()).device
        with torch.no_grad():
            return self(torch.full((1, 1), self.horizon, device=device),
                        torch.full((1, self.dimension), 1 / 3, device=device)).item()
