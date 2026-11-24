"""Continuous conservative heat PINN on the unit cube."""
from math import pi

import torch
from torch import nn


def gradient(value, inputs):
    return torch.autograd.grad(value.sum(), inputs, create_graph=True)[0]


def mlp(in_features, width, out_features):
    return nn.Sequential(
        nn.Linear(in_features, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, out_features),
    )


class InitialNetwork(nn.Module):
    """Fit point samples in cosine coordinates, with zero normal derivative."""

    def __init__(self, width):
        super().__init__()
        self.layers = mlp(3, width, 1)

    def forward(self, xyz):
        return self.layers(torch.cos(pi * xyz)).squeeze(-1)


class HeatNetwork(nn.Module):
    """Learn the time-integrated spatial gradient, with exactly sealed faces."""

    def __init__(self, initial, width):
        super().__init__()
        self.initial = initial.requires_grad_(False)
        self.transient = mlp(4, width, 3)
        self.terminal = mlp(3, width, 3)

    def flux(self, points):
        time = points[:, 3:]
        space = points[:, :3]
        coords = torch.cos(pi * space)
        decay = torch.exp(-pi**2 * time)
        transient = self.transient(torch.cat((coords, 2 * decay - 1), dim=-1))
        terminal = self.terminal(coords)
        return torch.sin(pi * space) * (
            -torch.expm1(-pi**2 * time) * terminal + time * decay * transient
        )

    def field(self, points):
        flux = self.flux(points)
        derivatives = [gradient(flux[:, i], points) for i in range(3)]
        divergence = sum(derivative[:, i] for i, derivative in enumerate(derivatives))
        return self.initial(points[:, :3]) + divergence, derivatives

    def forward(self, points):
        return self.field(points)[0]

    def residual(self, points):
        field, derivatives = self.field(points)
        flux_time = torch.stack([derivative[:, 3] for derivative in derivatives], dim=-1)
        return flux_time - gradient(field, points)[:, :3]


class HeatController(nn.Module):
    """Learn where large local defects occur and propose points to recheck."""

    def __init__(self, width, outputs):
        super().__init__()
        self.layers = mlp(4, width, outputs)

    def forward(self, points):
        return self.layers(points)
