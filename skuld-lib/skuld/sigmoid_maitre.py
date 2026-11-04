"""Sigmoid primitive network for Maître-style numerical integration.

The architecture and Xavier/output-scale initialization follow the original
sigmoid primitive experiment; training uses the shared mixed-partial loop.
"""
import torch
import torch.nn as nn

from ._maitre import MaitreIntegrator


class SigmoidPrimitiveNet(nn.Module):
    def __init__(self, n_params=4, n_int_vars=3, hidden_sizes=(128, 128, 128),
                 output_scale=1e4):
        super().__init__()
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        layers = []
        in_dim = n_params + n_int_vars
        for width in hidden_sizes:
            layers.extend((nn.Linear(in_dim, width), nn.Sigmoid()))
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(output_scale)
            self.net[-1].bias.mul_(output_scale)

    def forward(self, x):
        return self.net(x)


class SigmoidMaitreIntegrator(MaitreIntegrator):
    def __init__(self, n_params=4, n_int_vars=3, hidden_sizes=(128, 128, 128),
                 output_scale=1e4):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = SigmoidPrimitiveNet(n_params, n_int_vars, hidden_sizes,
                                       output_scale)
