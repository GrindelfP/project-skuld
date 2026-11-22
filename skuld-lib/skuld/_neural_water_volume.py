"""An end-to-end trained PyTorch surface with an analytic volume readout."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn
from torch.func import functional_call, jacrev

from ._exponential_water_volume import ExponentialWaterSurface


class NeuralWaterSurface(nn.Module):
    """Squared spatial layers, an exponential time layer and a positive output."""

    def __init__(self, start_rates: tuple[float, float]) -> None:
        super().__init__()
        dtype = torch.float64
        self.x_layer = nn.Linear(1, 3, dtype=dtype)
        self.y_layer = nn.Linear(1, 3, dtype=dtype)
        self.x_mixing = nn.Linear(3, 3, dtype=dtype)
        self.y_mixing = nn.Linear(3, 3, dtype=dtype)
        self.time_layer = nn.Linear(1, 2, bias=False, dtype=dtype)
        self.head = nn.Linear(27, 1, dtype=dtype)
        with torch.no_grad():
            for layer in (self.x_layer, self.y_layer):
                layer.weight[:, 0].copy_(torch.tensor([0.0, 0.7, -0.7], dtype=dtype))
                layer.bias.copy_(torch.tensor([1.0, 0.7, 0.7], dtype=dtype))
                layer.weight.add_(0.03 * torch.randn_like(layer.weight))
                layer.bias.add_(0.03 * torch.randn_like(layer.bias))
            for layer in (self.x_mixing, self.y_mixing):
                layer.weight.mul_(0.07).add_(torch.eye(3, dtype=dtype))
                layer.bias.mul_(0.07)
            self.head.weight.mul_(0.15)
            self.head.bias.zero_()
            self.time_layer.weight[:, 0].copy_(torch.tensor(start_rates, dtype=dtype))

    def log_field(self, points: torch.Tensor) -> torch.Tensor:
        x = self.x_mixing(self.x_layer(2 * points[:, 0:1] - 1).square())
        y = self.y_mixing(self.y_layer(2 * points[:, 1:2] - 1).square())
        spatial = (x[:, :, None] * y[:, None, :]).flatten(1)
        time = self.time_layer(points[:, 2:3] - 0.5).exp()
        time = torch.cat((torch.ones_like(time[:, :1]), time), dim=1)
        return self.head((time[:, :, None] * spatial[:, None, :]).flatten(1))[:, 0]

    def forward(self, points: torch.Tensor, *, log: bool = False) -> torch.Tensor:
        field = self.log_field(points)
        return field if log else field.exp()

    def moment_representation(self) -> ExponentialWaterSurface:
        """Expand the fitted layers algebraically, without evaluating data points."""
        def array(parameter: torch.Tensor) -> np.ndarray:
            return parameter.detach().cpu().numpy()

        def quadratic(layer: nn.Linear, mixing: nn.Linear) -> np.ndarray:
            w = array(layer.weight)[:, 0]
            b = array(layer.bias)
            polynomial = array(mixing.weight) @ np.column_stack((b**2, 2 * w * b, w**2))
            polynomial[:, 0] += array(mixing.bias)
            return polynomial

        spatial = np.einsum(
            "ip,jq->ijpq", quadratic(self.x_layer, self.x_mixing), quadratic(self.y_layer, self.y_mixing)
        ).reshape(9, 3, 3)
        coefficients = np.einsum("ij,jpq->ipq", array(self.head.weight).reshape(3, 9), spatial)
        coefficients[0, 0, 0] += array(self.head.bias)[0]
        return ExponentialWaterSurface(array(self.time_layer.weight)[:, 0].copy(), coefficients)


@dataclass(frozen=True)
class FitReport:
    network: NeuralWaterSurface
    initial_rmse: float
    final_rmse: float
    layer_updates: dict[str, float]
    jacobian_evaluations: int


def fit_neural_surface(
    points: np.ndarray, positive_values: np.ndarray, *, seed: int, start_rates: tuple[float, float],
) -> FitReport:
    """Train all layers jointly using a trust region and backpropagated Jacobians."""
    if (points.ndim != 2 or points.shape[1] != 3
            or positive_values.shape != (len(points),)
            or not np.all((0 <= points) & (points <= 1))
            or not np.all(np.isfinite(positive_values) & (positive_values > 0))):
        raise ValueError("Expected positive point samples in the unit cube")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = NeuralWaterSurface(start_rates)
    features = torch.as_tensor(points, dtype=torch.float64)
    target = torch.as_tensor(positive_values, dtype=torch.float64).log()
    parameters = dict(network.named_parameters())
    names = list(parameters)
    sizes = [parameter.numel() for parameter in parameters.values()]
    initial = torch.cat([parameter.detach().flatten() for parameter in parameters.values()])

    def unpack(flat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: value.reshape_as(parameters[name])
                for name, value in zip(names, flat.split(sizes), strict=True)}

    def residual(flat: torch.Tensor) -> torch.Tensor:
        return functional_call(network, unpack(flat), (features,), {"log": True}) - target

    derivative = jacrev(residual)
    lower = np.full(initial.numel(), -np.inf)
    upper = np.full(initial.numel(), np.inf)
    time_start = sum(sizes[:names.index("time_layer.weight")])
    lower[time_start:time_start + 2] = [.15, -8.]
    upper[time_start:time_start + 2] = [8., -.15]

    def objective(flat: np.ndarray) -> np.ndarray:
        return residual(torch.from_numpy(flat)).detach().numpy()

    initial_rmse = float(residual(initial).square().mean().sqrt())
    result = least_squares(
        objective, initial.numpy(),
        jac=lambda flat: derivative(torch.from_numpy(flat)).detach().numpy(),
        bounds=(lower, upper), max_nfev=220,
        xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    if not result.success:
        raise RuntimeError(result.message)
    final = torch.from_numpy(result.x)
    layer_updates = {}
    with torch.no_grad():
        for name, parameter in unpack(final).items():
            layer = name.split(".")[0]
            change = float((parameters[name] - parameter).abs().max())
            layer_updates[layer] = max(layer_updates.get(layer, 0), change)
            parameters[name].copy_(parameter)
    final_rmse = float(np.linalg.norm(result.fun) / np.sqrt(len(points)))
    return FitReport(network, initial_rmse, final_rmse, layer_updates, result.njev)
