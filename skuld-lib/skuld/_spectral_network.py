"""A separable polynomial network with an analytic integral readout."""

from collections.abc import Callable
from string import ascii_letters

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebvander
from numpy.typing import NDArray
from torch import nn

FloatArray = NDArray[np.float64]
GridFunction = Callable[..., FloatArray]


class SpectralNetwork(nn.Module):
    """Learn a bank of univariate features per coordinate and mix their products.

    Each branch is a Chebyshev activation bank followed by a learned linear
    map. The output layer mixes tensor products of the three branch outputs.
    Every output is a polynomial on the unit cube, with known basis moments.
    """

    def __init__(self, branches: list[FloatArray], readout: FloatArray) -> None:
        super().__init__()
        self.branches = nn.ParameterList(nn.Parameter(torch.from_numpy(w)) for w in branches)
        self.readout = nn.Parameter(torch.from_numpy(readout))
        size = branches[0].shape[0]
        even = torch.arange(0, size, 2, dtype=torch.float64)
        moments = torch.zeros(size, dtype=torch.float64)
        moments[::2] = 1 / (1 - even.square())
        self.register_buffer("moments", moments)

    @classmethod
    def fit(
        cls,
        function: GridFunction,
        *,
        degree: int,
        fibers: int,
        readout_nodes: int,
        rank_tolerance: float,
        seed: int,
        dimensions: int = 3,
    ) -> "SpectralNetwork":
        """Fit to vector-valued point samples on fibers and a readout grid.

        Each fiber varies one coordinate, fixing the other two at random
        positions. Truncated SVD learns the dominant univariate features
        across fibers and output channels. On a separate jittered grid,
        one least-squares solve per axis fits the product layer jointly.
        """
        size = degree + 1
        if degree < 1 or fibers < 1 or readout_nodes < size or not 0 < rank_tolerance < 1:
            raise ValueError("Expected a positive degree and fiber count, enough grid nodes, and 0 < tolerance < 1")
        if not 1 <= dimensions <= 25:
            raise ValueError("Expected 1 to 25 dimensions")

        rng = np.random.default_rng(seed)
        angles = np.pi * (np.arange(size) + 0.5) / size
        sites = (1 + np.cos(angles)) / 2
        # Discrete cosine transform at the Chebyshev roots.
        to_coefficients = (2 / size) * np.cos(np.outer(np.arange(size), angles))
        to_coefficients[0] *= 0.5
        branches = []
        for axis in range(dimensions):
            positions = (1 + np.cos(np.pi * rng.random((dimensions, 1, fibers)))) / 2
            coordinates = list(positions)
            coordinates[axis] = sites[:, None]
            values = function(*coordinates).reshape(size, -1)
            coefficients = to_coefficients @ values
            directions, singular_values, _ = np.linalg.svd(coefficients, full_matrices=False)
            width = max(1, np.count_nonzero(singular_values > rank_tolerance * singular_values[0]))
            branches.append(np.ascontiguousarray(directions[:, :width]))

        offsets = rng.uniform(0.25, 0.75, (dimensions, readout_nodes))
        angles = np.pi * (np.arange(readout_nodes) + offsets) / readout_nodes
        grid = (1 + np.cos(angles)) / 2
        values = function(*np.ix_(*grid))
        fit_maps = []
        for points, weights in zip(grid, branches, strict=True):
            features = chebvander(2 * points - 1, degree) @ weights
            inverse, _, rank, _ = np.linalg.lstsq(features, np.eye(readout_nodes), rcond=None)
            if rank != weights.shape[1]:
                raise ValueError("Readout samples do not span the learned branch")
            fit_maps.append(inverse)
        nodes = ascii_letters[:dimensions]
        modes = ascii_letters[dimensions:2 * dimensions]
        output = ascii_letters[2 * dimensions]
        maps = ",".join(mode + node for mode, node in zip(modes, nodes, strict=True))
        readout = np.einsum(f"{maps},{nodes}{output}->{modes}{output}", *fit_maps, values, optimize=True)
        return cls(branches, np.ascontiguousarray(readout))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate all outputs at a batch of points."""
        features = []
        for coordinate, weights in zip(points.unbind(dim=-1), self.branches, strict=True):
            scaled = 2 * coordinate - 1
            activations = [torch.ones_like(scaled), scaled]
            for _ in range(2, len(weights)):
                activations.append(2 * scaled * activations[-1] - activations[-2])
            features.append(torch.stack(activations, dim=-1) @ weights)
        modes = ascii_letters[:len(features)]
        features_spec = ",".join("Z" + mode for mode in modes)
        return torch.einsum(f"{features_spec},{modes}Y->ZY", *features, self.readout)

    def integrate(self) -> torch.Tensor:
        """Integrate each trained output over the unit cube by exact moments."""
        integrals = [self.moments @ weights for weights in self.branches]
        modes = ascii_letters[:len(integrals)]
        return torch.einsum(f"{','.join(modes)},{modes}Y->Y", *integrals, self.readout)
