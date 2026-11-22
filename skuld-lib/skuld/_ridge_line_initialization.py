"""Initialize sine ridge frequencies from parallel lines sampled in the cube."""

from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._water_volume import FloatArray


def infer_frequencies(
    oracle: Callable[[FloatArray], FloatArray], direction: FloatArray, limit: float,
) -> tuple[FloatArray | None, int]:
    """Recover phase shifts of candidate modes from evenly spaced parallel fibers."""
    n = 49
    dimension = len(direction)
    positions = np.linspace(-0.5, 0.5, n)
    spacing = float(positions[1] - positions[0])
    displacement = 1 / 16
    shifts = np.vstack((np.zeros(dimension), np.eye(dimension))) * displacement
    points = 0.5 + positions[:, None, None] * direction + shifts[None, :, :]
    values = oracle(points.reshape(-1, dimension)).reshape(n, -1)
    count = values.size
    scale = float(np.max(np.abs(values)))
    if not scale:
        return None, count
    values = values / scale

    # A sampled finite sine sum is a sum of complex geometric sequences.
    # The shift on the low-rank Hankel subspace has their poles as eigenvalues.
    windows = sliding_window_view(values.T, 21, axis=-1).reshape(-1, 21)
    left, singular, right = np.linalg.svd(windows[:, :-1], full_matrices=False)
    rank = int(np.count_nonzero(singular > 5e-12 * singular[0]))
    if rank < 2 or rank > 13:
        return None, count
    shift = (left[:, :rank].T @ windows[:, 1:] @ right[:rank].T) / singular[:rank, None]
    poles = np.linalg.eigvals(shift)
    if np.max(np.abs(np.abs(poles) - 1)) > 4e-3:
        return None, count
    angles = np.angle(poles)
    projections = np.sort(angles[angles > 1e-6]) / spacing
    width = len(projections)
    if not width or rank - 2 * width not in (0, 1):
        return None, count

    phases = positions[:, None] * projections
    basis = np.column_stack((np.ones(n), np.cos(phases), np.sin(phases)))
    coefficients = np.linalg.lstsq(basis, values, rcond=None)[0]
    if np.max(np.abs(basis @ coefficients - values)) > 2e-7:
        return None, count
    amplitudes = coefficients[1:width + 1] - 1j * coefficients[width + 1:]
    if np.min(np.abs(amplitudes)) < 1e-9:
        return None, count
    frequencies = np.angle(amplitudes[:, 1:] / amplitudes[:, :1]) / displacement
    if np.max(np.abs(frequencies @ direction - projections)) > 0.02:
        return None, count
    if np.max(np.abs(frequencies)) >= limit:
        return None, count
    return frequencies, count
