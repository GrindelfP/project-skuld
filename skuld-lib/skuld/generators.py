##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################
import itertools
import math
from typing import TypeAlias, Annotated, Callable, Any

import numpy as np
import torch
from torch import Tensor

# noinspection PyProtectedMember

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]


##########################################################################
###                     PER-DIMENSION SAMPLERS                         ###
###                                                                    ###
###  Each function returns a 1-D torch.Tensor of `n_samples` values   ###
###  drawn from [lower, upper] according to the given strategy.        ###
##########################################################################

def _sample_dim_std(lower: float, upper: float, n: int) -> torch.Tensor:
    """Uniform random (Monte-Carlo) samples."""
    return torch.rand(n) * (upper - lower) + lower


def _sample_dim_grd(lower: float, upper: float, n: int) -> torch.Tensor:
    """Uniform deterministic grid (linspace)."""
    return torch.linspace(lower, upper, n)


def _sample_dim_log(
        lower: float, upper: float, n: int, epsilon: float = 1e-6
) -> torch.Tensor:
    """Logarithmically-spaced grid."""
    start_val = lower if lower > 1e-9 else epsilon
    log_start  = np.log10(start_val)
    log_end    = np.log10(upper)
    samples    = torch.logspace(log_start, log_end, n)
    if lower == 0:
        samples[0] = 0.0
    return samples


def _sample_dim_mix(
        lower: float, upper: float, n: int,
        split_ratio: float = 0.2,
        density_ratio: float = 0.8,
) -> torch.Tensor:
    """Dense-near-lower + sparse-near-upper mixed grid."""
    n_dense  = int(n * density_ratio)
    n_sparse = n - n_dense
    if n_dense < 2 or n_sparse < 1:
        raise ValueError(
            f"n_samples={n} is too small for density_ratio={density_ratio}. "
            "Increase n_samples or adjust density_ratio."
        )
    split_val = lower + (upper - lower) * split_ratio
    x_dense   = torch.linspace(lower, split_val, n_dense)
    x_sparse  = torch.linspace(split_val, upper, n_sparse + 1)[1:]
    return torch.cat([x_dense, x_sparse])


def _sample_dim_exp(
        lower: float, upper: float, n: int, decay_rate: float = 5.0
) -> torch.Tensor:
    """Exponentially-biased random samples (dense near `lower`)."""
    R       = upper - lower
    max_val = 1.0 - torch.exp(torch.tensor(-decay_rate * R))
    u       = torch.rand(n)
    return -(1.0 / decay_rate) * torch.log(1.0 - u * max_val) + lower


# Map string tag → sampler callable
_STOCHASTIC_TYPES = {"std", "exp"}          # rows are independent random draws
_GRID_TYPES       = {"grd", "log", "mix"}   # rows form a deterministic axis

_DIM_SAMPLERS: dict[str, Callable] = {
    "std": _sample_dim_std,
    "grd": _sample_dim_grd,
    "log": _sample_dim_log,
    "mix": _sample_dim_mix,
    "exp": _sample_dim_exp,
}


##########################################################################
###                        HELPER: SINGLE DIM                          ###
##########################################################################
def _sample_single_dim(
        dis_type: str,
        lower: float,
        upper: float,
        n: int,
        **kwargs: Any,
) -> torch.Tensor:
    """
    Dispatch to the appropriate 1-D sampler.
    Extra kwargs (decay_rate, epsilon, split_ratio, density_ratio) are
    forwarded only to the samplers that accept them.
    """
    if dis_type not in _DIM_SAMPLERS:
        raise ValueError(
            f"Unknown distribution type '{dis_type}'. "
            f"Valid types: {sorted(_DIM_SAMPLERS)}"
        )
    sampler = _DIM_SAMPLERS[dis_type]

    # Build the subset of kwargs accepted by this sampler
    import inspect
    sig    = inspect.signature(sampler)
    params = set(sig.parameters) - {"lower", "upper", "n"}
    kw     = {k: v for k, v in kwargs.items() if k in params}

    return sampler(lower, upper, n, **kw)


##########################################################################
###                    ORIGINAL PER-TYPE GENERATORS                    ###
###   (kept for backward compatibility; called by generate_data when   ###
###    dis_type is a single string)                                     ###
##########################################################################
def generate_data_unfirm_grid(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
    Generates data on a uniform deterministic grid.

    :param func:        function to evaluate.
    :param lower:       lower bounds (length n_dim).
    :param upper:       upper bounds (length n_dim).
    :param n_samples:   points per axis for grid types; total points for
                        stochastic types.
    :param n_dim:       number of dimensions.
    :param func_params: additional keyword arguments forwarded to func.
    :returns: (X, y)
    """
    if n_dim == 1:
        X: Matrix = torch.linspace(lower[0], upper[0], n_samples).view(n_samples, 1)
        y: Vector = func(X, **func_params).view(n_samples, 1)
    else:
        ranges = [torch.linspace(lower[n], upper[n], n_samples).tolist()
                  for n in range(n_dim)]
        combinations: list = list(itertools.product(*ranges))
        X: Matrix = torch.tensor(combinations, dtype=torch.float32)
        y: Vector = func(X, **func_params).view(-1, 1)

    return X, y


def generate_data_uniform(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        **func_args: Any
) -> tuple[Matrix, Vector]:
    """
    Generates uniformly-distributed random samples.

    :param func:      function to evaluate.
    :param lower:     lower bounds (length n_dim).
    :param upper:     upper bounds (length n_dim).
    :param n_samples: number of random samples.
    :param n_dim:     number of dimensions.
    :param func_args: additional keyword arguments forwarded to func.
    :returns: (X, y)
    """
    if n_dim == 1:
        X: Matrix = torch.rand(n_samples, 1) * (upper[0] - lower[0]) + lower[0]
        y: Vector = func(X, **func_args).view(n_samples, 1)
    else:
        X: Matrix = torch.rand(n_samples, n_dim)
        for i in range(n_dim):
            X[:, i] = X[:, i] * (upper[i] - lower[i]) + lower[i]
        y: Vector = func(X, **func_args).view(n_samples, 1)
    return X, y


def generate_data_logarithmic(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        epsilon: float = 1e-6,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
    Generates data on a logarithmically-spaced grid.

    :param func:      function to evaluate.
    :param lower:     lower bounds (length n_dim).
    :param upper:     upper bounds (length n_dim).
    :param n_samples: points per axis.
    :param n_dim:     number of dimensions.
    :param epsilon:   substitute for zero lower bounds to avoid log(0).
    :param func_params: additional keyword arguments forwarded to func.
    :returns: (X, y)
    """
    ranges = []
    for i in range(n_dim):
        start_val = lower[i] if lower[i] > 1e-9 else epsilon
        log_start = np.log10(start_val)
        log_end   = np.log10(upper[i])
        dim_range = torch.logspace(log_start, log_end, n_samples)
        if lower[i] == 0:
            dim_range[0] = 0.0
        ranges.append(dim_range.tolist())

    if n_dim == 1:
        X: Matrix = torch.tensor(ranges[0], dtype=torch.float32).view(-1, 1)
    else:
        combinations: list = list(itertools.product(*ranges))
        X: Matrix = torch.tensor(combinations, dtype=torch.float32)

    y: Vector = func(X, **func_params).view(-1, 1)
    return X, y


def generate_data_mixed(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        split_ratio: float = 0.2,
        density_ratio: float = 0.8,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
    Generates data on a mixed dense-near-lower / sparse-near-upper grid.

    :param func:          function to evaluate.
    :param lower:         lower bounds (length n_dim).
    :param upper:         upper bounds (length n_dim).
    :param n_samples:     total points per axis.
    :param n_dim:         number of dimensions.
    :param split_ratio:   fraction of [lower, upper] that is the "dense" sub-interval.
    :param density_ratio: fraction of n_samples placed in the dense sub-interval.
    :param func_params:   additional keyword arguments forwarded to func.
    :returns: (X, y)
    """
    n_dense  = int(n_samples * density_ratio)
    n_sparse = n_samples - n_dense
    if n_dense < 2 or n_sparse < 1:
        raise ValueError("n_samples is too small for the given density_ratio")

    ranges = []
    for i in range(n_dim):
        l, u      = lower[i], upper[i]
        split_val = l + (u - l) * split_ratio
        x_dense   = torch.linspace(l, split_val, n_dense)
        x_sparse  = torch.linspace(split_val, u, n_sparse + 1)[1:]
        ranges.append(torch.cat([x_dense, x_sparse]).tolist())

    if n_dim == 1:
        X: Matrix = torch.tensor(ranges[0], dtype=torch.float32).view(-1, 1)
    else:
        combinations: list = list(itertools.product(*ranges))
        X: Matrix = torch.tensor(combinations, dtype=torch.float32)

    y: Vector = func(X, **func_params).view(-1, 1)
    return X, y


def generate_data_exponential(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 1000,
        n_dim: int = 1,
        decay_rate: float = 5.0,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
    Generates exponentially-biased random samples (dense near `lower`).

    :param func:        function to evaluate.
    :param lower:       lower bounds (length n_dim).
    :param upper:       upper bounds (length n_dim).
    :param n_samples:   number of random samples.
    :param n_dim:       number of dimensions.
    :param decay_rate:  controls how steeply density falls off from lower bound.
    :param func_params: additional keyword arguments forwarded to func.
    :returns: (X, y)
    """
    U       = torch.rand(n_samples, n_dim)
    X_parts = []
    for i in range(n_dim):
        R       = upper[i] - lower[i]
        max_val = 1.0 - torch.exp(torch.tensor(-decay_rate * R))
        xi      = -(1.0 / decay_rate) * torch.log(1.0 - U[:, i] * max_val)
        X_parts.append(xi + lower[i])

    X: Matrix = torch.stack(X_parts, dim=1)
    y: Vector = func(X, **func_params).view(-1, 1)
    return X, y


##########################################################################
###               PER-DIMENSION MIXED-STRATEGY GENERATOR               ###
##########################################################################
def generate_data_mixed_dims(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        dis_types: list[str],
        n_samples: int = 100,
        n_dim: int = 1,
        decay_rate:    float = 5.0,
        split_ratio:   float = 0.2,
        density_ratio: float = 0.8,
        epsilon:       float = 1e-6,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
    Generates training data where each dimension can have its own
    sampling strategy. Stochastic dimensions (``std``, ``exp``) are
    sampled jointly to produce ``n_samples`` rows; deterministic
    dimensions (``grd``, ``log``, ``mix``) each contribute an axis of
    ``n_samples`` points that is crossed (via Cartesian product) with
    all other deterministic axes — and then broadcast against the
    stochastic rows.

    In practice this means the total number of rows in X is::

        n_samples  ×  ∏(n_samples for each deterministic dim)

    which for a typical setup of 2 stochastic + 1 stochastic + k grid
    parameter dims gives ``n_samples × n_samples^k`` rows.

    :param func:          integrand / function to evaluate.
    :param lower:         lower bounds, one per dimension.
    :param upper:         upper bounds, one per dimension.
    :param dis_types:     list of distribution-type strings, one per dimension.
                          Supported: ``"std"``, ``"exp"``, ``"grd"``, ``"log"``, ``"mix"``.
    :param n_samples:     number of points per axis (stochastic dims) or per
                          grid axis (deterministic dims).
    :param n_dim:         number of dimensions (must equal ``len(dis_types)``).
    :param decay_rate:    ``exp`` sampler: exponential decay rate.
    :param split_ratio:   ``mix`` sampler: dense sub-interval fraction.
    :param density_ratio: ``mix`` sampler: fraction of points in dense region.
    :param epsilon:       ``log`` sampler: substitute for zero lower bound.
    :param func_params:   extra keyword arguments forwarded to func.
    :returns: ``(X, y)`` tensors.

    Example
    -------
    For a 4-dimensional integrand where the first two dimensions are
    integration variables sampled uniformly, the third is the variable
    ``t`` sampled with an exponential bias, and the fourth is a physical
    parameter swept on a coarse uniform grid::

        X, y = generate_data_mixed_dims(
            func=my_integrand,
            lower=[0.0, 0.0, 0.0, 0.1],
            upper=[1.0, 1.0, 50.0, 2.0],
            dis_types=["std", "std", "exp", "grd"],
            n_samples=200,
            n_dim=4,
            decay_rate=3.0,
        )
    """
    if len(dis_types) != n_dim:
        raise ValueError(
            f"len(dis_types)={len(dis_types)} must equal n_dim={n_dim}."
        )

    # Shared kwargs forwarded to per-dim samplers
    sampler_kwargs = dict(
        decay_rate    = decay_rate,
        split_ratio   = split_ratio,
        density_ratio = density_ratio,
        epsilon       = epsilon,
    )

    # ------------------------------------------------------------------ #
    # Separate stochastic and deterministic dimension indices             #
    # ------------------------------------------------------------------ #
    stochastic_idx   = [i for i, t in enumerate(dis_types) if t in _STOCHASTIC_TYPES]
    deterministic_idx = [i for i, t in enumerate(dis_types) if t in _GRID_TYPES]

    unknown = [t for t in dis_types if t not in _DIM_SAMPLERS]
    if unknown:
        raise ValueError(
            f"Unknown distribution type(s): {unknown}. "
            f"Valid: {sorted(_DIM_SAMPLERS)}"
        )

    # ------------------------------------------------------------------ #
    # Sample stochastic dimensions jointly (shape: n_samples × n_stoch)  #
    # ------------------------------------------------------------------ #
    if stochastic_idx:
        stoch_cols = []
        for i in stochastic_idx:
            col = _sample_single_dim(dis_types[i], lower[i], upper[i],
                                     n_samples, **sampler_kwargs)
            stoch_cols.append(col)
        # Stack: (n_samples, n_stoch)
        stoch_block = torch.stack(stoch_cols, dim=1)
    else:
        stoch_block = None   # no stochastic dims

    # ------------------------------------------------------------------ #
    # Sample deterministic dimensions, then take their Cartesian product  #
    # ------------------------------------------------------------------ #
    if deterministic_idx:
        det_axes = []
        for i in deterministic_idx:
            axis = _sample_single_dim(dis_types[i], lower[i], upper[i],
                                      n_samples, **sampler_kwargs)
            det_axes.append(axis.tolist())
        det_combinations = list(itertools.product(*det_axes))
        # Shape: (n_det_combos, n_det)
        det_block = torch.tensor(det_combinations, dtype=torch.float32)
    else:
        det_block = None   # no deterministic dims

    # ------------------------------------------------------------------ #
    # Combine: broadcast stochastic rows against deterministic combos     #
    # Result shape: (n_samples * n_det_combos, n_dim)                    #
    # ------------------------------------------------------------------ #
    if stoch_block is not None and det_block is not None:
        n_stoch_rows = stoch_block.shape[0]   # n_samples
        n_det_rows   = det_block.shape[0]     # n_det_combos

        # Repeat each stochastic row for every deterministic combo
        stoch_rep = stoch_block.repeat_interleave(n_det_rows, dim=0)
        # Tile the deterministic combos for every stochastic row
        det_rep   = det_block.repeat(n_stoch_rows, 1)

        # Build full X with columns in original dimension order
        total_rows = n_stoch_rows * n_det_rows
        X = torch.empty(total_rows, n_dim, dtype=torch.float32)
        for col_out, col_src in enumerate(stochastic_idx):
            src_col = stochastic_idx.index(col_src)
            X[:, col_src] = stoch_rep[:, src_col]
        for col_out, col_src in enumerate(deterministic_idx):
            X[:, col_src] = det_rep[:, col_out]

    elif stoch_block is not None:
        X = stoch_block
    else:
        # Only deterministic dims — det_block is already the full grid
        X = det_block

    y: Vector = func(X, **func_params).view(-1, 1)
    return X, y


##########################################################################
###                          MAIN DISPATCHER                           ###
##########################################################################
def generate_data(
        dis_type: str | list[str],
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        decay_rate:    float = 5.0,   # EXP distribution
        split_ratio:   float = 0.2,   # MIX distribution
        density_ratio: float = 0.8,   # MIX distribution
        epsilon:       float = 1e-6,  # LOG distribution
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
    Unified entry point for all data-generation strategies.

    ``dis_type`` can be either:

    * a **single string** — the same strategy is applied to all dimensions
      (original behaviour, fully backward-compatible); or
    * a **list of strings** — one strategy per dimension, enabling
      heterogeneous sampling (calls :func:`generate_data_mixed_dims`).

    Supported strategy tags: ``"std"``, ``"exp"``, ``"grd"``, ``"log"``, ``"mix"``.

    For a single-string ``dis_type`` the ``n_samples`` interpretation is:

    * ``"std"`` / ``"exp"`` — total number of random rows.
    * ``"grd"`` / ``"log"`` / ``"mix"`` — points *per axis*; total rows
      = ``n_samples ** n_dim``.

    When a list is supplied the same per-dim rule applies inside
    :func:`generate_data_mixed_dims`.

    :param dis_type:      strategy tag (str) or list of per-dim tags.
    :param func:          function / integrand to evaluate.
    :param lower:         lower bounds (length n_dim).
    :param upper:         upper bounds (length n_dim).
    :param n_samples:     points per axis (or total rows for stochastic types).
    :param n_dim:         number of input dimensions.
    :param decay_rate:    ``exp`` sampler parameter.
    :param split_ratio:   ``mix`` sampler parameter.
    :param density_ratio: ``mix`` sampler parameter.
    :param epsilon:       ``log`` sampler parameter (avoids log(0)).
    :param func_params:   extra keyword arguments forwarded to func.
    :returns: ``(X, y)`` — input tensor and function-value tensor.
    """
    # ------------------------------------------------------------------ #
    # Per-dimension mode                                                  #
    # ------------------------------------------------------------------ #
    if isinstance(dis_type, list):
        return generate_data_mixed_dims(
            func=func,
            lower=lower,
            upper=upper,
            dis_types=dis_type,
            n_samples=n_samples,
            n_dim=n_dim,
            decay_rate=decay_rate,
            split_ratio=split_ratio,
            density_ratio=density_ratio,
            epsilon=epsilon,
            **func_params
        )

    # ------------------------------------------------------------------ #
    # Original single-strategy mode (backward compatible)                #
    # ------------------------------------------------------------------ #
    if dis_type in _STOCHASTIC_TYPES:
        set_size = n_samples
    else:
        root: float = n_samples ** (1 / n_dim)
        set_size = math.ceil(root)

    if dis_type == "exp":
        return generate_data_exponential(
            func=func, lower=lower, upper=upper,
            n_samples=set_size, n_dim=n_dim,
            decay_rate=decay_rate, **func_params
        )
    elif dis_type == "log":
        return generate_data_logarithmic(
            func=func, lower=lower, upper=upper,
            n_samples=set_size, n_dim=n_dim,
            epsilon=epsilon, **func_params
        )
    elif dis_type == "mix":
        return generate_data_mixed(
            func=func, lower=lower, upper=upper,
            n_samples=set_size, n_dim=n_dim,
            split_ratio=split_ratio,
            density_ratio=density_ratio, **func_params
        )
    elif dis_type == "std":
        return generate_data_uniform(
            func=func, lower=lower, upper=upper,
            n_samples=set_size, n_dim=n_dim, **func_params
        )
    elif dis_type == "grd":
        return generate_data_unfirm_grid(
            func=func, lower=lower, upper=upper,
            n_samples=set_size, n_dim=n_dim, **func_params
        )
    else:
        raise ValueError(
            f"Unknown distribution type: '{dis_type}'. "
            f"Valid types: {sorted(_DIM_SAMPLERS)}"
        )
