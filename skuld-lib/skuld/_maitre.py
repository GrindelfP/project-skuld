"""Shared machinery for the Maitre et al. antiderivative integrators."""
import torch


def _nth_mixed_partial(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Differentiate g ([N], built from u) successively w.r.t. each column of u ([N, dim])."""
    for i in range(u.shape[1]):
        # retain_graph MUST stay True even on the final pass. In
        # torch.autograd.grad an omitted retain_graph defaults to the value of
        # create_graph, so the previous three-call implementation — which
        # omitted it on the last call — was in fact retaining the graph, and
        # loss.backward() depends on that. Passing False here frees a graph the
        # backward pass still needs and raises "backward through the graph a
        # second time".
        g = torch.autograd.grad(
            g, u,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0][:, i]

    return g


def mixed_partial_n(net: torch.nn.Module,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    d^n N / du_1 ... du_n by sequential autograd, where n is the number of
    integration variables (the columns of `batch` after the first
    `net.n_params` parameter columns).

    This replaced a former `mixed_partial_3` hardcoded to three columns.
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)

    dim = u.shape[1]
    declared = getattr(net, "n_int_vars", dim)
    if dim != declared:
        raise ValueError(
            f"batch has {dim} integration variables but net declares {declared}"
        )
    if dim < 1:
        raise ValueError(f"need at least 1 integration variable, got {dim}")

    g = net(torch.cat([s, u], dim=1)).squeeze(-1)

    return _nth_mixed_partial(g, u)


# Backwards-compatible alias: the name predates n-dimensional support.
mixed_partial_3 = mixed_partial_n
