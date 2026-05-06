import torch
import torch.nn as nn

from .integrand import center_value, integrand_transformed

from .physics import FLOATING_POINT_PRECISION

##############################################################################
# 5.  DERIVATIVE ∂³N/∂u₁∂u₂∂u₃ WITH AUTOGRAD
##############################################################################

def mixed_partial_3(net: nn.Module,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    ∂³N/∂u₁∂u₂∂u₃ with sequential autograd.

    batch: (N, n_params + 3)
        - [:, :n_params]  — s (non-differentable)
        - [:, n_params:]  — u₁,u₂,u₃ (differentable)
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)  # (N, 3)

    inp = torch.cat([s, u], dim=1)
    N_out = net(inp).squeeze(-1)  # (N,)

    ones_N = torch.ones_like(N_out)

    # ∂N/∂u₁
    g1 = torch.autograd.grad(
        N_out, u,
        grad_outputs=ones_N,
        create_graph=True,
        retain_graph=True,
    )[0][:, 0]  # (N,)

    # ∂²N/∂u₁∂u₂
    g12 = torch.autograd.grad(
        g1, u,
        grad_outputs=torch.ones_like(g1),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1]  # (N,)

    # ∂³N/∂u₁∂u₂∂u₃
    g123 = torch.autograd.grad(
        g12, u,
        grad_outputs=torch.ones_like(g12),
        create_graph=True,  # for backward() during training
    )[0][:, 2]  # (N,)

    return g123


##############################################################################
# 6.  BATCHES
##############################################################################

def make_batch(param_sets: list,
               n_per_param: int,
               device: torch.device,
               norm_cache: dict = None) -> tuple:
    if norm_cache is None:
        norm_cache = {}

    all_xu, all_f = [], []

    for (a, b, m, n) in param_sets:
        if (a, b, m, n) not in norm_cache:
            norm_cache[(a, b, m, n)] = center_value(a, b, m, n)
        fc = norm_cache[(a, b, m, n)]

        u = torch.rand(n_per_param, 3)
        f = integrand_transformed(u[:, 0], u[:, 1], u[:, 2], a, b, m, n) / fc
        s = torch.tensor([a, b, m, n], dtype=FLOATING_POINT_PRECISION).expand(n_per_param, -1)
        xu = torch.cat([s, u], dim=1)

        all_xu.append(xu)
        all_f.append(f)

    batch_xu = torch.cat(all_xu, dim=0).to(device)
    f_tilde = torch.cat(all_f, dim=0).to(device)
    return batch_xu, f_tilde, norm_cache


##############################################################################
# 8.  CHANGING SIGN SUM EVALUATION
##############################################################################

def compute_integral(net: nn.Module,
                     a: float, b: float, m: float, n: float,
                     norm_cache: dict = None,
                     device: torch.device = torch.device("cpu")) -> float:
    net.eval()
    net.to(device)

    if norm_cache and (a, b, m, n) in norm_cache:
        fc = norm_cache[(a, b, m, n)]
    else:
        fc = center_value(a, b, m, n)

    s_row = torch.tensor([[a, b, m, n]], dtype=FLOATING_POINT_PRECISION).to(device)
    I_tilde = 0.0

    with torch.no_grad():
        for (c1, c2, c3) in itertools.product([0.0, 1.0], repeat=3):
            sign = (-1) ** (3 - int(c1 + c2 + c3))
            u_t = torch.tensor([[c1, c2, c3]], dtype=FLOATING_POINT_PRECISION).to(device)
            inp = torch.cat([s_row, u_t], dim=1)
            I_tilde += sign * net(inp).item()

    return float(I_tilde * fc)
