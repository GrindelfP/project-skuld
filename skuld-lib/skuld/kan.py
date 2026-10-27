##############################################################################
#  KAN-based Neural Numerical Integration
#
#  Liu et al. 2024: "KAN: Kolmogorov-Arnold Networks"
#  https://arxiv.org/abs/2404.19756
#
#  KAN replaces fixed activation functions with learnable univariate
#  B-spline functions on each edge:
#      phi_{i->j}(x_i) = w_ij * SiLU(x_i) + c_{ij} . B(x_i)
#
#  B-spline derivatives are piecewise polynomials of lower degree,
#  making autograd through three differentiations numerically stable.
##############################################################################
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._maitre import MaitreIntegrator, mixed_partial_n, mixed_partial_3


##############################################################################
#  KAN Architecture
##############################################################################

class BSplineBasis(nn.Module):
    """Precomputes and evaluates a uniform B-spline basis over [-1, 1]."""

    def __init__(self, G: int = 8, k: int = 3):
        super().__init__()
        self.G = G
        self.k = k
        self.n_basis = G + k

        h = 2.0 / G
        inner = torch.linspace(-1.0, 1.0, G + 1)
        t = torch.cat([
            torch.full((k,), -1.0),
            inner,
            torch.full((k,), 1.0)
        ])
        self.register_buffer("t", t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions at x.

        x: (..., in_features) values in [-1, 1]
        returns: (..., in_features, n_basis)
        """
        t = self.t
        k = self.k

        x_exp = x.unsqueeze(-1)
        t_exp = t.unsqueeze(0)
        n_knots = t.shape[0]

        left = t[:-1].view(1, -1)
        right = t[1:].view(1, -1)
        B = ((x_exp >= left) & (x_exp < right)).to(x.dtype)
        last_mask = (x_exp == t[-1])
        B[..., -1] = B[..., -1] + last_mask.squeeze(-1).to(x.dtype)

        for d in range(1, k + 1):
            n_b = n_knots - 1 - d
            j = torch.arange(n_b, device=x.device)
            tj = t[j]
            tjd = t[j + d]
            tj1 = t[j + 1]
            tjd1 = t[j + d + 1]

            denom1 = (tjd - tj).clamp(min=1e-8)
            denom2 = (tjd1 - tj1).clamp(min=1e-8)

            alpha1 = (x_exp - tj) / denom1
            alpha2 = (tjd1 - x_exp) / denom2

            B_prev = B[..., :n_b]
            B_next = B[..., 1:n_b + 1]

            B = alpha1 * B_prev + alpha2 * B_next

        return B


class KANLayer(nn.Module):
    """One KAN layer using learned univariate B-spline functions on each edge."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 G: int = 8,
                 k: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        n_basis = G + k

        self.basis = BSplineBasis(G=G, k=k)

        self.w = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        self.c = nn.Parameter(
            torch.zeros(out_features, in_features, n_basis)
        )
        nn.init.normal_(self.c, std=0.1 / math.sqrt(in_features * n_basis))

        self.norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_clamped = torch.tanh(x)

        silu_x = F.silu(x_clamped)
        residual = torch.einsum('oi,ni->no', self.w, silu_x)

        B = self.basis(x_clamped)
        spline = torch.einsum('oig,nig->no', self.c, B)

        return residual + spline


class KANPrimitiveNet(nn.Module):
    """KAN-based primitive network N(s; u) ≈ F(s; u).

    Approximates the antiderivative such that
        d^3 N / (du1 du2 du3) ≈ f_tilde(s; u).
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 width: int = 32,
                 depth: int = 2,
                 G: int = 8,
                 k: int = 3,
                 output_scale: float = 1.0):
        super().__init__()
        self.n_params = n_params
        self.n_int_vars = n_int_vars

        in_dim = n_params + n_int_vars

        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers.append(KANLayer(d_in, width, G=G, k=k))
            d_in = width
        layers.append(KANLayer(d_in, 1, G=G, k=k))

        self.layers = nn.ModuleList(layers)
        self._output_scale = output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h * self._output_scale


##############################################################################
#  KANIntegrator: train + evaluate
##############################################################################

class KANIntegrator(MaitreIntegrator):
    """Application of KAN for numerical integration (Maitre et al. approach).

    Trains a KANPrimitiveNet to approximate the antiderivative, then
    evaluates the integral via an alternating-sign corner sum over the
    unit hypercube [0,1]^n_int_vars.
    """

    _clip_norm = 3.0

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 width: int = 32,
                 depth: int = 2,
                 G: int = 8,
                 k: int = 3,
                 output_scale: float = 1.0):
        self.n_params = n_params
        self.n_int_vars = n_int_vars
        self.net = KANPrimitiveNet(
            n_params=n_params,
            n_int_vars=n_int_vars,
            width=width,
            depth=depth,
            G=G,
            k=k,
            output_scale=output_scale,
        )

    def _make_scheduler(self, optimizer, n_epochs: int, lr: float):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=n_epochs,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e3,
        )

    def _make_loss(self):
        return nn.HuberLoss(delta=0.5)
