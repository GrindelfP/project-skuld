import itertools
import math
import time
import io
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate

##############################################################################
# 1.  PHYSICAL CONSTANTS  (identical to siren_nni.py)
##############################################################################
LA   = 1.0
M1   = 0.3 / LA
M2   = 0.3 / LA
M3   = 0.3 / LA
P1P1 = -(0.14 / LA) ** 2
P2P2 = -(0.14 / LA) ** 2
PP   = -(0.70 / LA) ** 2
P1P2 = (PP - P1P1 - P2P2) / 2

T_MAX = 80.0
L_LOG = math.log(1.0 + T_MAX)   # ~4.394

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]
PARAM_SETS = list(zip(A_VALS, B_VALS, M_VALS, N_VALS))

FLOATING_POINT_PRECISION = torch.float32


##############################################################################
# 2.  D(alpha1, alpha2) and R^2(alpha1, alpha2)
##############################################################################

def D_R2(alpha1: torch.Tensor, alpha2: torch.Tensor) -> tuple:
    """
    D  = alpha1*alpha2*P^2 + P1^2*alpha2*(1-alpha1-alpha2)
       + P2^2*alpha1*(1-alpha1-alpha2) + alpha1*m1^2 + alpha2*m2^2 + alpha3*m3^2
    R2 = alpha1^2*P2^2 + alpha2^2*P1^2
       - alpha1*alpha2*(P^2-P1^2-P2^2)
    """
    alpha3 = 1.0 - alpha1 - alpha2
    D = (alpha1 * alpha2 * PP
         + P1P1 * alpha2 * alpha3
         + P2P2 * alpha1 * alpha3
         + alpha1 * M1 ** 2
         + alpha2 * M2 ** 2
         + alpha3 * M3 ** 2)
    R2 = (alpha1 ** 2 * P2P2
          + alpha2 ** 2 * P1P1
          - alpha1 * alpha2 * (PP - P1P1 - P2P2))
    return D, R2


##############################################################################
# 3.  INTEGRAND IN UNIT CUBE  (u1, u2, u3) in [0,1]^3
##############################################################################

def korobov(u: torch.Tensor) -> torch.Tensor:
    return u * u * (3.0 - 2.0 * u)


def korobov_weight(u: torch.Tensor) -> torch.Tensor:
    return 6.0 * u * (1.0 - u)


def integrand_transformed(u1, u2, u3, a, b, m, n):
    x1 = korobov(u1);  x2 = korobov(u2)
    w1 = korobov_weight(u1);  w2 = korobov_weight(u2)

    alpha1  = x1
    alpha2  = (1.0 - x1) * x2
    J_alpha = (1.0 - x1) * w1 * w2

    L       = L_LOG
    exp_Lu3 = torch.exp(torch.tensor(L, dtype=u3.dtype, device=u3.device) * u3)
    t       = exp_Lu3 - 1.0
    Jt      = L * exp_Lu3

    D, R2    = D_R2(alpha1, alpha2)
    exponent = -(t * D + t / (1.0 + t) * R2)
    exponent = torch.clamp(exponent, min=-500.0, max=500.0)

    alpha1_a = torch.clamp(alpha1, min=0.0) ** a if a > 0 else torch.ones_like(alpha1)
    alpha2_b = torch.clamp(alpha2, min=0.0) ** b if b > 0 else torch.ones_like(alpha2)

    f_phys = (alpha1_a * alpha2_b
              * t ** m / (1.0 + t) ** n
              * torch.exp(exponent))

    return f_phys * J_alpha * Jt


def center_value(a, b, m, n):
    uc = torch.tensor([0.5])
    fc = integrand_transformed(uc, uc, uc, a, b, m, n).item()
    return fc if abs(fc) > 1e-30 else 1.0


##############################################################################
# 4.  KAN ARCHITECTURE
#
#  Kolmogorov-Arnold Networks  (Liu et al. 2024, arXiv:2404.19756)
#
#  Classical MLP:   y = sigma( W x + b )          -- fixed activation, learned W
#  KAN:             y = sum_i  phi_i( x_i )        -- learned activation per edge
#
#  Theoretical foundation
#  ----------------------
#  Kolmogorov-Arnold theorem (1957): any continuous f: [0,1]^n -> R can be
#  written as
#      f(x) = sum_{q=1}^{2n+1} Phi_q( sum_{p=1}^n phi_{q,p}(x_p) )
#  i.e. a two-layer network of univariate functions suffices.
#
#  Why KAN is well-suited for THIS integrand
#  ------------------------------------------
#  1. Our integrand factorises approximately:
#         f(u) ~ g_1(u1) * g_2(u2) * h(u3)
#     KAN's additive structure on each layer naturally captures such
#     near-separable dependencies without fighting the architecture.
#
#  2. B-spline basis functions phi_i are piecewise polynomials of degree k.
#     Their k-th derivative is a piecewise polynomial of degree k-k = 0
#     (piecewise constant), so autograd through three differentiations is
#     numerically trivial and never produces NaN or exploding gradients.
#     Compare with SIREN where d^3/dx^3 sin(omega*x) = -omega^3 cos(omega*x):
#     high omega amplifies curvature; KAN has no such issue.
#
#  3. Each KAN layer is interpretable: after training one can inspect phi_i
#     to understand how the network decomposes the integrand.
#
#  4. The trainable residual SiLU term  w * SiLU(x) on each edge adds a
#     smooth global backbone that absorbs the overall scale, while the
#     spline handles local corrections.
#
#  Implementation details
#  ----------------------
#  KANLayer: for each (input, output) pair, a B-spline basis of G+1 functions
#  over the interval [-1, 1] with k+1 order (degree k) is computed.
#  Coefficients c_{ij} are learned; the layer output is:
#
#      out_j = sum_i [ w_ij * SiLU(x_i) + sum_g c_{ijg} * B_{g,k}(x_i) ]
#
#  Input normalisation: the physical inputs span [0,1] (u-variables) and
#  {0,1,2,3} (integer params). We map everything to [-1,1] before entering
#  KAN layers, and apply a LayerNorm between layers so spline knots stay
#  in the calibrated range.
#
#  Stack: 3 KAN layers with widths [7, 64, 64, 1], G=8 grid points, k=3
#  (cubic B-splines). This gives a smooth C^2 function everywhere.
##############################################################################

class BSplineBasis(nn.Module):
    """
    Precomputes and evaluates a uniform B-spline basis over [-1, 1].

    For G internal intervals and degree k we need G + k knots on each side
    (extended knot vector). The number of basis functions is G + k.

    We use the Cox-de Boor recursion implemented as a differentiable
    forward pass via clamped linear interpolation.
    """

    def __init__(self, G: int = 8, k: int = 3):
        """
        G : number of grid intervals  (basis functions = G + k)
        k : spline degree (3 = cubic, C^2 continuous)
        """
        super().__init__()
        self.G = G
        self.k = k
        self.n_basis = G + k    # number of B-spline basis functions

        # Extended knot vector on [-1, 1]
        # k repeated knots at each end (clamped / open spline)
        h = 2.0 / G             # interval size
        # interior knots: G-1 points between -1 and 1 (exclusive)
        inner  = torch.linspace(-1.0, 1.0, G + 1)          # G+1 breakpoints
        # padded knot vector with k extra knots on each side
        t      = torch.cat([
            torch.full((k,), -1.0),
            inner,
            torch.full((k,), 1.0)
        ])   # length: G + 1 + 2k
        self.register_buffer("t", t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (..., in_features)  values in [-1, 1]
        return: (..., in_features, n_basis)  basis values B_{j,k}(x)
        """
        t = self.t   # (G + 1 + 2k,)
        k = self.k

        # We evaluate B-splines of degree k via the Cox-de Boor recurrence.
        # Start with degree-0 indicator functions:
        # B_{j,0}(x) = 1 if t_j <= x < t_{j+1}, else 0
        # with the convention that the last interval is closed on the right.

        x_exp = x.unsqueeze(-1)           # (..., in, 1)
        t_exp = t.unsqueeze(0)            # (1, n_knots)
        # For each (x, j): indicator B_{j,0}
        # We need x against n_knots-1 intervals
        n_knots = t.shape[0]

        # Degree-0: shape (..., in, n_knots-1)
        # Clamp last interval to be closed (include x==1)
        left  = t[:-1].view(1, -1)       # (1, n_knots-1)
        right = t[1: ].view(1, -1)       # (1, n_knots-1)
        B = ((x_exp >= left) & (x_exp < right)).to(x.dtype)
        # Close the rightmost interval
        last_mask = (x_exp == t[-1])
        B[..., -1] = B[..., -1] + last_mask.squeeze(-1).to(x.dtype)

        # Cox-de Boor recurrence up to degree k
        for d in range(1, k + 1):
            n_b = n_knots - 1 - d       # number of basis functions at this degree
            # left numerator / denominator
            t_left  = t[:n_b + d]       # t_j
            t_right = t[d: n_b + 2 * d] # t_{j+d}  -- but we need t_{j+d} for each j
            # Actually, for basis B_{j,d}: uses t[j] and t[j+d] and t[j+d+1]
            # Let's use index arrays
            j = torch.arange(n_b, device=x.device)
            tj   = t[j]                             # (n_b,)
            tjd  = t[j + d]                         # (n_b,)
            tj1  = t[j + 1]                         # (n_b,)
            tjd1 = t[j + d + 1]                     # (n_b,)

            denom1 = (tjd  - tj ).clamp(min=1e-8)
            denom2 = (tjd1 - tj1).clamp(min=1e-8)

            alpha1 = (x_exp - tj)  / denom1           # (..., in, n_b)
            alpha2 = (tjd1 - x_exp) / denom2          # (..., in, n_b)

            B_prev = B[..., :n_b]                     # B_{j,  d-1}
            B_next = B[..., 1:n_b + 1]                # B_{j+1,d-1}

            B = alpha1 * B_prev + alpha2 * B_next     # (..., in, n_b)

        return B   # (..., in_features, n_basis=G+k)


class KANLayer(nn.Module):
    """
    One KAN layer: maps R^{in_features} -> R^{out_features}
    using learned univariate B-spline functions on each edge.

    phi_{i->j}(x_i) = w_ij * SiLU(x_i) + c_{ij} . B(x_i)

    w_ij : scalar weight for the SiLU residual term  (shape: out x in)
    c_{ijg}: spline coefficients                     (shape: out x in x n_basis)
    """

    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 G:  int = 8,
                 k:  int = 3):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        n_basis = G + k

        self.basis = BSplineBasis(G=G, k=k)

        # SiLU (swish) residual weight -- one scalar per edge
        self.w = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        # Spline coefficients
        self.c = nn.Parameter(
            torch.zeros(out_features, in_features, n_basis)
        )
        nn.init.normal_(self.c, std=0.1 / math.sqrt(in_features * n_basis))

        # LayerNorm applied to input (keeps x in the spline's calibrated range)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_features)
        returns: (N, out_features)
        """
        x = self.norm(x)
        # Clamp to [-1, 1] for B-spline evaluation
        x_clamped = torch.tanh(x)                              # (N, in)

        # SiLU residual branch
        silu_x = F.silu(x_clamped)                            # (N, in)
        residual = torch.einsum('oi,ni->no', self.w, silu_x)  # (N, out)

        # B-spline branch
        B = self.basis(x_clamped)                             # (N, in, n_basis)
        # c: (out, in, n_basis)  B: (N, in, n_basis)
        spline = torch.einsum('oig,nig->no', self.c, B)       # (N, out)

        return residual + spline


class KANPrimitiveNet(nn.Module):
    """
    KAN-based primitive network  N(s; u) ~ F(s; u)

    Approximates the antiderivative such that
        d^3 N / (du1 du2 du3)  ~  f_tilde(s; u).

    Architecture
    ------------
    Input:  7-dim = [a, b, m, n, u1, u2, u3]
    Layer 1: KANLayer(7  -> width,  G, k)
    Layer 2: KANLayer(width -> width, G, k)
    Layer 3: KANLayer(width -> 1,     G, k)   (output layer)

    No final activation -- the network represents a smooth primitive F.

    Design rationale for KAN vs MLP
    ---------------------------------
    In an MLP, every hidden unit mixes ALL inputs through a dense weight
    matrix and applies a shared activation.  The 3rd-order autograd derivative
    of such a composition accumulates cross-terms from every layer.
    In a KAN, each edge is an independent 1D function; the network's output
    is a sum of compositions of 1D functions, whose derivatives are
    analytically clean: d/dx phi(x) = phi'(x) -- no cross-contamination.
    This structural inductive bias matches the near-separable structure of
    our integrand and makes the 3-fold autograd differentiation well-
    conditioned.
    """

    def __init__(self,
                 n_params:    int   = 4,
                 n_int_vars:  int   = 3,
                 width:       int   = 32,
                 depth:       int   = 2,
                 G:           int   = 8,
                 k:           int   = 3,
                 output_scale: float = 1.0):
        super().__init__()
        self.n_params   = n_params
        self.n_int_vars = n_int_vars

        in_dim = n_params + n_int_vars    # 7

        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers.append(KANLayer(d_in, width, G=G, k=k))
            d_in = width
        layers.append(KANLayer(d_in, 1, G=G, k=k))

        self.layers = nn.ModuleList(layers)

        # Scale output so initial d^3N is roughly O(output_scale)
        self._output_scale = output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h * self._output_scale


##############################################################################
# 5.  MIXED PARTIAL DERIVATIVE  d^3 N / (du1 du2 du3)  via autograd
##############################################################################

def mixed_partial_3(net: KANPrimitiveNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    batch: (N, n_params + 3)
        [:, :n_params]  -- s  (physical params, not differentiated)
        [:, n_params:]  -- u1, u2, u3  (differentiated)
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)   # (N, 3)

    inp   = torch.cat([s, u], dim=1)
    N_out = net(inp).squeeze(-1)                     # (N,)

    ones_N = torch.ones_like(N_out)

    # dN/du1
    g1 = torch.autograd.grad(
        N_out, u, grad_outputs=ones_N,
        create_graph=True, retain_graph=True,
    )[0][:, 0]

    # d^2N/(du1 du2)
    g12 = torch.autograd.grad(
        g1, u, grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]

    # d^3N/(du1 du2 du3)
    g123 = torch.autograd.grad(
        g12, u, grad_outputs=torch.ones_like(g12),
        create_graph=True,
    )[0][:, 2]

    return g123


##############################################################################
# 6.  BATCH GENERATION
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

        u  = torch.rand(n_per_param, 3)
        f  = integrand_transformed(u[:, 0], u[:, 1], u[:, 2], a, b, m, n) / fc
        s  = torch.tensor([a, b, m, n],
                           dtype=FLOATING_POINT_PRECISION).expand(n_per_param, -1)
        xu = torch.cat([s, u], dim=1)

        all_xu.append(xu)
        all_f.append(f)

    batch_xu = torch.cat(all_xu, dim=0).to(device)
    f_tilde  = torch.cat(all_f,  dim=0).to(device)
    return batch_xu, f_tilde, norm_cache


##############################################################################
# 7.  TRAINING
##############################################################################

def train(net: KANPrimitiveNet,
          param_sets:    list,
          n_epochs:      int   = 5000,
          n_per_param:   int   = 512,
          lr:            float = 1e-3,
          device:        torch.device = torch.device("cpu"),
          verbose_every: int   = 500) -> tuple:

    # LBFGS works extremely well for KAN because the problem is smooth and
    # low-dimensional per sample; but it requires closure().
    # We use Adam with OneCycleLR as a robust default:
    #   - OneCycleLR: fast warm-up + aggressive annealing, shown to find
    #     sharp optima in fewer epochs than CosineAnnealing.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr         = lr,
        total_steps    = n_epochs,
        pct_start      = 0.15,         # 15% warm-up
        anneal_strategy= 'cos',
        div_factor     = 25.0,         # start lr = max_lr / 25
        final_div_factor = 1e3,        # end   lr = start_lr / 1000
    )

    loss_fn    = nn.HuberLoss(delta=0.5)   # robust to large outlier f values
    net.to(device)
    net.train()

    history    = []
    norm_cache = {}
    t0         = time.time()

    header = (
        f"\n{'=' * 68}\n"
        f"  Training: {n_epochs} epochs | "
        f"{len(param_sets)} sets\n"
        f"  Per epoch: {len(param_sets) * n_per_param} pts | device: {device}\n"
        f"  Optimizer: Adam + OneCycleLR  (max_lr={lr}, warm-up 15%)\n"
        f"  Loss: HuberLoss (delta=0.5)\n"
        f"  Network: KAN  (B-spline edges, cubic k=3)\n"
        f"{'=' * 68}"
    )
    tee_print(header)

    for epoch in range(1, n_epochs + 1):
        batch_xu, f_tilde, norm_cache = make_batch(
            param_sets, n_per_param, device, norm_cache
        )

        dN   = mixed_partial_3(net, batch_xu)
        loss = loss_fn(dN, f_tilde)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        history.append(lv)

        if epoch % verbose_every == 0 or epoch == 1:
            lr_now  = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            tee_print(f"  Epoch {epoch:5d}/{n_epochs}  "
                      f"loss={lv:.4e}  lr={lr_now:.2e}  "
                      f"({elapsed:.1f}s)")

    elapsed_total = time.time() - t0
    tee_print(f"\n  Done in {elapsed_total:.1f}s | "
              f"min loss = {min(history):.4e}\n")
    return history, norm_cache


##############################################################################
# 8.  INTEGRAL EVALUATION  (alternating-sign corner sum)
##############################################################################

def compute_integral(net: KANPrimitiveNet,
                     a: float, b: float, m: float, n: float,
                     norm_cache: dict = None,
                     device: torch.device = torch.device("cpu")) -> float:
    net.eval()
    net.to(device)

    if norm_cache and (a, b, m, n) in norm_cache:
        fc = norm_cache[(a, b, m, n)]
    else:
        fc = center_value(a, b, m, n)

    s_row   = torch.tensor([[a, b, m, n]],
                            dtype=FLOATING_POINT_PRECISION).to(device)
    I_tilde = 0.0

    with torch.no_grad():
        for (c1, c2, c3) in itertools.product([0.0, 1.0], repeat=3):
            sign  = (-1) ** (3 - int(c1 + c2 + c3))
            u_t   = torch.tensor([[c1, c2, c3]],
                                  dtype=FLOATING_POINT_PRECISION).to(device)
            inp   = torch.cat([s_row, u_t], dim=1)
            I_tilde += sign * net(inp).item()

    return float(I_tilde * fc)


##############################################################################
# 9.  REFERENCE VALUES via scipy.integrate
##############################################################################

def reference_scipy(a, b, m, n, t_max=T_MAX, tol=1e-9):
    def f_inner(t, alpha2, alpha1):
        alpha3 = 1.0 - alpha1 - alpha2
        D  = (alpha1 * alpha2 * PP
              + P1P1 * alpha2 * alpha3
              + P2P2 * alpha1 * alpha3
              + alpha1 * M1 ** 2
              + alpha2 * M2 ** 2
              + alpha3 * M3 ** 2)
        R2 = (alpha1 ** 2 * P2P2
              + alpha2 ** 2 * P1P1
              - alpha1 * alpha2 * (PP - P1P1 - P2P2))
        exp_arg = -(t * D + t / (1.0 + t) * R2)
        if exp_arg > 700.0:
            return 0.0
        a1 = alpha1 ** a if a > 0 else 1.0
        a2 = alpha2 ** b if b > 0 else 1.0
        return a1 * a2 * t ** m / (1.0 + t) ** n * np.exp(exp_arg)

    def f_alpha2(alpha2, alpha1):
        v, _ = integrate.quad(f_inner, 0.0, t_max,
                              args=(alpha2, alpha1),
                              limit=300, epsabs=tol, epsrel=tol)
        return v

    def f_alpha1(alpha1):
        v, _ = integrate.quad(f_alpha2, 0.0, 1.0 - alpha1,
                              args=(alpha1,),
                              limit=200, epsabs=tol, epsrel=tol)
        return v

    result, err = integrate.quad(f_alpha1, 0.0, 1.0,
                                 limit=200, epsabs=tol, epsrel=tol)
    return float(result), float(err)


##############################################################################
# 10.  TEE: print to stdout AND accumulate in log buffer
##############################################################################

_log_buffer = io.StringIO()


def tee_print(*args, **kwargs):
    print(*args, **kwargs)
    kwargs_buf = {k: v for k, v in kwargs.items() if k != 'file'}
    print(*args, **kwargs_buf, file=_log_buffer)


##############################################################################
# 11.  MAIN
##############################################################################

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    torch.set_default_dtype(FLOATING_POINT_PRECISION)

    # ── Hyperparameters ───────────────────────────────────────────────────
    # KAN width: smaller than MLP because each KAN neuron is a rich
    # B-spline function rather than a simple dot-product + activation.
    WIDTH  = 32          # hidden width per KAN layer
    DEPTH  = 2           # number of hidden KAN layers (+ 1 output layer)
    G      = 8           # number of B-spline grid intervals
    K      = 3           # spline degree (cubic = C^2 continuity)
    N_EPOCHS    = 8000
    N_PER_PARAM = 1024
    LR          = 2e-3
    # d^3/du^3 of a degree-k spline is a step function (degree k-3 = 0);
    # the scale is set by the spline coefficient magnitudes, which we
    # initialise with std ~ 0.1/sqrt(in*n_basis) already small -- no
    # additional compensation needed beyond a modest output_scale.
    OUTPUT_SCALE = 1.0

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    tee_print(f"\n  Run started : {run_start.strftime('%Y-%m-%d %H:%M')}")
    tee_print(f"  Device      : {device}")

    net = KANPrimitiveNet(
        n_params     = 4,
        n_int_vars   = 3,
        width        = WIDTH,
        depth        = DEPTH,
        G            = G,
        k            = K,
        output_scale = OUTPUT_SCALE,
    )
    n_p = sum(p.numel() for p in net.parameters())
    n_basis = G + K

    tee_print(f"\n  Architecture : KAN (Kolmogorov-Arnold Network)")
    tee_print(f"  Layers       : [7] -> {DEPTH} x KANLayer({WIDTH}) -> [1]")
    tee_print(f"  B-spline     : G={G} intervals, k={K} (cubic), n_basis={n_basis}")
    tee_print(f"  Edge formula : phi_ij(x) = w_ij*SiLU(x) + c_ij . B(x)")
    tee_print(f"  Parameters   : {n_p:,}")
    tee_print(f"  Loss         : HuberLoss (delta=0.5)  -- robust to outliers")
    tee_print(f"  Scheduler    : OneCycleLR (max_lr={LR}, 15% warm-up)")
    tee_print(f"  t-substitution: logarithmic  "
              f"t = exp({L_LOG:.3f}*u3) - 1,  t in [0, {T_MAX:.0f}]")

    history, norm_cache = train(
        net,
        param_sets    = PARAM_SETS,
        n_epochs      = N_EPOCHS,
        n_per_param   = N_PER_PARAM,
        lr            = LR,
        device        = device,
        verbose_every = 500,
    )

    # ── Reference values ──────────────────────────────────────────────────
    tee_print("Computing reference values (scipy) ...\n")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
        tee_print(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  {r:.6e}  +/-  {e:.1e}")

    # ── Comparison table ──────────────────────────────────────────────────
    SEP = '=' * 84
    tee_print(f"\n{SEP}")
    tee_print(f"  {'(a,b,m,n)':^12}  {'KAN-NNI':^14}  {'scipy':^14}  "
              f"{'|Delta|':^11}  {'|Delta|/ref':^9}  {'Digits':^6}")
    tee_print(SEP)

    for (a, b, m, n) in PARAM_SETS:
        nni_val          = compute_integral(net, a, b, m, n,
                                            norm_cache=norm_cache,
                                            device=device)
        ref_val, ref_err = refs[(a, b, m, n)]
        abs_err          = abs(nni_val - ref_val)
        rel_err          = abs_err / (abs(ref_val) + 1e-30)
        correct_digits   = max(0, -math.floor(math.log10(abs_err + 1e-30)))

        tee_print(f"   ({int(a)},{int(b)},{int(m)},{int(n)})     "
                  f"{nni_val:>14.6e}  {ref_val:>14.6e}  "
                  f"{abs_err:>11.3e}  {rel_err:>9.3e}  "
                  f"{int(correct_digits):^6}")

    tee_print(SEP)
    tee_print(f"\n  Final loss   : {history[-1]:.4e}")
    tee_print(f"  Minimum loss : {min(history):.4e}")
    tee_print(f"  Loss @ epoch 1: {history[0]:.4e}")
    tee_print(f"\n  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── Save log file ─────────────────────────────────────────────────────
    log_filename = f"results_kan_{timestamp}.out"
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(_log_buffer.getvalue())
        print(f"\n  Log saved -> {log_filename}")
    except OSError as exc:
        print(f"\n  [WARNING] Could not save log: {exc}")


if __name__ == "__main__":
    main()
