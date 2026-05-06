"""
brok_upd_nni.py — Mixture-of-Experts SIREN for numerical integration
=================================================================

Architecture:
  BrokNet(a, b, m, n, u1, u2, u3)
      |
      |-- router:  (a,b,m,n) -> expert index i  (deterministic, no learned params)
      |
      `-- experts: [S_0(u1,u2,u3), S_1(u1,u2,u3), ..., S_{K-1}(u1,u2,u3)]
                    each S_i is a SindriNet — a SIREN with 3 inputs,
                    trained exclusively on its own integrand variant.

Each SindriNet S_i approximates the antiderivative F_i such that
    d^3 S_i / du1 du2 du3  ~  f_tilde_i(u1, u2, u3)

Routing: since (a,b,m,n) are discrete labels from a fixed known set,
the router is implemented as a plain dict with no learnable parameters.

During training the batch is split by label -> each group is forwarded
through its dedicated SindriNet -> per-expert losses are averaged into
a single scalar for the optimizer step.

Named after Brok and Sindri, the Huldra dwarves from God of War.
"""

import io
import itertools
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import integrate

##############################################################################
# 1.  PHYSICAL CONSTANTS
##############################################################################
LA   = 1.0
M1   = 0.3 / LA
M2   = 0.3 / LA
M3   = 0.3 / LA
P1P1 = -(0.14 / LA) ** 2    # p1^2
P2P2 = -(0.14 / LA) ** 2    # p2^2
PP   = -(0.70 / LA) ** 2    # P^2 = (p1+p2)^2
P1P2 = (PP - P1P1 - P2P2) / 2

T_MAX = 80.0
L_LOG = math.log(1.0 + T_MAX)    # ~4.394

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]
PARAM_SETS = list(zip(A_VALS, B_VALS, M_VALS, N_VALS))

FLOATING_POINT_PRECISION = torch.float32

##############################################################################
# 2.  D(alpha1, alpha2) AND R^2(alpha1, alpha2)
##############################################################################

def D_R2(alpha1: torch.Tensor,
         alpha2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    D  = alpha1*alpha2*P^2 + P1^2*alpha2*(1-alpha1-alpha2)
       + P2^2*alpha1*(1-alpha1-alpha2) + alpha1*m1^2 + alpha2*m2^2 + alpha3*m3^2
    R2 = alpha1^2*P2^2 + alpha2^2*P1^2 - alpha1*alpha2*(P^2-P1^2-P2^2)
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
# 3.  INTEGRAND IN UNIT CUBE (u1, u2, u3) in [0,1]^3
##############################################################################

def korobov(u: torch.Tensor) -> torch.Tensor:
    """Korobov variable transformation: x = u^2 * (3 - 2u)."""
    return u * u * (3.0 - 2.0 * u)


def korobov_weight(u: torch.Tensor) -> torch.Tensor:
    """Derivative of Korobov transform: dx/du = 6u(1-u)."""
    return 6.0 * u * (1.0 - u)


def integrand_transformed(u1: torch.Tensor,
                           u2: torch.Tensor,
                           u3: torch.Tensor,
                           a: float, b: float,
                           m: float, n: float) -> torch.Tensor:
    """
    Full transformed integrand in the unit cube [0,1]^3.

    Applies:
      - Korobov transform on (u1, u2) -> (alpha1, alpha2) with simplex Jacobian
      - Logarithmic substitution on u3 -> t:
            t = exp(L * u3) - 1,  u3 in [0,1] -> t in [0, T_MAX]
    """
    x1 = korobov(u1)
    x2 = korobov(u2)
    w1 = korobov_weight(u1)    # dx1/du1
    w2 = korobov_weight(u2)    # dx2/du2

    alpha1  = x1
    alpha2  = (1.0 - x1) * x2
    J_alpha = (1.0 - x1) * w1 * w2    # Jacobian of simplex mapping

    L       = L_LOG
    exp_Lu3 = torch.exp(torch.tensor(L, dtype=u3.dtype, device=u3.device) * u3)
    t       = exp_Lu3 - 1.0
    Jt      = L * exp_Lu3             # dt/du3

    D, R2 = D_R2(alpha1, alpha2)
    exponent = -(t * D + t / (1.0 + t) * R2)
    exponent = torch.clamp(exponent, min=-500.0, max=500.0)

    alpha1_a = torch.clamp(alpha1, min=0.0) ** a if a > 0 else torch.ones_like(alpha1)
    alpha2_b = torch.clamp(alpha2, min=0.0) ** b if b > 0 else torch.ones_like(alpha2)

    f_phys = (alpha1_a * alpha2_b
              * t ** m / (1.0 + t) ** n
              * torch.exp(exponent))

    return f_phys * J_alpha * Jt


def center_value(a: float, b: float, m: float, n: float) -> float:
    """
    Value of the transformed integrand at the center u=(0.5, 0.5, 0.5).
    Used as normalization constant: f_tilde = f / f_center.
    Falls back to 1.0 if the center value is essentially zero.
    """
    uc = torch.tensor([0.5])
    fc = integrand_transformed(uc, uc, uc, a, b, m, n).item()
    return fc if abs(fc) > 1e-30 else 1.0


##############################################################################
# 4.  SIREN LAYER
##############################################################################

class SirenLayer(nn.Module):
    """
    Single SIREN layer: Linear + sin(omega_0 * x).

    Weight initialization (Sitzmann et al. 2020):
      - First layer:  Uniform(-1/fan_in,  1/fan_in)
      - Hidden layers: Uniform(-sqrt(6/fan_in)/omega_0,  sqrt(6/fan_in)/omega_0)
        This preserves the distribution of activations across layers.

    Each differentiation of sin(omega_0 * W x) pulls down a factor omega_0,
    so three derivatives give omega_0^3 — accounted for in the output_scale
    of SindriNet.
    """

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        self.linear   = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, fan_in: int):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


##############################################################################
# 5.  SINDRIHET — EXPERT SUBNETWORK
##############################################################################

class SindriNet(nn.Module):
    """
    Expert subnetwork S_i(u1, u2, u3) — SIREN-based antiderivative approximator
    for a single fixed parameter set (a_i, b_i, m_i, n_i).

    Input:  integration variables only  (n_int_vars = 3 dimensions)
    Output: scalar F_i(u) such that d^3 F_i / du1 du2 du3  ~  f_tilde_i(u)

    Because the parameter label is NOT part of the input, SindriNet uses its
    full representational capacity exclusively for the shape of one integrand,
    without competing with other parameter variants for network width.
    """

    def __init__(self,
                 n_int_vars:   int       = 3,
                 hidden_sizes: List[int] = None,
                 omega_0:      float     = 30.0,
                 output_scale: float     = 1.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.n_int_vars = n_int_vars
        self.omega_0    = omega_0

        layers = []
        in_dim = n_int_vars
        for i, h in enumerate(hidden_sizes):
            layers.append(SirenLayer(in_dim, h,
                                     omega_0=omega_0,
                                     is_first=(i == 0)))
            in_dim = h

        # Final linear layer — no activation (output is a scalar antiderivative)
        final = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
            # Compensate for omega_0^3 accumulated over 3 autograd differentiations
            final.weight.data *= output_scale
            final.bias.data   *= output_scale
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            u: (N, n_int_vars) — integration variables in [0, 1]^3
        Returns:
            (N, 1) scalar antiderivative values
        """
        return self.net(u)


##############################################################################
# 6.  BROKNET — MIXTURE-OF-EXPERTS WRAPPER
##############################################################################

class BrokNet(nn.Module):
    """
    Brok — Mixture-of-Experts SIREN for parametric numerical integration.

    Accepts full input (a, b, m, n, u1, u2, u3), routes by the discrete key
    (a, b, m, n) to the corresponding SindriNet expert, which then operates
    only on (u1, u2, u3).

    The router is a plain dict — no learnable gating, no softmax. Since the
    parameter labels are known in advance and fully discrete, deterministic
    routing is exact and requires zero additional parameters.

    Args:
        param_sets:   list of (a, b, m, n) tuples — one per expert
        n_int_vars:   number of integration variables (3)
        hidden_sizes: hidden layer widths of each SindriNet expert
        omega_0:      SIREN frequency scaling factor
        output_scale: output layer scale (use 1/omega_0^3 to compensate derivatives)
    """

    def __init__(self,
                 param_sets:   List[Tuple],
                 n_int_vars:   int       = 3,
                 hidden_sizes: List[int] = None,
                 omega_0:      float     = 30.0,
                 output_scale: float     = 1.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.param_sets = param_sets
        self.n_int_vars = n_int_vars
        self.n_params   = 4    # a, b, m, n

        # One SindriNet per parameter set
        self.experts = nn.ModuleList([
            SindriNet(n_int_vars=n_int_vars,
                      hidden_sizes=hidden_sizes,
                      omega_0=omega_0,
                      output_scale=output_scale)
            for _ in param_sets
        ])

        # Deterministic router: key -> expert index
        self.router: Dict[Tuple, int] = {
            tuple(float(v) for v in ps): i
            for i, ps in enumerate(param_sets)
        }

    def expert_index(self, a: float, b: float, m: float, n: float) -> int:
        """Return the index of the expert responsible for parameter set (a,b,m,n)."""
        key = (float(a), float(b), float(m), float(n))
        if key not in self.router:
            raise KeyError(
                f"Parameter set {key} was not registered at construction time. "
                f"Available sets: {list(self.router.keys())}"
            )
        return self.router[key]

    def forward_expert(self, expert_idx: int, u: torch.Tensor) -> torch.Tensor:
        """
        Direct forward pass through a specific SindriNet expert.

        Args:
            expert_idx: index of the expert
            u:          (N, 3) integration variables
        Returns:
            (N, 1) antiderivative values
        """
        return self.experts[expert_idx](u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        General forward pass for a full input vector x = [a, b, m, n, u1, u2, u3].

        NOTE: intended for homogeneous batches at inference time (all rows share
        the same label). During training, batches are split by label explicitly
        inside train() — do not use this method for mixed-label training batches.

        Args:
            x: (N, 7) — columns [a, b, m, n, u1, u2, u3]
        Returns:
            (N, 1) antiderivative values from the routed expert
        """
        key = tuple(float(v) for v in x[0, :4].tolist())
        idx = self.router[key]
        u   = x[:, 4:]    # (N, 3)
        return self.experts[idx](u)


##############################################################################
# 7.  MIXED THIRD PARTIAL DERIVATIVE VIA AUTOGRAD
##############################################################################

def mixed_partial_3_expert(expert: SindriNet,
                            u: torch.Tensor) -> torch.Tensor:
    """
    Computes d^3 S_i / du1 du2 du3 for a single SindriNet expert
    using sequential reverse-mode autograd.

    Args:
        expert: the SindriNet subnetwork
        u:      (N, 3) integration variables (detached; grad is enabled internally)
    Returns:
        (N,) tensor of third mixed partial derivative values
    """
    u_d   = u.detach().requires_grad_(True)    # (N, 3)
    G_out = expert(u_d).squeeze(-1)            # (N,)
    ones  = torch.ones_like(G_out)

    # dG/du1
    g1 = torch.autograd.grad(
        G_out, u_d,
        grad_outputs=ones,
        create_graph=True, retain_graph=True,
    )[0][:, 0]    # (N,)

    # d^2 G / du1 du2
    g12 = torch.autograd.grad(
        g1, u_d,
        grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]    # (N,)

    # d^3 G / du1 du2 du3
    g123 = torch.autograd.grad(
        g12, u_d,
        grad_outputs=torch.ones_like(g12),
        create_graph=True,    # required so loss.backward() can propagate through
    )[0][:, 2]    # (N,)

    return g123


##############################################################################
# 8.  BATCH GENERATION
##############################################################################

def make_expert_batch(a: float, b: float, m: float, n: float,
                      n_samples: int,
                      device: torch.device,
                      norm_cache: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a training batch (u, f_tilde) for one parameter set.

    The integrand is normalized by its center value so that network outputs
    and targets are O(1) regardless of the physical scale of the integrand.

    Args:
        a, b, m, n:  parameter set identifying this expert
        n_samples:   number of random points to draw from [0,1]^3
        device:      target torch device
        norm_cache:  dict caching center_value() results; mutated in place

    Returns:
        u:       (n_samples, 3) uniform samples in [0, 1]^3
        f_tilde: (n_samples,)  normalized integrand values f / f_center
    """
    key = (float(a), float(b), float(m), float(n))
    if key not in norm_cache:
        norm_cache[key] = center_value(a, b, m, n)
    fc = norm_cache[key]

    u = torch.rand(n_samples, 3, dtype=FLOATING_POINT_PRECISION)
    f = integrand_transformed(u[:, 0], u[:, 1], u[:, 2], a, b, m, n) / fc

    return u.to(device), f.to(device)


##############################################################################
# 9.  TRAINING
##############################################################################

def train(brok: BrokNet,
          param_sets:    List[Tuple],
          n_epochs:      int   = 5000,
          n_per_expert:  int   = 512,
          lr:            float = 1e-3,
          device:        torch.device = torch.device("cpu"),
          verbose_every: int   = 500,
          log_fn=print) -> Tuple[List[float], Dict]:
    """
    Train BrokNet (all SindriNet experts jointly).

    Each epoch:
      1. For every SindriNet S_i, draw n_per_expert random points from [0,1]^3.
      2. Compute d^3 S_i / du1 du2 du3 via autograd.
      3. Compute per-expert MSE loss against the normalized integrand.
      4. Total loss = mean over all experts.
      5. Single optimizer step updates all parameters simultaneously.

    Gradients never flow between experts through the data, yet the optimizer
    sees a single aggregated loss — combining stability of joint training with
    the specialization benefits of independent experts.

    Args:
        brok:          BrokNet instance
        param_sets:    list of (a, b, m, n) tuples
        n_epochs:      number of training epochs
        n_per_expert:  random samples per expert per epoch
        lr:            initial learning rate (decays via CosineAnnealing to lr/10)
        device:        torch device
        verbose_every: print interval (epochs)
        log_fn:        logging callable (default: print)

    Returns:
        history:    list of total loss values, one per epoch
        norm_cache: dict mapping (a,b,m,n) -> center_value normalization constant
    """
    optimizer = torch.optim.Adam(brok.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 10
    )
    loss_fn = nn.MSELoss()
    brok.to(device)
    brok.train()

    n_experts    = len(param_sets)
    history      = []
    norm_cache: Dict = {}
    t0           = time.time()

    total_params = sum(p.numel() for p in brok.parameters())
    params_per   = total_params // n_experts

    log_fn(f"\n{'═' * 70}")
    log_fn(f"  BrokNet  |  {n_experts} SindriNet experts  |  "
           f"{params_per:,} params/expert  |  {total_params:,} total")
    log_fn(f"  Epochs: {n_epochs}  |  {n_per_expert} pts/expert/epoch  |  "
           f"lr: {lr} -> {lr / 10}  (CosineAnnealing)")
    log_fn(f"  Device: {device}")
    log_fn(f"{'═' * 70}")

    for epoch in range(1, n_epochs + 1):
        expert_losses = []

        for i, (a, b, m, n) in enumerate(param_sets):
            u, f_tilde = make_expert_batch(a, b, m, n,
                                           n_per_expert, device, norm_cache)
            expert = brok.experts[i]

            dG     = mixed_partial_3_expert(expert, u)
            loss_i = loss_fn(dG, f_tilde)
            expert_losses.append(loss_i)

        # Total loss: mean across all SindriNet experts
        total_loss = torch.stack(expert_losses).mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(brok.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        lv = total_loss.item()
        history.append(lv)

        if epoch % verbose_every == 0 or epoch == 1:
            lr_now  = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            per_expert_str = "  ".join(
                f"S{i}={el.item():.2e}"
                for i, el in enumerate(expert_losses)
            )
            log_fn(f"  Epoch {epoch:5d}/{n_epochs}  "
                   f"loss={lv:.4e}  lr={lr_now:.2e}  ({elapsed:.1f}s)")
            log_fn(f"    [{per_expert_str}]")

    log_fn(f"\n  Done in {time.time() - t0:.1f}s  |  "
           f"min loss = {min(history):.4e}\n")
    return history, norm_cache


##############################################################################
# 10.  INTEGRAL EVALUATION  (alternating-sign corner sum)
##############################################################################

def compute_integral(brok: BrokNet,
                     a: float, b: float, m: float, n: float,
                     norm_cache: Dict = None,
                     device: torch.device = torch.device("cpu")) -> float:
    """
    Recover the integral from the trained SindriNet antiderivative via the
    Newton-Leibniz inclusion-exclusion formula over the 8 corners of [0,1]^3:

        I_tilde = sum_{c in {0,1}^3}  (-1)^(3 - |c|)  *  S_i(c)
        I       = I_tilde * fc

    where fc = center_value(a, b, m, n) undoes the normalization applied
    during training.

    Args:
        brok:       trained BrokNet
        a, b, m, n: parameter set to evaluate
        norm_cache: optional dict of precomputed center values
        device:     torch device

    Returns:
        estimated integral value (float)
    """
    brok.eval()
    brok.to(device)

    key = (float(a), float(b), float(m), float(n))
    if norm_cache and key in norm_cache:
        fc = norm_cache[key]
    else:
        fc = center_value(a, b, m, n)

    expert_idx = brok.expert_index(a, b, m, n)
    expert     = brok.experts[expert_idx]

    I_tilde = 0.0
    with torch.no_grad():
        for (c1, c2, c3) in itertools.product([0.0, 1.0], repeat=3):
            sign  = (-1) ** (3 - int(c1 + c2 + c3))
            u_t   = torch.tensor([[c1, c2, c3]],
                                  dtype=FLOATING_POINT_PRECISION).to(device)
            I_tilde += sign * expert(u_t).item()

    return float(I_tilde * fc)


##############################################################################
# 11.  REFERENCE VALUES via scipy.integrate
##############################################################################

def reference_scipy(a: float, b: float, m: float, n: float,
                    t_max: float = T_MAX,
                    tol:   float = 1e-9) -> Tuple[float, float]:
    """
    Compute reference integral value using adaptive quadrature (scipy).

    Returns:
        (result, estimated_error)
    """
    def f_inner(t, alpha2, alpha1):
        alpha3  = 1.0 - alpha1 - alpha2
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
# 12.  TEE LOGGER
##############################################################################

_log_buffer = io.StringIO()


def tee_print(*args, **kwargs):
    """Print to stdout and accumulate in _log_buffer for later file export."""
    print(*args, **kwargs)
    kwargs_buf = {k: v for k, v in kwargs.items() if k != 'file'}
    print(*args, **kwargs_buf, file=_log_buffer)


##############################################################################
# 13.  MAIN
##############################################################################

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_default_dtype(FLOATING_POINT_PRECISION)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    # ── Hyperparameters ───────────────────────────────────────────────────
    HIDDEN       = [128, 128, 128]    # hidden layer widths per SindriNet expert
    OMEGA_0      = 30.0
    N_EPOCHS     = 8000
    N_PER_EXPERT = 1024            # random points per expert per epoch
    LR           = 5e-4
    # Compensate for omega_0^3 factor accumulated by three autograd derivatives
    OUTPUT_SCALE = 1.0 / (OMEGA_0 ** 3)

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    tee_print(f"\n  Run started  : {run_start.strftime('%Y-%m-%d %H:%M')}")
    tee_print(f"  Device       : {device}")

    # ── Build BrokNet ─────────────────────────────────────────────────────
    brok = BrokNet(
        param_sets   = PARAM_SETS,
        n_int_vars   = 3,
        hidden_sizes = HIDDEN,
        omega_0      = OMEGA_0,
        output_scale = OUTPUT_SCALE,
    )

    n_total  = sum(p.numel() for p in brok.parameters())
    n_expert = n_total // len(PARAM_SETS)
    tee_print(f"\n  Architecture : BrokNet  [{len(PARAM_SETS)} x SindriNet {HIDDEN}]")
    tee_print(f"  omega_0      : {OMEGA_0}")
    tee_print(f"  Parameters   : {n_total:,} total  /  {n_expert:,} per expert")
    tee_print(f"  Output scale : {OUTPUT_SCALE:.3e}")
    tee_print(f"  t-substitution: logarithmic  "
              f"t = exp({L_LOG:.3f} * u3) - 1,  t in [0, {T_MAX:.0f}]")

    # ── Training ──────────────────────────────────────────────────────────
    history, norm_cache = train(
        brok,
        param_sets    = PARAM_SETS,
        n_epochs      = N_EPOCHS,
        n_per_expert  = N_PER_EXPERT,
        lr            = LR,
        device        = device,
        verbose_every = 500,
        log_fn        = tee_print,
    )

    # ── Reference values ──────────────────────────────────────────────────
    tee_print("Computing reference values (scipy) ...\n")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
        tee_print(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  {r:.6e}  +-  {e:.1e}")

    # ── Results table ─────────────────────────────────────────────────────
    SEP = '═' * 84
    tee_print(f"\n{SEP}")
    tee_print(f"  {'(a,b,m,n)':^12}  {'BrokNet NNI':^14}  {'scipy':^14}  "
              f"{'|delta|':^11}  {'rel err':^9}  {'Digits':^6}")
    tee_print(SEP)

    for (a, b, m, n) in PARAM_SETS:
        nni_val          = compute_integral(brok, a, b, m, n,
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

    # ── Save log ──────────────────────────────────────────────────────────
    log_filename = f"results_brok_{timestamp}.out"
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(_log_buffer.getvalue())
        print(f"\n  Log saved -> {log_filename}")
    except OSError as exc:
        print(f"\n  [WARNING] Could not save log: {exc}")


if __name__ == "__main__":
    main()
