"""
13-brok_wire_nni.py — Mixture-of-Experts Wire-net for numerical integration
=========================================================================

Architecture:
  BrokNet(a, b, m, n, u1, u2, u3)
      |
      |-- router:  (a,b,m,n) -> expert index i  (deterministic, no learned params)
      |
      `-- experts: [S_0(u1,u2,u3), S_1(u1,u2,u3), ..., S_{K-1}(u1,u2,u3)]
                    each S_i is a SindriNet — a Wire net with 3 inputs,
                    trained exclusively on its own integrand variant.

Each SindriNet S_i approximates the antiderivative F_i such that
    d^3 S_i / du1 du2 du3  ~  f_tilde_i(u1, u2, u3)

Wire architecture (Saragadam et al. 2023):
  Each layer computes a complex Gabor / Morlet wavelet activation:
      Re(z) = cos(omega_0 * Wx) * exp(-(omega_0*Wx)^2 / (2*sigma_0^2))
      Im(z) = sin(omega_0 * Wx) * exp(-(omega_0*Wx)^2 / (2*sigma_0^2))
  Real and imaginary parts are concatenated, doubling the channel count.
  Residual skip connections are added between Wire blocks for stable
  gradient propagation through three levels of autograd differentiation.

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
# 4.  WIRE LAYERS
#
#  Wire = "Wavelet Implicit neural Representations"
#  Reference: Saragadam et al. 2023  https://arxiv.org/abs/2301.05187
#
#  Each Wire layer computes a complex Gabor / Morlet wavelet:
#      output = [cos(z)*env, sin(z)*env]   where
#      z   = omega_0 * (W*x + b)
#      env = exp(-z^2 / (2*sigma_0^2))
#
#  Properties relevant to this integrand:
#    - Gaussian envelope provides LOCAL support: neurons naturally specialize
#      to the active region of the integrand (simplex + exponential decay in t).
#    - Three autograd derivatives of a Gabor wavelet stay numerically stable:
#      d/dx[exp(i*omega*x)*exp(-sigma^2*x^2)] = (i*omega - 2*sigma^2*x)*same,
#      so the derivative is of the same functional form — no gradient blow-up.
#    - Independent tuning of frequency omega_0 and bandwidth sigma_0 gives
#      implicit multi-resolution decomposition.
#
#  We work in REAL arithmetic: real + imaginary parts as separate channels,
#  so a WireLayer with out_features neurons produces 2*out_features outputs.
##############################################################################

class WireLayer(nn.Module):
    """
    Single Wire (complex Gabor / Morlet wavelet) layer.

    Output dimension is 2 * out_features because we return
    [Re(Gabor), Im(Gabor)] as real tensors concatenated along the last dim.
    """

    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 omega_0:      float = 10.0,
                 sigma_0:      float = 10.0,
                 is_first:     bool  = False):
        super().__init__()
        self.omega_0      = omega_0
        self.sigma_0      = sigma_0
        self.is_first     = is_first
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, fan_in: int):
        with torch.no_grad():
            # First layer:  Uniform(-1/fan_in, 1/fan_in)  (as in SIREN)
            # Hidden layers: scale by 1/omega_0 so pre-activations are O(1)
            if self.is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z = omega_0 * (W*x + b)
        z        = self.omega_0 * self.linear(x)               # (N, out_features)
        envelope = torch.exp(-z ** 2 / (2.0 * self.sigma_0 ** 2))
        real     = torch.cos(z) * envelope                     # (N, out_features)
        imag     = torch.sin(z) * envelope                     # (N, out_features)
        return torch.cat([real, imag], dim=-1)                 # (N, 2*out_features)


class WireResidualBlock(nn.Module):
    """
    Two Wire layers with a linear residual skip connection.

    Input / output both have dimension 2*features (real+imag channels).
    Each inner WireLayer uses features//2 neurons so the output channel
    count stays constant at features = 2*(features//2).
    """

    def __init__(self, features: int, omega_0: float, sigma_0: float):
        super().__init__()
        half = features // 2
        self.layer1 = WireLayer(features, half, omega_0=omega_0, sigma_0=sigma_0)
        self.layer2 = WireLayer(features, half, omega_0=omega_0, sigma_0=sigma_0)
        # Linear skip — initialized near-identity to preserve gradient flow
        self.skip   = nn.Linear(features, features, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.skip.weight) if features == features else \
                nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(x) + self.layer2(x) + self.skip(x)


##############################################################################
# 5.  SINDRIHET — EXPERT SUBNETWORK  (Wire-based)
##############################################################################

class SindriNet(nn.Module):
    """
    Expert subnetwork S_i(u1, u2, u3) — Wire-based antiderivative approximator
    for a single fixed parameter set (a_i, b_i, m_i, n_i).

    Input:  integration variables only  (n_int_vars = 3 dimensions)
    Output: scalar F_i(u) such that d^3 F_i / du1 du2 du3  ~  f_tilde_i(u)

    Architecture
    ------------
    Wire entry layer : n_int_vars -> 2*entry_width   (is_first=True)
    n_blocks x WireResidualBlock : 2*entry_width -> 2*entry_width
    Linear output    : 2*entry_width -> 1            (no activation)

    Because the parameter label is NOT part of the input, SindriNet uses its
    full representational capacity exclusively for the shape of one integrand,
    without competing with other parameter variants for network width.

    Wire vs SIREN:
      - Gaussian envelope provides local support suited to the exponentially
        decaying integrand.
      - Gabor wavelet derivatives are self-similar — no blow-up over three
        levels of autograd differentiation.
      - output_scale compensates for the omega_0^3 factor accumulated by
        three sequential autograd differentiations.
    """

    def __init__(self,
                 n_int_vars:   int       = 3,
                 entry_width:  int       = 64,
                 n_blocks:     int       = 3,
                 omega_0:      float     = 10.0,
                 sigma_0:      float     = 10.0,
                 output_scale: float     = 1.0):
        super().__init__()
        self.n_int_vars = n_int_vars
        self.omega_0    = omega_0
        self.sigma_0    = sigma_0

        mid_dim = 2 * entry_width    # real + imag channels after entry layer

        # Entry Wire layer: n_int_vars -> mid_dim
        self.entry = WireLayer(n_int_vars, entry_width,
                               omega_0=omega_0, sigma_0=sigma_0,
                               is_first=True)

        # Residual Wire blocks: mid_dim -> mid_dim
        self.blocks = nn.ModuleList([
            WireResidualBlock(mid_dim, omega_0=omega_0, sigma_0=sigma_0)
            for _ in range(n_blocks)
        ])

        # Final linear output layer — no activation
        self.out = nn.Linear(mid_dim, 1)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.out.weight)
            # Compensate for omega_0^3 factor from three autograd derivatives
            self.out.weight.data *= output_scale
            nn.init.zeros_(self.out.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            u: (N, n_int_vars) — integration variables in [0, 1]^3
        Returns:
            (N, 1) scalar antiderivative values
        """
        h = self.entry(u)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


##############################################################################
# 6.  BROKNET — MIXTURE-OF-EXPERTS WRAPPER
##############################################################################

class BrokNet(nn.Module):
    """
    Brok — Mixture-of-Experts Wire-net for parametric numerical integration.

    Accepts full input (a, b, m, n, u1, u2, u3), routes by the discrete key
    (a, b, m, n) to the corresponding SindriNet expert, which then operates
    only on (u1, u2, u3).

    The router is a plain dict — no learnable gating, no softmax. Since the
    parameter labels are known in advance and fully discrete, deterministic
    routing is exact and requires zero additional parameters.

    Args:
        param_sets:   list of (a, b, m, n) tuples — one per expert
        n_int_vars:   number of integration variables (3)
        entry_width:  neurons per Wire layer (output channels = 2 * entry_width)
        n_blocks:     number of WireResidualBlocks per SindriNet
        omega_0:      Wire frequency scaling factor
        sigma_0:      Wire Gaussian envelope bandwidth
        output_scale: output layer scale (use 1/omega_0^3 to compensate derivatives)
    """

    def __init__(self,
                 param_sets:   List[Tuple],
                 n_int_vars:   int   = 3,
                 entry_width:  int   = 64,
                 n_blocks:     int   = 3,
                 omega_0:      float = 10.0,
                 sigma_0:      float = 10.0,
                 output_scale: float = 1.0):
        super().__init__()
        self.param_sets = param_sets
        self.n_int_vars = n_int_vars
        self.n_params   = 4    # a, b, m, n

        # One SindriNet per parameter set
        self.experts = nn.ModuleList([
            SindriNet(n_int_vars=n_int_vars,
                      entry_width=entry_width,
                      n_blocks=n_blocks,
                      omega_0=omega_0,
                      sigma_0=sigma_0,
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
          n_epochs:      int   = 8000,
          n_per_expert:  int   = 1024,
          lr:            float = 5e-4,
          mode:          str   = "joint",   # "joint" or "individual"
          device:        torch.device = torch.device("cpu"),
          verbose_every: int   = 500,
          log_fn=print) -> Tuple[List[float], Dict]:
    """
    Train BrokNet experts.

    Args:
        brok:         BrokNet model instance.
        param_sets:   List of (a, b, m, n) tuples for each expert.
        n_epochs:     Total number of training epochs.
        n_per_expert: Number of random samples generated per expert per epoch.
        lr:           Initial learning rate for Adam optimizer.
        mode:         "joint" for single update per epoch using mean loss,
                      "individual" for updating each expert sub-network separately.
        device:       Torch device (cpu, cuda, or mps).
        verbose_every: Frequency of logging results to console.
        log_fn:       Function used for logging (default is print).

    Returns:
        history:    List of average loss values per epoch.
        norm_cache: Dictionary of normalization constants (center values).
    """

    # AdamW with weight decay matches the Wire optimizer from wire_nni.py
    optimizer = torch.optim.AdamW(brok.parameters(), lr=lr, weight_decay=1e-5)

    # CosineAnnealingWarmRestarts: four restarts over training — helps escape
    # shallow local minima that arise with Gabor wavelet activations
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, n_epochs // 4), T_mult=1, eta_min=lr / 50
    )

    loss_fn = nn.MSELoss()
    brok.to(device)
    brok.train()

    n_experts = len(param_sets)
    history: List[float] = []
    norm_cache: Dict     = {}
    t0 = time.time()

    total_params = sum(p.numel() for p in brok.parameters())
    params_per   = total_params // n_experts

    log_fn(f"\n{'═' * 70}")
    log_fn(f"  Training Mode: {mode.upper()}")
    log_fn(f"  BrokNet  |  {n_experts} SindriNet experts  |  "
           f"{params_per:,} params/expert  |  {total_params:,} total")
    log_fn(f"  Epochs: {n_epochs}  |  {n_per_expert} pts/expert/epoch  |  "
           f"lr: {lr} -> {lr / 50}")
    log_fn(f"  Device: {device}")
    log_fn(f"{'═' * 70}")

    for epoch in range(1, n_epochs + 1):
        current_expert_losses: List[torch.Tensor] = []
        epoch_loss_val = 0.0

        if mode == "joint":
            # --- JOINT TRAINING MODE ---
            # Compute all expert losses, average them, then update once
            optimizer.zero_grad()
            expert_losses_tensors: List[torch.Tensor] = []

            for i, (a, b, m, n) in enumerate(param_sets):
                u, f_tilde = make_expert_batch(a, b, m, n, n_per_expert,
                                               device, norm_cache)
                dG     = mixed_partial_3_expert(brok.experts[i], u)
                loss_i = loss_fn(dG, f_tilde)
                expert_losses_tensors.append(loss_i)
                current_expert_losses.append(loss_i.detach())

            total_loss = torch.stack(expert_losses_tensors).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(brok.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss_val = total_loss.item()

        elif mode == "individual":
            # --- INDIVIDUAL TRAINING MODE ---
            # Update each expert independently within the epoch
            temp_losses: List[torch.Tensor] = []
            for i, (a, b, m, n) in enumerate(param_sets):
                optimizer.zero_grad()

                u, f_tilde = make_expert_batch(a, b, m, n, n_per_expert,
                                               device, norm_cache)
                dG     = mixed_partial_3_expert(brok.experts[i], u)
                loss_i = loss_fn(dG, f_tilde)

                loss_i.backward()
                torch.nn.utils.clip_grad_norm_(brok.experts[i].parameters(),
                                               max_norm=5.0)
                optimizer.step()

                dl = loss_i.detach()
                temp_losses.append(dl)
                current_expert_losses.append(dl)

            epoch_loss_val = torch.stack(temp_losses).mean().item()

        # Learning rate step: CosineAnnealingWarmRestarts uses epoch index
        scheduler.step(epoch)
        history.append(epoch_loss_val)

        if epoch % verbose_every == 0 or epoch == 1:
            lr_now   = scheduler.get_last_lr()[0]
            elapsed  = time.time() - t0
            per_expert_str = "  ".join(
                f"S{i}={el.item():.2e}"
                for i, el in enumerate(current_expert_losses)
            )
            log_fn(f"  Epoch {epoch:5d}/{n_epochs}  "
                   f"loss={epoch_loss_val:.4e}  lr={lr_now:.2e}  ({elapsed:.1f}s)")
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
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    # ── Hyperparameters ───────────────────────────────────────────────────
    ENTRY_WIDTH  = 64      # neurons per WireLayer (channels = 2 * ENTRY_WIDTH)
    N_BLOCKS     = 3       # WireResidualBlocks per SindriNet expert
    OMEGA_0      = 10.0    # Wire frequency — lower than SIREN; Gaussian envelope
    SIGMA_0      = 10.0    # Wire Gaussian bandwidth
    N_EPOCHS     = 8000
    N_PER_EXPERT = 8192    # random points per expert per epoch
    LR           = 5e-4
    # Three autograd derivatives of Gabor accumulate ~omega_0^3; compensate:
    OUTPUT_SCALE = 1.0 / (OMEGA_0 ** 3)

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    tee_print(f"\n  Run started  : {run_start.strftime('%Y-%m-%d %H:%M')}")
    tee_print(f"  Device       : {device}")

    # ── Build BrokNet ─────────────────────────────────────────────────────
    brok = BrokNet(
        param_sets   = PARAM_SETS,
        n_int_vars   = 3,
        entry_width  = ENTRY_WIDTH,
        n_blocks     = N_BLOCKS,
        omega_0      = OMEGA_0,
        sigma_0      = SIGMA_0,
        output_scale = OUTPUT_SCALE,
    )

    n_total  = sum(p.numel() for p in brok.parameters())
    n_expert = n_total // len(PARAM_SETS)
    mid_dim  = 2 * ENTRY_WIDTH

    tee_print(f"\n  Architecture : BrokNet  [{len(PARAM_SETS)} x SindriNet (Wire)]")
    tee_print(f"  Expert layout: {N_INT_VARS} -> {mid_dim}  (WireLayer entry, is_first=True)")
    tee_print(f"                 {N_BLOCKS} x WireResidualBlock ({mid_dim} channels)")
    tee_print(f"                 {mid_dim} -> 1  (linear output)")
    tee_print(f"  omega_0      : {OMEGA_0}   sigma_0: {SIGMA_0}")
    tee_print(f"  Parameters   : {n_total:,} total  /  {n_expert:,} per expert")
    tee_print(f"  Output scale : {OUTPUT_SCALE:.3e}")
    tee_print(f"  Optimizer    : AdamW  (weight_decay=1e-5)")
    tee_print(f"  Scheduler    : CosineAnnealingWarmRestarts  (T_0={N_EPOCHS // 4})")
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
        # mode          = "individual"
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


# Number of integration variables — used in the architecture printout
N_INT_VARS = 3

if __name__ == "__main__":
    main()
