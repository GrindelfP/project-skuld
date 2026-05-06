import itertools
import math
import time
import io
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
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
    """Smoothing map x = u^2(3-2u)."""
    return u * u * (3.0 - 2.0 * u)


def korobov_weight(u: torch.Tensor) -> torch.Tensor:
    """Jacobian dx/du = 6u(1-u)."""
    return 6.0 * u * (1.0 - u)


def integrand_transformed(u1, u2, u3, a, b, m, n):
    x1 = korobov(u1)
    x2 = korobov(u2)
    w1 = korobov_weight(u1)
    w2 = korobov_weight(u2)

    alpha1 = x1
    alpha2 = (1.0 - x1) * x2
    J_alpha = (1.0 - x1) * w1 * w2

    L = L_LOG
    exp_Lu3 = torch.exp(torch.tensor(L, dtype=u3.dtype, device=u3.device) * u3)
    t  = exp_Lu3 - 1.0
    Jt = L * exp_Lu3

    D, R2 = D_R2(alpha1, alpha2)
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
# 4.  WIRE ARCHITECTURE
#
#  Wire = "Wavelet Implicit neural Representations"
#  Reference: Saragadam et al. 2023  https://arxiv.org/abs/2301.05187
#
#  Core idea:
#    Instead of  sin(omega * Wx + b)   (SIREN)
#    Wire uses   exp(i*omega*(Wx+b)) * exp(-s^2*(Wx+b)^2)
#               = complex Gabor / Morlet wavelet
#
#  Why better for this integrand?
#  ---------------------------------------------------------------
#  1. SIREN uses purely periodic activations — every neuron "sees"
#     the entire domain equally. Our integrand decays exponentially
#     in t and is supported on the simplex (alpha1+alpha2<=1), so
#     large portions of the domain carry very little weight. Wire's
#     Gaussian envelope naturally provides LOCAL support, letting
#     neurons specialize to the active region without fighting the
#     global periodicity of sin.
#
#  2. Three autograd derivatives of a Gabor wavelet:
#       d/dx [exp(i*omega*x)*exp(-sigma^2*x^2)]
#       = (i*omega - 2*sigma^2*x) * same
#     — the derivative is of the same functional form (no increase
#     in polynomial degree), so gradients stay numerically stable
#     across all three differentiation steps. SIREN derivatives
#     accumulate powers of omega without the balancing Gaussian
#     damping.
#
#  3. The frequency omega and bandwidth sigma are separately tunable
#     per layer, giving the network an implicit multi-resolution
#     decomposition akin to a wavelet frame.
#
#  Implementation note:
#  We work in REAL arithmetic by keeping the real and imaginary
#  parts as separate channels and concatenating them.
#  A Wire layer with in_features input produces 2*out_features real
#  outputs = [Re(Gabor), Im(Gabor)].  The final linear layer maps
#  back to a single real scalar (the primitive F).
#
#  Residual connections:
#  Added between every pair of Wire blocks to improve gradient flow
#  through three levels of autograd differentiation.
##############################################################################

class WireLayer(nn.Module):
    """
    Single Wire (complex Gabor / Morlet wavelet) layer.

    Output dimension is 2 * out_features because we return
    [Re(Gabor), Im(Gabor)] as real tensors.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 omega_0: float = 10.0,
                 sigma_0: float = 10.0,
                 is_first: bool = False):
        super().__init__()
        self.omega_0     = omega_0
        self.sigma_0     = sigma_0
        self.is_first    = is_first
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, fan_in: int):
        with torch.no_grad():
            # First layer: uniform in [-1/fan_in, 1/fan_in] (as in SIREN)
            # Hidden layers: scale by 1/omega_0 so pre-activations ~ O(1)
            if self.is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z = omega_0 * (Wx + b)
        z = self.omega_0 * self.linear(x)              # (N, out_features)
        # Morlet / Gabor:  exp(i*z) * exp(-z^2 / (2*sigma_0^2))
        # Real part:  cos(z) * exp(-z^2 / (2*sigma_0^2))
        # Imag part:  sin(z) * exp(-z^2 / (2*sigma_0^2))
        envelope = torch.exp(-z ** 2 / (2.0 * self.sigma_0 ** 2))
        real = torch.cos(z) * envelope                 # (N, out_features)
        imag = torch.sin(z) * envelope                 # (N, out_features)
        return torch.cat([real, imag], dim=-1)         # (N, 2*out_features)


class WireResidualBlock(nn.Module):
    """
    Two Wire layers with a linear residual skip connection.

    Input/output both have dimension 2*features (real+imag channels).
    """

    def __init__(self, features: int, omega_0: float, sigma_0: float):
        super().__init__()
        # Each WireLayer doubles the channel count; we use features//2 neurons
        # so the output stays at 'features' total real channels.
        half = features // 2
        self.layer1 = WireLayer(features,  half, omega_0=omega_0, sigma_0=sigma_0)
        self.layer2 = WireLayer(features,  half, omega_0=omega_0, sigma_0=sigma_0)
        # Linear projection for skip (no activation)
        self.skip   = nn.Linear(features, features, bias=False)

        # Initialize skip as near-identity to preserve gradient flow
        with torch.no_grad():
            nn.init.eye_(self.skip.weight) if features == features else \
                nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(x) + self.layer2(x) + self.skip(x)


class WirePrimitiveNet(nn.Module):
    """
    Wire-based primitive network  N(s; u) approx F(s; u)

    Approximates the antiderivative such that
        d^3 N / (du1 du2 du3)  approx  f_tilde(s; u).

    Architecture
    ------------
    Input dim  = n_params + n_int_vars = 4 + 3 = 7
    Wire entry layer:  7 -> 2*entry_width   (WireLayer, is_first=True)
    n_blocks x WireResidualBlock:  2*entry_width -> 2*entry_width
    Linear output:  2*entry_width -> 1          (no activation)
    """

    def __init__(self,
                 n_params:    int   = 4,
                 n_int_vars:  int   = 3,
                 entry_width: int   = 64,    # neurons per Wire layer (output = 2x)
                 n_blocks:    int   = 3,
                 omega_0:     float = 10.0,
                 sigma_0:     float = 10.0,
                 output_scale: float = 1.0):
        super().__init__()
        self.n_params   = n_params
        self.n_int_vars = n_int_vars
        self.omega_0    = omega_0
        self.sigma_0    = sigma_0

        in_dim   = n_params + n_int_vars   # 7
        mid_dim  = 2 * entry_width          # real+imag channels

        # Entry Wire layer: in_dim -> mid_dim
        self.entry = WireLayer(in_dim, entry_width,
                               omega_0=omega_0, sigma_0=sigma_0,
                               is_first=True)

        # Residual Wire blocks: mid_dim -> mid_dim
        self.blocks = nn.ModuleList([
            WireResidualBlock(mid_dim, omega_0=omega_0, sigma_0=sigma_0)
            for _ in range(n_blocks)
        ])

        # Final linear output layer
        self.out = nn.Linear(mid_dim, 1)
        with torch.no_grad():
            # Scale so initial d^3N ~ O(output_scale)
            # Three derivatives of Gabor bring down ~omega_0^3
            nn.init.xavier_uniform_(self.out.weight)
            self.out.weight.data *= output_scale
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.entry(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


##############################################################################
# 5.  MIXED PARTIAL DERIVATIVE  d^3 N / (du1 du2 du3)  via autograd
##############################################################################

def mixed_partial_3(net: WirePrimitiveNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    Compute  d^3 N / (du1 du2 du3)  by sequential autograd.

    batch: (N, n_params + 3)
        [:, :n_params]  -- s  (physical params, not differentiated)
        [:, n_params:]  -- u1, u2, u3  (differentiated)
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)   # (N, 3)

    inp   = torch.cat([s, u], dim=1)
    N_out = net(inp).squeeze(-1)                      # (N,)

    ones_N = torch.ones_like(N_out)

    # dN/du1
    g1 = torch.autograd.grad(
        N_out, u,
        grad_outputs=ones_N,
        create_graph=True, retain_graph=True,
    )[0][:, 0]

    # d^2N/(du1 du2)
    g12 = torch.autograd.grad(
        g1, u,
        grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]

    # d^3N/(du1 du2 du3)
    g123 = torch.autograd.grad(
        g12, u,
        grad_outputs=torch.ones_like(g12),
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

def train(net: WirePrimitiveNet,
          param_sets:    list,
          n_epochs:      int   = 5000,
          n_per_param:   int   = 512,
          lr:            float = 1e-3,
          device:        torch.device = torch.device("cpu"),
          verbose_every: int   = 500) -> tuple:

    # AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)

    # Cosine annealing with warm restarts — helps escape local minima
    # T_0=n_epochs//4: four restarts over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, n_epochs // 4), T_mult=1, eta_min=lr / 50
    )

    loss_fn    = nn.MSELoss()
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
        f"  lr: {lr} -> {lr / 50}  (CosineAnnealingWarmRestarts)\n"
        f"  Network: Wire  omega_0={net.omega_0}  sigma_0={net.sigma_0}\n"
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
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(epoch)

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

def compute_integral(net: WirePrimitiveNet,
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

    # ── Hyperparameters ────────────────────────────────────────────────────
    # Wire frequency omega_0: lower than SIREN because the Gaussian envelope
    # already provides decay; omega_0 in [5, 15] works well here.
    OMEGA_0      = 10.0
    SIGMA_0      = 10.0   # bandwidth of the Gaussian envelope
    ENTRY_WIDTH  = 64     # neurons per WireLayer  (output = 2*64 = 128 channels)
    N_BLOCKS     = 3      # residual Wire blocks
    N_EPOCHS     = 8000
    N_PER_PARAM  = 1024
    LR           = 5e-4
    # Three derivatives of Gabor bring down ~omega_0^3; compensate:
    OUTPUT_SCALE = 1.0 / (OMEGA_0 ** 3)

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    tee_print(f"\n  Run started : {run_start.strftime('%Y-%m-%d %H:%M')}")
    tee_print(f"  Device      : {device}")

    net = WirePrimitiveNet(
        n_params    = 4,
        n_int_vars  = 3,
        entry_width = ENTRY_WIDTH,
        n_blocks    = N_BLOCKS,
        omega_0     = OMEGA_0,
        sigma_0     = SIGMA_0,
        output_scale = OUTPUT_SCALE,
    )
    n_p = sum(p.numel() for p in net.parameters())
    mid_dim = 2 * ENTRY_WIDTH

    tee_print(f"\n  Architecture : Wire (Morlet wavelet) + Residual blocks")
    tee_print(f"  Entry layer  : 7 -> {mid_dim}  (WireLayer, is_first=True)")
    tee_print(f"  Residual blocks: {N_BLOCKS} x WireResidualBlock ({mid_dim} channels each)")
    tee_print(f"  Output layer : {mid_dim} -> 1  (linear)")
    tee_print(f"  omega_0      : {OMEGA_0}   sigma_0: {SIGMA_0}")
    tee_print(f"  Parameters   : {n_p:,}")
    tee_print(f"  Output scale : {OUTPUT_SCALE:.3e}")
    tee_print(f"  Optimizer    : AdamW  (weight_decay=1e-5)")
    tee_print(f"  Scheduler    : CosineAnnealingWarmRestarts  (T_0={N_EPOCHS // 4})")
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

    # ── Reference values ───────────────────────────────────────────────────
    tee_print("Computing reference values (scipy) ...\n")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
        tee_print(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  {r:.6e}  +/-  {e:.1e}")

    # ── Comparison table ───────────────────────────────────────────────────
    SEP = '=' * 84
    tee_print(f"\n{SEP}")
    tee_print(f"  {'(a,b,m,n)':^12}  {'Wire-NNI':^14}  {'scipy':^14}  "
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

    # ── Save log file ──────────────────────────────────────────────────────
    log_filename = f"results_wire_{timestamp}.out"
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(_log_buffer.getvalue())
        print(f"\n  Log saved -> {log_filename}")
    except OSError as exc:
        print(f"\n  [WARNING] Could not save log: {exc}")


if __name__ == "__main__":
    main()
