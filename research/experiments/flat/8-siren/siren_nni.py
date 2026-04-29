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
# 1.  PHYSICAL CONSTANTS
##############################################################################
LA = 1.0
M1 = 0.3 / LA
M2 = 0.3 / LA
M3 = 0.3 / LA
P1P1 = -(0.14 / LA) ** 2   # p₁²
P2P2 = -(0.14 / LA) ** 2   # p₂²
PP   = -(0.70 / LA) ** 2   # P² = (p₁+p₂)²
P1P2 = (PP - P1P1 - P2P2) / 2

# LOGARITHMIC SUBSTITUTION
# t = exp(L·u) - 1,  u∈[0,1] → t∈[0, T_MAX]
T_MAX  = 80.0
L_LOG  = math.log(1.0 + T_MAX)   # ≈ 4.394

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]
PARAM_SETS = list(zip(A_VALS, B_VALS, M_VALS, N_VALS))

FLOATING_POINT_PRECISION = torch.float32
torch.device("cpu")


##############################################################################
# 2.  D(α₁,α₂) и R²(α₁,α₂)
##############################################################################

def D_R2(alpha1: torch.Tensor, alpha2: torch.Tensor) -> tuple:
    """
    D  = α₁α₂P² + P₁²α₂(1-α₁-α₂) + P₂²α₁(1-α₁-α₂) + α₁m₁² + α₂m₂² + α₃m₃²
    R² = α₁²P₂² + α₂²P₁² - α₁α₂(P²-P₁²-P₂²)
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
# 3.  INTEGRAND IN UNIT CUBE (u₁, u₂, u₃) ∈ [0,1]³
##############################################################################

def korobov(u: torch.Tensor) -> torch.Tensor:
    """x = u²(3-2u)."""
    return u * u * (3.0 - 2.0 * u)


def korobov_weight(u: torch.Tensor) -> torch.Tensor:
    """dx/du = 6u(1-u)."""
    return 6.0 * u * (1.0 - u)


def integrand_transformed(u1: torch.Tensor,
                           u2: torch.Tensor,
                           u3: torch.Tensor,
                           a: float, b: float,
                           m: float, n: float) -> torch.Tensor:
    x1 = korobov(u1)
    x2 = korobov(u2)
    w1 = korobov_weight(u1)   # dx₁/du₁
    w2 = korobov_weight(u2)   # dx₂/du₂

    alpha1 = x1
    alpha2 = (1.0 - x1) * x2
    J_alpha = (1.0 - x1) * w1 * w2

    L = L_LOG
    exp_Lu3 = torch.exp(torch.tensor(L, dtype=u3.dtype, device=u3.device) * u3)
    t  = exp_Lu3 - 1.0
    Jt = L * exp_Lu3   # dt/du₃

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
    """Value of integrand at u=(0.5,0.5,0.5) — used for normalization."""
    uc = torch.tensor([0.5])
    fc = integrand_transformed(uc, uc, uc, a, b, m, n).item()
    return fc if abs(fc) > 1e-30 else 1.0


##############################################################################
# 4.  SIREN NETWORK
#     Sitzmann et al. 2020: "Implicit Neural Representations with Periodic
#     Activation Functions" — https://arxiv.org/abs/2006.09661
#
#     Key design choices for using SIREN as a primitive network
#     (i.e. approximating the antiderivative F such that ∂³F/∂u₁∂u₂∂u₃ ≈ f̃):
#
#       • First layer:  weights ~ Uniform(-1/fan_in, 1/fan_in)
#       • Hidden layers: weights ~ Uniform(-√(6/fan_in)/ω, √(6/fan_in)/ω)
#         This preserves the distribution of activations across layers.
#       • ω₀ (omega_0): frequency scaling — the main tuning knob.
#         Higher ω₀ → the net can represent higher-frequency content.
#         For smooth integrands, ω₀ ∈ [10, 30] usually works well.
#
#     The derivative ∂sin(ω·x)/∂x = ω·cos(ω·x), so each autograd
#     differentiation brings down a factor ω — three derivatives ⇒ ω³.
#     This is accounted for by the output_scale initialisation.
##############################################################################

class SirenLayer(nn.Module):
    """Single SIREN linear + sin layer."""

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


class SirenPrimitiveNet(nn.Module):
    """
    SIREN-based primitive network N(a, b, m, n, u₁, u₂, u₃) ≈ F(s; u).

    Approximates the antiderivative such that
        ∂³N/∂u₁∂u₂∂u₃  ≈  f̃(s; u).

    Architecture: SIREN hidden layers + linear output (no final sin).
    """

    def __init__(self,
                 n_params:    int   = 4,
                 n_int_vars:  int   = 3,
                 hidden_sizes: list  = None,
                 omega_0:     float = 30.0,
                 output_scale: float = 1.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.n_params   = n_params
        self.n_int_vars = n_int_vars
        self.omega_0    = omega_0

        layers = []
        in_dim = n_params + n_int_vars
        for i, h in enumerate(hidden_sizes):
            layers.append(SirenLayer(in_dim, h,
                                     omega_0=omega_0,
                                     is_first=(i == 0)))
            in_dim = h

        # Final linear layer — no sin activation (output is a scalar primitive)
        final = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_dim) / omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
            # Scale output so that ∂³N values are O(1) initially.
            # Three derivatives each pull down ω₀, so we compensate:
            final.weight.data *= output_scale
            final.bias.data   *= output_scale
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


##############################################################################
# 5.  DERIVATIVE ∂³N/∂u₁∂u₂∂u₃ VIA AUTOGRAD
##############################################################################

def mixed_partial_3(net: SirenPrimitiveNet,
                    batch: torch.Tensor) -> torch.Tensor:
    """
    ∂³N/∂u₁∂u₂∂u₃  computed by sequential autograd.

    batch: (N, n_params + 3)
        - [:, :n_params]  — s (not differentiated)
        - [:, n_params:]  — u₁,u₂,u₃ (differentiated)
    """
    k = net.n_params
    s = batch[:, :k]
    u = batch[:, k:].detach().requires_grad_(True)   # (N, 3)

    inp   = torch.cat([s, u], dim=1)
    N_out = net(inp).squeeze(-1)                      # (N,)

    ones_N = torch.ones_like(N_out)

    # ∂N/∂u₁
    g1 = torch.autograd.grad(
        N_out, u,
        grad_outputs=ones_N,
        create_graph=True, retain_graph=True,
    )[0][:, 0]   # (N,)

    # ∂²N/∂u₁∂u₂
    g12 = torch.autograd.grad(
        g1, u,
        grad_outputs=torch.ones_like(g1),
        create_graph=True, retain_graph=True,
    )[0][:, 1]   # (N,)

    # ∂³N/∂u₁∂u₂∂u₃
    g123 = torch.autograd.grad(
        g12, u,
        grad_outputs=torch.ones_like(g12),
        create_graph=True,   # needed for loss.backward()
    )[0][:, 2]   # (N,)

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

def train(net: SirenPrimitiveNet,
          param_sets:   list,
          n_epochs:     int   = 5000,
          n_per_param:  int   = 512,
          lr:           float = 1e-3,
          device:       torch.device = torch.device("cpu"),
          verbose_every: int  = 500) -> tuple:

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 10
    )
    loss_fn = nn.MSELoss()
    net.to(device)
    net.train()

    history    = []
    norm_cache = {}
    t0         = time.time()

    header = (
        f"\n{'═' * 64}\n"
        f"  Training: {n_epochs} epochs | {n_per_param} pts/set | "
        f"{len(param_sets)} sets\n"
        f"  Per epoch: {len(param_sets) * n_per_param} pts | device: {device}\n"
        f"  lr: {lr} -> {lr / 10}  (CosineAnnealing)\n"
        f"  Network: SIREN  omega_0={net.omega_0}\n"
        f"{'═' * 64}"
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
        scheduler.step()

        lv = loss.item()
        history.append(lv)

        if epoch % verbose_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
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

def compute_integral(net: SirenPrimitiveNet,
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
            sign = (-1) ** (3 - int(c1 + c2 + c3))
            u_t  = torch.tensor([[c1, c2, c3]],
                                 dtype=FLOATING_POINT_PRECISION).to(device)
            inp   = torch.cat([s_row, u_t], dim=1)
            I_tilde += sign * net(inp).item()

    return float(I_tilde * fc)


##############################################################################
# 9.  REFERENCE VALUES via scipy.integrate
##############################################################################

def reference_scipy(a: float, b: float, m: float, n: float,
                    t_max: float = T_MAX,
                    tol:   float = 1e-9) -> tuple:
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
# 10.  TEE: print to stdout AND buffer for log file
##############################################################################

_log_buffer = io.StringIO()


def tee_print(*args, **kwargs):
    """Print to console and accumulate in _log_buffer."""
    print(*args, **kwargs)
    # Replicate to buffer
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
    HIDDEN       = [128, 128, 128]
    OMEGA_0      = 20.0     # SIREN frequency; try 20–40 for this integrand
    N_EPOCHS     = 8000
    N_PER_PARAM  = 1024
    LR           = 5e-4
    OUTPUT_SCALE = 1.0 / (OMEGA_0 ** 3)   # compensate for ω³ from 3 derivatives

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    tee_print(f"\n  Run started : {run_start.strftime('%Y-%m-%d %H:%M')}")
    tee_print(f"  Device      : {device}")

    net = SirenPrimitiveNet(
        n_params    = 4,
        n_int_vars  = 3,
        hidden_sizes = HIDDEN,
        omega_0     = OMEGA_0,
        output_scale = OUTPUT_SCALE,
    )
    n_p = sum(p.numel() for p in net.parameters())
    tee_print(f"\n  Architecture : SIREN  {HIDDEN}  omega_0={OMEGA_0}")
    tee_print(f"  Parameters   : {n_p:,}")
    tee_print(f"  Output scale : {OUTPUT_SCALE:.3e}")
    tee_print(f"  t-substitution: logarithmic  "
              f"t = exp({L_LOG:.3f}·u₃) - 1,  t ∈ [0, {T_MAX:.0f}]")

    history, norm_cache = train(
        net,
        param_sets   = PARAM_SETS,
        n_epochs     = N_EPOCHS,
        n_per_param  = N_PER_PARAM,
        lr           = LR,
        device       = device,
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
    SEP = '═' * 84
    tee_print(f"\n{SEP}")
    tee_print(f"  {'(a,b,m,n)':^12}  {'NNI':^14}  {'scipy':^14}  "
              f"{'|Δ|':^11}  {'|Δ|/ref':^9}  {'Digits':^6}")
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
    log_filename = f"results_siren_{timestamp}.out"
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(_log_buffer.getvalue())
        print(f"\n  Log saved -> {log_filename}")
    except OSError as exc:
        print(f"\n  [WARNING] Could not save log: {exc}")


if __name__ == "__main__":
    main()
