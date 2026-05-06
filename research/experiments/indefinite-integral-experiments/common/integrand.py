import torch
from .physics import L_LOG, P1P1, P2P2, PP, M1, M2, M3


##############################################################################
# 2.  D(α₁,α₂) И R²(α₁,α₂)
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
# 3.  INTEGRAND IN UNIT CUBE В (u₁, u₂, u₃) ∈ [0,1]³
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
    w1 = korobov_weight(u1)  # dx₁/du₁
    w2 = korobov_weight(u2)  # dx₂/du₂

    alpha1 = x1
    alpha2 = (1.0 - x1) * x2
    J_alpha = (1.0 - x1) * w1 * w2

    L = L_LOG
    exp_Lu3 = torch.exp(torch.tensor(L, dtype=u3.dtype, device=u3.device) * u3)
    t = exp_Lu3 - 1.0
    Jt = L * exp_Lu3  # dt/du₃

    D, R2 = D_R2(alpha1, alpha2)
    exponent = -(t * D + t / (1.0 + t) * R2)

    exponent = torch.clamp(exponent, min=-500.0, max=500.0)

    # α₁^a и α₂^b: with a, b=0 is 1; with a, b=1 is α; clamp for edge cases
    alpha1_a = torch.clamp(alpha1, min=0.0) ** a if a > 0 else torch.ones_like(alpha1)
    alpha2_b = torch.clamp(alpha2, min=0.0) ** b if b > 0 else torch.ones_like(alpha2)

    f_phys = (alpha1_a * alpha2_b
              * t ** m / (1.0 + t) ** n
              * torch.exp(exponent))

    return f_phys * J_alpha * Jt


def center_value(a: float, b: float, m: float, n: float) -> float:
    """
    Value of integrand at u=(0.5,0.5,0.5).
    For normalization (f̃ = f / f_center) и restoration (I = Ĩ × f_center).
    """
    uc = torch.tensor([0.5])
    fc = integrand_transformed(uc, uc, uc, a, b, m, n).item()
    return fc if abs(fc) > 1e-30 else 1.0
