"""
physics.py — shared physics integrand and reference values for SIREN experiments.

This module contains the integrand definition, coordinate transformations,
and scipy reference values that are common across all SIREN-based experiments.
"""
import math

import numpy as np
import torch
from scipy import integrate

##############################################################################
# 1.  PHYSICAL CONSTANTS
##############################################################################
LA = 1.0
M1 = 0.3 / LA
M2 = 0.3 / LA
M3 = 0.3 / LA
P1P1 = -(0.14 / LA) ** 2
P2P2 = -(0.14 / LA) ** 2
PP   = -(0.70 / LA) ** 2
P1P2 = (PP - P1P1 - P2P2) / 2

# Logarithmic substitution: t = exp(L·u) - 1,  u∈[0,1] → t∈[0, T_MAX]
T_MAX = 80.0
L_LOG = math.log(1.0 + T_MAX)

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]
PARAM_SETS = list(zip(A_VALS, B_VALS, M_VALS, N_VALS))

FLOATING_POINT_PRECISION = torch.float32


##############################################################################
# 2.  D(α₁,α₂) and R²(α₁,α₂)
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
# 3.  INTEGRAND IN UNIT CUBE
##############################################################################

def korobov(u: torch.Tensor) -> torch.Tensor:
    """x = u²(3-2u)."""
    return u * u * (3.0 - 2.0 * u)


def korobov_weight(u: torch.Tensor) -> torch.Tensor:
    """dx/du = 6u(1-u)."""
    return 6.0 * u * (1.0 - u)


def integrand_transformed(u: torch.Tensor, a: float, b: float,
                          m: float, n: float) -> torch.Tensor:
    """
    Integrand in the unit cube (u1, u2, u3) ∈ [0,1]³.
    Accepts a single tensor u of shape (N, 3) for batch evaluation.
    """
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

    x1 = korobov(u1)
    x2 = korobov(u2)
    w1 = korobov_weight(u1)
    w2 = korobov_weight(u2)

    alpha1 = x1
    alpha2 = (1.0 - x1) * x2
    J_alpha = (1.0 - x1) * w1 * w2

    exp_Lu3 = torch.exp(torch.tensor(L_LOG, dtype=u.dtype, device=u.device) * u3)
    t = exp_Lu3 - 1.0
    Jt = L_LOG * exp_Lu3

    D, R2 = D_R2(alpha1, alpha2)
    exponent = -(t * D + t / (1.0 + t) * R2)
    exponent = torch.clamp(exponent, min=-500.0, max=500.0)

    alpha1_a = torch.clamp(alpha1, min=0.0) ** a if a > 0 else torch.ones_like(alpha1)
    alpha2_b = torch.clamp(alpha2, min=0.0) ** b if b > 0 else torch.ones_like(alpha2)

    f_phys = (alpha1_a * alpha2_b
              * t ** m / (1.0 + t) ** n
              * torch.exp(exponent))

    return f_phys * J_alpha * Jt


##############################################################################
# 4.  REFERENCE VALUES via scipy.integrate
##############################################################################

def reference_scipy(a: float, b: float, m: float, n: float,
                    t_max: float = T_MAX, tol: float = 1e-9) -> tuple:
    def f_inner(t, alpha2, alpha1):
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
