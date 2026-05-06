import numpy as np
import scipy.integrate as integrate
from .physics import P1P1, P2P2, PP, M1, M2, M3, T_MAX

##############################################################################
# 9.  scipy.integrate
##############################################################################

def reference_scipy(a: float, b: float, m: float, n: float,
                    t_max: float = T_MAX,
                    tol: float = 1e-9) -> tuple:
    def f_inner(t, alpha2, alpha1):
        alpha3 = 1.0 - alpha1 - alpha2
        D = (alpha1 * alpha2 * PP + P1P1 * alpha2 * alpha3 + P2P2 * alpha1 * alpha3
             + alpha1 * M1 ** 2 + alpha2 * M2 ** 2 + alpha3 * M3 ** 2)
        R2 = (alpha1 ** 2 * P2P2 + alpha2 ** 2 * P1P1
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
