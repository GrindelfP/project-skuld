import numpy as np
from scipy.integrate import quad

# Global variables (simulating FORTRAN COMMON blocks)
param1 = {'m': 0, 'n': 0, 'k': 0, 'l': 0}
param3 = {'alp1': 0.0}
param4 = {'alp2': 0.0}

la = 1.0
m1_global = (1.4 / la)
m2_global = (1.4 / la)
m3_global = (1.4 / la)
p11_global = - (0.14 / la)**2
p22_global = - (0.14 / la)**2
PP_global = - (1.2 / la)**2

def funcI3(t):
    """The integral function (FORTRAN funcI3)"""
    alp1 = param3['alp1']
    alp2 = param4['alp2']
    m = param1['m']
    n = param1['n']
    k = param1['k']
    l = param1['l']

    RR = alp1**2.0 * p11_global + alp2**2.0 * p22_global - alp1 * alp2 * (PP_global - p11_global - p22_global)
    DD = alp1 * (p11_global + m1_global**2) + alp2 * (p22_global + m2_global**2) + (1.0 - alp1 - alp2) * (m3_global**2) - RR
    z0 = t * DD + t / (1.0 + t) * RR
    Fz0 = np.exp(-3.0 * z0)
    return (alp1**m) * (alp2**n) * (t**k / (1.0 + t)**l) * Fz0

def funcI31(alp2a):
    """The first integration (FORTRAN funcI31)"""
    param4['alp2'] = alp2a
    errabs = 0.01
    errrel = 0.001
    result = quad(funcI3, 0.0, 100.0, epsabs=errabs, epsrel=errrel)
    return result

def ff(x):
    """FORTRAN function ff"""
    return 1.0 - 2.0 * x

def funcI32(alp1a):
    """The second integration (FORTRAN funcI32)"""
    param3['alp1'] = alp1a
    errabs = 0.001
    errrel = 0.0001
    aaa = 0.0
    bbb = 1.0 - ff(alp1a)
    result = quad(funcI31, aaa, bbb, epsabs=errabs, epsrel=errrel)
    return result

def funcI33(mm, nn, kk, ll):
    """The third integration (FORTRAN funcI33)"""
    param1['m'] = mm
    param1['n'] = nn
    param1['k'] = kk
    param1['l'] = ll
    errabs = 0.001
    errrel = 0.0001
    aaa = 0.0
    bbb = 1.0
    result = quad(funcI32, aaa, bbb, epsabs=errabs, epsrel=errrel)
    return result

if __name__ == "__main__":
    # The main program (FORTRAN program integralI3)
    # Note: The original FORTRAN code had a commented-out print statement
    # print *, funcI3(0.1)
    # Since funcI3 relies on global variables set within the integration functions,
    # directly calling it here without that context might not yield the intended result.

    print(funcI33(0, 0, 0, 0))
    print(funcI33(0, 0, 1, 2))
    print(funcI33(0, 1, 1, 2))
    print(funcI33(1, 0, 1, 2))
    print(funcI33(1, 1, 1, 2))

    print(funcI33(0, 0, 2, 3))
    print(funcI33(1, 0, 2, 3))
    print(funcI33(1, 1, 2, 3))
