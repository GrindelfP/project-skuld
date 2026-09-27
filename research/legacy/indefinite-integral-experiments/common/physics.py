import torch
import math

##############################################################################
# 1.  PHYSICAL CONSTANTS
##############################################################################
LA = 1.0
M1 = 0.3 / LA
M2 = 0.3 / LA
M3 = 0.3 / LA
P1P1 = -(0.14 / LA) ** 2  # p₁²
P2P2 = -(0.14 / LA) ** 2  # p₂²
PP = -(0.70 / LA) ** 2  # P² = (p₁+p₂)²
P1P2 = (PP - P1P1 - P2P2) / 2

# LOGARITHMIC SUBSTITUTION
# t = exp(L·u) - 1,  u∈[0,1] → t∈[0, T_MAX]
# T_MAX=80: min(D+R²)≈0.07, exp[-0.07·80] ≈ 0.004 → Negligibly small.
T_MAX = 80.0
L_LOG = math.log(1.0 + T_MAX)  # ≈ 4.394

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]
PARAM_SETS = list(zip(A_VALS, B_VALS, M_VALS, N_VALS))

FLOATING_POINT_PRECISION = torch.float32
