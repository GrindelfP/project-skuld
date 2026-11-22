"""High-precision scoring references for the finite-cutoff meson integrals."""

import mpmath as mp

from physics import D_R2, PARAM_SETS, T_MAX


def physical_references(order: int) -> list[mp.mpf]:
    """Independent Gauss integration on the physical simplex and linear t panels."""
    nodes, weights = mp.gauss_quadrature(order, "legendre01")
    panels = (0, 1, 4, 16, 40, T_MAX)
    time_nodes = []
    for lo, hi in zip(panels[:-1], panels[1:], strict=True):
        for z, w in zip(nodes, weights, strict=True):
            t = lo + (hi - lo) * z
            ratio = t / (1 + t)
            time_nodes.append((t, ratio, (hi - lo) * w * ratio / (1 + t)))

    totals = [mp.mpf(0) for _ in PARAM_SETS]
    for x, wx in zip(nodes, weights, strict=True):
        for y, wy in zip(nodes, weights, strict=True):
            alpha1, alpha2 = x, (1 - x) * y
            d, r2 = D_R2(alpha1, alpha2)
            m1 = m2 = mp.mpf(0)
            for t, ratio, wt in time_nodes:
                value = wt * mp.exp(-t * d - ratio * r2)
                m1 += value
                m2 += value * ratio
            spatial = wx * wy * (1 - x)
            for index, (a, b, m, n) in enumerate(PARAM_SETS):
                if (m, n) not in {(1, 2), (2, 3)}:
                    raise ValueError("Reference time weights require the standard meson exponents")
                totals[index] += spatial * alpha1**a * alpha2**b * (m1 if m == 1 else m2)
    return totals
