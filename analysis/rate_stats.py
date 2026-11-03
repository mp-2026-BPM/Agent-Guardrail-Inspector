"""Small-sample statistics for success rates (binary outcomes per conversation).

Everything here works on counts so it can be applied per cell, per scenario or
stratified. Intervals are 95% unless `z` says otherwise.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def newcombe_diff(
    k1: int, n1: int, k0: int, n0: int, z: float = Z95
) -> tuple[float, float, float]:
    """Difference p1 - p0 with Newcombe's hybrid score interval (method 10)."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1, z)
    l0, u0 = wilson(k0, n0, z)
    d = p1 - p0
    return (
        d,
        d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2),
        d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2),
    )


def fisher_p(k1: int, n1: int, k0: int, n0: int) -> float:
    """Two-sided Fisher exact test on the 2x2 table."""
    return stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]]).pvalue


def mantel_haenszel_rd(strata: pd.DataFrame, z: float = Z95) -> dict:
    """Mantel-Haenszel risk difference across strata, with Sato's variance
    and the Cochran-Mantel-Haenszel test (no continuity correction).

    `strata` has one row per stratum with columns k1, n1 (treated) and k0, n0
    (reference). Strata where either arm is empty are dropped.
    """
    s = strata[(strata.n1 > 0) & (strata.n0 > 0)]
    a, n1, c, n0 = (s[col].to_numpy(float) for col in ("k1", "n1", "k0", "n0"))
    n = n1 + n0
    w = (n1 * n0 / n).sum()
    rd = ((a * n0 - c * n1) / n).sum() / w
    p_term = ((n1**2 * c - n0**2 * a + n1 * n0 * (n0 - n1) / 2) / n**2).sum()
    q_term = ((a * (n0 - c) + c * (n1 - a)) / (2 * n)).sum()
    se = math.sqrt(max(rd * p_term + q_term, 0.0)) / w

    m1 = a + c
    expected = n1 * m1 / n
    var = n1 * n0 * m1 * (n - m1) / (n**2 * (n - 1))
    chi2 = (a - expected).sum() ** 2 / var.sum() if var.sum() > 0 else math.nan
    return {
        "rd": rd,
        "lo": rd - z * se,
        "hi": rd + z * se,
        "p_cmh": stats.chi2.sf(chi2, 1) if not math.isnan(chi2) else math.nan,
        "strata": len(s),
    }


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted
