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
    # Clamp so floating-point error never puts p outside its own interval (p = 0 or 1).
    return max(0.0, min(centre - half, p)), min(1.0, max(centre + half, p))


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


def mean_ci(values, z: float = Z95) -> tuple[float, float, float]:
    """Mean with a normal-approximation interval (t would be wider for tiny n)."""
    x = np.asarray(values, float)
    if len(x) < 2:
        return (x.mean() if len(x) else math.nan), math.nan, math.nan
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    return m, m - z * se, m + z * se


def welch_diff(x1, x0, z: float = Z95) -> tuple[float, float, float, float]:
    """Difference of means x1 - x0, normal-approximation interval and Welch t-test p."""
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    d = x1.mean() - x0.mean()
    se = math.sqrt(x1.var(ddof=1) / len(x1) + x0.var(ddof=1) / len(x0))
    if se == 0:
        return d, d, d, (1.0 if d == 0 else 0.0)
    p = stats.ttest_ind(x1, x0, equal_var=False).pvalue
    return d, d - z * se, d + z * se, p


def stratified_mean_diff(strata: list[tuple], z: float = Z95) -> dict:
    """Difference of means pooled over strata, each weighted by n1*n0/(n1+n0)
    (the Mantel-Haenszel weight), with a normal-approximation interval and z-test.

    `strata` is a list of (x1, x0) value arrays; strata with an empty arm are dropped.
    """
    num = den = var = 0.0
    used = 0
    for x1, x0 in strata:
        x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
        n1, n0 = len(x1), len(x0)
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0 / (n1 + n0)
        v1 = x1.var(ddof=1) if n1 > 1 else 0.0
        v0 = x0.var(ddof=1) if n0 > 1 else 0.0
        num += w * (x1.mean() - x0.mean())
        den += w
        var += w * w * (v1 / n1 + v0 / n0)
        used += 1
    d = num / den
    se = math.sqrt(var) / den
    p = 2 * stats.norm.sf(abs(d) / se) if se > 0 else (1.0 if d == 0 else 0.0)
    return {"diff": d, "lo": d - z * se, "hi": d + z * se, "p": p, "strata": used}


def homogeneity_p(ks, ns, n_sim: int = 20_000, seed: int = 0) -> float:
    """Test that several groups share one success rate (k_i successes out of n_i).

    Monte Carlo version of the exact conditional test: with the total number of
    successes fixed, the successes are spread over the groups at random
    (multivariate hypergeometric), and the observed chi-square statistic is
    compared with those draws. Unlike the chi-square approximation, it stays
    valid for cells of a few conversations.
    """
    ks, ns = np.asarray(ks, int), np.asarray(ns, int)
    total = int(ks.sum())
    if total in (0, int(ns.sum())):
        return 1.0
    rate = total / ns.sum()
    e1, e0 = ns * rate, ns * (1 - rate)

    def chi2(k):
        return ((k - e1) ** 2 / e1 + (k - e1) ** 2 / e0).sum(axis=-1)

    draws = np.random.default_rng(seed).multivariate_hypergeometric(
        ns, total, size=n_sim
    )
    # Tolerance so draws that tie with the observed table count as at least as extreme.
    extreme = (chi2(draws) >= chi2(ks) - 1e-9).sum()
    return (1 + extreme) / (n_sim + 1)


def cochran_q_p(estimates, ses) -> float:
    """Cochran's Q test that several independent estimates agree, given their
    standard errors. Estimates with a zero standard error are dropped."""
    d, se = np.asarray(estimates, float), np.asarray(ses, float)
    keep = se > 0
    d, w = d[keep], 1 / se[keep] ** 2
    if len(d) < 2:
        return math.nan
    q = (w * (d - (w * d).sum() / w.sum()) ** 2).sum()
    return stats.chi2.sf(q, len(d) - 1)
