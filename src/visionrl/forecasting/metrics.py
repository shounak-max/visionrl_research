"""Statistical metrics, uncertainty estimation, conformal calibration, and hypothesis test functions."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def rank_vals(x: Any) -> np.ndarray:
    """Average rank values."""
    return pd.Series(np.asarray(x, float)).rank(method="average").values


def pearson(a: Any, b: Any) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: Any, b: Any) -> float:
    return pearson(rank_vals(a), rank_vals(b))


def auroc(score: Any, label: Any) -> float:
    """Rank-based AUROC. Returns NaN when only one class is present (metric is undefined, not 0.5)."""
    s, y = np.asarray(score, float), np.asarray(label).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank_vals(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score: Any, label: Any) -> float:
    """Average precision with ties grouped (thresholds sit at distinct score values)."""
    s, y = np.asarray(score, float), np.asarray(label).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]                  # group ends
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def tpr_at_fpr(score: Any, label: Any, fpr: float = 0.05) -> float:
    s, y = np.asarray(score, float), np.asarray(label).astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    tau = np.quantile(s[~y], 1 - fpr)
    return float((s[y] > tau).mean())


def brier(p: Any, y: Any) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def metrics_bundle(score: Any, fail: Any) -> Dict[str, Any]:
    """Report failure predictor metrics with prevalence, random reference and status."""
    fail = np.asarray(fail).astype(int)
    prev = float(fail.mean()) if len(fail) else float("nan")
    status = "ok"
    if fail.sum() == 0:
        status = "undefined: no failures"
    elif fail.sum() == len(fail):
        status = "undefined: no successes"
    ap = auprc(score, fail)
    return dict(
        n=int(len(fail)),
        prevalence=prev,
        auroc=auroc(score, fail),
        auprc=ap,
        auprc_random=prev,
        auprc_lift=(ap / prev if prev and prev > 0 else float("nan")),
        tpr_at_5fpr=tpr_at_fpr(score, fail, 0.05),
        status=status,
    )


def cluster_bootstrap(fn: Callable[[pd.DataFrame], float], df: pd.DataFrame, group_col: str, B: int, seed: int = 0) -> Tuple[float, float]:
    """Percentile CI of fn(df) resampling whole groups (episodes sharing a seed are dependent)."""
    rng = np.random.default_rng(seed)
    groups = {g: d for g, d in df.groupby(group_col)}
    keys = list(groups)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        v = fn(pd.concat([groups[keys[i]] for i in pick], ignore_index=True))
        if v == v:
            vals.append(v)
    if len(vals) < max(10, B // 10):
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def paired_boot_diff(a: Any, b: Any, B: int = 2000, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    """Mean paired difference a - b with a percentile CI; units are episodes paired by seed."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def mcnemar_exact(x: Any, y: Any) -> Tuple[float, int, int]:
    """Exact two-sided McNemar test for paired binary outcomes x, y (True = event)."""
    x, y = np.asarray(x).astype(bool), np.asarray(y).astype(bool)
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, p)), b, c


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    n1, n2, k, N = a + b, c + d, a + c, a + b + c + d
    pmf = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    p_obs = pmf(a)
    return float(sum(pmf(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pmf(x) <= p_obs + 1e-12))


def n_per_arm(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    return math.ceil((z(1 - alpha / 2) + z(power)) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)


def holm(pvals: Any) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def critical_severity(sev: Any, sr: Any, frac: float = 0.5) -> Tuple[float, bool]:
    """Smallest severity at which success rate falls to frac * (clean success rate), by linear interpolation.
    Returns (value, censored). If it never falls that far, returns (max severity, True).
    """
    sev, sr = np.asarray(sev, float), np.asarray(sr, float)
    o = np.argsort(sev)
    sev, sr = sev[o], sr[o]
    base = sr[0]
    if not base > 0:
        return float("nan"), False
    thr = frac * base
    for i in range(1, len(sev)):
        if sr[i] <= thr:
            lo, hi = sr[i - 1], sr[i]
            w = 0.0 if lo == hi else (lo - thr) / (lo - hi)
            return float(sev[i - 1] + w * (sev[i] - sev[i - 1])), False
    return float(sev[-1]), True


def cliffness(sev: Any, sr: Any) -> float:
    """Share of the total success drop that happens in the single steepest step: 1 = pure cliff,
    about 1/(number of steps) = perfectly gradual. NaN if there is no drop.
    """
    sr = np.asarray(sr, float)[np.argsort(sev)]
    total = sr[0] - sr.min()
    if total <= 1e-9 or len(sr) < 3:
        return float("nan")
    return float(np.max(-np.diff(sr)) / total)


def conformal_threshold(vals: Any, alpha: float) -> float:
    """Split-conformal threshold from clean calibration scores: ceil((n+1)(1-alpha))-th smallest value.
    Controls the clean false-alarm rate at alpha.
    """
    v = np.sort(np.asarray(vals, float))
    n = len(v)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha) - 1e-9))
    return float(v[min(max(k, 1), n) - 1])


def ema_smooth(x: Any, ema: float) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.empty(len(x), float)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = ema * y[t - 1] + (1 - ema) * x[t]
    return y


def paired_signflip_p(a: Any, b: Any, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test for mean difference a - b (episodes paired by agent and seed).
    Returns exactly 1.0 when every pair is identical.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan")
    obs = abs(d.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d[None, :]).mean(1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))
