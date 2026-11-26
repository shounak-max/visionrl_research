"""Prevalence portability theory (Proposition 2) and AUROC mixture formulas."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from visionrl.forecasting.metrics import auroc, critical_severity, rank_vals
from visionrl.forecasting.predictor import PROPOSED


def auroc_mixture(q: float, fp: float, fc: float, a: float) -> float:
    """Proposition 2: AUROC under a mixture of plateau (1-q, fail rate fp) and cliff (q, fail rate fc) regimes.
    AUROC = 1/2 + (a - 1/2) * (pf_c - pn_c) = 1/2 + (a - 1/2) * J(q)
    where J(q) is the Youden index of the oracle regime indicator.
    """
    den_f = q * fc + (1 - q) * fp
    den_n = q * (1 - fc) + (1 - q) * (1 - fp)
    if den_f <= 0.0 or den_n <= 0.0:
        return float("nan")
    pf_c = q * fc / den_f
    pn_c = q * (1 - fc) / den_n
    pf_p, pn_p = 1 - pf_c, 1 - pn_c
    return float(pf_c * pn_p * a + pf_p * pn_c * (1 - a) + 0.5 * (pf_c * pn_c + pf_p * pn_p))


def regime_summary(g: pd.DataFrame, col: str, s_crit: float) -> Optional[Dict[str, Any]]:
    """The three-number, prevalence-portable summary of one (policy, shift kind): (a, fp, fc) split at cliff."""
    z = g["z_" + col].values
    cliff = g.sev.values >= s_crit
    zc, zp = z[cliff], z[~cliff]
    if len(zc) < 2 or len(zp) < 2:
        return None
    a = float((rank_vals(np.r_[zc, zp])[: len(zc)].sum() - len(zc) * (len(zc) + 1) / 2) / (len(zc) * len(zp)))
    return dict(
        a=a,
        fp=float(g.fail.values[~cliff].mean()),
        fc=float(g.fail.values[cliff].mean()),
        n_cliff=len(zc),
        n_plateau=len(zp),
    )


def portable_check(
    d: pd.DataFrame,
    col: str = PROPOSED,
    q_grid: Tuple[float, ...] = (0.05, 0.15, 0.3, 0.5, 0.7),
    n_draw: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Empirical verification of Proposition 2: Re-mix real episodes to target q and compare with auroc_mixture."""
    rng = np.random.default_rng(seed)
    rows = []
    t = d[d.split == "test"]
    for (_aid, kind), g in t.groupby(["agent_id", "kind"]):
        sr = g.groupby("sev").success.mean().sort_index()
        sc, cens = critical_severity(sr.index.values, sr.values)
        if cens or sc != sc:
            continue
        sm = regime_summary(g, col, sc)
        if sm is None or sm["fp"] >= 1 or sm["fc"] <= 0:
            continue
        cl, pl = g[g.sev >= sc], g[g.sev < sc]
        for q in q_grid:
            take_c = rng.random(n_draw) < q
            s_ = np.r_[
                cl["z_" + col].values[rng.integers(0, len(cl), take_c.sum())],
                pl["z_" + col].values[rng.integers(0, len(pl), (~take_c).sum())],
            ]
            y_ = np.r_[
                cl.fail.values[rng.integers(0, len(cl), take_c.sum())],
                pl.fail.values[rng.integers(0, len(pl), (~take_c).sum())],
            ]
            emp = auroc(s_, y_)
            pred = auroc_mixture(q, sm["fp"], sm["fc"], sm["a"])
            rows.append(dict(
                agent_id=_aid,
                kind=kind,
                q=q,
                auroc_empirical=emp,
                auroc_predicted=pred,
                abs_err=abs(emp - pred),
                **sm,
            ))
    return pd.DataFrame(rows)
