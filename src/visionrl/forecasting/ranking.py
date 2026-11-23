"""Policy ranking, selection regret, and Proposition 1 identifiability analysis."""

from typing import Any

import numpy as np
import pandas as pd

from visionrl.forecasting.predictor import SCORE_COLS, forecast_cliffs


def kendall_tau_b(x: Any, y: Any) -> float:
    """Kendall tau-b with ties. NaN if undefined (fewer than 2 valid points)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 2:
        return float("nan")
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                tx += 1
            elif sy == 0:
                ty += 1
            elif sx == sy:
                c += 1
            else:
                d += 1
    den = np.sqrt((c + d + tx) * (c + d + ty))
    return float("nan") if den == 0 else float((c - d) / den)


def selection_regret(pred: Any, true: Any) -> float:
    """true[best] minus the mean true value over the policies the forecast ranks first (ties averaged)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if len(pred) < 2 or np.isnan(pred).any() or np.isnan(true).any():
        return float("nan")
    top = np.isclose(pred, pred.max())
    return float(true.max() - true[top].mean())


def ranking_table(d: pd.DataFrame, col: str, proto: str = "strict", method: str = "iso") -> pd.DataFrame:
    """Across policies, per shift kind: does the forecast order policies by true critical severity?"""
    fc = forecast_cliffs(d, col, proto)
    if len(fc) == 0:
        return pd.DataFrame()
    fc = fc[fc.method == method]
    rows = []
    for kind, g in fc.groupby("kind"):
        rows.append(dict(
            predictor=col,
            kind=kind,
            n_policies=len(g),
            tau=kendall_tau_b(g.s_crit_pred, g.s_crit_true),
            regret=selection_regret(g.s_crit_pred, g.s_crit_true),
            true_spread=float(g.s_crit_true.std(ddof=0)),
            pred_spread=float(g.s_crit_pred.std(ddof=0)),
        ))
    return pd.DataFrame(rows)


def identifiability_table(d: pd.DataFrame) -> pd.DataFrame:
    """Spread across policies of the forecast critical severity versus true spread.
    Policy-agnostic scores computed from observation statistics alone barely separate policies (Proposition 1).
    """
    rows = []
    for col in SCORE_COLS:
        fc = forecast_cliffs(d, col, "strict")
        if len(fc) == 0:
            continue
        fc = fc[fc.method == "iso"]
        rows.append(dict(
            predictor=col,
            pred_spread=fc.groupby("kind").s_crit_pred.std(ddof=0).mean(),
            true_spread=fc.groupby("kind").s_crit_true.std(ddof=0).mean(),
        ))
    return pd.DataFrame(rows).set_index("predictor")
