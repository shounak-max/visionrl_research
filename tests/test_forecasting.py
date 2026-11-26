"""Unit tests for forecasting statistics, prevalence portability, and ranking metrics."""

import math
import numpy as np
import pytest

from visionrl.forecasting.metrics import (
    auroc,
    auprc,
    brier,
    cliffness,
    conformal_threshold,
    critical_severity,
    ema_smooth,
    paired_signflip_p,
)
from visionrl.forecasting.prevalence import auroc_mixture
from visionrl.forecasting.ranking import kendall_tau_b, selection_regret


def test_basic_metrics():
    score = [0.1, 0.4, 0.35, 0.8]
    label = [0, 0, 1, 1]
    auc = auroc(score, label)
    assert 0.0 <= auc <= 1.0

    prc = auprc(score, label)
    assert 0.0 <= prc <= 1.0

    b = brier([0.2, 0.8], [0, 1])
    assert np.isclose(b, 0.04)


def test_ranking_and_regret():
    assert kendall_tau_b([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau_b([1, 2, 3], [30, 20, 10]) == -1.0
    assert np.isclose(selection_regret([0.9, 0.1], [0.2, 0.8]), 0.6)
    assert np.isclose(selection_regret([0.1, 0.9], [0.2, 0.8]), 0.0)


def test_proposition_2_mixture():
    # Invariance condition: when regime determines failure completely
    auc1 = auroc_mixture(0.05, 0.0, 1.0, 0.9)
    auc2 = auroc_mixture(0.60, 0.0, 1.0, 0.9)
    assert np.isclose(auc1, auc2)

    # Vanishing cliff share
    assert np.isclose(auroc_mixture(1e-9, 0.01, 0.97, 0.9), 0.5, atol=1e-5)


def test_auroc_mixture_edge_cases():
    """auroc_mixture must not raise ZeroDivisionError when denominators vanish."""
    assert math.isnan(auroc_mixture(0.0, 0.0, 0.8, 0.9))
    assert math.isnan(auroc_mixture(1.0, 0.1, 1.0, 0.9))
    assert math.isnan(auroc_mixture(0.5, 0.0, 0.0, 0.9))
    assert math.isnan(auroc_mixture(0.5, 1.0, 1.0, 0.9))


def test_fit_predict_guards():
    """fit_predict seed_out branch must safely handle < 2 valid cal rows or NaNs."""
    import pandas as pd
    from visionrl.config import make_cfg
    from visionrl.forecasting.predictor import SCORE_COLS, fit_predict

    cfg = make_cfg("smoke", "pursuit")
    cal_df = pd.DataFrame({
        "agent_id": ["a0", "a0"],
        **{c: [0.1, 0.2] for c in SCORE_COLS},
    })
    df = pd.DataFrame({
        "agent_id": ["a0", "a0", "a0"],
        "split": ["cal", "cal", "test"],
        "kind": ["noise", "noise", "noise"],
        "fail": [0, 1, 1],
        "z_knn_policy": [0.1, float("nan"), 0.5],
    })
    out = fit_predict(df, cfg, cal_df, cols=["knn_policy"])
    # Only 1 valid cal row -> seed_out should remain NaN without raising ValueError
    assert out.loc[out["split"] == "test", "phat_knn_policy_seed_out"].isna().all()


def test_conformal_and_signflip():
    vals = np.arange(1, 101)
    thr = conformal_threshold(vals, 0.05)
    assert thr == 96.0

    a = np.ones(20) * 2.0
    b = np.zeros(20)
    p_val = paired_signflip_p(a, b, n_perm=1000)
    assert p_val < 0.01
    assert paired_signflip_p(a, a, n_perm=1000) == 1.0
