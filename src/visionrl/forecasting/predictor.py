"""Failure forecasting, split-conformal calibration, cliff location, and alarm streaming."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from visionrl.config import Cfg
from visionrl.forecasting.metrics import (
    auroc,
    brier,
    cliffness,
    cluster_bootstrap,
    conformal_threshold,
    critical_severity,
    ema_smooth,
    metrics_bundle,
    pearson,
    spearman,
)

SCORE_COLS: List[str] = [
    "knn_policy",
    "pac_drift_l0",
    "pac_drift_l1",
    "pac_drift",
    "gcm_drift",
    "maha_policy",
    "cent_policy",
    "sem_knn_policy",
    "sem_knn_uniform",
    "knn_random",
    "knn_pixel",
    "act_disagree",
    "q_drop",
    "critic_gap",
]
PROPOSED: str = "knn_policy"


def cal_stats(cal: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per-agent mean and standard deviation of every score on dedicated CLEAN calibration episodes."""
    return {col: cal.groupby("agent_id")[col].agg(["mean", "std"]) for col in SCORE_COLS}


def cal_z(cal: pd.DataFrame) -> pd.DataFrame:
    st = cal_stats(cal)
    z = cal[["agent_id"]].copy()
    for col in SCORE_COLS:
        z["z_" + col] = (cal[col] - cal.agent_id.map(st[col]["mean"])) / (cal.agent_id.map(st[col]["std"]).fillna(0) + 1e-06)
    return z


def add_flags_and_z(df: pd.DataFrame, cfg: Cfg, cal: pd.DataFrame) -> pd.DataFrame:
    """Split into calibration seeds vs test seeds (disjoint episodes).
    z-scores use the dedicated clean calibration episodes.
    """
    d = df.copy()
    half = max(1, cfg.eps_per_cell // 2)
    d["split"] = np.where(d.seed_idx < half, "cal", "test")
    d["fail"] = (~d.success.astype(bool)).astype(int)
    st = cal_stats(cal)
    for col in SCORE_COLS:
        mu = d.agent_id.map(st[col]["mean"])
        sd = d.agent_id.map(st[col]["std"]).fillna(0) + 1e-06
        d["z_" + col] = (d[col] - mu) / sd
    return d


def fit_predict(df: pd.DataFrame, cfg: Cfg, cal: pd.DataFrame, cols: List[str] = SCORE_COLS) -> pd.DataFrame:
    """Two calibration protocols evaluated on held-out seeds:
       'seed_out' : fit on calibration seeds of all shift kinds, predict on test seeds.
       'strict'   : additionally hold out the whole shift KIND (leave-one-kind-out).
    Zero-shot alarm rule: split-conformal (1 - clean_fpr) threshold of clean calibration scores.
    """
    d = df.copy()
    kinds = sorted(d.kind.unique())
    cz = cal_z(cal)

    for col in cols:
        z = "z_" + col
        tau = {_aid: conformal_threshold(g[z].values, cfg.clean_fpr) for _aid, g in cz.groupby("agent_id")}
        d["tau_" + col] = d.agent_id.map(tau)
        d["alarm_" + col] = (d[z] > d["tau_" + col]).astype(float)

        for proto in ("seed_out", "strict"):
            d[f"phat_{col}_{proto}"] = np.nan

        cal_rows = d.split == "cal"
        test = d.split == "test"

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1).fit(
            d.loc[cal_rows, z], d.loc[cal_rows, "fail"]
        )
        d.loc[test, f"phat_{col}_seed_out"] = iso.predict(d.loc[test, z])

        for kind in kinds:
            tr = cal_rows & (d.kind != kind)
            te = test & (d.kind == kind)
            if tr.sum() < 2 or te.sum() == 0:
                continue
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1).fit(
                d.loc[tr, z], d.loc[tr, "fail"]
            )
            d.loc[te, f"phat_{col}_strict"] = iso.predict(d.loc[te, z])

    return d


def pooled_rows(d: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Keep one copy of clean (severity 0) episodes when pooling across shift kinds."""
    x = d[d.split == split]
    if len(x) == 0:
        return x
    k0 = sorted(d.kind.unique())[0]
    return x[(x.sev > 0) | (x.kind == k0)]


def predictor_table(d: pd.DataFrame, cfg: Cfg, proto: str = "strict", cols: List[str] = SCORE_COLS, ci: bool = True) -> pd.DataFrame:
    x = pooled_rows(d)
    rows = []
    for col in cols:
        z = "z_" + col
        m = metrics_bundle(x[z].values, x.fail.values)
        clean = x[x.sev == 0]
        bad = x[x.fail == 1]
        ph = x[f"phat_{col}_{proto}"]
        r = dict(
            predictor=col,
            **{k: m[k] for k in ["n", "prevalence", "auroc", "auprc", "auprc_random", "auprc_lift", "tpr_at_5fpr", "status"]},
            false_alarm_rate_clean=float(clean["alarm_" + col].mean()) if len(clean) else float("nan"),
            alarm_rate_on_failures=float(bad["alarm_" + col].mean()) if len(bad) else float("nan"),
            brier=brier(ph[ph.notna()], x.loc[ph.notna(), "fail"]) if ph.notna().any() else float("nan"),
        )
        if ci and m["status"] == "ok":
            lo, hi = cluster_bootstrap(lambda df_: auroc(df_[z].values, df_.fail.values), x, "seed_idx", cfg.boot_B)
            r["auroc_ci_lo"], r["auroc_ci_hi"] = (lo, hi)
        rows.append(r)
    return pd.DataFrame(rows).set_index("predictor")


def dose_response(d: pd.DataFrame, split: Optional[str] = None) -> pd.DataFrame:
    x = d if split is None else d[d.split == split]
    return x.groupby(["agent_id", "kind", "sev"]).agg(
        success_rate=("success", "mean"),
        mean_cle=("mean_cle", "mean"),
        collapse_rate=("collapse", "mean"),
        n=("success", "size"),
    ).reset_index()


def cliff_table(d: pd.DataFrame, split: Optional[str] = None) -> pd.DataFrame:
    dr = dose_response(d, split)
    rows = []
    for (a, k), g in dr.groupby(["agent_id", "kind"]):
        g = g.sort_values("sev")
        sc, cens = critical_severity(g.sev.values, g.success_rate.values)
        rows.append(dict(
            agent_id=a,
            kind=k,
            clean_success=g.success_rate.iloc[0],
            s_crit=sc,
            censored=cens,
            cliffness=cliffness(g.sev.values, g.success_rate.values),
            total_drop=g.success_rate.iloc[0] - g.success_rate.min(),
        ))
    return pd.DataFrame(rows)


def forecast_cliffs(d: pd.DataFrame, col: str = PROPOSED, proto: str = "strict") -> pd.DataFrame:
    """Forecast the success-rate curve from observations only, then locate the competence cliff on it."""
    t = d[d.split == "test"]
    rows = []
    for (a, k), g in t.groupby(["agent_id", "kind"]):
        agg = g.groupby("sev").agg(
            sr=("success", "mean"),
            phat=(f"phat_{col}_{proto}", "mean"),
            alarm=("alarm_" + col, "mean"),
        ).sort_index()

        if agg.phat.isna().all():
            continue

        sev = agg.index.values
        sr_iso = 1.0 - agg.phat.values
        sr_rule = agg.sr.values[0] * (1.0 - agg.alarm.values)
        true_c, true_cens = critical_severity(sev, agg.sr.values)

        for name, pred in [("iso", sr_iso), ("rule", sr_rule)]:
            pc, pcens = critical_severity(sev, pred)
            rows.append(dict(
                agent_id=a,
                kind=k,
                method=name,
                s_crit_true=true_c,
                s_crit_pred=pc,
                censored_true=true_cens,
                censored_pred=pcens,
                abs_err=abs(pc - true_c) if pc == pc and true_c == true_c else float("nan"),
                curve_mae=float(np.mean(np.abs(pred - agg.sr.values))),
                rank_corr=spearman(pred, agg.sr.values),
            ))
    return pd.DataFrame(rows)


def correlation_vs_threshold(d: pd.DataFrame, cols: Tuple[str, ...] = ("knn_policy", "maha_policy", "knn_pixel")) -> pd.DataFrame:
    """Correlation of proxy with success next to held-out failure prediction metrics."""
    rows = []
    for kind, g in d[d.sev > 0].groupby("kind"):
        gg = pd.concat([g, d[(d.sev == 0) & (d.kind == kind)]])
        for col in cols:
            z = "z_" + col
            m = metrics_bundle(gg[z].values, gg.fail.values)
            rows.append(dict(
                kind=kind,
                predictor=col,
                pearson=pearson(gg[z], gg.success.astype(float)),
                spearman=spearman(gg[z], gg.success.astype(float)),
                auroc=m["auroc"],
                auprc=m["auprc"],
                prevalence=m["prevalence"],
                auprc_lift=m["auprc_lift"],
                status=m["status"],
            ))
    return pd.DataFrame(rows)


def alarm_timing(
    frame_scores: Dict[Tuple[str, str, float, int], np.ndarray],
    closed: Dict[Tuple[str, str, float, int], int],
    cfg: Cfg,
    clean_tau: Dict[str, float],
    ema: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Streaming view of alarms based on EMA-smoothed scores and split-conformal per-episode maximum thresholds."""
    ema = cfg.ema if ema is None else ema
    rows = []
    for (a, k, s, seed), x in frame_scores.items():
        y = ema_smooth(x, ema)
        above = y > clean_tau[a]
        onsets = int(above[0]) + int(np.sum(above[1:] & ~above[:-1]))
        first = int(np.argmax(above)) if above.any() else len(x)
        fail = closed.get((a, k, s, seed))
        rows.append(dict(
            agent_id=a,
            kind=k,
            sev=s,
            seed=seed,
            onsets=onsets,
            ever=bool(above.any()),
            first_alarm=first,
            quarter=bool(above[: max(1, len(x) // 4)].any()),
            fail=fail,
        ))
    t = pd.DataFrame(rows)
    summary = t.groupby(["kind", "sev"]).agg(
        onsets_per_ep=("onsets", "mean"),
        ever_alarmed=("ever", "mean"),
        alarmed_first_quarter=("quarter", "mean"),
        n=("ever", "size"),
    ).reset_index()
    return summary, t
