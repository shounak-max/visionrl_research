"""Step F: Pre-registered hypothesis evaluation and cross-environment replication."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from visionrl.config import Cfg, PREREG
from visionrl.environments import ENV_REGISTRY
from visionrl.forecasting.metrics import cluster_bootstrap


def paired_error_gain(fc: pd.DataFrame, base: str, proposed: str, B: int, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    """Per (policy, shift kind): |error of base| minus |error of proposed|.
    Positive favours the proposed predictor. Confidence interval resamples whole policies.
    """
    nan_res = (float("nan"), (float("nan"), float("nan")))
    if len(fc) == 0:
        return nan_res
    piv = fc[fc.method == "iso"].pivot_table(index=["agent_id", "kind"], columns="predictor", values="abs_err")
    if proposed not in piv.columns or base not in piv.columns:
        return nan_res
    d = (piv[base] - piv[proposed]).dropna().reset_index(name="gain")
    if len(d) == 0:
        return nan_res
    mean_gain = float(d.gain.mean())
    if len(d) < 5:
        # Graceful small-sample handling (Bug 3 fix)
        ci = (float(d.gain.min()), float(d.gain.max()))
        return mean_gain, ci
    return mean_gain, cluster_bootstrap(lambda x: float(x.gain.mean()), d, "agent_id", B, seed)


def evaluate_hypotheses(
    cfg: Cfg,
    gate_df: pd.DataFrame,
    cliff_df: pd.DataFrame,
    fc_base: pd.DataFrame,
    rank_df: pd.DataFrame,
    portable_df: pd.DataFrame,
    tta_cmp: pd.DataFrame,
    tta_df: pd.DataFrame,
    tab_strict: pd.DataFrame,
    alarm_df: pd.DataFrame,
    agents_count: int,
    prereg: Dict[str, Any] = PREREG,
) -> pd.DataFrame:
    """Pre-registered decision rules decided mechanically."""
    P = prereg
    rows = []
    smoke = cfg.mode == "smoke"
    gate_known = cfg.train_gate_success > 0
    pre_ok = bool(gate_df.gate_ok.all()) if (gate_known and len(gate_df)) else False

    def add(h: str, statement: str, ok: Optional[bool], evidence: str) -> None:
        if h != "H0" and (smoke or not pre_ok):
            ok = None
            evidence = (
                "smoke mode: agents are untrained plumbing check"
                if smoke
                else "precondition H0 not met: agents do not solve the clean task"
            )
        rows.append(dict(
            hypothesis=h,
            verdict="UNDEFINED" if ok is None else ("SUPPORTED" if ok else "NOT SUPPORTED"),
            statement=statement,
            evidence=evidence,
        ))

    add("H0", "Agents solve the clean task (training gate)", pre_ok if gate_known else None,
        f"min clean success {gate_df.clean_success.min():.2f}, gate {cfg.train_gate_success}" if len(gate_df) else "no agents")

    real = cliff_df[cliff_df.total_drop >= P["h1_min_total_drop"]] if len(cliff_df) else pd.DataFrame()
    if len(real) == 0:
        add("H1", "Failure is abrupt in a majority of curves that show a real drop", None, "no curve with a real drop")
    else:
        share = float((real.cliffness >= P["h1_min_cliffness"]).mean())
        add("H1", f"Failure is abrupt (cliffness >= {P['h1_min_cliffness']}) in at least {P['h1_min_share']:.0%} of curves with a real drop",
            share >= P["h1_min_share"], f"{share:.2f} of {len(real)} curves")

    pm = P["proposed"]
    iso = fc_base[fc_base.method == "iso"] if len(fc_base) else fc_base
    med = iso.groupby("predictor").abs_err.median() if len(iso) else pd.Series(dtype=float)
    gains = {b: paired_error_gain(fc_base, b, pm, cfg.boot_B) for b in P["h2_require_beats"]}
    beats = [None if np.isnan(g[1][0]) else bool(g[1][0] > 0) for g in gains.values()]
    ok_h2 = (
        None
        if (np.isnan(med.get(pm, np.nan)) or any(x is None for x in beats))
        else bool(med[pm] <= P["h2_max_median_abs_err"] and all(beats))
    )
    add("H2", f"{pm} forecasts critical severity with median error <= {P['h2_max_median_abs_err']} and beats baselines",
        ok_h2, f"median error {med.get(pm, np.nan):.3f}; " + "; ".join(f"gain vs {b}: {g[0]:.3f} [{g[1][0]:.3f}, {g[1][1]:.3f}]" for b, g in gains.items()))

    rs = rank_df.groupby("predictor").agg(tau=("tau", "mean"), regret=("regret", "mean")) if len(rank_df) else pd.DataFrame(columns=["tau", "regret"])
    tp, rp = rs.tau.get(pm, np.nan), rs.regret.get(pm, np.nan)
    add("H3", f"{pm} ranks policies by true critical severity (tau >= {P['h3_min_tau']}) with selection regret <= {P['h3_max_regret']}",
        None if (np.isnan(tp) or np.isnan(rp)) else bool(tp >= P["h3_min_tau"] and rp <= P["h3_max_regret"]),
        f"tau {tp:.3f}, regret {rp:.3f}, {agents_count} policies")

    ta = float(pd.Series([rs.tau.get(b, np.nan) for b in P["agnostic"]]).mean()) if len(rs) else np.nan
    tc = float(pd.Series([rs.tau.get(b, np.nan) for b in P["policy_conditional"]]).max()) if len(rs) else np.nan
    add("H4", f"Policy-conditional signals rank policies better than policy-agnostic ones (tau gap >= {P['h4_min_tau_gap']})",
        None if (np.isnan(ta) or np.isnan(tc)) else bool(tc - ta >= P["h4_min_tau_gap"]),
        f"best conditional tau {tc:.3f} vs mean agnostic tau {ta:.3f}")

    err = float(portable_df.abs_err.mean()) if len(portable_df) else np.nan
    add("H5", f"The (a, fp, fc) summary predicts AUROC at another regime mix within {P['h5_max_auroc_error']}",
        None if np.isnan(err) else bool(err <= P["h5_max_auroc_error"]), f"mean absolute error {err:.3f} over {len(portable_df)} cases")

    sub = tta_cmp[tta_cmp.contrast == "ttss vs ttt"] if len(tta_cmp) else tta_cmp
    ttss = tta_df[tta_df.arm == "ttss"] if len(tta_df) else pd.DataFrame()
    engaged = float((ttss.rejected > 0).mean()) if len(ttss) else 0.0
    accepted = float((ttss.accepted > 0).mean()) if len(ttss) else 0.0
    if len(sub) == 0 or sub[["p_success_holm", "p_collapse_holm"]].isna().all().all():
        add("H6", "Safeguarded TTA improves on naive TTA (secondary)", None, "no comparison available")
    elif engaged == 0:
        add("H6", "Safeguarded TTA improves on naive TTA (secondary)", None,
            "safeguard never rejected an update (raise adaptation step or lower accept_delta)")
    else:
        sig = bool(((sub.p_success_holm < P["h6_alpha"]) | (sub.p_collapse_holm < P["h6_alpha"])).any())
        add("H6", "Safeguarded TTA improves on naive TTA with accepted and rejected updates (secondary)", bool(sig and accepted > 0.5),
            f"min Holm p {np.nanmin(np.r_[sub.p_success_holm, sub.p_collapse_holm]):.3f}; episodes with accepted updates {accepted:.2f}, with rejected updates {engaged:.2f}")

    far_ep = float(tab_strict.loc[pm, "false_alarm_rate_clean"]) if (len(tab_strict) and pm in tab_strict.index) else float("nan")
    far_st = float(alarm_df[alarm_df.sev == 0].ever_alarmed.mean()) if len(alarm_df) else float("nan")
    worst = np.nanmax([far_ep, far_st]) if not (np.isnan(far_ep) and np.isnan(far_st)) else float("nan")
    add("H7", f"Clean false-alarm rate within {P['h7_max_false_alarm_factor']}x nominal {cfg.clean_fpr}",
        None if np.isnan(worst) else bool(worst <= P["h7_max_false_alarm_factor"] * cfg.clean_fpr),
        f"episode-score rate {far_ep:.3f}, streaming ever-alarmed rate {far_st:.3f}")

    # Novel Extension: Policy-Action Curvature Drift (PAC-Drift)
    p_novel = P.get("novel", "pac_drift")
    med_novel = iso.groupby("predictor").abs_err.median() if len(iso) else pd.Series(dtype=float)
    gains_novel = paired_error_gain(fc_base, pm, p_novel, cfg.boot_B)
    ok_h_novel1 = (
        None
        if (np.isnan(med_novel.get(p_novel, np.nan)) or np.isnan(gains_novel[1][0]))
        else bool(med_novel[p_novel] <= P.get("h_novel_max_err", 0.10) and gains_novel[1][0] > 0)
    )
    add(
        "H_novel.1",
        f"PAC-Drift ({p_novel}) forecasts critical severity with error <= {P.get('h_novel_max_err', 0.10)} and beats {pm}",
        ok_h_novel1,
        f"median error {med_novel.get(p_novel, np.nan):.3f}; gain vs {pm}: {gains_novel[0]:.3f} [{gains_novel[1][0]:.3f}, {gains_novel[1][1]:.3f}]",
    )

    tp_n, rp_n = rs.tau.get(p_novel, np.nan), rs.regret.get(p_novel, np.nan)
    ok_h_novel2 = (
        None
        if (np.isnan(tp_n) or np.isnan(rp_n))
        else bool(tp_n >= P.get("h_novel_min_tau", 0.65) and rp_n <= P.get("h_novel_max_regret", 0.05))
    )
    add(
        "H_novel.2",
        f"PAC-Drift ranks policies with tau >= {P.get('h_novel_min_tau', 0.65)} and regret <= {P.get('h_novel_max_regret', 0.05)}",
        ok_h_novel2,
        f"tau {tp_n:.3f}, regret {rp_n:.3f}, {agents_count} policies",
    )

    return pd.DataFrame(rows)


def cross_env_summary(current: Optional[Tuple[str, pd.DataFrame]] = None, mode: str = "smoke", root: str = ".") -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Evaluates the pre-registered replication rule across environments."""
    found = {}
    for env in ENV_REGISTRY:
        path = os.path.join(root, f"results_{mode}_{env}", "hypotheses.csv")
        if os.path.exists(path):
            found[env] = pd.read_csv(path).set_index("hypothesis")
    if current is not None:
        found[current[0]] = current[1].set_index("hypothesis")
    if not found:
        return None, []
    piv = pd.DataFrame({env: h.verdict for env, h in found.items()})
    need = PREREG["min_envs_supported"]
    rows = []
    for h in ["H1", "H2", "H3", "H4", "H5", "H_novel.1", "H_novel.2"]:
        if h not in piv.index:
            continue
        n_ok = int((piv.loc[h] == "SUPPORTED").sum())
        n_bad = int((piv.loc[h] == "NOT SUPPORTED").sum())
        if len(found) < need:
            verdict = f"INSUFFICIENT: only {len(found)} environment(s) run"
        elif n_ok >= need:
            verdict = "REPLICATED"
        elif n_bad > 0:
            verdict = "NOT REPLICATED"
        else:
            verdict = "UNDEFINED"
        rows.append(dict(hypothesis=h, **piv.loc[h].to_dict(), supported_in=f"{n_ok}/{len(found)}", replication=verdict))
    return pd.DataFrame(rows), sorted(found)
