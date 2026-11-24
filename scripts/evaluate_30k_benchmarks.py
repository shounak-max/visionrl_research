"""
Evaluate failure forecasting AUROC and policy ranking Kendall tau on the 30k models:
- DrQ-v2 (30k, seed 0)
- SVEA (30k, seed 0)
- TACO (30k, seed 0)

Evaluates:
1. Ground truth closed-loop performance across 5 shift families x 11 severities (20 seeds per cell).
2. Rollout-free candidate scores (knn_policy, maha_policy, cent_policy, sem_knn_policy,
   knn_random, knn_pixel, act_disagree, q_drop, critic_gap, pac_drift).
3. Failure forecasting AUROC, AUPRC, TPR@5%FPR for each candidate score overall and per policy.
4. Policy ranking Kendall tau and selection regret across the 3 policies.
5. Per-policy Kendall tau and rank correlation of predicted boundary vs true boundary.

Outputs:
- results/predictor_table_30k.csv & .json
- results/ranking_table_30k.csv & .json
- results/per_agent_forecasting_30k.csv & .json
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataclasses import replace
import numpy as np
import pandas as pd
import torch

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env, is_collapse
from visionrl.experiments.study import study_one_agent
from visionrl.experiments.train import random_floor_cle
from visionrl.forecasting.metrics import auroc, auprc, tpr_at_fpr, metrics_bundle
from visionrl.forecasting.predictor import (
    SCORE_COLS,
    PROPOSED,
    add_flags_and_z,
    fit_predict,
    predictor_table,
    forecast_cliffs,
    pooled_rows,
)
from visionrl.forecasting.ranking import kendall_tau_b, selection_regret, ranking_table
from visionrl.models.drq_agent import DrQAgent


def main():
    cfg = replace(
        make_cfg("sandbox", "pursuit"),
        img=48,
        ep_len=50,
        train_steps=30000,
        seed_steps=1000,
        batch=128,
        feature_dim=50,
        hidden=256,
        lr=3e-4,
        taco_dim=50,
        eps_per_cell=20,
        ref_eps=10,
        cal_clean_eps=20,
        knn_k=5,
        shift_kinds=("background", "noise", "distractor", "viewpoint", "color_gain"),
        severities=tuple(np.round(np.linspace(0.0, 1.0, 11), 2)),
    )
    device = get_device()
    env = make_env(cfg)
    floor = random_floor_cle(cfg)

    print(f"=== Starting 30k Benchmark Study across DrQ-v2, SVEA, TACO on {device} ===")

    checkpoints = {
        "drqv2": Path("checkpoints/drqv2_30k_s0.pt"),
        "svea": Path("checkpoints/svea_30k_s0.pt"),
        "taco": Path("checkpoints/taco_30k_s0.pt"),
    }

    agents = {}
    for aid, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ag = DrQAgent(cfg, variant=aid, seed=0)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        ag.load(state)
        ag.eval()
        agents[aid] = ag
        print(f"Loaded {aid} from {ckpt_path}")

    # Collect study data for each agent
    raw_rows = []
    cal_rows = []
    cal_frames = {}

    for aid, ag in agents.items():
        print(f"\n--- Running Study Grid for {aid} (55 cells x 20 episodes) ---")
        t0 = time.time()
        res = study_one_agent(aid, ag, cfg, floor, env)
        agent_rows, frame_scores, closed, grass, cal_dict = res
        raw_rows.extend(agent_rows)
        cal_rows.extend([dict(agent_id=aid, **x) for x in cal_dict["rows"]])
        cal_frames[aid] = cal_dict["frames"]
        print(f"Completed {aid} in {time.time() - t0:.1f}s: {len(agent_rows)} episodes")

    raw_df = pd.DataFrame(raw_rows)
    cal_df = pd.DataFrame(cal_rows)
    print(f"\nTotal episodes gathered: {len(raw_df)} across {len(agents)} agents.")

    # Conformal calibration & z-scoring
    print("\n--- Performing Split-Conformal Calibration & Score Standardization ---")
    df_with_z = add_flags_and_z(raw_df, cfg, cal_df)
    episodes = fit_predict(df_with_z, cfg, cal_df)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(results_dir / "episodes_30k.csv", index=False)
    print(f"Saved full episodes table to {results_dir / 'episodes_30k.csv'}")

    # 1. Benchmark-wide Predictor Table (Overall AUROC & Ranking across 30k models)
    print("\n--- Computing Overall Candidate Score Predictor Table (30k Benchmark) ---")
    pred_tab = predictor_table(episodes, cfg, proto="strict", cols=SCORE_COLS, ci=False)

    # Add Kendall tau and regret across policies for each predictor
    ranking_summaries = {}
    for col in SCORE_COLS:
        rt = ranking_table(episodes, col, proto="strict", method="iso")
        if len(rt) > 0 and not rt["tau"].isna().all():
            mean_tau = float(rt["tau"].mean())
            mean_regret = float(rt["regret"].mean())
        else:
            mean_tau = float("nan")
            mean_regret = float("nan")
        ranking_summaries[col] = {"tau": round(mean_tau, 3), "regret": round(mean_regret, 3)}

    pred_tab["tau"] = [ranking_summaries[c]["tau"] for c in pred_tab.index]
    pred_tab["regret"] = [ranking_summaries[c]["regret"] for c in pred_tab.index]

    pred_tab.round(3).to_csv(results_dir / "predictor_table_30k.csv")
    pred_tab.round(3).reset_index().to_json(results_dir / "predictor_table_30k.json", orient="records", indent=2)
    print(f"Saved predictor table to {results_dir / 'predictor_table_30k.csv'} and .json")

    # 2. Ranking Table (Per Shift Kind Kendall tau across the 3 policies)
    print("\n--- Computing Multi-Policy Ranking Table by Shift Family ---")
    all_ranking_rows = []
    for col in SCORE_COLS:
        rt = ranking_table(episodes, col, proto="strict", method="iso")
        if len(rt) > 0:
            for _, r in rt.iterrows():
                all_ranking_rows.append({
                    "predictor": col,
                    "kind": r["kind"],
                    "n_policies": int(r["n_policies"]),
                    "tau": round(float(r["tau"]), 3) if not np.isnan(r["tau"]) else None,
                    "regret": round(float(r["regret"]), 3) if not np.isnan(r["regret"]) else None,
                    "true_spread": round(float(r["true_spread"]), 3),
                    "pred_spread": round(float(r["pred_spread"]), 3),
                })

    df_ranking = pd.DataFrame(all_ranking_rows)
    df_ranking.to_csv(results_dir / "ranking_table_30k.csv", index=False)
    with open(results_dir / "ranking_table_30k.json", "w", encoding="utf-8") as f:
        json.dump(all_ranking_rows, f, indent=2)
    print(f"Saved ranking table to {results_dir / 'ranking_table_30k.csv'} and .json")

    # 3. Per-Agent AUROC and Kendall tau Breakdown
    print("\n--- Computing Per-Agent AUROC and Failure Boundary Ranking Breakdown ---")
    per_agent_rows = []

    for aid in ["drqv2", "svea", "taco"]:
        agent_eps = episodes[episodes.agent_id == aid]
        agent_pooled = pooled_rows(agent_eps)
        fail_vec = agent_pooled.fail.values

        for col in SCORE_COLS:
            z_col = "z_" + col
            m = metrics_bundle(agent_pooled[z_col].values, fail_vec)

            # Measure rank correlation (Kendall tau-b) between proxy score and severity across episodes
            tau_sev = kendall_tau_b(agent_pooled[z_col].values, agent_pooled.sev.values)

            # Measure rank correlation between forecasted cliff and true critical severity across shift kinds
            fc_agent = forecast_cliffs(agent_eps, col, proto="strict")
            if len(fc_agent) > 0:
                fc_iso = fc_agent[fc_agent.method == "iso"]
                tau_cliff = kendall_tau_b(fc_iso.s_crit_pred.values, fc_iso.s_crit_true.values)
                cliff_mae = float(fc_iso.abs_err.mean()) if not fc_iso.abs_err.isna().all() else None
            else:
                tau_cliff = None
                cliff_mae = None

            per_agent_rows.append({
                "agent": aid.upper() if aid != "drqv2" else "DrQ-v2",
                "variant": aid,
                "predictor": col,
                "n_episodes": int(m["n"]),
                "prevalence": round(float(m["prevalence"]), 3),
                "auroc": round(float(m["auroc"]), 3) if not np.isnan(m["auroc"]) else None,
                "auprc": round(float(m["auprc"]), 3) if not np.isnan(m["auprc"]) else None,
                "tpr_at_5fpr": round(float(m["tpr_at_5fpr"]), 3) if not np.isnan(m["tpr_at_5fpr"]) else None,
                "tau_severity": round(float(tau_sev), 3) if not np.isnan(tau_sev) else None,
                "tau_cliff": round(float(tau_cliff), 3) if tau_cliff is not None and not np.isnan(tau_cliff) else None,
                "cliff_mae": round(cliff_mae, 3) if cliff_mae is not None else None,
            })

    df_per_agent = pd.DataFrame(per_agent_rows)
    df_per_agent.to_csv(results_dir / "per_agent_forecasting_30k.csv", index=False)
    with open(results_dir / "per_agent_forecasting_30k.json", "w", encoding="utf-8") as f:
        json.dump(per_agent_rows, f, indent=2)
    print(f"Saved per-agent forecasting to {results_dir / 'per_agent_forecasting_30k.csv'} and .json")

    # 4. Update Summary JSON files with AUROC and Kendall tau
    print("\n--- Updating Summary JSONs with Genuine AUROC and Kendall tau ---")

    # Update drqv2_30k_summary.json
    drq_metrics = df_per_agent[df_per_agent.variant == "drqv2"].set_index("predictor")
    drq_summary_path = results_dir / "drqv2_30k_summary.json"
    if drq_summary_path.exists():
        with open(drq_summary_path, "r", encoding="utf-8") as f:
            drq_sum = json.load(f)
        drq_sum["failure_forecasting_metrics"] = {
            "knn_policy": {
                "auroc": drq_metrics.loc["knn_policy", "auroc"],
                "auprc": drq_metrics.loc["knn_policy", "auprc"],
                "tau_severity": drq_metrics.loc["knn_policy", "tau_severity"],
                "tau_cliff": drq_metrics.loc["knn_policy", "tau_cliff"],
            },
            "act_disagree": {
                "auroc": drq_metrics.loc["act_disagree", "auroc"],
                "tau_severity": drq_metrics.loc["act_disagree", "tau_severity"],
            },
            "q_drop": {
                "auroc": drq_metrics.loc["q_drop", "auroc"],
                "tau_severity": drq_metrics.loc["q_drop", "tau_severity"],
            },
            "pac_drift": {
                "auroc": drq_metrics.loc["pac_drift", "auroc"],
                "tau_severity": drq_metrics.loc["pac_drift", "tau_severity"],
            },
        }
        with open(drq_summary_path, "w", encoding="utf-8") as f:
            json.dump(drq_sum, f, indent=2)
        print("Updated drqv2_30k_summary.json")

    # Update svea_30k_summary.json
    svea_metrics = df_per_agent[df_per_agent.variant == "svea"].set_index("predictor")
    svea_summary_path = results_dir / "svea_30k_summary.json"
    if svea_summary_path.exists():
        with open(svea_summary_path, "r", encoding="utf-8") as f:
            svea_sum = json.load(f)
        svea_sum["failure_forecasting_metrics"] = {
            "knn_policy": {
                "auroc": svea_metrics.loc["knn_policy", "auroc"],
                "auprc": svea_metrics.loc["knn_policy", "auprc"],
                "tau_severity": svea_metrics.loc["knn_policy", "tau_severity"],
                "tau_cliff": svea_metrics.loc["knn_policy", "tau_cliff"],
            },
            "act_disagree": {
                "auroc": svea_metrics.loc["act_disagree", "auroc"],
                "tau_severity": svea_metrics.loc["act_disagree", "tau_severity"],
            },
            "q_drop": {
                "auroc": svea_metrics.loc["q_drop", "auroc"],
                "tau_severity": svea_metrics.loc["q_drop", "tau_severity"],
            },
            "pac_drift": {
                "auroc": svea_metrics.loc["pac_drift", "auroc"],
                "tau_severity": svea_metrics.loc["pac_drift", "tau_severity"],
            },
        }
        with open(svea_summary_path, "w", encoding="utf-8") as f:
            json.dump(svea_sum, f, indent=2)
        print("Updated svea_30k_summary.json")

    # Update taco_summary.json
    taco_metrics = df_per_agent[df_per_agent.variant == "taco"].set_index("predictor")
    taco_summary_path = results_dir / "taco_summary.json"
    if taco_summary_path.exists():
        with open(taco_summary_path, "r", encoding="utf-8") as f:
            taco_sum = json.load(f)
        taco_sum["failure_forecasting_metrics"] = {
            "knn_policy": {
                "auroc": taco_metrics.loc["knn_policy", "auroc"],
                "auprc": taco_metrics.loc["knn_policy", "auprc"],
                "tau_severity": taco_metrics.loc["knn_policy", "tau_severity"],
                "tau_cliff": taco_metrics.loc["knn_policy", "tau_cliff"],
            },
            "act_disagree": {
                "auroc": taco_metrics.loc["act_disagree", "auroc"],
                "tau_severity": taco_metrics.loc["act_disagree", "tau_severity"],
            },
            "q_drop": {
                "auroc": taco_metrics.loc["q_drop", "auroc"],
                "tau_severity": taco_metrics.loc["q_drop", "tau_severity"],
            },
            "pac_drift": {
                "auroc": taco_metrics.loc["pac_drift", "auroc"],
                "tau_severity": taco_metrics.loc["pac_drift", "tau_severity"],
            },
        }
        with open(taco_summary_path, "w", encoding="utf-8") as f:
            json.dump(taco_sum, f, indent=2)
        print("Updated taco_summary.json")

    # Print summary tables
    print("\n" + "=" * 80)
    print("30,000-STEP BENCHMARK: OVERALL PREDICTOR AUROC & KENDALL TAU")
    print("=" * 80)
    print(pred_tab[["n", "prevalence", "auroc", "auprc", "tpr_at_5fpr", "tau", "regret"]].to_string())

    print("\n" + "=" * 80)
    print("PER-AGENT AUROC BREAKDOWN (DrQ-v2 vs SVEA vs TACO at 30k Steps)")
    print("=" * 80)
    key_predictors = ["knn_policy", "act_disagree", "q_drop", "pac_drift", "maha_policy", "knn_pixel"]
    sub_df = df_per_agent[df_per_agent.predictor.isin(key_predictors)]
    print(sub_df.pivot(index="predictor", columns="agent", values="auroc").to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
