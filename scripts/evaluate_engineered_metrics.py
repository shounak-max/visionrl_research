"""
Evaluate all engineered representation, drift, and forecasting metrics on the 30k trained agents:
- DrQ-v2 (30k, seed 0)
- SVEA (30k, seed 0)
- TACO (30k, seed 0)

Engineered metrics evaluated:
1. Clean Performance: Mean CLE (px), Clean Success Rate (<5px), Mean Return
2. Manifold & Geodesic Representation Drift:
   - knn_policy (Latent representation kNN distance, Eq. 3)
   - pac_drift_l0 (PAC-Drift First-Order Jacobian Pullback, Score 10)
   - pac_drift_l1 (PAC-Drift First + Second-Order Hessian Curvature, Score 10)
   - gcm_drift (Geodesic Covariance Manifold AIRM distance, Score 11)
   - grassmann_distance (Subspace principal angle geodesic distance)
   - maha_policy (Mahalanobis distance to clean feature distribution)
   - cent_policy (Centroid distance, Eq. 1)
   - sem_knn_policy (Semantic-pooled entity-focused kNN distance)
3. Action & Value Sensitivity:
   - act_disagree (Action deviation under shift ||mu_s - mu_c||_2)
   - q_drop (Critic value drop under shift)
   - critic_gap (Twin-critic disagreement |Q1 - Q2|)
4. Baseline Controls:
   - knn_random (Untrained random CNN feature drift)
   - knn_pixel (Downsampled pixel drift)

Saves:
- results/engineered_metrics_comparison_30k.json
- results/engineered_metrics_comparison_30k.csv
"""

import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes.feature_bank import FeatureBank, RefBank, collect_clean, score_episode
from visionrl.probes.metrics import grassmann_distance
from visionrl.experiments.train import eval_cell

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_metrics():
    device = get_device()
    print(f"=== Evaluating Engineered Metrics on 30k Baselines on {device} ===")

    cfg = dataclasses.replace(
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
        ref_eps=10,
        knn_k=5,
    )

    env = make_env(cfg)

    agents_info = [
        ("DrQ-v2", "drqv2", 0, "results/drqv2_30k_checkpoint_s0.pt"),
        ("SVEA", "svea", 0, "results/svea_30k_checkpoint_s0.pt"),
        ("TACO", "taco", 0, "results/taco_checkpoint_s0.pt"),
    ]

    shifts = [
        ("clean", 0.0),
        ("noise", 0.5),
        ("noise", 1.0),
        ("background", 0.5),
        ("background", 1.0),
        ("distractor", 0.5),
        ("distractor", 1.0),
        ("viewpoint", 0.5),
        ("color_gain", 0.5),
    ]

    eval_seeds = [seed_for("eval_curve", i) for i in range(20)]
    all_agent_metrics = {}
    table_rows = []

    for name, variant, seed, ckpt_path in agents_info:
        print(f"\n--- Loading and Evaluating {name} ({ckpt_path}) ---")
        agent = DrQAgent(cfg, variant=variant, seed=seed)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load(state)
        agent.eval()

        # 1. Measure Clean Performance
        print("  Evaluating clean performance across 20 held-out seeds...")
        clean_recs = eval_cell(agent.act, cfg, None, 0.0, eval_seeds, env=env)
        mean_cle = float(np.mean([r["mean_cle"] for r in clean_recs]))
        cle_std = float(np.std([r["mean_cle"] for r in clean_recs]))
        succ_rate = float(np.mean([float(r["success"]) for r in clean_recs]))
        mean_ret = float(np.mean([r["ret"] for r in clean_recs]))

        # 2. Collect Reference clean episodes for feature bank
        print("  Building feature bank & reference bank...")
        ref_recs = collect_clean(agent, cfg, "ref", cfg.ref_eps, env=env)
        bank = FeatureBank(agent, cfg)
        ref = RefBank(bank, ref_recs, cfg)

        # 3. Test episodes for shift evaluation
        test_clean_recs = collect_clean(agent, cfg, "test", 5, env=env)

        shift_scores = {}
        for kind, sev in shifts:
            shift_tag = f"{kind}_{sev}" if sev > 0 else "clean"
            ep_rows = []
            z_clean_list = []
            z_shift_list = []

            for r in test_clean_recs:
                scores, _, z_s = score_episode(bank, ref, r, kind, sev, cfg)
                ep_rows.append(scores)
                z_clean_list.append(r["_cf"]["policy"])
                z_shift_list.append(z_s)

            # Average scores over test episodes
            mean_scores = {k: float(np.mean([row[k] for row in ep_rows])) for k in ep_rows[0]}

            # Compute Grassmannian geodesic distance
            all_zc = np.concatenate(z_clean_list, axis=0)
            all_zs = np.concatenate(z_shift_list, axis=0)
            mean_scores["grassmann_dist"] = float(grassmann_distance(all_zc, all_zs, k=10))

            shift_scores[shift_tag] = mean_scores

            # Append to table rows
            row_entry = {
                "agent": name,
                "variant": variant,
                "shift": kind,
                "severity": sev,
                **mean_scores,
            }
            table_rows.append(row_entry)

        # Compute aggregate mean across all shifted conditions (excluding clean)
        shifted_keys = [k for k in shift_scores.keys() if k != "clean"]
        overall_shift_avg = {}
        for score_key in shift_scores["clean"].keys():
            overall_shift_avg[score_key] = float(np.mean([shift_scores[sk][score_key] for sk in shifted_keys]))

        all_agent_metrics[name] = {
            "variant": variant,
            "training_steps": 30000,
            "clean_performance": {
                "mean_cle_px": round(mean_cle, 2),
                "cle_std_px": round(cle_std, 2),
                "success_rate": round(succ_rate, 3),
                "mean_return": round(mean_ret, 2),
                "competence_status": "Passes competence gate (<5.0 px)" if mean_cle < 5.0 else "Sub-threshold / Developing",
            },
            "overall_ood_drift_means": {k: round(v, 4) for k, v in overall_shift_avg.items()},
            "shift_breakdown": shift_scores,
        }

        print(f"  [{name} Summary]")
        print(f"    Clean CLE: {mean_cle:.2f} px | Clean Success: {succ_rate:.1%} | Return: {mean_ret:.2f}")
        print(f"    OOD kNN Policy Drift: {overall_shift_avg['knn_policy']:.3f}")
        print(f"    OOD PAC-Drift (L0): {overall_shift_avg['pac_drift_l0']:.4f} | PAC-Drift (L1): {overall_shift_avg['pac_drift_l1']:.4f}")
        print(f"    OOD GCM-Drift (AIRM): {overall_shift_avg['gcm_drift']:.3f} | Grassmann Dist: {overall_shift_avg['grassmann_dist']:.3f}")
        print(f"    OOD Action Disagreement: {overall_shift_avg['act_disagree']:.3f} | Q-Drop: {overall_shift_avg['q_drop']:.3f}")

    # Save detailed JSON and CSV
    json_path = RESULTS_DIR / "engineered_metrics_comparison_30k.json"
    with open(json_path, "w") as f:
        json.dump(all_agent_metrics, f, indent=2)

    df_table = pd.DataFrame(table_rows)
    csv_path = RESULTS_DIR / "engineered_metrics_comparison_30k.csv"
    df_table.to_csv(csv_path, index=False)

    print(f"\nSaved metrics comparison to {json_path} and {csv_path}")


if __name__ == "__main__":
    evaluate_metrics()
