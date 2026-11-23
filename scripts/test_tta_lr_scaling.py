"""
Test whether action deviation scales with tta_lr on diagnostic agents.
Tests tta_lr in [1e-4, 3e-4, 1e-3] without retraining.
Measures:
- action deviation distribution (mean, median, p90, max)
- moment matching loss reduction
- acceptance rates under different delta thresholds (0.20, 0.30, 0.40, 0.50)
Saves results to results/tta_lr_scaling.json and results/tta_lr_scaling.csv.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from dataclasses import replace

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env, rollout
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes.adaptation import AdaptedPolicy, clean_feature_stats
from visionrl.probes.feature_bank import collect_clean
from visionrl.shifts.observation_shifts import Shift


def test_lr_scaling():
    device = get_device()
    print(f"=== Testing TTA Action Deviation Scaling with tta_lr on {device} ===")

    cfg = replace(
        make_cfg("sandbox"),
        img=48,
        ep_len=50,
        batch=64,
        feature_dim=50,
        hidden=256,
        tta_buffer=8,
    )
    env = make_env(cfg)

    # We test on drqv2_s0 and svea_s0
    agents_to_test = [("drqv2", 0), ("svea", 0)]
    lrs = [1e-4, 3e-4, 1e-3]
    deltas = [0.20, 0.30, 0.40, 0.50]
    n_episodes = 5  # 5 episodes * 49 steps = 245 steps per cell

    all_step_rows = []
    summary_by_lr = {}

    for variant, seed in agents_to_test:
        aid = f"{variant}_s{seed}"
        ckpt_path = Path(f"checkpoints/{variant}_30k_s{seed}.pt")
        if not ckpt_path.exists():
            ckpt_path = Path(f"results/{variant}_30k_checkpoint_s{seed}.pt")
        if not ckpt_path.exists():
            print(f"Skipping {aid}: checkpoint not found")
            continue

        agent = DrQAgent(cfg, variant=variant, seed=seed)
        # Load full agent state (encoder, actor, critic)
        agent.load(torch.load(ckpt_path, map_location=device, weights_only=False))
        agent.eval()

        ref_recs = collect_clean(agent, cfg, "ref", 5, env)
        mu, var, clean_stacks = clean_feature_stats(agent, ref_recs, cfg)

        for lr in lrs:
            lr_cfg = replace(cfg, tta_lr=lr, tta_accept_delta=10.0)  # do not reject so we observe full deviation
            step_devs = []
            first_step_devs = []
            step_loss_drops = []

            for ep_idx in range(n_episodes):
                s = seed_for("test", ep_idx)
                sh = Shift("noise", 0.6, s, cfg.img)
                pol = AdaptedPolicy(agent, lr_cfg, "ttss", (mu, var), clean_stacks)
                rollout(env, pol.act, s, cfg, shift=sh)

                if pol.step_diagnostics:
                    first_step_devs.append(pol.step_diagnostics[0]["dev"])

                for step_i, d in enumerate(pol.step_diagnostics):
                    step_devs.append(d["dev"])
                    loss_drop = (d["before"] - d["after"]) / max(1e-6, d["before"])
                    step_loss_drops.append(loss_drop)
                    all_step_rows.append({
                        "agent": aid,
                        "lr": lr,
                        "ep_idx": ep_idx,
                        "step_in_ep": step_i,
                        "dev": d["dev"],
                        "loss_before": d["before"],
                        "loss_after": d["after"],
                        "loss_rel_drop": loss_drop,
                    })

            devs_arr = np.array(step_devs)
            first_devs_arr = np.array(first_step_devs)
            drops_arr = np.array(step_loss_drops)

            key = f"{aid}_lr_{lr}"
            summary_by_lr[key] = {
                "agent": aid,
                "tta_lr": lr,
                "n_steps": len(devs_arr),
                "first_step_median_dev": round(float(np.median(first_devs_arr)), 4) if len(first_devs_arr) else None,
                "first_step_mean_dev": round(float(np.mean(first_devs_arr)), 4) if len(first_devs_arr) else None,
                "mean_dev": round(float(np.mean(devs_arr)), 4),
                "median_dev": round(float(np.median(devs_arr)), 4),
                "p90_dev": round(float(np.percentile(devs_arr, 90)), 4),
                "max_dev": round(float(np.max(devs_arr)), 4),
                "mean_rel_loss_drop": round(float(np.mean(drops_arr)), 4),
            }
            # Acceptance rate under various deltas
            for d in deltas:
                acc_rate = float(np.mean(devs_arr <= d))
                summary_by_lr[key][f"acc_rate_delta_{d:.2f}"] = round(acc_rate, 4)

            print(f"[{aid} lr={lr:.0e}] Mean Dev: {summary_by_lr[key]['mean_dev']:.4f} | "
                  f"Median: {summary_by_lr[key]['median_dev']:.4f} | "
                  f"Acc@0.20: {summary_by_lr[key]['acc_rate_delta_0.20']:.1%} | "
                  f"Acc@0.40: {summary_by_lr[key]['acc_rate_delta_0.40']:.1%} | "
                  f"Loss Drop: {summary_by_lr[key]['mean_rel_loss_drop']:.1%}")

    df_steps = pd.DataFrame(all_step_rows)
    df_steps.to_csv("results/tta_lr_scaling.csv", index=False)

    with open("results/tta_lr_scaling.json", "w") as f:
        json.dump(summary_by_lr, f, indent=2)

    print("\nScaling test complete! Artifacts saved to results/tta_lr_scaling.json and .csv")


if __name__ == "__main__":
    test_lr_scaling()
