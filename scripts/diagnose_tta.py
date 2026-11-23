"""
Diagnostic script for Test-Time Adaptation (TTA / TTSS).
Runs paired arms (none, ttt, ttss) on GPU across DrQ-v2 and SVEA,
records exact step-level rejection causes, and saves persistent artifacts:
- results/tta_diagnosis_summary.json
- results/tta_diagnosis_steps.csv
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from dataclasses import replace

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env, rollout
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes.adaptation import AdaptedPolicy, clean_feature_stats, run_tta
from visionrl.probes.feature_bank import collect_clean
from visionrl.shifts.observation_shifts import Shift
from visionrl.experiments.train import train_agent


def run_tta_diagnosis():
    device = get_device()
    print(f"[TTA Diagnosis] Running on device: {device}")
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cfg = replace(
        make_cfg("sandbox"),
        img=48,
        ep_len=50,
        train_steps=12000,
        seed_steps=600,
        batch=64,
        feature_dim=50,
        hidden=256,
        tta_eps=15,
        tta_kinds=("noise", "background"),
        tta_severities=(0.6, 1.0),
        tta_lr=1e-3,
        tta_accept_delta=0.20,
    )

    env = make_env(cfg)
    floor = 22.05

    agents_to_eval = [("drqv2", 0), ("svea", 0)]
    all_res_rows = []
    all_step_diagnostics = []

    summary_stats = {
        "config": {
            "img": cfg.img,
            "ep_len": cfg.ep_len,
            "tta_lr": cfg.tta_lr,
            "tta_accept_delta": cfg.tta_accept_delta,
            "tta_kinds": list(cfg.tta_kinds),
            "tta_severities": list(cfg.tta_severities),
            "tta_eps_per_cell": cfg.tta_eps,
        },
        "agents": {}
    }

    for variant, seed in agents_to_eval:
        aid = f"{variant}_s{seed}"
        ckpt_path = f"checkpoints/{variant}_30k_s{seed}.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"results/{variant}_30k_checkpoint_s{seed}.pt"

        if os.path.exists(ckpt_path):
            print(f"\nLoading existing agent {aid} from {ckpt_path}...")
            agent = DrQAgent(cfg, variant=variant, seed=seed)
            agent.load(torch.load(ckpt_path, map_location=device, weights_only=False))
            agent.eval()
        else:
            print(f"\nTraining agent {aid} for {cfg.train_steps} steps on {device}...")
            agent, tr_df = train_agent(cfg, variant=variant, seed=seed, verbose=False)
            agent.eval()
            torch.save(agent.state(), ckpt_path)
            print(f"Saved full checkpoint to {ckpt_path}. Final eval: {tr_df.iloc[-1].to_dict() if len(tr_df) else 'N/A'}")

        ref_recs = collect_clean(agent, cfg, "ref", cfg.ref_eps, env)
        mu, var, clean_stacks = clean_feature_stats(agent, ref_recs, cfg)

        # 1. Run standard paired evaluation across arms
        print(f"Running paired arms for {aid}...")
        res_df = run_tta(agent, aid, cfg, ref_recs, floor)
        all_res_rows.append(res_df)

        # 2. Detailed step diagnostic per episode
        agent_step_logs = []
        for kind in cfg.tta_kinds:
            for sev in cfg.tta_severities:
                ttss_pol = AdaptedPolicy(agent, cfg, "ttss", (mu, var), clean_stacks)
                for ep_idx in range(cfg.tta_eps):
                    s = seed_for("test", ep_idx)
                    sh = Shift(kind, sev, s, cfg.img)
                    ttss_pol.reset_episode()
                    r = rollout(env, ttss_pol.act, s, cfg, shift=sh)

                    diag = pd.DataFrame(ttss_pol.step_diagnostics)
                    if len(diag) > 0:
                        diag["agent_id"] = aid
                        diag["variant"] = variant
                        diag["kind"] = kind
                        diag["sev"] = sev
                        diag["ep_seed"] = s
                        diag["ep_idx"] = ep_idx
                        agent_step_logs.append(diag)

        if agent_step_logs:
            agent_step_df = pd.concat(agent_step_logs, ignore_index=True)
            all_step_diagnostics.append(agent_step_df)

            # Compute summary distributions
            total_steps = len(agent_step_df)
            accepted_steps = int(agent_step_df["accepted"].sum())
            loss_failed = int((~agent_step_df["loss_ok"]).sum())
            dev_failed = int((~agent_step_df["dev_ok"]).sum())
            both_failed = int(((~agent_step_df["loss_ok"]) & (~agent_step_df["dev_ok"])).sum())

            summary_stats["agents"][aid] = {
                "total_steps_evaluated": total_steps,
                "accepted_steps": accepted_steps,
                "acceptance_rate_pct": float(np.round(100.0 * accepted_steps / max(1, total_steps), 2)),
                "rejection_breakdown": {
                    "dev_exceeded_count": dev_failed,
                    "dev_exceeded_pct": float(np.round(100.0 * dev_failed / max(1, total_steps), 2)),
                    "loss_increased_count": loss_failed,
                    "loss_increased_pct": float(np.round(100.0 * loss_failed / max(1, total_steps), 2)),
                    "both_failed_count": both_failed,
                },
                "dev_distribution": {
                    "mean": float(np.round(agent_step_df["dev"].mean(), 4)),
                    "median": float(np.round(agent_step_df["dev"].median(), 4)),
                    "p75": float(np.round(agent_step_df["dev"].quantile(0.75), 4)),
                    "p90": float(np.round(agent_step_df["dev"].quantile(0.90), 4)),
                    "max": float(np.round(agent_step_df["dev"].max(), 4)),
                },
                "loss_reduction_pct_mean": float(np.round(100.0 * ((agent_step_df["before"] - agent_step_df["after"]) / np.maximum(agent_step_df["before"], 1e-6)).mean(), 2))
            }

    # Save combined results
    combined_res = pd.concat(all_res_rows, ignore_index=True)
    combined_res.to_csv("results/tta_diagnosis_arms.csv", index=False)

    if all_step_diagnostics:
        combined_steps = pd.concat(all_step_diagnostics, ignore_index=True)
        combined_steps.to_csv("results/tta_diagnosis_steps.csv", index=False)

    with open("results/tta_diagnosis_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 70)
    print("TTA DIAGNOSIS SUMMARY REPORT")
    print("=" * 70)
    print(json.dumps(summary_stats, indent=2))
    print("\nSaved artifacts:")
    print(" - results/tta_diagnosis_summary.json")
    print(" - results/tta_diagnosis_steps.csv")
    print(" - results/tta_diagnosis_arms.csv")


if __name__ == "__main__":
    run_tta_diagnosis()
