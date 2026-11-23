"""
Diagnostic script for Test-Time Adaptation (TTA / TTSS).
Runs paired arms (none, ttt, ttss) on GPU and records exact step-level rejection causes:
- Did the moment-matching loss fail to decrease (after >= before)?
- Did the action deviation on clean references exceed tta_accept_delta?
- What are the empirical distributions of before, after, and dev?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from dataclasses import replace

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes.adaptation import AdaptedPolicy, clean_feature_stats, run_tta
from visionrl.probes.feature_bank import collect_clean
from visionrl.experiments.train import train_agent


def run_tta_diagnosis():
    device = get_device()
    print(f"[TTA Diagnosis] Running on device: {device}")

    # Use pilot/sandbox config with CUDA
    cfg = replace(
        make_cfg("sandbox"),
        img=48,
        ep_len=50,
        train_steps=10000,
        seed_steps=500,
        batch=64,
        feature_dim=50,
        hidden=256,
        tta_eps=10,
        tta_kinds=("noise", "background"),
        tta_severities=(0.6, 1.0),
        tta_lr=1e-3,
        tta_accept_delta=0.20,
    )

    print(f"Training an evaluation agent for {cfg.train_steps} steps on {device}...")
    agent, tr_df = train_agent(cfg, "drqv2", seed=0, verbose=False)
    print(f"Agent trained. Final eval: {tr_df.iloc[-1].to_dict() if len(tr_df) else 'N/A'}")

    env = make_env(cfg)
    ref_recs = collect_clean(agent, cfg, "ref", cfg.ref_eps, env)
    floor = 22.05

    print(f"\nRunning TTA evaluation across {cfg.tta_kinds} x {cfg.tta_severities} ({cfg.tta_eps} eps each)...")
    res_df = run_tta(agent, "drqv2_s0", cfg, ref_recs, floor)

    print("\n--- Summary of Arms ---")
    summary = res_df.groupby(["kind", "sev", "arm"])[["success", "mean_cle", "accepted", "rejected", "rejected_loss_inc", "rejected_dev_exceeded"]].mean()
    print(summary)

    # Detailed step diagnostic from a dedicated TTSS run
    mu, var, clean_stacks = clean_feature_stats(agent, ref_recs, cfg)
    for kind in cfg.tta_kinds:
        for sev in cfg.tta_severities:
            ttss_pol = AdaptedPolicy(agent, cfg, "ttss", (mu, var), clean_stacks)
            from visionrl.shifts.observation_shifts import Shift
            from visionrl.environments import rollout

            for ep_idx in range(3):
                s = seed_for("test", ep_idx)
                sh = Shift(kind, sev, s, cfg.img)
                ttss_pol.reset_episode()
                r = rollout(env, ttss_pol.act, s, cfg, shift=sh)

                diag = pd.DataFrame(ttss_pol.step_diagnostics)
                if len(diag) > 0:
                    n_steps = len(diag)
                    n_acc = diag["accepted"].sum()
                    n_loss_fail = (~diag["loss_ok"]).sum()
                    n_dev_fail = (~diag["dev_ok"]).sum()
                    mean_dev = diag["dev"].mean()
                    max_dev = diag["dev"].max()
                    mean_before = diag["before"].mean()
                    mean_after = diag["after"].mean()
                    print(f"\n[{kind} s={sev} ep={ep_idx}] Steps: {n_steps} | Accepted: {n_acc} ({100*n_acc/n_steps:.1f}%) | "
                          f"Loss Inc: {n_loss_fail} | Dev Exceeded: {n_dev_fail} | Mean Dev: {mean_dev:.3f} (max {max_dev:.3f}) | "
                          f"Loss before->after: {mean_before:.4f} -> {mean_after:.4f}")

    return res_df


if __name__ == "__main__":
    run_tta_diagnosis()
