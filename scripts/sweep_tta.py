"""
Sweep script for TTA / TTSS on CUDA.
Sweeps:
- tta_lr in [1e-4, 3e-4, 1e-3]
- tta_accept_delta in [0.20, 0.30, 0.40, 0.50]
Measures:
- accepted rate (%)
- mean_cle of none, ttt, and ttss
- whether ttss outperforms baseline when accept_delta is relaxed or lr is lowered.
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
from visionrl.probes.adaptation import clean_feature_stats, run_tta, compare_arms
from visionrl.probes.feature_bank import collect_clean
from visionrl.experiments.train import train_agent


def run_sweep():
    device = get_device()
    print(f"[TTA Sweep] Running on {device}")

    cfg_base = replace(
        make_cfg("sandbox"),
        img=48,
        ep_len=50,
        train_steps=20000,
        seed_steps=1000,
        batch=128,
        feature_dim=50,
        hidden=256,
        tta_eps=15,
        tta_kinds=("noise", "background"),
        tta_severities=(0.6, 1.0),
    )

    print(f"Training agent for {cfg_base.train_steps} steps on {device}...")
    agent, tr_df = train_agent(cfg_base, "drqv2", seed=0, verbose=False)
    print(f"Agent trained. Final: {tr_df.iloc[-1].to_dict() if len(tr_df) else 'N/A'}")

    env = make_env(cfg_base)
    ref_recs = collect_clean(agent, cfg_base, "ref", cfg_base.ref_eps, env)
    floor = 22.05

    lrs = [1e-4, 3e-4, 1e-3]
    deltas = [0.20, 0.30, 0.40]

    all_results = []
    all_comparisons = []

    for lr in lrs:
        for delta in deltas:
            print(f"\n--- Testing lr={lr}, delta={delta} ---")
            cfg = replace(cfg_base, tta_lr=lr, tta_accept_delta=delta)
            res_df = run_tta(agent, "drqv2_s0", cfg, ref_recs, floor)
            res_df["tta_lr"] = lr
            res_df["tta_accept_delta"] = delta
            all_results.append(res_df)

            cmp = compare_arms(res_df)
            cmp["tta_lr"] = lr
            cmp["tta_accept_delta"] = delta
            all_comparisons.append(cmp)

            summary = res_df.groupby(["kind", "sev", "arm"])[["mean_cle", "success", "accepted", "rejected"]].mean()
            print(summary)

    full_res = pd.concat(all_results, ignore_index=True)
    full_res.to_csv("results/tta_sweep_results.csv", index=False)

    full_cmp = pd.concat(all_comparisons, ignore_index=True)
    full_cmp.to_csv("results/tta_sweep_comparisons.csv", index=False)
    print("\nSweep completed! Saved results to results/tta_sweep_results.csv and results/tta_sweep_comparisons.csv")


if __name__ == "__main__":
    run_sweep()
