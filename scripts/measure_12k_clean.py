"""
Evaluate clean tracking performance of the 12k-step diagnostic agents.
Measures clean success rate (< 5.0 px CLE), mean CLE, and return across 20 held-out seeds.
Saves results to results/clean_performance_12k.json.
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
from visionrl.environments import make_env
from visionrl.experiments.train import eval_cell
from visionrl.models.drq_agent import DrQAgent


def main():
    cfg = replace(
        make_cfg("sandbox"),
        img=48,
        ep_len=50,
        train_steps=12000,
        seed_steps=600,
        batch=64,
        feature_dim=50,
        hidden=256,
    )
    device = get_device()
    env = make_env(cfg)
    eval_seeds = [seed_for("eval_curve", 500 + i) for i in range(20)]

    checkpoints = {
        "diag_drqv2_s0": Path("checkpoints/diag_drqv2_s0.pt"),
        "diag_svea_s0": Path("checkpoints/diag_svea_s0.pt"),
    }

    results = {}
    print("=== Evaluating Clean Performance of 12k Diagnostic Agents ===")

    for name, path in checkpoints.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        variant = "svea" if "svea" in name else "drqv2"
        agent = DrQAgent(cfg, variant=variant, seed=0)
        state = torch.load(path, map_location=device, weights_only=False)
        agent.load(state)
        agent.eval()

        recs = eval_cell(agent.act, cfg, None, 0.0, eval_seeds, env=env)
        cle_vals = [r["mean_cle"] for r in recs]
        succ_vals = [float(r["success"]) for r in recs]
        ret_vals = [r["ret"] for r in recs]

        results[name] = {
            "variant": variant,
            "training_steps": 12000,
            "n_eval_episodes": len(recs),
            "clean_mean_cle_px": round(float(np.mean(cle_vals)), 2),
            "clean_cle_std_px": round(float(np.std(cle_vals)), 2),
            "clean_success_rate": round(float(np.mean(succ_vals)), 3),
            "mean_return": round(float(np.mean(ret_vals)), 2),
            "random_floor_cle_px": 22.05,
            "target_threshold_px": 5.0,
            "competence_status": "Competent (<5.0 px)" if np.mean(cle_vals) < 5.0 else "Sub-threshold / Near random floor",
        }
        print(f"  {name}: Mean CLE = {results[name]['clean_mean_cle_px']} px (std {results[name]['clean_cle_std_px']}), "
              f"Success Rate = {results[name]['clean_success_rate']:.1%}, Status: {results[name]['competence_status']}")

    out_file = Path("results/clean_performance_12k.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
