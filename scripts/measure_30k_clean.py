"""
Evaluate clean tracking performance of the 30,000-step baseline agents:
- DrQ-v2 (30k, seed 0)
- SVEA (30k, seed 0)
- TACO (30k, seed 0)

Measures clean success rate (< 5.0 px CLE), mean CLE, and return across 20 held-out seeds.
Saves results to results/clean_performance_30k.json.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataclasses import replace
import numpy as np
import pandas as pd
import torch

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env
from visionrl.experiments.train import eval_cell
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
    )
    device = get_device()
    env = make_env(cfg)
    eval_seeds = [seed_for("eval_curve", i) for i in range(20)]

    checkpoints = {
        "drqv2_s0": Path("checkpoints/drqv2_30k_s0.pt"),
        "svea_s0": Path("checkpoints/svea_30k_s0.pt"),
        "taco_s0": Path("checkpoints/taco_30k_s0.pt"),
    }

    results = {}
    print("=== Evaluating Clean Performance of 30k Baseline Agents ===")

    for name, path in checkpoints.items():
        if not path.exists():
            # Try results/ alternative
            alt_path = Path(f"results/{name.split('_')[0]}_30k_checkpoint_{name.split('_')[1]}.pt")
            if alt_path.exists():
                path = alt_path
            elif name.startswith("taco"):
                alt_taco = Path("results/taco_checkpoint_s0.pt")
                if alt_taco.exists():
                    path = alt_taco
                else:
                    print(f"Skipping {name}: {path} not found")
                    continue
            else:
                print(f"Skipping {name}: {path} not found")
                continue

        variant = name.split("_")[0]
        agent = DrQAgent(cfg, variant=variant, seed=0)
        state = torch.load(path, map_location=device, weights_only=False)
        agent.load(state)
        agent.eval()

        recs = eval_cell(agent.act, cfg, None, 0.0, eval_seeds, env=env)
        cle_vals = [r["mean_cle"] for r in recs]
        succ_vals = [float(r["success"]) for r in recs]
        ret_vals = [r["ret"] for r in recs]

        mean_cle = float(np.mean(cle_vals))
        std_cle = float(np.std(cle_vals))
        succ_rate = float(np.mean(succ_vals))
        mean_ret = float(np.mean(ret_vals))

        status = "Passes competence gate (<5.0 px)" if mean_cle < 5.0 else "Sub-threshold / Near random floor"

        results[name] = {
            "algorithm": "TACO" if variant == "taco" else ("DrQ-v2" if variant == "drqv2" else "SVEA"),
            "variant": variant,
            "seed": 0,
            "training_steps": 30000,
            "n_eval_episodes": len(recs),
            "clean_mean_cle_px": round(mean_cle, 2),
            "clean_cle_std_px": round(std_cle, 2),
            "clean_success_rate": round(succ_rate, 3),
            "mean_return": round(mean_ret, 2),
            "random_floor_cle_px": 22.05,
            "target_threshold_px": 5.0,
            "competence_status": status,
        }
        print(f"  {name}: Mean CLE = {results[name]['clean_mean_cle_px']} px (std {results[name]['clean_cle_std_px']}), "
              f"Success Rate = {results[name]['clean_success_rate']:.1%}, Status: {results[name]['competence_status']}")

    out_file = Path("results/clean_performance_30k.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
