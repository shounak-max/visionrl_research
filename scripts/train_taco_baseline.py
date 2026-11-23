"""Train a short real TACO baseline run on Visual Pursuit to verify modern baseline execution."""

import dataclasses
import json
import time
from pathlib import Path
import pandas as pd
import torch

from visionrl.config import make_cfg
from visionrl.experiments.train import train_agent
from visionrl.environments.pursuit import PursuitEnv

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run():
    print("=== Training TACO Baseline (Modern Temporal Action-Contrastive RL) ===")
    base_cfg = make_cfg("smoke", "pursuit")
    cfg = dataclasses.replace(
        base_cfg,
        train_steps=1000,
        seed_steps=200,
        eval_every_eps=4,
        n_eval_eps=5,
        batch=64,
        taco_dim=50,
        taco_weight=1.0,
        taco_temp=0.1
    )

    t0 = time.time()
    agent, curve = train_agent(cfg, variant="taco", seed=0, verbose=True)
    dt = time.time() - t0

    # Save curve and checkpoint
    curve.to_csv(RESULTS_DIR / "taco_training_curve.csv", index=False)
    torch.save(agent.state(), RESULTS_DIR / "taco_checkpoint_s0.pt")

    # Evaluate on clean environment using standard eval_cell
    from visionrl.experiments.train import eval_cell
    from visionrl.config import seed_for
    env = PursuitEnv(cfg)
    eval_seeds = [seed_for("eval_curve", i) for i in range(10)]
    ev_results = eval_cell(agent.act, cfg, None, 0.0, eval_seeds, env=env)
    
    clean_cle_list = [r["mean_cle"] for r in ev_results]
    clean_success_list = [r["success"] for r in ev_results]

    summary = {
        "status": "completed",
        "algorithm": "TACO (Temporal Action-Contrastive RL)",
        "env": "Visual Pursuit",
        "steps_trained": 1000,
        "elapsed_seconds": round(dt, 2),
        "steps_per_second": round(1000 / dt, 2),
        "seed": 0,
        "taco_hyperparams": {
            "taco_dim": cfg.taco_dim,
            "taco_weight": cfg.taco_weight,
            "taco_temp": cfg.taco_temp,
            "batch_size": cfg.batch,
            "learning_rate": cfg.lr
        },
        "evaluation_clean": {
            "n_episodes": len(ev_results),
            "mean_cle_px": round(float(pd.Series(clean_cle_list).mean()), 2),
            "success_rate": round(float(pd.Series(clean_success_list).mean()), 2),
            "random_floor_cle_px": 22.05
        },
        "curve_summary": {
            "initial_eval_cle": round(float(curve["mean_cle"].iloc[0]), 2) if len(curve) else None,
            "final_eval_cle": round(float(curve["mean_cle"].iloc[-1]), 2) if len(curve) else None,
            "initial_success": round(float(curve["success"].iloc[0]), 2) if len(curve) else None,
            "final_success": round(float(curve["success"].iloc[-1]), 2) if len(curve) else None,
        }
    }

    with open(RESULTS_DIR / "taco_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"TACO trained and evaluated successfully in {dt:.1f}s.")
    print(f"Clean Mean CLE: {summary['evaluation_clean']['mean_cle_px']} px")
    print("Wrote results/taco_summary.json, taco_training_curve.csv, taco_checkpoint_s0.pt")

if __name__ == "__main__":
    run()
