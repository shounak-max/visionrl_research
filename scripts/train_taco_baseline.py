"""
Train a full-scale TACO baseline on Visual Pursuit (continuous control).
Runs on CUDA to competence (30,000 steps, img=48, ep_len=50).
Saves:
- checkpoints/taco_30k_s0.pt
- results/taco_training_curve.csv
- results/taco_checkpoint_s0.pt
- results/taco_summary.json
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import torch

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.experiments.train import train_agent, eval_cell
from visionrl.environments import make_env

RESULTS_DIR = Path("results")
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def run():
    device = get_device()
    print(f"=== Training TACO Baseline on {device} (30,000 steps) ===")
    
    # 48x48 continuous Visual Pursuit configuration matching the empirical benchmark
    base_cfg = make_cfg("sandbox", "pursuit")
    cfg = dataclasses.replace(
        base_cfg,
        img=48,
        ep_len=50,
        train_steps=30000,
        seed_steps=1000,
        std_decay_steps=15000,
        eval_every_eps=25,
        n_eval_eps=10,
        batch=128,
        feature_dim=50,
        hidden=256,
        lr=3e-4,
        taco_dim=50,
        taco_weight=1.0,
        taco_temp=0.1,
    )

    t0 = time.time()
    agent, curve = train_agent(cfg, variant="taco", seed=0, verbose=True)
    dt = time.time() - t0

    # Save training curve and checkpoint
    curve.to_csv(RESULTS_DIR / "taco_training_curve.csv", index=False)
    taco_state = agent.state()
    torch.save(taco_state, CHECKPOINTS_DIR / "taco_30k_s0.pt")
    torch.save(taco_state, RESULTS_DIR / "taco_checkpoint_s0.pt")

    # Evaluate on clean environment across 20 held-out evaluation seeds
    env = make_env(cfg)
    eval_seeds = [seed_for("eval_curve", i) for i in range(20)]
    ev_results = eval_cell(agent.act, cfg, None, 0.0, eval_seeds, env=env)

    clean_cle_list = [r["mean_cle"] for r in ev_results]
    clean_success_list = [float(r["success"]) for r in ev_results]
    mean_clean_cle = float(pd.Series(clean_cle_list).mean())
    clean_success_rate = float(pd.Series(clean_success_list).mean())

    summary = {
        "status": "completed",
        "algorithm": "TACO (Temporal Action-Contrastive RL)",
        "env": "Visual Pursuit",
        "steps_trained": cfg.train_steps,
        "elapsed_seconds": round(dt, 2),
        "steps_per_second": round(cfg.train_steps / max(1e-3, dt), 2),
        "seed": 0,
        "device": str(device),
        "taco_hyperparams": {
            "taco_dim": cfg.taco_dim,
            "taco_weight": cfg.taco_weight,
            "taco_temp": cfg.taco_temp,
            "batch_size": cfg.batch,
            "learning_rate": cfg.lr,
            "img_resolution": f"{cfg.img}x{cfg.img}",
            "episode_length": cfg.ep_len,
        },
        "evaluation_clean": {
            "n_episodes": len(ev_results),
            "mean_cle_px": round(mean_clean_cle, 2),
            "success_rate": round(clean_success_rate, 2),
            "random_floor_cle_px": 22.05,
            "tracking_competence": "Passes competence gate (mean CLE < 5.0 px)" if mean_clean_cle < 5.0 else "Sub-threshold",
        },
        "evaluation_scope": f"Trained on CUDA for {cfg.train_steps} steps. Achieved clean success rate {clean_success_rate:.2f} (mean CLE {mean_clean_cle:.2f} px)."
    }

    taco_sum_file = RESULTS_DIR / "taco_summary.json"
    if taco_sum_file.exists():
        with open(taco_sum_file, "r") as f:
            existing_taco_summary = json.load(f)
        if "failure_forecasting_metrics" in existing_taco_summary:
            summary["failure_forecasting_metrics"] = existing_taco_summary["failure_forecasting_metrics"]
    with open(taco_sum_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TACO BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Steps: {cfg.train_steps} | Time: {dt:.1f}s ({cfg.train_steps/dt:.1f} steps/s)")
    print(f"Clean Success Rate: {clean_success_rate:.2f} | Clean Mean CLE: {mean_clean_cle:.2f} px")
    print("Artifacts saved to checkpoints/taco_30k_s0.pt, results/taco_summary.json, results/taco_checkpoint_s0.pt, results/taco_training_curve.csv")


if __name__ == "__main__":
    run()
