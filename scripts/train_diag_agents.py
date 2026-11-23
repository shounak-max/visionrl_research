"""
Train and evaluate 12k diagnostic agents on CUDA with full state saving.
Saves:
- checkpoints/diag_drqv2_s0.pt (full agent state)
- checkpoints/diag_svea_s0.pt (full agent state)
- results/clean_performance_12k.json (clean evaluation across 20 held-out seeds)
"""

import json
from pathlib import Path
import sys
import time
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env
from visionrl.experiments.train import eval_cell, train_agent
from visionrl.models.drq_agent import DrQAgent


def main():
    device = get_device()
    print(f"=== Training 12k Diagnostic Agents on {device} with Full State Saving ===")
    
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
    eval_seeds = [seed_for("eval_curve", 500 + i) for i in range(20)]
    agents_to_train = [("drqv2", 0), ("svea", 0)]
    clean_eval_results = {}

    for variant, seed in agents_to_train:
        aid = f"{variant}_s{seed}"
        ckpt_path = Path(f"checkpoints/diag_{aid}.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Training {aid} ({cfg.train_steps} steps) ---")
        t0 = time.time()
        agent, curve = train_agent(cfg, variant=variant, seed=seed, verbose=True)
        dt = time.time() - t0

        # Save FULL agent state (encoder, actor, critic)
        torch.save(agent.state(), ckpt_path)
        print(f"Saved FULL state to {ckpt_path} in {dt:.1f}s ({cfg.train_steps/dt:.1f} steps/s)")

        # Verify load round-trip
        test_agent = DrQAgent(cfg, variant=variant, seed=seed)
        test_agent.load(torch.load(ckpt_path, map_location=device, weights_only=False))
        test_agent.eval()

        # Evaluate clean performance on 20 held-out seeds
        recs = eval_cell(test_agent.act, cfg, None, 0.0, eval_seeds, env=env)
        cle_vals = [r["mean_cle"] for r in recs]
        succ_vals = [float(r["success"]) for r in recs]
        ret_vals = [r["ret"] for r in recs]

        mean_cle = float(np.mean(cle_vals))
        std_cle = float(np.std(cle_vals))
        succ_rate = float(np.mean(succ_vals))
        mean_ret = float(np.mean(ret_vals))

        clean_eval_results[aid] = {
            "variant": variant,
            "seed": seed,
            "train_steps": cfg.train_steps,
            "elapsed_seconds": round(dt, 2),
            "steps_per_second": round(cfg.train_steps / max(1e-3, dt), 2),
            "n_eval_episodes": len(recs),
            "clean_mean_cle_px": round(mean_cle, 2),
            "clean_cle_std_px": round(std_cle, 2),
            "clean_success_rate": round(succ_rate, 3),
            "mean_return": round(mean_ret, 2),
            "random_floor_cle_px": 22.05,
            "competence_gate_threshold_px": 5.0,
            "status": "Competent (<5.0 px)" if mean_cle < 5.0 else "Sub-threshold / Developing",
        }
        print(f"[{aid}] Clean Mean CLE: {mean_cle:.2f} px (std {std_cle:.2f}) | Success Rate: {succ_rate:.1%} | Status: {clean_eval_results[aid]['status']}")

    results_path = Path("results/clean_performance_12k.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(clean_eval_results, f, indent=2)

    print(f"\nAll 12k diagnostic agents trained and saved. Clean performance saved to {results_path}")


if __name__ == "__main__":
    main()
