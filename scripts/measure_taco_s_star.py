"""
Measure critical severity s* and failure boundaries for TACO (30k, seed 0).

Evaluates closed-loop performance across all 5 shift families:
- background
- noise
- distractor
- viewpoint
- color_gain
across 11 severities (s in [0.0, 1.0] with step 0.1) on 20 held-out evaluation seeds.

Computes:
- Critical severity s* via linear interpolation (inf {s : SR(s) <= 0.5 * SR(0)})
- Censoring flag (censored = True if SR never halves)
- Cliffness metric (fraction of drop in single steepest step)
- Total success drop (SR(0) - min SR(s))
- Mean CLE and return across all cells

Saves outputs to:
- results/taco_s_star.json
- results/taco_s_star.csv
- results/taco_severity_sweep.csv
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataclasses import replace
import numpy as np
import pandas as pd
import torch

from visionrl.config import make_cfg, get_device, seed_for
from visionrl.environments import make_env, is_collapse
from visionrl.experiments.train import eval_cell, random_floor_cle
from visionrl.forecasting.metrics import critical_severity, cliffness
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
    floor = random_floor_cle(cfg)
    eval_seeds = [seed_for("eval_curve", i) for i in range(20)]

    ckpt_path = Path("checkpoints/taco_30k_s0.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading TACO checkpoint: {ckpt_path} on {device}")
    agent = DrQAgent(cfg, variant="taco", seed=0)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load(state)
    agent.eval()

    shift_kinds = ["background", "noise", "distractor", "viewpoint", "color_gain"]
    severities = [float(round(s, 2)) for s in np.linspace(0.0, 1.0, 11)]

    print(f"Evaluating TACO closed-loop across {len(shift_kinds)} shift families and {len(severities)} severities (20 seeds each)...")

    # 1. Clean baseline evaluation (s = 0.0)
    print("\n--- Evaluating Clean Baseline (s = 0.0) ---")
    clean_recs = eval_cell(agent.act, cfg, None, 0.0, eval_seeds, env=env)
    clean_sr = float(np.mean([float(r["success"]) for r in clean_recs]))
    clean_cle = float(np.mean([float(r["mean_cle"]) for r in clean_recs]))
    clean_ret = float(np.mean([float(r["ret"]) for r in clean_recs]))
    print(f"Clean Success Rate: {clean_sr:.3f} ({int(clean_sr * len(clean_recs))}/{len(clean_recs)})")
    print(f"Clean Mean CLE: {clean_cle:.2f} px (Target gate: < 5.0 px)")
    print(f"Clean Mean Return: {clean_ret:.2f}")

    half_sr_threshold = 0.5 * clean_sr
    print(f"Critical Halving Threshold: SR <= {half_sr_threshold:.4f}")

    # Sweep storage
    sweep_rows = []
    s_star_results = []

    for kind in shift_kinds:
        print(f"\n--- Sweeping shift family: {kind} ---")
        kind_sr_list = []
        kind_cle_list = []

        for sev in severities:
            if sev == 0.0:
                recs = clean_recs
            else:
                recs = eval_cell(agent.act, cfg, kind, sev, eval_seeds, env=env)

            sr = float(np.mean([float(r["success"]) for r in recs]))
            mean_cle = float(np.mean([float(r["mean_cle"]) for r in recs]))
            cle_std = float(np.std([float(r["mean_cle"]) for r in recs]))
            mean_ret = float(np.mean([float(r["ret"]) for r in recs]))
            collapse_rate = float(np.mean([float(is_collapse(r["mean_cle"], cfg, floor)) for r in recs]))

            kind_sr_list.append(sr)
            kind_cle_list.append(mean_cle)

            sweep_rows.append({
                "agent": "TACO",
                "variant": "taco",
                "seed": 0,
                "shift_kind": kind,
                "severity": sev,
                "success_rate": round(sr, 3),
                "mean_cle_px": round(mean_cle, 2),
                "cle_std_px": round(cle_std, 2),
                "mean_return": round(mean_ret, 2),
                "collapse_rate": round(collapse_rate, 3),
                "n_episodes": len(recs),
            })
            print(f"  sev={sev:0.1f} | SR={sr:.3f} | CLE={mean_cle:.2f} px | Ret={mean_ret:.2f} | Collapse={collapse_rate:.2f}")

        # Compute s* and cliff metrics
        s_crit, is_censored = critical_severity(severities, kind_sr_list, frac=0.5)
        c_metric = cliffness(severities, kind_sr_list)
        tot_drop = float(clean_sr - min(kind_sr_list))

        s_star_results.append({
            "agent": "TACO",
            "variant": "taco",
            "seed": 0,
            "shift_kind": kind,
            "clean_success_rate": round(clean_sr, 3),
            "halving_threshold": round(half_sr_threshold, 4),
            "s_crit": round(float(s_crit), 4),
            "censored": bool(is_censored),
            "cliffness": round(float(c_metric), 4) if not np.isnan(c_metric) else None,
            "total_drop": round(tot_drop, 4),
            "min_success_rate": round(float(min(kind_sr_list)), 3),
            "max_mean_cle_px": round(float(max(kind_cle_list)), 2),
        })
        status_str = f"s* = {s_crit:.4f}" if not is_censored else f"s* = {s_crit:.4f} (CENSORED: never dropped below {half_sr_threshold:.3f})"
        print(f"  => Summary for {kind}: {status_str}, Cliffness={c_metric:.3f}, Total Drop={tot_drop:.3f}")

    # Output paths
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(results_dir / "taco_severity_sweep.csv", index=False)
    print(f"\nSaved sweep grid to {results_dir / 'taco_severity_sweep.csv'}")

    df_s_star = pd.DataFrame(s_star_results)
    df_s_star.to_csv(results_dir / "taco_s_star.csv", index=False)
    print(f"Saved s* summary to {results_dir / 'taco_s_star.csv'}")

    s_star_dict = {
        "metadata": {
            "agent": "TACO",
            "variant": "taco",
            "training_steps": 30000,
            "seed": 0,
            "clean_success_rate": round(clean_sr, 3),
            "clean_mean_cle_px": round(clean_cle, 2),
            "halving_threshold": round(half_sr_threshold, 4),
            "n_eval_episodes_per_cell": len(eval_seeds),
            "random_floor_cle_px": round(floor, 2),
        },
        "critical_severities": s_star_results,
    }

    with open(results_dir / "taco_s_star.json", "w", encoding="utf-8") as f:
        json.dump(s_star_dict, f, indent=2)
    print(f"Saved s* json to {results_dir / 'taco_s_star.json'}")

    # Print summary table
    print("\n" + "=" * 70)
    print("TACO Critical Severity s* Results (Nominal Clean SR = 0.550, Halving Threshold = 0.275)")
    print("=" * 70)
    print(f"{'Shift Family':<15} | {'s*':<8} | {'Censored':<10} | {'Cliffness':<10} | {'Total Drop':<10} | {'Min SR':<8}")
    print("-" * 70)
    for r in s_star_results:
        cliff_str = f"{r['cliffness']:.3f}" if r['cliffness'] is not None else "N/A"
        print(f"{r['shift_kind']:<15} | {r['s_crit']:<8.3f} | {str(r['censored']):<10} | {cliff_str:<10} | {r['total_drop']:<10.3f} | {r['min_success_rate']:<8.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
