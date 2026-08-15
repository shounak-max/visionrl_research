"""
Phase 0 Baseline Reproduction & Sanity Check Runner.

Reproduces the two reference correlations using plain d_rep:
1. Visual Pursuit Tracking (smooth continuous control): target Pearson r ≈ -0.81 (strong correlation).
2. Procgen CoinRun (discrete non-smooth transitions): target Spearman rho ≈ -0.11 (metric collapse).
"""

import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch

from envs import make_env, PerturbationConfig
from policies import VisualPursuitExpertPolicy, CoinRunBaselinePolicy
from encoders import FrozenResNet18Encoder
from rollout_collector import RolloutCollector, save_rollout_batch
from evaluation import DistanceEngine, EvaluationEngine, CorrelationResult


def evaluate_tracking_correlation(
    seed: int = 42, n_episodes: int = 5
) -> CorrelationResult:
    """Evaluates d_rep correlation on smooth Visual Pursuit Tracking under photometric shifts."""
    print("\n--- Running Visual Pursuit Tracking (Smooth Control) Evaluation ---")
    encoder = FrozenResNet18Encoder(embedding_dim=512)
    policy = VisualPursuitExpertPolicy()

    # Collect clean baseline rollouts
    clean_cfg = PerturbationConfig(axis="clean", intensity=0.0)
    clean_batch = RolloutCollector.collect(
        env_id="visual_pursuit",
        policy=policy,
        perturbation_config=clean_cfg,
        n_episodes=n_episodes,
        seed=seed,
    )
    save_rollout_batch(clean_batch)

    # Compute clean embeddings and performance (negative tracking error in px)
    clean_obs_t = torch.tensor(clean_batch.observations, dtype=torch.float32)
    clean_emb = encoder.embed(clean_obs_t)
    clean_perf = clean_batch.rewards  # reward = -tracking_error_px

    d_rep_list = []
    perf_drop_list = []

    # Sweep photometric intensities
    intensities = [0.2, 0.4, 0.6, 0.8, 1.0]
    for inst in intensities:
        pert_cfg = PerturbationConfig(axis="photometric", intensity=inst)
        pert_batch = RolloutCollector.collect(
            env_id="visual_pursuit",
            policy=policy,
            perturbation_config=pert_cfg,
            n_episodes=n_episodes,
            seed=seed,
        )
        pert_obs_t = torch.tensor(pert_batch.observations, dtype=torch.float32)
        pert_emb = encoder.embed(pert_obs_t)
        pert_perf = pert_batch.rewards

        # Calculate elementwise d_rep L2 distance
        d_rep_vals = DistanceEngine.d_rep(clean_emb, pert_emb).numpy()

        # Performance drop: clean reward - perturbed reward
        # (higher drop means worse tracking performance under perturbation)
        perf_drops = clean_perf - pert_perf

        d_rep_list.extend(d_rep_vals.tolist())
        perf_drop_list.extend(perf_drops.tolist())

    result = EvaluationEngine.correlate(
        distance_metric=d_rep_list,
        performance_drop=perf_drop_list,
        metric_name="d_rep_visual_pursuit",
    )

    print(
        f"[Visual Pursuit Tracking] Pearson r: {result.pearson_r:.4f} (p={result.pearson_pvalue:.4e}), Spearman rho: {result.spearman_rho:.4f}"
    )
    return result


def evaluate_coinrun_correlation(
    seed: int = 42, n_episodes: int = 5
) -> CorrelationResult:
    """Evaluates d_rep correlation on Procgen CoinRun under discrete background/texture shifts."""
    print("\n--- Running Procgen CoinRun (Discrete Non-Smooth) Evaluation ---")
    encoder = FrozenResNet18Encoder(embedding_dim=512)
    policy = CoinRunBaselinePolicy()

    clean_cfg = PerturbationConfig(axis="clean", intensity=0.0)
    clean_batch = RolloutCollector.collect(
        env_id="procgen_ood",
        policy=policy,
        perturbation_config=clean_cfg,
        n_episodes=n_episodes,
        seed=seed,
    )
    save_rollout_batch(clean_batch)

    clean_obs_t = torch.tensor(clean_batch.observations, dtype=torch.float32)
    clean_emb = encoder.embed(clean_obs_t)
    clean_perf = clean_batch.rewards

    d_rep_list = []
    perf_drop_list = []
    axes_list = []

    # Sweep holdout axes (background shift vs wall_texture vs sprite)
    for axis in ["background", "wall_texture", "sprite"]:
        for inst in [0.3, 0.6, 0.9]:
            pert_cfg = PerturbationConfig(
                axis=axis, intensity=inst, holdout_split="test"
            )
            pert_batch = RolloutCollector.collect(
                env_id="procgen_ood",
                policy=policy,
                perturbation_config=pert_cfg,
                n_episodes=n_episodes,
                seed=seed,
            )
            pert_obs_t = torch.tensor(pert_batch.observations, dtype=torch.float32)
            for ep in range(n_episodes):
                clean_indices = np.where(clean_batch.episode_ids == ep)[0]
                pert_indices = np.where(pert_batch.episode_ids == ep)[0]

                if len(clean_indices) == 0 or len(pert_indices) == 0:
                    continue

                min_len = min(len(clean_indices), len(pert_indices))
                c_obs = clean_obs_t[clean_indices[:min_len]]
                p_obs = pert_obs_t[pert_indices[:min_len]]

                c_emb = encoder.embed(c_obs)
                p_emb = encoder.embed(p_obs)

                d_rep_ep = float(torch.mean(DistanceEngine.d_rep(c_emb, p_emb)).item())

                clean_ret = float(np.sum(clean_batch.rewards[clean_indices]))
                pert_ret = float(np.sum(pert_batch.rewards[pert_indices]))
                perf_drop = clean_ret - pert_ret

                d_rep_list.append(d_rep_ep)
                perf_drop_list.append(perf_drop)
                axes_list.append(axis)

    result = EvaluationEngine.correlate(
        distance_metric=d_rep_list,
        performance_drop=perf_drop_list,
        metric_name="d_rep_coinrun",
        axes=axes_list,
    )

    print(
        f"[Procgen CoinRun] Spearman rho: {result.spearman_rho:.4f} (p={result.spearman_pvalue:.4e}), Pearson r: {result.pearson_r:.4f}"
    )
    return result


def main():
    print("=" * 70)
    print("      VISUAL RL GENERALIZATION BOUNDS - PHASE 0 SANITY CHECK")
    print("=" * 70)

    seed = 42
    n_episodes = 5

    # 1. Run smooth tracking task
    vp_res = evaluate_tracking_correlation(seed=seed, n_episodes=n_episodes)

    # 2. Run non-smooth CoinRun task
    cr_res = evaluate_coinrun_correlation(seed=seed, n_episodes=n_episodes)

    # Compile results summary
    summary = {
        "visual_pursuit": {
            "pearson_r": vp_res.pearson_r,
            "pearson_pvalue": vp_res.pearson_pvalue,
            "spearman_rho": vp_res.spearman_rho,
            "target_pearson_r": -0.8099,
            "num_samples": vp_res.num_samples,
        },
        "procgen_coinrun": {
            "spearman_rho": cr_res.spearman_rho,
            "spearman_pvalue": cr_res.spearman_pvalue,
            "pearson_r": cr_res.pearson_r,
            "target_spearman_rho": -0.1094,
            "num_samples": cr_res.num_samples,
            "axis_breakdown": cr_res.axis_breakdown,
        },
    }

    results_file = "phase0_sanity_check_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("                 PHASE 0 SANITY CHECK RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"1. Visual Pursuit Tracking (Smooth): Measured Pearson r = {vp_res.pearson_r:.4f} (Target: ≈ -0.8099)"
    )
    print(
        f"2. Procgen CoinRun (Non-smooth): Measured Spearman rho = {cr_res.spearman_rho:.4f} (Target: ≈ -0.1094)"
    )
    print("=" * 70)
    print(f"Machine-readable results written to '{results_file}'.")

    # Verification threshold check
    vp_pass = abs(vp_res.pearson_r - (-0.8099)) < 0.25 or vp_res.pearson_r < -0.60
    cr_pass = abs(cr_res.spearman_rho - (-0.1094)) < 0.25 or abs(cr_res.spearman_rho) < 0.35

    if vp_pass and cr_pass:
        print("[SUCCESS] Phase 0 baseline reproduction confirmed within tolerance!")
    else:
        print("[WARNING] Measured baseline correlations diverged beyond expected tolerance.")


if __name__ == "__main__":
    main()
