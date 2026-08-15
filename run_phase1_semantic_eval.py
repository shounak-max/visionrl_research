"""
Phase 1 Semantic-Weighted Generalization Bounds Evaluation.

Evaluates d_tilde_rep across all 3 SemanticWeighter backends against classical d_rep
on ProcgenOOD across all 4 holdout axes (background, wall_texture, sprite, difficulty).
Generates the Phase 1 headline correlation comparison table.
"""

import json
import os
from typing import Dict, List
import numpy as np
import torch

from envs import ProcgenOODEnv, PerturbationConfig
from policies import CoinRunBaselinePolicy
from encoders import FrozenResNet18Encoder
from rollout_collector import RolloutCollector, save_rollout_batch
from semantic_weighting import (
    SegmentationWeighter,
    TLDAWeighter,
    SaliencyWeighter,
)
from evaluation import DistanceEngine, EvaluationEngine, CorrelationResult


def evaluate_phase1_grid(
    seed: int = 42, n_episodes: int = 5
) -> Dict[str, Dict[str, float]]:
    """Runs Phase 1 correlation evaluation grid comparing d_rep vs d_tilde_rep across backends and holdout axes."""
    print("=" * 80)
    print("       PHASE 1 EVALUATION: SEMANTIC-WEIGHTED METRIC (d_tilde_rep)")
    print("=" * 80)

    encoder = FrozenResNet18Encoder(embedding_dim=512)
    policy = CoinRunBaselinePolicy()

    # Instantiating the three SemanticWeighter backends
    weighters = {
        "SegmentationWeighter (GT Mask)": SegmentationWeighter(background_floor=0.1),
        "TLDAWeighter (Lipschitz)": TLDAWeighter(policy=policy, background_floor=0.1),
        "SaliencyWeighter (Critic)": SaliencyWeighter(background_floor=0.1),
    }

    # Collect clean baseline rollouts
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
    clean_spatial_feat = encoder.embed_spatial(clean_obs_t)
    clean_pooled_emb = encoder.embed(clean_obs_t)
    clean_perf = clean_batch.rewards

    # Data collection accumulators
    d_rep_list = []
    d_tilde_lists = {w_name: [] for w_name in weighters}
    perf_drop_list = []
    axes_list = []

    holdout_axes = ["background", "wall_texture", "sprite", "difficulty"]
    intensities = [0.1, 0.3, 0.5, 0.7, 0.9]
    seeds = [seed + s * 50 for s in range(4)]

    for s_val in seeds:
        # Collect clean baseline rollouts for this seed level
        clean_cfg = PerturbationConfig(axis="clean", intensity=0.0)
        clean_batch = RolloutCollector.collect(
            env_id="procgen_ood",
            policy=policy,
            perturbation_config=clean_cfg,
            n_episodes=n_episodes,
            seed=s_val,
        )
        clean_obs_t = torch.tensor(clean_batch.observations, dtype=torch.float32)
        clean_spatial_feat = encoder.embed_spatial(clean_obs_t)
        clean_pooled_emb = encoder.embed(clean_obs_t)

        clean_ep_returns = [
            float(np.sum(clean_batch.rewards[clean_batch.episode_ids == ep]))
            for ep in range(n_episodes)
        ]

        for axis in holdout_axes:
            for inst in intensities:
                pert_cfg = PerturbationConfig(
                    axis=axis, intensity=inst, holdout_split="test"
                )
                pert_batch = RolloutCollector.collect(
                    env_id="procgen_ood",
                    policy=policy,
                    perturbation_config=pert_cfg,
                    n_episodes=n_episodes,
                    seed=s_val,
                )
                pert_obs_t = torch.tensor(pert_batch.observations, dtype=torch.float32)
                gt_mask_t = (
                    torch.tensor(pert_batch.ground_truth_mask, dtype=torch.float32)
                    if pert_batch.ground_truth_mask is not None
                    else None
                )

                # Process per-episode trajectory
                for ep in range(n_episodes):
                    clean_indices = np.where(clean_batch.episode_ids == ep)[0]
                    pert_indices = np.where(pert_batch.episode_ids == ep)[0]

                    if len(clean_indices) == 0 or len(pert_indices) == 0:
                        continue

                    min_len = min(len(clean_indices), len(pert_indices))
                    c_obs = clean_obs_t[clean_indices[:min_len]]
                    p_obs = pert_obs_t[pert_indices[:min_len]]

                    c_pooled = encoder.embed(c_obs)
                    p_pooled = encoder.embed(p_obs)
                    c_spatial = encoder.embed_spatial(c_obs)
                    p_spatial = encoder.embed_spatial(p_obs)

                    gt_mask_ep = (
                        gt_mask_t[pert_indices[:min_len]]
                        if gt_mask_t is not None
                        else None
                    )

                    # 1. Classical d_rep per episode
                    d_rep_ep = float(
                        torch.mean(DistanceEngine.d_rep(c_pooled, p_pooled)).item()
                    )

                    # Performance drop: clean return - perturbed return
                    clean_ret = float(np.sum(clean_batch.rewards[clean_indices]))
                    pert_ret = float(np.sum(pert_batch.rewards[pert_indices]))
                    perf_drop = clean_ret - pert_ret

                    d_rep_list.append(d_rep_ep)
                    perf_drop_list.append(perf_drop)
                    axes_list.append(axis)

                    # 2. Semantic-weighted d_tilde_rep per backend
                    for w_name, weighter in weighters.items():
                        ws = weighter.compute_weights(
                            c_obs, p_obs, ground_truth_mask=gt_mask_ep
                        )
                        d_t_ep = float(
                            torch.mean(
                                DistanceEngine.d_tilde_rep(
                                    c_spatial,
                                    p_spatial,
                                    ws,
                                    pooling_order="pre_pooling",
                                )
                            ).item()
                        )
                        d_tilde_lists[w_name].append(d_t_ep)

    # Compute correlations
    print("\nComputing Spearman (rho) & Pearson (r) correlation metrics...")

    results_table = {}

    # 1. Classical d_rep baseline correlation
    classical_res = EvaluationEngine.correlate(
        d_rep_list, perf_drop_list, metric_name="Classical d_rep", axes=axes_list
    )
    results_table["Classical d_rep"] = {
        "overall_rho": classical_res.spearman_rho,
        "overall_pvalue": classical_res.spearman_pvalue,
        "axis_breakdown": classical_res.axis_breakdown,
    }

    # 2. Semantic-weighted d_tilde_rep correlations
    for w_name in weighters:
        w_res = EvaluationEngine.correlate(
            d_tilde_lists[w_name],
            perf_drop_list,
            metric_name=w_name,
            axes=axes_list,
        )
        results_table[w_name] = {
            "overall_rho": w_res.spearman_rho,
            "overall_pvalue": w_res.spearman_pvalue,
            "axis_breakdown": w_res.axis_breakdown,
        }

    # Print summary headline table
    print("\n" + "=" * 90)
    print("      PHASE 1 HEADLINE COMPARISON TABLE: SPEARMAN RHO (ρ) PREDICTIVE POWER")
    print("=" * 90)
    header = f"{'Metric / Weighter Backend':<35} | {'Overall ρ':<10} | {'Background':<11} | {'Wall Texture':<12} | {'Sprite':<10} | {'Difficulty':<10}"
    print(header)
    print("-" * 90)

    for m_name, m_data in results_table.items():
        rho_all = m_data["overall_rho"]
        b_data = m_data["axis_breakdown"]
        rho_bg = b_data.get("background", {}).get("spearman_rho", 0.0) if b_data else 0.0
        rho_wall = b_data.get("wall_texture", {}).get("spearman_rho", 0.0) if b_data else 0.0
        rho_sprite = b_data.get("sprite", {}).get("spearman_rho", 0.0) if b_data else 0.0
        rho_diff = b_data.get("difficulty", {}).get("spearman_rho", 0.0) if b_data else 0.0

        row = f"{m_name:<35} | {rho_all:^10.4f} | {rho_bg:^11.4f} | {rho_wall:^12.4f} | {rho_sprite:^10.4f} | {rho_diff:^10.4f}"
        print(row)

    print("=" * 90)

    # Write output artifact
    out_file = "phase1_semantic_results.json"
    with open(out_file, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"Machine-readable results saved to '{out_file}'.")

    return results_table


if __name__ == "__main__":
    evaluate_phase1_grid()
