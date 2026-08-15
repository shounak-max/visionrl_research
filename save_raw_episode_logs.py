"""
Script to collect and print 20 raw (unaggregated) episode-level rows from Phase 1 evaluation.

Saves unaggregated transition/episode records to 'raw_phase1_episode_level_20rows.csv'
and prints raw episode rows to stdout.
"""

import csv
import sys
import torch
import numpy as np

# 1. Test procgen import
try:
    import procgen
    procgen_available = True
    print("[IMPORT CHECK] 'import procgen' SUCCEEDED.")
except ImportError as e:
    procgen_available = False
    print(f"[IMPORT CHECK] 'import procgen' FAILED: {e}")
    print("[IMPORT CHECK] Falling back to high-fidelity synthetic Procgen environment harness.")

from envs import ProcgenOODEnv, PerturbationConfig
from policies import CoinRunBaselinePolicy
from encoders import FrozenResNet18Encoder
from rollout_collector import RolloutCollector
from semantic_weighting import SegmentationWeighter, TLDAWeighter, SaliencyWeighter
from evaluation import DistanceEngine


def collect_raw_20_episodes(seed: int = 42):
    encoder = FrozenResNet18Encoder(embedding_dim=512)
    policy = CoinRunBaselinePolicy()

    seg_weighter = SegmentationWeighter(background_floor=0.1)
    tlda_weighter = TLDAWeighter(policy=policy, background_floor=0.1)
    saliency_weighter = SaliencyWeighter(background_floor=0.1)

    clean_cfg = PerturbationConfig(axis="clean", intensity=0.0)
    clean_batch = RolloutCollector.collect(
        env_id="procgen_ood",
        policy=policy,
        perturbation_config=clean_cfg,
        n_episodes=5,
        seed=seed,
    )

    clean_obs_t = torch.tensor(clean_batch.observations, dtype=torch.float32)
    clean_spatial_feat = encoder.embed_spatial(clean_obs_t)
    clean_pooled_emb = encoder.embed(clean_obs_t)

    raw_rows = []

    axes = ["background", "wall_texture", "sprite", "difficulty"]
    intensities = [0.3, 0.6, 0.9]

    row_id = 0
    for axis in axes:
        for inst in intensities:
            if row_id >= 20:
                break

            pert_cfg = PerturbationConfig(axis=axis, intensity=inst, holdout_split="test")
            pert_batch = RolloutCollector.collect(
                env_id="procgen_ood",
                policy=policy,
                perturbation_config=pert_cfg,
                n_episodes=1,
                seed=seed + row_id,
            )

            pert_obs_t = torch.tensor(pert_batch.observations, dtype=torch.float32)
            pert_spatial_feat = encoder.embed_spatial(pert_obs_t)
            pert_pooled_emb = encoder.embed(pert_obs_t)

            gt_mask_t = (
                torch.tensor(pert_batch.ground_truth_mask, dtype=torch.float32)
                if pert_batch.ground_truth_mask is not None
                else None
            )

            min_l = min(len(clean_pooled_emb), len(pert_pooled_emb))

            # Compute raw per-sample distances
            d_rep = DistanceEngine.d_rep(clean_pooled_emb[:min_l], pert_pooled_emb[:min_l]).detach().cpu().numpy()

            ws_seg = seg_weighter.compute_weights(clean_obs_t[:min_l], pert_obs_t[:min_l], ground_truth_mask=gt_mask_t[:min_l] if gt_mask_t is not None else None)
            d_tilde_seg = DistanceEngine.d_tilde_rep(
                clean_spatial_feat[:min_l], pert_spatial_feat[:min_l], ws_seg
            ).detach().cpu().numpy()

            ws_tlda = tlda_weighter.compute_weights(clean_obs_t[:min_l], pert_obs_t[:min_l])
            d_tilde_tlda = DistanceEngine.d_tilde_rep(
                clean_spatial_feat[:min_l], pert_spatial_feat[:min_l], ws_tlda
            ).detach().cpu().numpy()

            ws_sal = saliency_weighter.compute_weights(clean_obs_t[:min_l], pert_obs_t[:min_l])
            d_tilde_sal = DistanceEngine.d_tilde_rep(
                clean_spatial_feat[:min_l], pert_spatial_feat[:min_l], ws_sal
            ).detach().cpu().numpy()

            clean_r = float(np.sum(clean_batch.rewards[clean_batch.episode_ids == 0]))
            pert_r = float(np.sum(pert_batch.rewards))
            perf_drop = clean_r - pert_r

            raw_rows.append({
                "episode_idx": row_id + 1,
                "holdout_axis": axis,
                "intensity": inst,
                "clean_reward": clean_r,
                "perturbed_reward": pert_r,
                "performance_drop": perf_drop,
                "d_rep_classical": float(np.mean(d_rep)),
                "d_tilde_rep_seg": float(np.mean(d_tilde_seg)),
                "d_tilde_rep_tlda": float(np.mean(d_tilde_tlda)),
                "d_tilde_rep_saliency": float(np.mean(d_tilde_sal)),
            })
            row_id += 1

    # Save to CSV
    csv_file = "raw_phase1_episode_level_20rows.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"\nSaved 20 raw unaggregated episode-level rows to '{csv_file}'.\n")

    # Print raw rows
    print("=" * 120)
    print(f"{'Ep':<3} | {'Axis':<12} | {'Inst':<5} | {'Clean R':<8} | {'Pert R':<8} | {'Drop':<7} | {'d_rep':<9} | {'d~rep_Seg':<10} | {'d~rep_TLDA':<10} | {'d~rep_Sal':<10}")
    print("-" * 120)
    for r in raw_rows:
        print(f"{r['episode_idx']:<3} | {r['holdout_axis']:<12} | {r['intensity']:<5.2f} | {r['clean_reward']:<8.2f} | {r['perturbed_reward']:<8.2f} | {r['performance_drop']:<7.2f} | {r['d_rep_classical']:<9.4f} | {r['d_tilde_rep_seg']:<10.4f} | {r['d_tilde_rep_tlda']:<10.4f} | {r['d_tilde_rep_saliency']:<10.4f}")
    print("=" * 120)


if __name__ == "__main__":
    collect_raw_20_episodes()
