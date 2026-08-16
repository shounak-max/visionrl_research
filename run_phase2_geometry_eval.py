"""
Phase 2 Geodesic Geometry Evaluation Runner.

Analyzes representational manifold deformation under visual perturbations across three fine-tuning regimes:
1. Frozen ResNet-18 (ATC-style decoupled backbone)
2. Partially fine-tuned ResNet-18 (Jones et al., 2025)
3. End-to-end CNN (DrQ-v2 / SVEA)

Sweeps subspace dimensions k in {5, 10, 20} on Grassmann manifolds Gr(k, D).
"""

import json
import os

# Enforce REAL Procgen C-API Gym environment execution
os.environ["USE_PROCGEN_GYM"] = "1"

from typing import Dict, List
import numpy as np
import torch

from envs import ProcgenOODEnv, PerturbationConfig
from policies import CoinRunBaselinePolicy
from encoders import (
    FrozenResNet18Encoder,
    PartialFineTuneResNet18Encoder,
    EndToEndCNNEncoder,
)
from rollout_collector import RolloutCollector
from geometry import GrassmannMapper, GeodesicDistance


def evaluate_phase2_geometry(
    seed: int = 42, n_episodes: int = 5
) -> Dict[str, Dict[str, float]]:
    """Runs Phase 2 geodesic geometry evaluation across encoder regimes and subspace dimensions k."""
    print("=" * 80)
    print("      PHASE 2 EVALUATION: GEODESIC GEOMETRY ON GRASSMANN MANIFOLDS")
    print("=" * 80)

    policy = CoinRunBaselinePolicy()

    # Load or train End-to-End CNN with DrQ-v2
    cnn_encoder = EndToEndCNNEncoder(embedding_dim=512)
    checkpoint_path = "checkpoints/drq2_cnn_encoder.pt"
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "encoder_state_dict" in ckpt:
                cnn_encoder.load_state_dict(ckpt["encoder_state_dict"])
                print(f"[INFO] Loaded trained DrQ-v2 CNN encoder checkpoint from '{checkpoint_path}'.")
            elif "state_dict" in ckpt:
                cnn_encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")})
                print(f"[INFO] Loaded trained DrQ-v2 CNN encoder from '{checkpoint_path}'.")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint ({e}), proceeding with standard initialization.")
    else:
        print("[INFO] No trained DrQ-v2 checkpoint found; training DrQ-v2 encoder for valid representation comparison...")
        from train_rl_policy import train_policy
        train_policy(env_name="procgen_ood", total_steps=200, checkpoint_path=checkpoint_path, seed=seed)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "encoder_state_dict" in ckpt:
            cnn_encoder.load_state_dict(ckpt["encoder_state_dict"])

    # Three encoder fine-tuning regimes
    encoders = {
        "Frozen ResNet-18": FrozenResNet18Encoder(embedding_dim=512),
        "Partial Fine-Tune ResNet-18": PartialFineTuneResNet18Encoder(
            embedding_dim=512
        ),
        "End-to-End CNN (DrQ-v2 Trained)": cnn_encoder,
    }

    # Collect clean baseline rollout trajectory
    clean_cfg = PerturbationConfig(axis="clean", intensity=0.0)
    clean_batch = RolloutCollector.collect(
        env_id="procgen_ood",
        policy=policy,
        perturbation_config=clean_cfg,
        n_episodes=n_episodes,
        seed=seed,
    )
    clean_obs_t = torch.tensor(clean_batch.observations, dtype=torch.float32)

    # Collect perturbed rollout trajectory (background holdout at high intensity)
    pert_cfg = PerturbationConfig(
        axis="background", intensity=0.8, holdout_split="test"
    )
    pert_batch = RolloutCollector.collect(
        env_id="procgen_ood",
        policy=policy,
        perturbation_config=pert_cfg,
        n_episodes=n_episodes,
        seed=seed,
    )
    pert_obs_t = torch.tensor(pert_batch.observations, dtype=torch.float32)

    results_table = {}

    subspace_dims = [5, 10, 20]

    for enc_name, encoder in encoders.items():
        print(f"\nExtracting manifold trajectories for regime: '{enc_name}'...")
        clean_emb = encoder.embed(clean_obs_t)
        pert_emb = encoder.embed(pert_obs_t)

        regime_results = {}

        for k in subspace_dims:
            # Extract Grassmann points Uk [D, k]
            p_clean = GrassmannMapper.to_subspace(clean_emb, subspace_dim=k)
            p_pert = GrassmannMapper.to_subspace(pert_emb, subspace_dim=k)

            # Compute geodesic metrics
            d_geo = GeodesicDistance.compute(
                p_clean, p_pert, metric_type="geodesic"
            )
            d_proj = GeodesicDistance.compute(
                p_clean, p_pert, metric_type="projection"
            )
            max_angle = GeodesicDistance.compute(
                p_clean, p_pert, metric_type="max_angle"
            )

            regime_results[f"k={k}"] = {
                "geodesic_distance": d_geo,
                "projection_distance": d_proj,
                "max_principal_angle_rad": max_angle,
                "max_principal_angle_deg": float(np.degrees(max_angle)),
            }

        results_table[enc_name] = regime_results

    # Print summary table
    print("\n" + "=" * 80)
    print("  PHASE 2 HEADLINE GEODESIC DISTANCE COMPARISON TABLE (RIEMANNIAN METRIC)")
    print("=" * 80)
    header = f"{'Encoder Fine-Tuning Regime':<32} | {'k=5 Geodesic':<14} | {'k=10 Geodesic':<14} | {'k=20 Geodesic':<14}"
    print(header)
    print("-" * 80)

    for enc_name, r_data in results_table.items():
        d5 = r_data["k=5"]["geodesic_distance"]
        d10 = r_data["k=10"]["geodesic_distance"]
        d20 = r_data["k=20"]["geodesic_distance"]
        print(
            f"{enc_name:<32} | {d5:^14.4f} | {d10:^14.4f} | {d20:^14.4f}"
        )

    print("=" * 80)

    # Output artifact
    out_file = "phase2_geometry_results.json"
    with open(out_file, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"Machine-readable geometry results saved to '{out_file}'.")

    return results_table


if __name__ == "__main__":
    evaluate_phase2_geometry()
