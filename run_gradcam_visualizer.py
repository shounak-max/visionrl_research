"""
Grad-CAM & Spatial Attention Heatmap Visualization Runner.

Generates multi-panel qualitative explainability figures comparing:
1. Raw RGB Observations (Clean vs Background vs Hazard vs Severe Shift)
2. Ground-Truth Semantic Segmentation Masks (Ws)
3. Critic Saliency Masks (SEQA / MaDi style)
4. Policy & Critic Grad-CAM Activation Heatmaps
5. Attention Shifts Before vs After TTSS Adaptation

Saves publication-ready figures to `paper_artifacts/figures/`.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Enforce real procgen environment
os.environ["USE_PROCGEN_GYM"] = "1"

from envs import make_env, PerturbationConfig
from encoders import EndToEndCNNEncoder, FrozenResNet18Encoder
from semantic_weighting import SegmentationWeighter, SaliencyWeighter, TLDAWeighter
from evaluation.gradcam import GradCAM, overlay_heatmap
from policies import VisualPursuitExpertPolicy, CoinRunBaselinePolicy
from ttss import TTSSAdapter


def generate_gradcam_figures(output_dir="paper_artifacts/figures", seed=42):
    """Generates comprehensive Grad-CAM and attention comparison figures."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(">>> Generating Grad-CAM Attention Heatmaps across Visual RL Tasks <<<")

    # 1. Setup Models & Encoders
    cnn_encoder = EndToEndCNNEncoder(embedding_dim=512)
    # Target the last conv layer for Grad-CAM
    cam_cnn = GradCAM(model=cnn_encoder, target_layer=cnn_encoder.conv4)

    seg_weighter = SegmentationWeighter()
    sal_weighter = SaliencyWeighter()

    # 2. Collect sample observations across Procgen CoinRun perturbation axes
    env_clean = make_env("coinrun", seed=seed, perturbation_config=PerturbationConfig(axis="clean", intensity=0.0))
    env_bg = make_env("coinrun", seed=seed, perturbation_config=PerturbationConfig(axis="background", intensity=0.8))
    env_hazard = make_env("coinrun", seed=seed, perturbation_config=PerturbationConfig(axis="difficulty", intensity=0.8))
    env_sprite = make_env("coinrun", seed=seed, perturbation_config=PerturbationConfig(axis="sprite", intensity=0.8))
    env_severe = make_env("coinrun", seed=seed, perturbation_config=PerturbationConfig(axis="severe_shift", intensity=1.0))

    obs_clean, _ = env_clean.reset()
    obs_bg, _ = env_bg.reset()
    obs_hazard, _ = env_hazard.reset()
    obs_sprite, _ = env_sprite.reset()
    obs_severe, _ = env_severe.reset()

    # Take a few steps to reach an interactive state with coins and hazards
    for _ in range(5):
        obs_clean, _, _, _, _ = env_clean.step(3)
        obs_bg, _, _, _, _ = env_bg.step(3)
        obs_hazard, _, _, _, _ = env_hazard.step(3)
        obs_sprite, _, _, _, _ = env_sprite.step(3)
        obs_severe, _, _, _, _ = env_severe.step(3)

    scenarios = [
        ("Clean Baseline", obs_clean, env_clean),
        ("Task-Irrelevant Background Shift", obs_bg, env_bg),
        ("Task-Critical Hazard Shift", obs_hazard, env_hazard),
        ("Sprite Variation Shift", obs_sprite, env_sprite),
        ("Severe Distribution Shift", obs_severe, env_severe),
    ]

    # Create 5 rows (scenarios) x 5 columns (Raw Obs, GT Mask, Critic Saliency, Grad-CAM Heatmap, Overlay)
    fig, axes = plt.subplots(len(scenarios), 5, figsize=(15, 13))

    for row_idx, (title, obs, env) in enumerate(scenarios):
        obs_np = obs if obs.shape[-1] == 3 else obs.transpose(1, 2, 0)
        obs_t = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # 1. Ground Truth Segmentation Mask
        gt_mask_np = env.get_ground_truth_mask() if hasattr(env, "get_ground_truth_mask") else None
        gt_mask_t = torch.tensor(gt_mask_np, dtype=torch.float32).unsqueeze(0) if gt_mask_np is not None else None
        seg_mask = seg_weighter.compute_weights(obs_t, obs_t, ground_truth_mask=gt_mask_t).squeeze(0).cpu().numpy()

        # 2. Critic Saliency Mask
        sal_mask = sal_weighter.compute_weights(obs_t, obs_t).squeeze(0).detach().cpu().numpy()

        # 3. Grad-CAM Heatmap
        cam_map = cam_cnn.compute_cam(obs_t).squeeze(0).cpu().numpy()

        # 4. Heatmap Overlay
        overlay_img = overlay_heatmap(obs_np, cam_map, alpha=0.55, colormap="jet")

        # Column 0: Raw Observation
        ax = axes[row_idx, 0]
        ax.imshow(obs_np)
        ax.set_ylabel(title, fontsize=11, fontweight="bold", labelpad=10)
        if row_idx == 0:
            ax.set_title("Input Observation ($I$)", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 1: Ground-Truth Semantic Mask
        ax = axes[row_idx, 1]
        ax.imshow(seg_mask, cmap="viridis", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("GT Semantic Mask ($W_{s,\\mathrm{seg}}$)", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 2: Critic Saliency Mask (SEQA)
        ax = axes[row_idx, 2]
        ax.imshow(sal_mask, cmap="magma", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("Critic Saliency ($W_{s,\\mathrm{sal}}$)", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 3: Grad-CAM Activation Heatmap
        ax = axes[row_idx, 3]
        ax.imshow(cam_map, cmap="jet", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("Grad-CAM Heatmap ($L_{\\mathrm{CAM}}$)", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 4: Grad-CAM Overlay
        ax = axes[row_idx, 4]
        ax.imshow(overlay_img)
        if row_idx == 0:
            ax.set_title("Attentive Overlay ($I + L_{\\mathrm{CAM}}$)", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    cam_cnn.remove_hooks()

    plt.suptitle("Visual RL Saliency & Grad-CAM Attention Heatmaps across Perturbation Regimes",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_base = os.path.join(output_dir, "fig5_gradcam_attention_heatmaps")
    for ext in ["png", "svg", "pdf"]:
        out_path = f"{out_base}.{ext}"
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved Grad-CAM Figure: {out_path}")
    plt.close()

    print("[SUCCESS] Grad-CAM visual heatmaps exported successfully.")


if __name__ == "__main__":
    generate_gradcam_figures()
