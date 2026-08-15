"""
TLDAWeighter: Reimplements Task-Aware Lipschitz Data Augmentation (Yuan et al., 2022, arXiv:2202.09982).

Measures policy-output sensitivity per pixel/patch under localized perturbation.
Pixels that induce large policy output changes get high weight Ws (task-relevant);
pixels with small induced changes get low weight (task-irrelevant background).
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_weighting.base_weighter import BaseSemanticWeighter
from policies.base_policy import BasePolicy


class TLDAWeighter(BaseSemanticWeighter):
    """TLDA policy sensitivity weighting backend."""

    def __init__(
        self,
        policy: Optional[BasePolicy] = None,
        patch_size: int = 8,
        background_floor: float = 0.0001,
    ):
        super().__init__(weighter_id="tlda_weighter")
        self.policy = policy
        self.patch_size = patch_size
        self.background_floor = background_floor

    def compute_weights(
        self,
        obs_clean: torch.Tensor,
        obs_perturbed: torch.Tensor,
        ground_truth_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs_clean = self.preprocess_obs(obs_clean)
        obs_pert = self.preprocess_obs(obs_perturbed)
        B, C, H, W = obs_clean.shape

        if self.policy is None:
            # Fallback sensitivity metric if policy model is unattached: observation pixel difference
            pixel_diff = torch.mean(torch.abs(obs_clean - obs_pert), dim=1)
            norm_diff = (pixel_diff - pixel_diff.min()) / (
                pixel_diff.max() - pixel_diff.min() + 1e-8
            )
            return torch.clamp(
                norm_diff + (1.0 - norm_diff) * self.background_floor, 0.0, 1.0
            )

        # Patch-wise policy output sensitivity calculation
        weights = torch.zeros((B, H, W), device=obs_clean.device)

        # Compute baseline policy output for clean observation
        with torch.no_grad():
            base_act = self.policy.act(obs_clean.permute(0, 2, 3, 1))

        # Sweep grid patches
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        for i in range(h_patches):
            for j in range(w_patches):
                h_slice = slice(i * self.patch_size, (i + 1) * self.patch_size)
                w_slice = slice(j * self.patch_size, (j + 1) * self.patch_size)

                # Create perturbed version with patched zeroed out
                obs_patch_pert = obs_clean.clone()
                obs_patch_pert[:, :, h_slice, w_slice] = 0.0

                with torch.no_grad():
                    patch_act = self.policy.act(obs_patch_pert.permute(0, 2, 3, 1))

                # Calculate action distribution / vector discrepancy
                if isinstance(base_act, np.ndarray) or isinstance(base_act, (int, float)):
                    diff_val = np.abs(np.array(base_act) - np.array(patch_act)).sum()
                else:
                    diff_val = torch.sum(torch.abs(base_act - patch_act)).item()

                weights[:, h_slice, w_slice] = float(diff_val)

        # Normalize per batch item to [background_floor, 1.0]
        for b in range(B):
            w_b = weights[b]
            max_w = torch.max(w_b)
            if max_w > 0.0:
                weights[b] = self.background_floor + (1.0 - self.background_floor) * (
                    w_b / max_w
                )
            else:
                weights[b] = self.background_floor

        return torch.clamp(weights, 0.0, 1.0)
