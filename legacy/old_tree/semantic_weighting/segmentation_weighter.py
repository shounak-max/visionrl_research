"""
SegmentationWeighter: Ground-truth entity/hazard mask backed semantic weighting backend.

Uses simulator asset/entity positions directly (Procgen C API / tracking reticle mask)
for hazard/foreground weighting. Removes segmentation-model error as a confound.
"""

from typing import Optional
import torch
import torch.nn.functional as F

from semantic_weighting.base_weighter import BaseSemanticWeighter


class SegmentationWeighter(BaseSemanticWeighter):
    """Ground-truth mask backed semantic weighter backend."""

    def __init__(self, background_floor: float = 0.0001):
        super().__init__(weighter_id="segmentation_weighter")
        self.background_floor = background_floor

    def compute_weights(
        self,
        obs_clean: torch.Tensor,
        obs_perturbed: torch.Tensor,
        ground_truth_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs_clean = self.preprocess_obs(obs_clean)
        B, C, H, W = obs_clean.shape

        if ground_truth_mask is not None:
            if not isinstance(ground_truth_mask, torch.Tensor):
                mask = torch.tensor(ground_truth_mask, dtype=torch.float32)
            else:
                mask = ground_truth_mask.float()

            if mask.ndim == 2:
                mask = mask.unsqueeze(0).repeat(B, 1, 1)

            # Ensure mask is in [0.0, 1.0] and apply floor
            mask = torch.clamp(mask, 0.0, 1.0)
            weighted_mask = mask + (1.0 - mask) * self.background_floor
            return torch.clamp(weighted_mask, 0.0, 1.0)

        # Heuristic fallback if ground truth mask is not supplied: image gradient magnitude
        grad_x = torch.abs(obs_clean[:, :, :, 1:] - obs_clean[:, :, :, :-1])
        grad_y = torch.abs(obs_clean[:, :, 1:, :] - obs_clean[:, :, :-1, :])
        grad_mag = torch.mean(grad_x[:, :, :-1, :], dim=1) + torch.mean(
            grad_y[:, :, :, :-1], dim=1
        )
        grad_mag = F.pad(grad_mag, (0, 1, 0, 1), mode="replicate")
        norm_mask = (grad_mag - grad_mag.min()) / (
            grad_mag.max() - grad_mag.min() + 1e-8
        )
        return torch.clamp(
            norm_mask + (1.0 - norm_mask) * self.background_floor, 0.0, 1.0
        )
