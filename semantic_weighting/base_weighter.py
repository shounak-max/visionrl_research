"""
Base interface for Semantic Weighter backends.

Computes a spatial weight matrix Ws in [0.0, 1.0] [B, H, W] to down-weight background/appearance shift
and up-weight shift near decision-critical (hazard/target) regions.
"""

from typing import Optional
import torch
import torch.nn as nn


class BaseSemanticWeighter(nn.Module):
    """Abstract base class for semantic weighting backends."""

    def __init__(self, weighter_id: str = "base_weighter"):
        super().__init__()
        self.weighter_id = weighter_id

    def compute_weights(
        self,
        obs_clean: torch.Tensor,
        obs_perturbed: torch.Tensor,
        ground_truth_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes spatial per-pixel weight map Ws in [0.0, 1.0] [B, H, W].

        Args:
            obs_clean: Clean observation tensor [B, H, W, C] or [B, C, H, W].
            obs_perturbed: Perturbed observation tensor [B, H, W, C] or [B, C, H, W].
            ground_truth_mask: Optional ground-truth asset/entity mask [B, H, W] (Procgen).

        Returns:
            Tensor [B, H, W] with values strictly bounded in [0.0, 1.0].
        """
        raise NotImplementedError

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Standardizes tensor to float [B, C, H, W] in [0, 1]."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] in [1, 3, 4] and obs.shape[1] not in [1, 3, 4]:
            obs = obs.permute(0, 3, 1, 2)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs
