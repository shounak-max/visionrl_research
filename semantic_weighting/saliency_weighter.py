"""
SaliencyWeighter: End-to-end critic-loss supervised mask-decoder backend (SEQA / MaDi family).

Reimplements Bertoin et al. (2022) and MaDi (arXiv:2312.15339).
Outputs soft spatial mask Ws in [0.0, 1.0] [B, H, W] trained end-to-end using ONLY gradients
from the critic loss (no ground-truth segmentation labels, no separate supervision loss).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_weighting.base_weighter import BaseSemanticWeighter


class SaliencyMaskDecoder(nn.Module):
    """Lightweight 2-layer convolutional mask decoder predicting soft per-pixel mask in [0, 1]."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1, H, W] soft mask."""
        return self.net(obs)


class SaliencyWeighter(BaseSemanticWeighter):
    """SEQA/MaDi-style saliency mask-decoder weighting backend."""

    def __init__(self, in_channels: int = 3, background_floor: float = 0.0001):
        super().__init__(weighter_id="saliency_weighter")
        self.mask_decoder = SaliencyMaskDecoder(in_channels=in_channels)
        self.background_floor = background_floor

    def compute_weights(
        self,
        obs_clean: torch.Tensor,
        obs_perturbed: torch.Tensor,
        ground_truth_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs_clean = self.preprocess_obs(obs_clean)
        B, C, H, W = obs_clean.shape
        # Spatial saliency via decoder and color variance
        color_var = torch.var(obs_clean, dim=1, keepdim=True)
        raw_mask = self.mask_decoder(obs_clean) * (color_var + 0.1)
        min_v = raw_mask.amin(dim=(-2, -1), keepdim=True)
        max_v = raw_mask.amax(dim=(-2, -1), keepdim=True)
        norm_mask = (raw_mask - min_v) / (max_v - min_v + 1e-8)
        mask = norm_mask.squeeze(1)
        # Apply background floor
        weighted_mask = mask + (1.0 - mask) * self.background_floor
        return torch.clamp(weighted_mask, 0.0, 1.0)

    def train_step_critic_gradient(
        self,
        critic_fn: nn.Module,
        encoder_fn: nn.Module,
        obs: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        sparsity_weight: float = 0.01,
    ) -> float:
        """Trains the mask decoder end-to-end via gradients from the critic loss alone.

        If the masker suppresses a pixel the critic actually needed, critic loss goes up,
        and the masker learns to stop suppressing it.
        """
        self.mask_decoder.train()
        obs = self.preprocess_obs(obs)

        # 1. Compute soft spatial mask
        mask = self.mask_decoder(obs)  # [B, 1, H, W]

        # 2. Mask the input observation
        masked_obs = obs * mask

        # 3. Compute Q-values via encoder + critic
        embeddings = encoder_fn(masked_obs)
        if hasattr(critic_fn, "predict"):
            q_pred = critic_fn.predict(embeddings, actions)
        else:
            q_pred = critic_fn(embeddings)

        # 4. Critic MSE loss + mild mask L1 sparsity regularization
        critic_loss = F.mse_loss(q_pred, target_q)
        sparsity_loss = sparsity_weight * torch.mean(mask)
        total_loss = critic_loss + sparsity_loss

        # 5. Backpropagate gradients to mask decoder
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return float(critic_loss.item())
