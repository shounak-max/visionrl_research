"""
Base Encoder interface for visual RL generalization bounds research.

Every encoder provides:
1. `embed(obs)` -> `[B, D]` global embedding vector.
2. `embed_spatial(obs)` -> `[B, C_spatial, H_spatial, W_spatial]` pre-GAP spatial feature map.
"""

from typing import Tuple
import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Abstract base class for perception encoders."""

    def __init__(self, embedding_dim: int = 512, encoder_id: str = "base_encoder"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_id = encoder_id

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias for embed(obs)."""
        return self.embed(obs)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes global pooled embedding [B, D].

        Accepts observation tensor in either [B, H, W, C] or [B, C, H, W].
        Format is automatically normalized to PyTorch [B, C, H, W] in [0, 1].
        """
        raise NotImplementedError

    def embed_spatial(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes last spatial feature map [B, C_feat, H_feat, W_feat] before GAP pooling."""
        raise NotImplementedError

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalizes observation tensor to PyTorch float tensor [B, C, H, W] in [0.0, 1.0]."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()

        # Handle missing batch dimension
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        # Convert [B, H, W, C] -> [B, C, H, W] if needed
        if obs.shape[-1] in [1, 3, 4] and obs.shape[1] not in [1, 3, 4]:
            obs = obs.permute(0, 3, 1, 2)

        # Scale uint8 [0, 255] -> float [0, 1] if needed
        if obs.max() > 1.0:
            obs = obs / 255.0

        return obs
