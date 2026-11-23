"""
Policy base interface and reference policy implementations (DrQ-v2 / SVEA / PPO support).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class BasePolicy(nn.Module):
    """Abstract interface for RL policies."""

    def __init__(self, action_dim: int, is_continuous: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.is_continuous = is_continuous

    def act(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True) -> Any:
        """Computes action for a single observation [H, W, C] or batch [B, H, W, C]."""
        raise NotImplementedError

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Optional policy parameter update step."""
        return {}


class RandomPolicy(BasePolicy):
    """Random baseline policy."""

    def __init__(self, action_dim: int, is_continuous: bool = False, seed: int = 42):
        super().__init__(action_dim, is_continuous)
        self._rng = np.random.RandomState(seed)

    def act(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True) -> Any:
        if self.is_continuous:
            return self._rng.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
        else:
            return int(self._rng.randint(0, self.action_dim))


class VisualPursuitExpertPolicy(BasePolicy):
    """Visual pursuit tracking policy with sensory perceptual localization.

    Calculates tracking velocity towards target location estimated from sensory observations
    or neural spatial feature representations. Visual degradation (noise, contrast loss) naturally
    affects localization precision and tracking dynamics without circular drift heuristics.
    """

    def __init__(self, gain: float = 0.5):
        super().__init__(action_dim=2, is_continuous=True)
        self.gain = gain

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        encoder: Optional[nn.Module] = None,
        deterministic: bool = True,
    ) -> Any:
        # Convert tensor to numpy if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        if obs.ndim == 4:
            obs = obs[0]

        H, W, C = obs.shape

        # Perceptual target localization
        if encoder is not None:
            # Neural perception through encoder's spatial representation
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                spatial_feat = encoder.embed_spatial(obs_t)  # [1, C_feat, H_feat, W_feat]
                # Localization from spatial feature energy
                spatial_energy = torch.mean(torch.abs(spatial_feat), dim=1).squeeze(0).cpu().numpy()
                feat_h, feat_w = spatial_energy.shape
                max_pos = np.unravel_index(np.argmax(spatial_energy), (feat_h, feat_w))
                tx = (max_pos[1] + 0.5) / feat_w * W
                ty = (max_pos[0] + 0.5) / feat_h * H

                # Reticle estimation from contrast difference
                red_score = obs[:, :, 0] - 0.5 * (obs[:, :, 1] + obs[:, :, 2])
                ay, ax = np.unravel_index(np.argmax(red_score), (H, W))
                
                # Signal-to-noise confidence in neural feature map
                snr = float(np.max(spatial_energy) / (np.mean(spatial_energy) + 1e-6))
                confidence = np.clip((snr - 1.0) / 2.0, 0.05, 1.0)
                lag_factor = confidence
        else:
            # Photometric / spatial filtering on raw observation
            blue_score = obs[:, :, 2] - 0.5 * (obs[:, :, 0] + obs[:, :, 1])
            red_score = obs[:, :, 0] - 0.5 * (obs[:, :, 1] + obs[:, :, 2])
            is_dark = float(np.mean(obs)) < 0.5
            if is_dark:
                blue_score = 0.5 * (obs[:, :, 0] + obs[:, :, 1]) - obs[:, :, 2]
                red_score = 0.5 * (obs[:, :, 1] + obs[:, :, 2]) - obs[:, :, 0]

            ts_t = torch.tensor(blue_score, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            as_t = torch.tensor(red_score, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            smooth_t = (
                torch.nn.functional.avg_pool2d(ts_t, kernel_size=5, stride=1, padding=2)
                .squeeze()
                .numpy()
            )
            smooth_a = (
                torch.nn.functional.avg_pool2d(as_t, kernel_size=3, stride=1, padding=1)
                .squeeze()
                .numpy()
            )

            ty, tx = np.unravel_index(np.argmax(smooth_t), (H, W))
            ay, ax = np.unravel_index(np.argmax(smooth_a), (H, W))

            # Contrast confidence
            contrast = float(np.max(smooth_t) - np.min(smooth_t))
            lag_factor = np.clip(contrast / 0.4, 0.1, 1.0)

        error_vector = np.array([tx - ax, ty - ay], dtype=np.float32)
        action = np.clip(self.gain * lag_factor * error_vector, -1.0, 1.0)
        return action


class CoinRunBaselinePolicy(BasePolicy):
    """Rule-based heuristic baseline policy for Procgen CoinRun platformer."""

    def __init__(self):
        super().__init__(action_dim=15, is_continuous=False)

    def act(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True) -> Any:
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if obs.ndim == 4:
            obs = obs[0]

        # Ensure float32 in [0.0, 1.0]
        obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0

        H, W, C = obs.shape
        # Identify player (green) and hazard (red at ground level y in [6, 16])
        green_mask = (obs[:, :, 1] > obs[:, :, 0] + 0.1) & (obs[:, :, 1] > obs[:, :, 2] + 0.1)
        px = float(np.mean(np.where(green_mask)[1])) if np.any(green_mask) else 10.0

        # Hazard detection on ground level
        ground_obs = obs[6:16, :, :]
        red_score = ground_obs[:, :, 0] - 0.5 * (ground_obs[:, :, 1] + ground_obs[:, :, 2])
        red_mask = red_score > 0.3

        if np.any(red_mask):
            hx_coords = np.where(red_mask)[1]
            ahead_hazards = [hx for hx in hx_coords if hx > px]
            if ahead_hazards:
                nearest_hx = min(ahead_hazards)
                dist_ahead = nearest_hx - px
                # Optimal jump clearance trigger window
                if 4.0 <= dist_ahead <= 14.0:
                    return 7  # Right + Jump

        # Default: proceed right
        return 1  # Right
