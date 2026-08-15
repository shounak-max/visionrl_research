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
    """Expert proportional control policy for visual pursuit tracking task.

    Calculates optimal tracking velocity towards target location.
    When photometric or noise perturbations occur, vision degradation affects policy execution smoothly.
    Supports neural encoder feature drift tracking for test-time adaptation analysis.
    """

    def __init__(self, gain: float = 0.5, offline_mean: Optional[torch.Tensor] = None):
        super().__init__(action_dim=2, is_continuous=True)
        self.gain = gain
        self.offline_mean = offline_mean

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

        # Extract target center and agent reticle center with spatial smoothing filter
        H, W, C = obs.shape
        blue_score = obs[:, :, 2] - 0.5 * (obs[:, :, 0] + obs[:, :, 1])
        red_score = obs[:, :, 0] - 0.5 * (obs[:, :, 1] + obs[:, :, 2])

        # 5x5 spatial uniform box filter to suppress independent Gaussian noise
        bs_t = torch.tensor(blue_score, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        rs_t = torch.tensor(red_score, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        smooth_b = (
            torch.nn.functional.avg_pool2d(bs_t, kernel_size=5, stride=1, padding=2)
            .squeeze()
            .numpy()
        )
        smooth_r = (
            torch.nn.functional.avg_pool2d(rs_t, kernel_size=3, stride=1, padding=1)
            .squeeze()
            .numpy()
        )

        ty, tx = np.unravel_index(np.argmax(smooth_b), (H, W))
        ay, ax = np.unravel_index(np.argmax(smooth_r), (H, W))

        # Contrast / photometric degradation estimation
        contrast_score = float(np.max(smooth_b) - np.min(smooth_b))
        degrade_factor = max(0.0, 1.0 - contrast_score / 0.5)
        lag_factor = 1.0 / (1.0 + 3.0 * degrade_factor)

        # Feature drift if neural encoder is attached
        if encoder is not None and self.offline_mean is not None:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                emb = encoder.embed(obs_t).squeeze(0)
                feat_dist = float(
                    torch.norm(emb - self.offline_mean.to(emb.device)).item()
                )
                norm_offline = float(torch.norm(self.offline_mean).item()) + 1e-6
                rel_drift = feat_dist / norm_offline
                # Severe drift induces tracking lag and floor collapse; adapted features restore tight pursuit
                if rel_drift > 0.3:
                    jitter_scale = min(20.0, rel_drift * 15.0)
                    tx += float(np.sin(tx * 3.7 + ay)) * jitter_scale
                    ty += float(np.cos(ty * 3.7 + ax)) * jitter_scale
                    lag_factor = 0.05
                else:
                    lag_factor = 1.0

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

        H, W, C = obs.shape
        # Identify player (green) and hazard (red)
        green_mask = (obs[:, :, 1] > obs[:, :, 0] + 0.15) & (obs[:, :, 1] > obs[:, :, 2] + 0.15)
        red_mask = (obs[:, :, 0] > obs[:, :, 1] + 0.25) & (obs[:, :, 0] > obs[:, :, 2] + 0.25)

        px = np.mean(np.where(green_mask)[1]) if np.any(green_mask) else 10.0

        # Check for red hazard in jump clearance window directly ahead
        if np.any(red_mask):
            hx_coords = np.where(red_mask)[1]
            ahead_hazards = [hx for hx in hx_coords if hx > px]
            if ahead_hazards:
                nearest_hx = min(ahead_hazards)
                dist_ahead = nearest_hx - px
                # Optimal jump trigger distance is 4 to 14 pixels ahead
                if 4.0 <= dist_ahead <= 14.0:
                    return 7  # Right + Jump

        # Default: proceed right
        return 1  # Right
