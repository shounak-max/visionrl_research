"""
Base environment interface and perturbation configuration data contract.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class PerturbationConfig:
    """Configuration for visual and environmental domain perturbations.

    Axes include:
    - background_type / background_intensity (ProcgenOOD / CoinRun)
    - wall_texture (ProcgenOOD)
    - sprite (ProcgenOOD)
    - difficulty (Procgen)
    - photometric_noise_std (Visual Pursuit)
    - camera_jitter_std (Visual Pursuit)
    """

    axis: str = "clean"  # e.g., 'clean', 'background', 'wall_texture', 'sprite', 'difficulty', 'photometric'
    intensity: float = 0.0  # magnitude of perturbation in [0.0, 1.0]
    holdout_split: str = "train"  # 'train' or 'test'
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_tag(self) -> str:
        """Returns a string identifier for cache keying."""
        if self.axis == "clean":
            return "clean"
        return f"{self.axis}_inst{self.intensity:.2f}_{self.holdout_split}"


class BaseVisualEnvironment:
    """Abstract base interface for visual RL research environments."""

    def __init__(
        self,
        env_id: str,
        seed: int = 42,
        perturbation_config: Optional[PerturbationConfig] = None,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
    ):
        self.env_id = env_id
        self.seed = seed
        self.perturbation_config = perturbation_config or PerturbationConfig()
        self.image_shape = image_shape

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment state and returns initial observation [H, W, C] and info dict.

        Info dict may contain 'ground_truth_mask': np.ndarray [H, W] (0.0 to 1.0) if supported.
        """
        raise NotImplementedError

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes action in environment.

        Returns:
            (observation [H, W, C], reward, terminated, truncated, info)
        """
        raise NotImplementedError

    @property
    def action_space_dim(self) -> int:
        """Returns action dimension."""
        raise NotImplementedError

    def close(self):
        """Clean up environment resources."""
        pass
