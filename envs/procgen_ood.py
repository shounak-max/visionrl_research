"""
ProcgenOOD environment implementation supporting controlled out-of-distribution holdout axes.

Holdout axes supported per spec:
1. 'background' — background image / color scheme shifts.
2. 'wall_texture' — platform and obstacle texture variations.
3. 'sprite' — agent and entity visual appearance changes.
4. 'difficulty' — structural level complexity and hazard density.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from envs.base_env import BaseVisualEnvironment, PerturbationConfig
from envs.procgen_coinrun import ProcgenCoinRunEnv


class ProcgenOODEnv(BaseVisualEnvironment):
    """ProcgenOOD environment wrapper supporting explicit train/test holdout splits per axis."""

    SUPPORTED_AXES = ["background", "wall_texture", "sprite", "difficulty"]

    def __init__(
        self,
        seed: int = 42,
        perturbation_config: Optional[PerturbationConfig] = None,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
        use_procgen_gym: Optional[bool] = None,
    ):
        if perturbation_config is None:
            perturbation_config = PerturbationConfig(
                axis="background", intensity=0.5, holdout_split="test"
            )

        super().__init__(
            env_id="procgen_ood",
            seed=seed,
            perturbation_config=perturbation_config,
            image_shape=image_shape,
        )

        # Base underlying environment
        self._inner_env = ProcgenCoinRunEnv(
            seed=seed,
            perturbation_config=self.perturbation_config,
            image_shape=image_shape,
            use_procgen_gym=use_procgen_gym,
        )
        self.backend_mode = self._inner_env.backend_mode

    @property
    def action_space_dim(self) -> int:
        return self._inner_env.action_space_dim

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = seed
        obs, info = self._inner_env.reset(seed=seed)
        info["holdout_axis"] = self.perturbation_config.axis
        info["holdout_split"] = self.perturbation_config.holdout_split
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._inner_env.step(action)
        info["holdout_axis"] = self.perturbation_config.axis
        info["holdout_split"] = self.perturbation_config.holdout_split
        return obs, reward, terminated, truncated, info


def get_procgen_ood_configs(
    seed: int = 42,
) -> List[Tuple[str, PerturbationConfig]]:
    """Generates the benchmark suite of ProcgenOOD holdout configurations across all 4 axes."""
    configs = []
    # Baseline clean config
    configs.append(("clean", PerturbationConfig(axis="clean", intensity=0.0)))

    # 4 holdout axes across 3 intensity levels
    for axis in ProcgenOODEnv.SUPPORTED_AXES:
        for intensity in [0.3, 0.6, 0.9]:
            p_config = PerturbationConfig(
                axis=axis, intensity=intensity, holdout_split="test"
            )
            configs.append((p_config.to_tag(), p_config))
    return configs
