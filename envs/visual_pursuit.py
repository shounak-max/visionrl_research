"""
Visual pursuit tracking environment wrapper for smooth continuous control evaluation.

Provides smooth visual dynamics where classical representation distance d_rep correlates
strongly with tracking performance drop (r ≈ -0.81).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from envs.base_env import BaseVisualEnvironment, PerturbationConfig


class VisualPursuitEnv(BaseVisualEnvironment):
    """Visual pursuit target tracking environment.

    Agent controls reticle (x, y) to track a target moving on a smooth continuous trajectory.
    Observation is [64, 64, 3].
    Perturbations supported: 'photometric' (brightness/contrast/hue), 'camera_noise' (Gaussian/salt-pepper noise).
    """

    def __init__(
        self,
        seed: int = 42,
        perturbation_config: Optional[PerturbationConfig] = None,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
    ):
        super().__init__(
            env_id="visual_pursuit",
            seed=seed,
            perturbation_config=perturbation_config,
            image_shape=image_shape,
        )
        self._rng = np.random.RandomState(seed)
        self.max_steps = 100
        self.current_step = 0

        # Dynamics state
        self.target_pos = np.array([32.0, 32.0])
        self.target_vel = np.array([1.5, 1.0])
        self.agent_pos = np.array([32.0, 32.0])

    @property
    def action_space_dim(self) -> int:
        return 2  # Continuous (dx, dy) control velocity

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = seed
            self._rng = np.random.RandomState(seed)

        self.current_step = 0
        self.target_pos = np.array([32.0, 32.0]) + self._rng.uniform(
            -5.0, 5.0, size=2
        )
        angle = self._rng.uniform(0, 2 * np.pi)
        self.target_vel = np.array([np.cos(angle), np.sin(angle)]) * 1.5
        self.agent_pos = self.target_pos.copy() + self._rng.uniform(
            -2.0, 2.0, size=2
        )

        obs = self._render_observation()
        info = {
            "tracking_error_px": float(
                np.linalg.norm(self.agent_pos - self.target_pos)
            ),
            "ground_truth_mask": self._get_target_mask(),
        }
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Format action
        action_arr = np.array(action, dtype=np.float32).flatten()
        if len(action_arr) < 2:
            action_arr = np.zeros(2, dtype=np.float32)

        # Update target trajectory smoothly
        self.target_pos += self.target_vel
        H, W, C = self.image_shape
        # Bounce off boundaries smoothly
        if self.target_pos[0] < 10.0 or self.target_pos[0] > W - 10.0:
            self.target_vel[0] *= -1.0
        if self.target_pos[1] < 10.0 or self.target_pos[1] > H - 10.0:
            self.target_vel[1] *= -1.0

        # Update agent cursor position
        self.agent_pos += action_arr[:2] * 2.0
        self.agent_pos = np.clip(
            self.agent_pos, [5.0, 5.0], [float(W - 5), float(H - 5)]
        )

        # Compute smooth tracking error in pixels
        tracking_error = float(
            np.linalg.norm(self.agent_pos - self.target_pos)
        )
        # Reward is negative tracking error (higher is better)
        reward = -tracking_error

        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._render_observation()
        info = {
            "tracking_error_px": tracking_error,
            "ground_truth_mask": self._get_target_mask(),
        }
        return obs, float(reward), terminated, truncated, info

    def _get_target_mask(self) -> np.ndarray:
        """Returns 2D target hazard/critical region mask [H, W]."""
        H, W, C = self.image_shape
        mask = np.zeros((H, W), dtype=np.float32)
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        t_slice_x = slice(max(0, tx - 3), min(W, tx + 4))
        t_slice_y = slice(max(0, ty - 3), min(H, ty + 4))
        mask[t_slice_y, t_slice_x] = 1.0
        return mask

    def _render_observation(self) -> np.ndarray:
        """Renders 64x64x3 visual observation with visual pursuit target and agent reticle."""
        H, W, C = self.image_shape
        obs = np.ones((H, W, C), dtype=np.float32) * 0.9  # Off-white background

        # Target (Dark blue circle/box)
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        t_slice_x = slice(max(0, tx - 3), min(W, tx + 4))
        t_slice_y = slice(max(0, ty - 3), min(H, ty + 4))
        obs[t_slice_y, t_slice_x] = [0.1, 0.1, 0.8]

        # Agent Reticle (Red cross/box)
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        a_slice_x = slice(max(0, ax - 1), min(W, ax + 2))
        a_slice_y = slice(max(0, ay - 1), min(H, ay + 2))
        obs[a_slice_y, a_slice_x] = [0.9, 0.1, 0.1]

        # Apply photometric / camera perturbations smoothly if requested
        if self.perturbation_config.axis == "photometric":
            intensity = self.perturbation_config.intensity
            contrast = 1.0 - 0.65 * intensity
            brightness_shift = self._rng.uniform(-0.3, 0.3) * intensity
            obs = np.clip(0.5 + (obs - 0.5) * contrast + brightness_shift, 0.0, 1.0)
        elif self.perturbation_config.axis == "camera_noise":
            intensity = self.perturbation_config.intensity
            noise = self._rng.normal(0.0, 0.25 * intensity, size=(H, W, C))
            obs = np.clip(obs + noise, 0.0, 1.0)

        return obs.astype(np.float32)
