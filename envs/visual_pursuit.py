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

        # Distractor state (for visual distractors perturbation)
        self.num_distractors = 2
        self.distractor_pos = [
            np.array([16.0 + i * 24.0, 48.0 - i * 16.0]) for i in range(self.num_distractors)
        ]
        self.distractor_vel = [
            np.array([self._rng.choice([-1.2, 1.2]), self._rng.choice([-1.0, 1.0])])
            for _ in range(self.num_distractors)
        ]

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
        for i in range(self.num_distractors):
            self.distractor_pos[i] = np.array([16.0 + i * 24.0, 48.0 - i * 16.0]) + self._rng.uniform(-4.0, 4.0, size=2)
            d_angle = self._rng.uniform(0, 2 * np.pi)
            self.distractor_vel[i] = np.array([np.cos(d_angle), np.sin(d_angle)]) * 1.2

        obs = self._render_observation()
        cle = float(np.linalg.norm(self.agent_pos - self.target_pos))
        info = {
            "tracking_error_px": cle,
            "cle_px": cle,
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

        # Update distractors if active
        for i in range(self.num_distractors):
            self.distractor_pos[i] += self.distractor_vel[i]
            if self.distractor_pos[i][0] < 8.0 or self.distractor_pos[i][0] > W - 8.0:
                self.distractor_vel[i][0] *= -1.0
            if self.distractor_pos[i][1] < 8.0 or self.distractor_pos[i][1] > H - 8.0:
                self.distractor_vel[i][1] *= -1.0

        # Update agent cursor position
        self.agent_pos += action_arr[:2] * 2.0
        self.agent_pos = np.clip(
            self.agent_pos, [5.0, 5.0], [float(W - 5), float(H - 5)]
        )

        # Compute smooth tracking error (CLE) in pixels
        cle = float(np.linalg.norm(self.agent_pos - self.target_pos))
        # Reward is negative tracking error (higher is better)
        reward = -cle

        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._render_observation()
        info = {
            "tracking_error_px": cle,
            "cle_px": cle,
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
        """Renders 64x64x3 visual observation with visual pursuit target, distractors, and agent reticle."""
        H, W, C = self.image_shape
        obs = np.ones((H, W, C), dtype=np.float32) * 0.9  # Off-white background

        # Background color/pattern shift if configured
        if self.perturbation_config.axis == "background":
            intensity = self.perturbation_config.intensity
            freq = 2.0 + 3.0 * intensity
            bg_pattern = (
                np.sin(np.linspace(0, freq * np.pi, W))
                .reshape(1, W, 1)
                .repeat(H, axis=0)
            )
            obs = np.clip(obs * (1.0 - 0.4 * intensity) + bg_pattern * (0.2 + 0.3 * intensity), 0.0, 1.0)

        # Target (Dark blue circle/box)
        tx, ty = int(np.clip(self.target_pos[0], 4, W - 5)), int(np.clip(self.target_pos[1], 4, H - 5))
        t_slice_x = slice(max(0, tx - 3), min(W, tx + 4))
        t_slice_y = slice(max(0, ty - 3), min(H, ty + 4))
        obs[t_slice_y, t_slice_x] = [0.1, 0.1, 0.8]

        # Distractor objects (if axis is distractors or video_distractors)
        if self.perturbation_config.axis in ["distractors", "visual_distractors", "video_distractors"]:
            intensity = self.perturbation_config.intensity
            num_active = max(1, int(np.ceil(self.num_distractors * intensity)))
            for i in range(num_active):
                dx, dy = int(np.clip(self.distractor_pos[i][0], 4, W - 5)), int(np.clip(self.distractor_pos[i][1], 4, H - 5))
                d_slice_x = slice(max(0, dx - 3), min(W, dx + 4))
                d_slice_y = slice(max(0, dy - 3), min(H, dy + 4))
                obs[d_slice_y, d_slice_x] = [0.2, 0.7, 0.2]  # Green distractor

        # Agent Reticle (Red cross/box)
        ax, ay = int(np.clip(self.agent_pos[0], 2, W - 3)), int(np.clip(self.agent_pos[1], 2, H - 3))
        a_slice_x = slice(max(0, ax - 1), min(W, ax + 2))
        a_slice_y = slice(max(0, ay - 1), min(H, ay + 2))
        obs[a_slice_y, a_slice_x] = [0.9, 0.1, 0.1]

        # Apply photometric / camera perturbations smoothly if requested
        if self.perturbation_config.axis == "photometric":
            intensity = self.perturbation_config.intensity
            contrast = 1.0 - 0.65 * intensity
            brightness_shift = self._rng.uniform(-0.3, 0.3) * intensity
            obs = np.clip(0.5 + (obs - 0.5) * contrast + brightness_shift, 0.0, 1.0)
        elif self.perturbation_config.axis in ["camera_noise", "gaussian_noise"]:
            intensity = self.perturbation_config.intensity
            noise = self._rng.normal(0.0, 0.25 * intensity, size=(H, W, C))
            obs = np.clip(obs + noise, 0.0, 1.0)
        elif self.perturbation_config.axis == "severe_shift":
            intensity = self.perturbation_config.intensity
            # Color inversion, contrast reduction, and out-of-distribution camera noise
            contrast = max(0.05, 1.0 - 0.75 * intensity)
            obs = 1.0 - obs
            obs = np.clip(0.5 + (obs - 0.5) * contrast, 0.0, 1.0)
            noise = self._rng.normal(0.0, 0.25 * intensity, size=(H, W, C))
            obs = np.clip(obs + noise, 0.0, 1.0)

        return obs.astype(np.float32)
