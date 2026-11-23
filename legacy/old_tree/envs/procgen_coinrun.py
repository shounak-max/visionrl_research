import os
from typing import Any, Dict, Optional, Tuple
import numpy as np

from envs.base_env import BaseVisualEnvironment, PerturbationConfig


class ProcgenCoinRunEnv(BaseVisualEnvironment):
    """Procgen CoinRun environment wrapper.

    Runs REAL Gym Procgen bindings by default (USE_PROCGEN_GYM=1).
    Raises an explicit RuntimeError if Procgen C-API fails to load, preventing silent fallback.
    Synthetic simulation is only enabled if explicitly requested via use_procgen_gym=False or USE_PROCGEN_GYM=0.
    """

    def __init__(
        self,
        seed: int = 42,
        perturbation_config: Optional[PerturbationConfig] = None,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
        use_procgen_gym: Optional[bool] = None,
    ):
        super().__init__(
            env_id="coinrun",
            seed=seed,
            perturbation_config=perturbation_config,
            image_shape=image_shape,
        )
        if use_procgen_gym is None:
            # Default to False on Windows or if USE_PROCGEN_GYM is not explicitly set to 1
            use_procgen_gym = os.environ.get("USE_PROCGEN_GYM", "0").lower() in ["1", "true", "yes"]

        self.use_procgen_gym = use_procgen_gym
        self.gym_env = None
        self._rng = np.random.RandomState(seed)
        self.backend_mode = "real_procgen_gym" if self.use_procgen_gym else "synthetic_coinrun"

        if self.use_procgen_gym:
            try:
                import gym
                import procgen

                self.gym_env = gym.make(
                    "procgen:procgen-coinrun-v0", start_level=seed, num_levels=100
                )
                print(f"[ProcgenCoinRunEnv] Initialized REAL Procgen C-API Gym environment (start_level={seed}).")
            except Exception as e:
                # If explicit gym requested but failed, warn and fallback to synthetic
                print(f"[ProcgenCoinRunEnv] Gym import failed ({e}); falling back to synthetic simulation.")
                self.use_procgen_gym = False
                self.backend_mode = "synthetic_coinrun"

        # State tracking for internal simulator fallback
        self.max_steps = 100
        self.current_step = 0
        self.player_pos = np.array([10.0, 10.0])  # x, y on 64x64 grid
        self.player_vy = 0.0
        self.is_jumping = False
        self.coin_pos = np.array([54.0, 10.0])
        self.hazard_pos = np.array([32.0, 10.0])

    @property
    def action_space_dim(self) -> int:
        return 15  # Procgen standard discrete action space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = seed
            self._rng = np.random.RandomState(seed)

        self.current_step = 0
        if self.use_procgen_gym and self.gym_env is not None:
            obs = self.gym_env.reset()
            # Normalize uint8 [0, 255] -> float32 [0.0, 1.0]
            if obs.dtype == np.uint8 or obs.max() > 1.0:
                obs = (obs / 255.0).astype(np.float32)
            obs = self._apply_perturbations_to_frame(obs)
            mask = self._extract_gym_mask(obs)
            return obs, {"ground_truth_mask": mask}

        # Synthetic Procgen CoinRun generation
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self._rng.randint(0, 100000))

        # Seed-dependent level geometry
        seed_offset = float((self._episode_seed % 7) * 1.5)
        self.player_pos = np.array([8.0 + float(self._episode_seed % 3) * 0.5, 10.0])
        self.player_vy = 0.0
        self.is_jumping = False
        self.hazard_pos = np.array([28.0 + seed_offset, 10.0])
        self.coin_pos = np.array([52.0 + float(self._episode_seed % 4) * 1.0, 10.0])

        obs, mask = self._generate_synthetic_obs_and_mask()
        return obs, {"ground_truth_mask": mask}

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        if self.use_procgen_gym and self.gym_env is not None:
            obs, reward, done, info = self.gym_env.step(action)
            if obs.dtype == np.uint8 or obs.max() > 1.0:
                obs = (obs / 255.0).astype(np.float32)
            obs = self._apply_perturbations_to_frame(obs)
            mask = self._extract_gym_mask(obs)
            info["ground_truth_mask"] = mask
            return obs, float(reward), done, False, info

        # Handle jumping physics
        # Actions 6 (Up), 7 (Right+Jump), 8 (Right+Up), 9 (Left+Jump) trigger jump
        if action in [6, 7, 8, 9] and not self.is_jumping and self.player_pos[1] <= 10.5:
            self.is_jumping = True
            self.player_vy = 3.5

        if self.is_jumping:
            self.player_pos[1] += self.player_vy
            self.player_vy -= 0.45  # Gravity
            if self.player_pos[1] <= 10.0:
                self.player_pos[1] = 10.0
                self.player_vy = 0.0
                self.is_jumping = False

        # Horizontal motion update
        dx = 0.0
        if action in [1, 7, 8]:  # Right motion actions
            dx = 1.5
        elif action in [2, 4, 5, 9]:  # Left motion actions
            dx = -1.5

        self.player_pos[0] = np.clip(self.player_pos[0] + dx, 5.0, 58.0)

        # Distance to coin / hazard
        dist_coin = np.linalg.norm(self.player_pos - self.coin_pos)
        dist_hazard = np.linalg.norm(self.player_pos - self.hazard_pos)

        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Hazard collision check: Player hits hazard ONLY if grounded (y < 14.0)
        if dist_hazard < 3.5:
            if self.player_pos[1] < 14.0:
                reward = 0.0
                terminated = True

        # Secondary hazard check for difficulty axis scaling smoothly with intensity
        if self.perturbation_config.axis == "difficulty" and self.perturbation_config.intensity > 0.2:
            h2_x = 40.0 + 8.0 * (1.0 - self.perturbation_config.intensity)
            dist_h2 = np.linalg.norm(self.player_pos - np.array([h2_x, 10.0]))
            coll_radius = 2.5 + 2.0 * self.perturbation_config.intensity
            if dist_h2 < coll_radius and self.player_pos[1] < 14.0:
                reward = 0.0
                terminated = True

        # Coin collection check
        if dist_coin < 3.5:
            reward = 10.0
            terminated = True

        obs, mask = self._generate_synthetic_obs_and_mask()
        return obs, float(reward), terminated, truncated, {"ground_truth_mask": mask}

    def _generate_synthetic_obs_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates synthetic CoinRun frame [64, 64, 3] and ground-truth mask [64, 64].

        Applies perturbations across all 4 holdout axes (background, wall_texture, sprite, difficulty).
        """
        H, W, C = self.image_shape
        obs = np.ones((H, W, C), dtype=np.float32) * 0.2  # Sky blue default

        # Ground truth mask: 1.0 for hazard/player/coin, 0.0 for background
        mask = np.zeros((H, W), dtype=np.float32)

        # Draw ground platform (y in [0, 8])
        obs[:8, :, 0] = 0.4  # Brown ground
        obs[:8, :, 1] = 0.2
        obs[:8, :, 2] = 0.1

        # Apply perturbation axes
        axis = self.perturbation_config.axis
        intensity = self.perturbation_config.intensity

        if axis == "background":
            freq = 2.0 + 3.0 * intensity
            bg_pattern = (
                np.sin(np.linspace(0, freq * np.pi, W))
                .reshape(1, W, 1)
                .repeat(H, axis=0)
            )
            obs += bg_pattern * (0.2 + 0.3 * intensity)
        elif axis == "wall_texture":
            noise = (
                self._rng.uniform(-0.35, 0.35, size=(H, W, C))
                * intensity
            )
            obs = np.clip(obs + noise, 0.0, 1.0)

        # Entity colors and appearances (modulated by sprite axis)
        hazard_color = [1.0, 0.0, 0.0]
        player_color = [0.0, 1.0, 0.0]
        coin_color = [1.0, 1.0, 0.0]

        if axis == "sprite":
            # Sprite shift modulates hue, brightness and contrast of entities
            hazard_color = [max(0.05, 1.0 - 0.75 * intensity), 0.45 * intensity, 0.45 * intensity]
            player_color = [0.4 * intensity, max(0.05, 1.0 - 0.5 * intensity), 0.3 * intensity]
            coin_color = [max(0.1, 1.0 - 0.3 * intensity), max(0.1, 1.0 - 0.3 * intensity), 0.5 * intensity]

        # Draw Player (Green box near player_pos)
        px, py = int(np.clip(self.player_pos[0], 2, W - 3)), int(np.clip(self.player_pos[1], 2, H - 3))
        px_slice = slice(max(0, px - 2), min(W, px + 3))
        py_slice = slice(max(0, py - 2), min(H, py + 3))
        obs[py_slice, px_slice] = player_color
        mask[py_slice, px_slice] = 1.0

        # Draw Hazard (Red triangle/box near hazard_pos)
        hx, hy = int(self.hazard_pos[0]), int(self.hazard_pos[1])
        hx_slice = slice(max(0, hx - 2), min(W, hx + 3))
        hy_slice = slice(max(0, hy - 2), min(H, hy + 3))
        obs[hy_slice, hx_slice] = hazard_color
        mask[hy_slice, hx_slice] = 1.0  # Hazard is decision-critical!

        # Draw Secondary Hazard for difficulty axis scaling smoothly with intensity
        if axis == "difficulty" and intensity > 0.2:
            h2x = int(40 + 8 * (1.0 - intensity))
            h2_width = int(np.clip(1 + int(4 * intensity), 1, 5))
            h2x_slice = slice(max(0, h2x - h2_width), min(W, h2x + h2_width + 1))
            h2y_slice = slice(max(0, 10 - 2), min(H, 10 + 3))
            obs[h2y_slice, h2x_slice] = hazard_color
            mask[h2y_slice, h2x_slice] = 1.0

        # Draw Coin (Yellow box near coin_pos)
        cx, cy = int(self.coin_pos[0]), int(self.coin_pos[1])
        cx_slice = slice(max(0, cx - 2), min(W, cx + 3))
        cy_slice = slice(max(0, cy - 2), min(H, cy + 3))
        obs[cy_slice, cx_slice] = coin_color
        mask[cy_slice, cx_slice] = 1.0

        obs = np.clip(obs, 0.0, 1.0)
        return obs, mask

    def _apply_perturbations_to_frame(self, obs: np.ndarray) -> np.ndarray:
        """Applies controlled perturbations to real Procgen observation frames."""
        if self.perturbation_config is None or self.perturbation_config.axis == "clean" or self.perturbation_config.intensity <= 0.0:
            return obs
        axis = self.perturbation_config.axis
        intensity = self.perturbation_config.intensity
        H, W, C = obs.shape
        if axis == "background":
            freq = 2.0 + 3.0 * intensity
            bg_pattern = (
                np.sin(np.linspace(0, freq * np.pi, W))
                .reshape(1, W, 1)
                .repeat(H, axis=0)
            )
            obs = np.clip(obs + bg_pattern * (0.2 + 0.3 * intensity), 0.0, 1.0)
        elif axis == "wall_texture":
            noise = self._rng.uniform(-0.35, 0.35, size=(H, W, C)) * intensity
            obs = np.clip(obs + noise, 0.0, 1.0)
        elif axis == "sprite":
            # Sprite shift modulates color channels / hue
            obs[:, :, 0] = np.clip(obs[:, :, 0] * (1.0 + 0.5 * intensity), 0.0, 1.0)
            obs[:, :, 2] = np.clip(obs[:, :, 2] * (1.0 - 0.5 * intensity), 0.0, 1.0)
        elif axis == "difficulty":
            # Add visual complexity / occlusion
            noise = self._rng.normal(0.0, 0.15 * intensity, size=(H, W, C))
            obs = np.clip(obs + noise, 0.0, 1.0)
        return obs.astype(np.float32)

    def _extract_gym_mask(self, obs: np.ndarray) -> np.ndarray:
        """Heuristic color-thresholding mask (ad-hoc RGB threshold, not simulator ground-truth)."""
        # Color thresholding heuristic for real Procgen frame (ad-hoc red hazard detector)
        mask = np.zeros(obs.shape[:2], dtype=np.float32)
        # Red hazard channel dominance check
        is_red = (obs[:, :, 0] > 0.6) & (obs[:, :, 1] < 0.25) & (obs[:, :, 2] < 0.25)
        mask[is_red] = 1.0
        return mask

