"""
Procgen CoinRun environment implementation with asset tracking and ground-truth mask support.

Supports live gym/procgen bindings when available, with a high-fidelity synthetic fallback
for fast offline testing and non-C-API platform execution.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from envs.base_env import BaseVisualEnvironment, PerturbationConfig


class ProcgenCoinRunEnv(BaseVisualEnvironment):
    """Procgen CoinRun environment wrapper.

    Provides observations [64, 64, 3] in uint8 or float32 [0, 1] format,
    and returns asset ground-truth hazard/entity mask [64, 64] (float32 [0.0, 1.0]).
    """

    def __init__(
        self,
        seed: int = 42,
        perturbation_config: Optional[PerturbationConfig] = None,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
        use_procgen_gym: bool = False,
    ):
        super().__init__(
            env_id="coinrun",
            seed=seed,
            perturbation_config=perturbation_config,
            image_shape=image_shape,
        )
        self.use_procgen_gym = use_procgen_gym
        self.gym_env = None
        self._rng = np.random.RandomState(seed)

        if self.use_procgen_gym:
            try:
                import gym
                import procgen

                self.gym_env = gym.make(
                    "procgen:procgen-coinrun-v0", start_level=seed, num_levels=100
                )
            except Exception as e:
                print(
                    f"[Warning] Procgen C API loading failed ({e}). Falling back to synthetic CoinRun simulation."
                )
                self.use_procgen_gym = False

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
            mask = self._extract_gym_mask(obs)
            return obs, {"ground_truth_mask": mask}

        # Synthetic Procgen CoinRun generation
        self.player_pos = np.array([10.0, 10.0]) + self._rng.uniform(
            -0.5, 0.5, size=2
        )
        self.player_pos[1] = 10.0
        self.player_vy = 0.0
        self.is_jumping = False
        self.coin_pos = np.array([54.0, 10.0])
        self.hazard_pos = np.array([32.0, 10.0]) + self._rng.uniform(
            -1.0, 1.0, size=2
        )
        self.hazard_pos[1] = 10.0

        obs, mask = self._generate_synthetic_obs_and_mask()
        return obs, {"ground_truth_mask": mask}

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        if self.use_procgen_gym and self.gym_env is not None:
            obs, reward, done, info = self.gym_env.step(action)
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

        # Secondary hazard check for difficulty axis
        if self.perturbation_config.axis == "difficulty" and self.perturbation_config.intensity > 0.3:
            h2_pos = np.array([44.0, 10.0])
            dist_h2 = np.linalg.norm(self.player_pos - h2_pos)
            if dist_h2 < 3.5 and self.player_pos[1] < 14.0:
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
            bg_pattern = (
                np.sin(np.linspace(0, 4 * np.pi, W))
                .reshape(1, W, 1)
                .repeat(H, axis=0)
            )
            obs += bg_pattern * intensity * 0.4
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
            hazard_color = [max(0.0, 1.0 - 0.75 * intensity), 0.4 * intensity, 0.4 * intensity]
            player_color = [0.4 * intensity, max(0.0, 1.0 - 0.5 * intensity), 0.3 * intensity]
            coin_color = [max(0.0, 1.0 - 0.3 * intensity), max(0.0, 1.0 - 0.3 * intensity), 0.5 * intensity]

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

        # Draw Secondary Hazard for difficulty axis if present
        if axis == "difficulty" and intensity > 0.3:
            h2x, h2y = 44, 10
            h2x_slice = slice(max(0, h2x - 2), min(W, h2x + 3))
            h2y_slice = slice(max(0, h2y - 2), min(H, h2y + 3))
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

    def _extract_gym_mask(self, obs: np.ndarray) -> np.ndarray:
        """Extracts ground-truth hazard/entity mask from Procgen observation or asset layer."""
        # Simple color thresholding heuristic for real Procgen frame if asset C API not bound
        H, W, C = obs.shape[:2], obs.shape[1], obs.shape[2]
        mask = np.zeros(obs.shape[:2], dtype=np.float32)
        # Red hazard channel dominance check
        is_red = (obs[:, :, 0] > 180) & (obs[:, :, 1] < 50) & (obs[:, :, 2] < 50)
        mask[is_red] = 1.0
        return mask
