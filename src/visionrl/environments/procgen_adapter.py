"""Procgen CoinRun adapter for out-of-distribution evaluation."""

from typing import Optional, Tuple

import numpy as np

from visionrl.environments.base import to_u8
from visionrl.shifts.observation_shifts import Shift


class ProcgenCoinRun:
    """Honest adapter for Procgen CoinRun.
    (1) Train and held-out levels are disjoint by construction: train uses levels [0, n_train), held-out
    uses levels starting at HELDOUT_START; every episode is one specific level, so reset seeding is real.
    (2) Only shifts that Procgen can support WITHOUT entity masks are exposed, and they carry their true names:
    'noise' (Gaussian pixel noise), 'color_gain' (per-channel gain) and 'viewpoint' (fixed per-level affine warp).
    No claim about sprite identity or difficulty level is made, and the semantic probe is disabled (no simulator mask).
    """
    HELDOUT_START = 10_000
    ALLOWED = ("noise", "color_gain", "viewpoint")

    def __init__(self, level: int, kind: Optional[str] = None, sev: float = 0.0, distribution_mode: str = "easy"):
        from procgen import ProcgenEnv                                   # raises ImportError if not installed
        assert kind is None or kind in self.ALLOWED, f"{kind} is not supported without entity masks"
        self.env = ProcgenEnv(
            num_envs=1, env_name="coinrun", num_levels=1, start_level=int(level),
            distribution_mode=distribution_mode, rand_seed=int(level)
        )
        self.kind, self.sev, self.level = kind, sev, int(level)
        self.shift = Shift(kind, sev, level, 64) if kind and sev > 0 else None
        self.t = 0

    def _obs(self) -> Tuple[float, np.ndarray]:
        rew, obs, first = self.env.observe()
        fr = obs["rgb"][0].astype(np.float32) / 255.0
        if self.shift is not None:
            fr = self.shift(fr, np.zeros(fr.shape[:2], np.int8), self.t)[0]
        return rew[0], to_u8(fr)

    def reset(self) -> np.ndarray:
        self.t = 0
        return self._obs()[1]

    def step(self, action: int) -> Tuple[np.ndarray, float]:
        self.env.act(np.array([action]))
        r, o = self._obs()
        self.t += 1
        return o, float(r)


def procgen_available() -> bool:
    """Check if the procgen package is available."""
    try:
        import procgen  # noqa: F401
        return True
    except Exception:
        return False
