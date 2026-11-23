"""Discrete action pursuit environment for PPO / action-rule testing."""

import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np

from visionrl.config import Cfg
from visionrl.environments.base import to_u8
from visionrl.environments.pursuit import PursuitEnv
from visionrl.shifts.observation_shifts import Shift


class DiscretePursuit:
    """The same pursuit task with 5 discrete actions {stay, +x, -x, +y, -y}.
    It exists so that the PPO code can be tested without Procgen, and so that the action rule
    (greedy or sampled) can be audited.
    """
    ACTS = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)
    n_actions = 5

    def __init__(self, cfg: Cfg, kind: Optional[str] = None, sev: float = 0.0):
        self.cfg = cfg
        self.env = PursuitEnv(cfg)
        self.kind = kind
        self.sev = sev
        self.shift = None
        self.t = 0
        self.cles = []
        self.stack = collections.deque(maxlen=cfg.n_stack)

    def reset(self, seed: int) -> np.ndarray:
        self.shift = Shift(self.kind, self.sev, seed, self.cfg.img) if (self.kind and self.sev > 0) else None
        f, m = self.env.reset(seed)
        self.t = 0
        self.cles = []
        v = to_u8(f if self.shift is None else self.shift(f, m, 0)[0])
        self.stack = collections.deque([v] * self.cfg.n_stack, maxlen=self.cfg.n_stack)
        return np.concatenate(list(self.stack), 0)

    def step(self, a: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        f, m, r, cle, done = self.env.step(self.ACTS[int(a)])
        self.t += 1
        self.cles.append(cle)
        self.stack.append(to_u8(f if self.shift is None else self.shift(f, m, self.t)[0]))
        return np.concatenate(list(self.stack), 0), r, done, dict(cle=cle)
