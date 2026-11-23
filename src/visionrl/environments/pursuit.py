"""Visual pursuit environment."""

from typing import Tuple

import numpy as np

from visionrl.config import Cfg


class PursuitEnv:
    """Visual pursuit task.
    A red disc (the agent) is moved by a 2-D velocity command and must stay on a green disc (the target).
    The target performs a smooth random walk that depends ONLY on the episode seed, so two policies run on the
    same seed face an identical target trajectory (which makes paired evaluation possible).
    The renderer also returns an entity mask (0 background, 1 agent, 2 target, 3 distractor),
    so ground-truth segmentation comes from the simulator and not from a colour threshold.
    """
    BG = np.array([26, 26, 26], np.float32) / 255.0
    AGENT = np.array([230, 26, 26], np.float32) / 255.0
    TARGET = np.array([26, 230, 26], np.float32) / 255.0

    def __init__(self, cfg: Cfg):
        self.c = cfg
        H = cfg.img
        self.yy, self.xx = np.mgrid[0:H, 0:H].astype(np.float32)
        self.p = np.zeros(2, np.float32)
        self.q = np.zeros(2, np.float32)
        self.heading = 0.0
        self.t = 0
        self.trng = np.random.default_rng(0)

    def reset(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        c, H = self.c, self.c.img
        r = np.random.default_rng([int(seed), 7])
        self.trng = np.random.default_rng([int(seed), 11])      # target stream, independent of the agent
        m = 6.0
        self.p = r.uniform(m, H - m, 2).astype(np.float32)
        self.q = r.uniform(m, H - m, 2).astype(np.float32)
        self.heading = float(r.uniform(0, 2 * np.pi))
        self.t = 0
        return self.render()

    def _disc(self, c: np.ndarray, rad: float) -> np.ndarray:
        return (self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2 <= rad ** 2

    def render(self) -> Tuple[np.ndarray, np.ndarray]:
        H = self.c.img
        img = np.empty((H, H, 3), np.float32)
        img[:] = self.BG
        mask = np.zeros((H, H), np.int8)
        d_target = self._disc(self.q, self.c.agent_radius)
        img[d_target] = self.TARGET
        mask[d_target] = 2
        d_agent = self._disc(self.p, self.c.agent_radius)
        img[d_agent] = self.AGENT
        mask[d_agent] = 1
        return img, mask

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        c, H = self.c, self.c.img
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.p = np.clip(self.p + c.max_step_px * a, c.agent_radius, H - c.agent_radius)
        self.heading += float(self.trng.normal(0, 0.3))
        v = c.target_speed * np.array([np.cos(self.heading), np.sin(self.heading)], np.float32)
        q = self.q + v
        lo, hi = c.agent_radius, H - c.agent_radius
        for k in range(2):                                      # reflect at the walls
            if q[k] < lo or q[k] > hi:
                q[k] = np.clip(q[k], lo, hi)
                self.heading = (np.pi - self.heading) if k == 0 else (-self.heading)
        self.q = q.astype(np.float32)
        self.t += 1
        cle = float(np.linalg.norm(self.p - self.q))
        reward = -cle / (H / 2.0)
        img, mask = self.render()
        done = self.t >= c.ep_len
        return img, mask, reward, cle, done
