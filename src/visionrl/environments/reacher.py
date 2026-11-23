"""2-link planar reacher environment."""

from typing import Tuple

import numpy as np

from visionrl.config import Cfg
from visionrl.environments.pursuit import PursuitEnv


class ReacherEnv:
    """Second task, structurally different from pursuit: a 2-link planar arm with its base at the image center.
    Actions are joint-angle velocities in [-1,1]^2. The target is a static disc placed in the arm's reachable annulus
    (it depends only on the episode seed). The task is to bring the fingertip onto the target and keep it there.
    CLE is the fingertip-to-target distance in pixels. The renderer returns an entity mask (0 background, 1 arm, 2 target),
    so the same shifts, probes and outcome definitions apply unchanged. This is a DMControl-STYLE analogue implemented
    without MuJoCo; it is NOT the DeepMind Control Suite and must not be described as such.
    """
    BG, AGENT, TARGET = PursuitEnv.BG, PursuitEnv.AGENT, PursuitEnv.TARGET

    def __init__(self, cfg: Cfg):
        self.c = cfg
        H = cfg.img
        self.yy, self.xx = np.mgrid[0:H, 0:H].astype(np.float32)
        self.base = np.array([H / 2.0, H / 2.0], np.float32)
        self.L = cfg.link_frac * H
        self.th = np.zeros(2, np.float32)
        self.q = np.zeros(2, np.float32)
        self.t = 0

    def reset(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        c, H = self.c, self.c.img
        r = np.random.default_rng([int(seed), 21])
        self.th = r.uniform(-np.pi, np.pi, 2).astype(np.float32)
        rad, ang = r.uniform(0.4, 1.8) * self.L, r.uniform(0, 2 * np.pi)
        self.q = np.clip(self.base + rad * np.array([np.cos(ang), np.sin(ang)], np.float32), 4.0, H - 4.0).astype(np.float32)
        self.t = 0
        return self.render()

    def _joints(self) -> Tuple[np.ndarray, np.ndarray]:
        a1, a2 = float(self.th[0]), float(self.th[0] + self.th[1])
        elbow = self.base + self.L * np.array([np.cos(a1), np.sin(a1)], np.float32)
        tip = elbow + self.L * np.array([np.cos(a2), np.sin(a2)], np.float32)
        return elbow, tip

    def _capsule(self, A: np.ndarray, B: np.ndarray, w: float) -> np.ndarray:
        AB = B - A
        denom = float(AB @ AB) + 1e-9
        t = np.clip(((self.xx - A[0]) * AB[0] + (self.yy - A[1]) * AB[1]) / denom, 0.0, 1.0)
        return (self.xx - (A[0] + t * AB[0])) ** 2 + (self.yy - (A[1] + t * AB[1])) ** 2 <= w ** 2

    def render(self) -> Tuple[np.ndarray, np.ndarray]:
        H = self.c.img
        img = np.empty((H, H, 3), np.float32)
        img[:] = self.BG
        mask = np.zeros((H, H), np.int8)
        d_target = (self.xx - self.q[0]) ** 2 + (self.yy - self.q[1]) ** 2 <= 2.0 ** 2
        img[d_target] = self.TARGET
        mask[d_target] = 2
        elbow, tip = self._joints()
        for A, B in ((self.base, elbow), (elbow, tip)):
            d_arm = self._capsule(A, B, 1.3)
            img[d_arm] = self.AGENT
            mask[d_arm] = 1
        d_tip = (self.xx - tip[0]) ** 2 + (self.yy - tip[1]) ** 2 <= 2.0 ** 2
        img[d_tip] = self.AGENT
        mask[d_tip] = 1
        return img, mask

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        c, H = self.c, self.c.img
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.th = ((self.th + c.max_dtheta * a + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
        self.t += 1
        _, tip = self._joints()
        cle = float(np.linalg.norm(tip - self.q))
        img, mask = self.render()
        done = self.t >= c.ep_len
        return img, mask, -cle / (H / 2.0), cle, done
