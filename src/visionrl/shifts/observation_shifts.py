"""Controlled observation shifts (background, noise, distractor, viewpoint, color_gain)."""

from typing import Any, Dict, Tuple

import numpy as np

from visionrl.environments.base import from_u8, to_u8
from visionrl.environments.pursuit import PursuitEnv

KIND_ID: Dict[str, int] = {
    "background": 1,
    "noise": 2,
    "distractor": 3,
    "viewpoint": 4,
    "color_gain": 5,
}


class Shift:
    """A controlled observation shift of a given kind and severity s in [0,1].
    * The shift acts on the rendered frame and entity mask only; dynamics and reward are untouched.
    * All random parameters are drawn from an RNG seeded by (episode seed, kind), so they are FIXED within an
      episode and identical wherever the same episode is re-rendered (closed loop or post hoc).
    * Per-frame randomness (sensor noise) is seeded by (episode seed, kind, frame index).
    * severity 0 is the exact identity.
    """

    def __init__(self, kind: str, severity: float, ep_seed: int, img: int):
        assert kind in KIND_ID, f"Invalid shift kind: {kind}. Expected one of {list(KIND_ID.keys())}"
        self.kind, self.s, self.seed, self.img = kind, float(severity), int(ep_seed), int(img)
        r = np.random.default_rng([self.seed, KIND_ID[kind], 99])

        if kind == "background":
            self.col = r.uniform(0, 1, 3).astype(np.float32)
        elif kind == "distractor":
            k = int(round(self.s * 5))
            self.k = k
            n = max(k, 1)
            self.pos0 = r.uniform(0, img, (n, 2))
            self.vel = r.uniform(-1.2, 1.2, (n, 2))
            self.rad = r.uniform(2.0, 3.5, n)
            self.dcol = r.uniform(0, 1, (n, 3)).astype(np.float32)
        elif kind == "viewpoint":
            self.angle = np.deg2rad(30.0 * self.s) * (1.0 if r.random() < 0.5 else -1.0)
            phi = r.uniform(0, 2 * np.pi)
            self.shift_xy = 0.12 * img * self.s * np.array([np.cos(phi), np.sin(phi)])
            c = (img - 1) / 2.0
            yy, xx = np.mgrid[0:img, 0:img].astype(np.float64)
            ox, oy = xx - c - self.shift_xy[0], yy - c - self.shift_xy[1]
            ca, sa = np.cos(-self.angle), np.sin(-self.angle)           # inverse rotation
            sx, sy = ca * ox - sa * oy + c, sa * ox + ca * oy + c
            self.sx, self.sy = np.rint(sx).astype(int), np.rint(sy).astype(int)
            self.valid = (self.sx >= 0) & (self.sx < img) & (self.sy >= 0) & (self.sy < img)
            self.sx, self.sy = np.clip(self.sx, 0, img - 1), np.clip(self.sy, 0, img - 1)
        elif kind == "color_gain":
            self.gain = (1.0 + self.s * r.uniform(-0.8, 0.8, 3)).astype(np.float32)

        yy, xx = np.mgrid[0:img, 0:img].astype(np.float32)
        self._yy, self._xx = yy, xx

    def params(self) -> Dict[str, np.ndarray]:
        """Episode-level parameters, used by tests to verify they never change within an episode."""
        keys = ["col", "k", "angle", "shift_xy", "gain"]
        return {k: np.array(getattr(self, k)).copy() for k in keys if hasattr(self, k)}

    def __call__(self, frame: np.ndarray, mask: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.s == 0.0:
            return frame, mask
        k = self.kind
        if k == "background":
            out = frame.copy()
            bg = mask == 0
            out[bg] = (1 - self.s) * PursuitEnv.BG + self.s * self.col
            return out, mask
        if k == "noise":
            rng = np.random.default_rng([self.seed, KIND_ID[k], 1000 + int(t)])
            return np.clip(frame + rng.normal(0, 0.6 * self.s, frame.shape).astype(np.float32), 0.0, 1.0), mask
        if k == "distractor":
            out, m2 = frame.copy(), mask.copy()
            for i in range(self.k):
                pos = (self.pos0[i] + self.vel[i] * t) % self.img
                d = ((self._xx - pos[0]) ** 2 + (self._yy - pos[1]) ** 2 <= self.rad[i] ** 2) & (mask == 0)
                out[d] = self.dcol[i]
                m2[d] = 3
            return out, m2
        if k == "viewpoint":
            out = frame[self.sy, self.sx].copy()
            m2 = mask[self.sy, self.sx].copy()
            out[~self.valid] = PursuitEnv.BG
            m2[~self.valid] = 0
            return out, m2
        if k == "color_gain":
            return np.clip(frame * self.gain[None, None, :], 0.0, 1.0), mask
        raise ValueError(f"Unknown kind: {k}")


def apply_shift_episode(
    clean_u8: np.ndarray,
    masks: np.ndarray,
    kind: str,
    severity: float,
    ep_seed: int,
    img: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-hoc injection: re-render a recorded CLEAN episode under a shift.
    Uses the very same Shift object as the closed-loop path, so the two paths cannot drift apart.
    Returns (view_u8, masks_shifted).
    """
    sh = Shift(kind, severity, ep_seed, img)
    out, mo = [], []
    for t in range(clean_u8.shape[0]):
        f, m = sh(from_u8(clean_u8[t]), masks[t], t)
        out.append(to_u8(f))
        mo.append(m)
    return np.stack(out), np.stack(mo)
