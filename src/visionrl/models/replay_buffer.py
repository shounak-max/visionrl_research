"""Replay buffer storing whole fixed-length episodes with n-step returns."""

from typing import Any, Dict, Tuple

import numpy as np
import torch

from visionrl.config import Cfg, get_device


class EpisodeReplay:
    """Stores whole fixed-length episodes (single frames, not stacks) and builds stacks and n-step targets at sample time.
    Episodes have a fixed horizon, so time-limit truncation is bootstrapped (no terminal flag), as in DrQ-v2.
    """

    def __init__(self, max_eps: int, cfg: Cfg, act_dim: int = 2):
        self.c, self.max = cfg, max_eps
        T, H = cfg.ep_len, cfg.img
        self.fr = np.zeros((max_eps, T + 1, 3, H, H), np.uint8)
        self.ac = np.zeros((max_eps, T, act_dim), np.float32)
        self.rw = np.zeros((max_eps, T), np.float32)
        self.n = 0
        self.last_actseq = None

    def add(self, rec: Dict[str, Any]) -> None:
        i = self.n % self.max
        self.fr[i], self.ac[i], self.rw[i] = rec["view_u8"], rec["actions"], rec["rewards"]
        self.n += 1

    def sample(self, B: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.c
        T, ns = c.ep_len, c.n_stack
        device = get_device()
        e = rng.integers(0, min(self.n, self.max), B)
        t = rng.integers(0, T - c.n_step + 1, B)
        off = np.arange(ns - 1, -1, -1)
        io = np.clip(t[:, None] - off[None, :], 0, None)
        inx = np.clip(t[:, None] + c.n_step - off[None, :], 0, None)
        obs = self.fr[e[:, None], io].reshape(B, -1, c.img, c.img)
        nxt = self.fr[e[:, None], inx].reshape(B, -1, c.img, c.img)
        act = self.ac[e, t]
        self.last_actseq = self.ac[e[:, None], t[:, None] + np.arange(c.n_step)[None, :]].reshape(B, -1)  # n-step action sequence
        disc = c.discount ** np.arange(c.n_step)
        rew = sum(disc[j] * self.rw[e, t + j] for j in range(c.n_step))
        to_t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        return (to_t(obs), to_t(act), to_t(rew[:, None]), to_t(np.full((B, 1), c.discount ** c.n_step)), to_t(nxt))
