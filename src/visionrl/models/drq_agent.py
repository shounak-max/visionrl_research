"""Pixel-based RL agents: DrQ-v2, SVEA, NoAug ablation, and TACO auxiliary recipe."""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visionrl.config import Cfg, get_device
from visionrl.models.actor_critic import Actor, Critic, _noisy_action
from visionrl.models.augmentations import RandomShift, random_overlay
from visionrl.models.encoder import Encoder


class DrQAgent:
    """Pixel-based agent supporting:
    - 'drqv2': random shift data augmentation
    - 'svea': DrQ-v2 plus decoupled strong-augmentation critic term (Hansen et al. 2021)
    - 'noaug': unaugmented baseline (ablation for inductive bias ranking)
    - 'taco': DrQ-v2 with temporal action-conditioned contrastive auxiliary loss
    """

    def __init__(self, cfg: Cfg, variant: str = "drqv2", seed: int = 0, act_dim: int = 2):
        self.cfg = cfg
        self.variant = variant
        self.seed = seed
        self.act_dim = act_dim
        device = get_device()
        self.device = device

        torch.manual_seed(seed)
        np.random.seed(seed)

        in_ch = 3 * cfg.n_stack
        self.encoder = Encoder(in_ch, cfg.img).to(device)
        d = self.encoder.repr_dim
        self.actor = Actor(d, act_dim, cfg.feature_dim, cfg.hidden).to(device)
        self.critic = Critic(d, act_dim, cfg.feature_dim, cfg.hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.aux = None
        if variant == "taco":
            D, K = cfg.taco_dim, cfg.n_step
            self.aux = nn.ModuleDict(dict(
                s=nn.Linear(d, D),
                a=nn.Sequential(nn.Linear(K * act_dim, D), nn.ReLU(), nn.Linear(D, D)),
                g=nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, D)),
            )).to(device)

        _enc_params = list(self.encoder.parameters()) + (list(self.aux.parameters()) if self.aux is not None else [])
        self.enc_opt = torch.optim.Adam(_enc_params, lr=cfg.lr)
        self.act_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.cri_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Scale padding dynamically if needed, respecting resolution
        scaled_pad = max(2, int(round(cfg.img * 4.0 / 84.0))) if cfg.aug_pad is None else cfg.aug_pad
        self.aug = RandomShift(scaled_pad)

    def eval(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        return self

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    @torch.no_grad()
    def act(self, obs_u8: np.ndarray, noise_std: Optional[float] = None) -> np.ndarray:
        """Deterministic evaluation rule is the mean action (noise_std=None).
        When noise_std is provided, adds exploration noise.
        """
        x = torch.as_tensor(obs_u8, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu = self.actor(self.encoder(x))
        a = mu if not noise_std else _noisy_action(mu, noise_std, self.cfg.std_clip * 10)
        return a.squeeze(0).cpu().numpy()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.variant == "noaug" else self.aug(x)

    def update(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        std: float,
        aseq: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        c = self.cfg
        obs, act, rew, disc, nxt = batch
        obs_c, nxt_c = self._prep(obs), self._prep(nxt)
        h = self.encoder(obs_c)

        with torch.no_grad():
            nh = self.encoder(nxt_c)
            na = _noisy_action(self.actor(nh), std, c.std_clip)
            tq1, tq2 = self.critic_target(nh, na)
            target = rew + disc * torch.min(tq1, tq2)

        q1, q2 = self.critic(h, act)
        loss_clean = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        if self.variant == "svea":
            hs = self.encoder(random_overlay(obs_c))
            s1, s2 = self.critic(hs, act)
            loss_aug = F.mse_loss(s1, target) + F.mse_loss(s2, target)
            critic_loss = c.svea_alpha * loss_clean + c.svea_beta * loss_aug
        else:
            critic_loss = loss_clean

        if self.variant == "taco" and aseq is not None:
            critic_loss = critic_loss + c.taco_weight * self._taco_loss(h, nh, aseq)

        self.enc_opt.zero_grad(set_to_none=True)
        self.cri_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.cri_opt.step()
        self.enc_opt.step()

        hd = h.detach()
        a_pi = _noisy_action(self.actor(hd), std, c.std_clip)
        pq1, pq2 = self.critic(hd, a_pi)
        actor_loss = -torch.min(pq1, pq2).mean()

        self.act_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.act_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, c.tau)

        return float(critic_loss.item()), float(actor_loss.item())

    def _taco_loss(self, h: torch.Tensor, nh: torch.Tensor, aseq: torch.Tensor) -> torch.Tensor:
        """TACO InfoNCE loss between predicted state-action representation and true next state."""
        a = self.aux
        q = F.normalize(a["g"](torch.cat([a["s"](h), a["a"](aseq)], 1)), dim=1)
        with torch.no_grad():
            k = F.normalize(a["s"](nh), dim=1)
        logits = q @ k.t() / self.cfg.taco_temp
        return F.cross_entropy(logits, torch.arange(len(q), device=q.device))

    def state(self) -> Dict[str, Any]:
        return dict(
            encoder=self.encoder.state_dict(),
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            aux=(self.aux.state_dict() if self.aux is not None else None),
        )

    def load(self, st: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(st["encoder"])
        self.actor.load_state_dict(st["actor"])
        self.critic.load_state_dict(st["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        if self.aux is not None and st.get("aux") is not None:
            self.aux.load_state_dict(st["aux"])
