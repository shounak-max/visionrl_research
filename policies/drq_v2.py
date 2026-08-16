"""
DrQ-v2 / SVEA Visual Reinforcement Learning Policy Implementation.

Implements DrQ-v2 (Yarats et al., 2021) with:
1. Convolutional encoder trunk (EndToEndCNNEncoder)
2. Random shift image data augmentation
3. Truncated Quantile / Double-Q Actor-Critic architecture
4. Transition Replay Buffer and self-contained gradient training loop
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.cnn_encoder import EndToEndCNNEncoder
from policies.base_policy import BasePolicy


class RandomShiftsAug(nn.Module):
    """Random shift data augmentation for image observations (DrQ-v2 / SVEA standard)."""

    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies random pixel translation [B, C, H, W] -> [B, C, H, W]."""
        n, c, h, w = x.size()
        assert h == w, "Observations must be square"
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class ReplayBuffer:
    """Cyclic transition replay buffer for visual RL training."""

    def __init__(self, capacity: int = 10000, obs_shape: Tuple[int, ...] = (64, 64, 3), action_dim: int = 2):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.idx] = obs
        self.action_buf[self.idx] = action
        self.reward_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_obs
        self.done_buf[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 64) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs_buf[idxs], dtype=torch.float32),
            "action": torch.tensor(self.action_buf[idxs], dtype=torch.float32),
            "reward": torch.tensor(self.reward_buf[idxs], dtype=torch.float32),
            "next_obs": torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32),
            "done": torch.tensor(self.done_buf[idxs], dtype=torch.float32),
        }


class DrQv2Policy(BasePolicy):
    """DrQ-v2 Actor-Critic Policy with shared CNN encoder trunk and data augmentation."""

    def __init__(
        self,
        action_dim: int = 2,
        embedding_dim: int = 512,
        is_continuous: bool = True,
        lr: float = 1e-4,
        gamma: float = 0.99,
    ):
        super().__init__(action_dim=action_dim, is_continuous=is_continuous)
        self.embedding_dim = embedding_dim
        self.gamma = gamma

        self.encoder = EndToEndCNNEncoder(embedding_dim=embedding_dim)
        self.aug = RandomShiftsAug(pad=4)

        # Actor head
        if self.is_continuous:
            self.actor = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )

        # Critic head (Q1, Q2)
        critic_in = embedding_dim + (action_dim if self.is_continuous else 0)
        self.q1 = nn.Sequential(
            nn.Linear(critic_in, 256),
            nn.ReLU(),
            nn.Linear(256, 1 if self.is_continuous else action_dim),
        )
        self.q2 = nn.Sequential(
            nn.Linear(critic_in, 256),
            nn.ReLU(),
            nn.Linear(256, 1 if self.is_continuous else action_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def act(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True) -> Any:
        self.eval()
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32)
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
            emb = self.encoder.embed(obs)
            out = self.actor(emb)
            if self.is_continuous:
                action = out.squeeze(0).cpu().numpy()
                if not deterministic:
                    action += np.random.normal(0, 0.1, size=action.shape)
                return np.clip(action, -1.0, 1.0)
            else:
                probs = F.softmax(out, dim=-1)
                if deterministic:
                    return int(torch.argmax(probs, dim=-1).item())
                else:
                    return int(torch.multinomial(probs, 1).item())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Performs a single DrQ-v2 actor-critic update step with visual augmentation."""
        self.train()
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # 1. Preprocess & apply random shift data augmentation
        obs_p = self.encoder.preprocess_obs(obs)
        next_obs_p = self.encoder.preprocess_obs(next_obs)
        obs_aug = self.aug(obs_p)
        next_obs_aug = self.aug(next_obs_p)

        # 2. Compute embeddings
        # Note: preprocess_obs is already applied, so pass to embed trunk
        emb = self.encoder.embed(obs_aug)
        with torch.no_grad():
            next_emb = self.encoder.embed(next_obs_aug)
            if self.is_continuous:
                next_action = self.actor(next_emb)
                target_q1 = self.q1(torch.cat([next_emb, next_action], dim=-1))
                target_q2 = self.q2(torch.cat([next_emb, next_action], dim=-1))
                target_q = torch.min(target_q1, target_q2)
                target_value = reward + (1.0 - done) * self.gamma * target_q
            else:
                next_q1 = self.q1(next_emb)
                next_q2 = self.q2(next_emb)
                target_q = torch.min(next_q1, next_q2).max(dim=-1, keepdim=True)[0]
                target_value = reward + (1.0 - done) * self.gamma * target_q

        # 3. Critic loss
        if self.is_continuous:
            current_q1 = self.q1(torch.cat([emb, action], dim=-1))
            current_q2 = self.q2(torch.cat([emb, action], dim=-1))
            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        else:
            current_q1 = self.q1(emb).gather(1, action.long())
            current_q2 = self.q2(emb).gather(1, action.long())
            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        # 4. Actor loss
        if self.is_continuous:
            pred_action = self.actor(emb.detach())
            actor_loss = -self.q1(torch.cat([emb.detach(), pred_action], dim=-1)).mean()
        else:
            logits = self.actor(emb.detach())
            log_probs = F.log_softmax(logits, dim=-1)
            q_vals = self.q1(emb.detach())
            actor_loss = -(log_probs * F.softmax(q_vals, dim=-1)).sum(dim=-1).mean()

        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            if "state_dict" in checkpoint:
                self.load_state_dict(checkpoint["state_dict"])
            elif "encoder_state_dict" in checkpoint:
                self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
