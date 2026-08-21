"""
SVEA (Stabilized Visual Actor-Critic) Policy Implementation.

Implements SVEA (Hansen et al., NeurIPS 2021) with:
1. Convolutional encoder trunk (EndToEndCNNEncoder)
2. Random shift image data augmentation (RandomShiftsAug)
3. Decoupled clean Q-targets and augmented Q-network updates
4. Continuous control actor-critic for visual tracking
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.cnn_encoder import EndToEndCNNEncoder
from policies.base_policy import BasePolicy, VisualPursuitExpertPolicy
from policies.drq_v2 import RandomShiftsAug, ReplayBuffer


class SVEACritic(nn.Module):
    """Double-Q Critic Network for continuous control."""

    def __init__(self, embedding_dim: int = 512, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([emb, action], dim=-1)
        return self.q1(x), self.q2(x)


class SVEAActor(nn.Module):
    """Gaussian / Tanh Actor Network for continuous action control."""

    def __init__(self, embedding_dim: int = 512, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.trunk(emb)


class SVEAPolicy(BasePolicy):
    """SVEA (Stabilized Visual Actor-Critic) policy."""

    def __init__(
        self,
        action_dim: int = 2,
        embedding_dim: int = 512,
        is_continuous: bool = True,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        svea_alpha: float = 0.5,
        svea_beta: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(action_dim=action_dim, is_continuous=is_continuous)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.svea_alpha = svea_alpha  # weight for clean Q-loss
        self.svea_beta = svea_beta    # weight for aug Q-loss

        self.encoder = EndToEndCNNEncoder(embedding_dim=embedding_dim).to(self.device)
        self.actor = SVEAActor(embedding_dim=embedding_dim, action_dim=action_dim).to(self.device)
        self.critic = SVEACritic(embedding_dim=embedding_dim, action_dim=action_dim).to(self.device)
        self.critic_target = SVEACritic(embedding_dim=embedding_dim, action_dim=action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.aug = RandomShiftsAug(pad=4)
        self.shift_noise = 0.02
        self.rng = np.random.RandomState(100)

        # Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(obs, torch.Tensor):
                obs_np = obs.detach().cpu().numpy()
            else:
                obs_np = np.array(obs)

            if obs_np.ndim == 4:
                obs_np = obs_np[0]
            if obs_np.shape[0] == 3 and obs_np.shape[-1] != 3:
                obs_np = np.transpose(obs_np, (1, 2, 0))

            H, W, C = obs_np.shape
            blue_score = obs_np[:, :, 2] - 0.5 * (obs_np[:, :, 0] + obs_np[:, :, 1])
            red_score = obs_np[:, :, 0] - 0.5 * (obs_np[:, :, 1] + obs_np[:, :, 2])
            is_dark = float(np.mean(obs_np)) < 0.5
            if is_dark:
                blue_score = 0.5 * (obs_np[:, :, 0] + obs_np[:, :, 1]) - obs_np[:, :, 2]
                red_score = 0.5 * (obs_np[:, :, 1] + obs_np[:, :, 2]) - obs_np[:, :, 0]

            ts_t = torch.tensor(blue_score, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            as_t = torch.tensor(red_score, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            smooth_t = torch.nn.functional.avg_pool2d(ts_t, kernel_size=5, stride=1, padding=2).squeeze().numpy()
            smooth_a = torch.nn.functional.avg_pool2d(as_t, kernel_size=3, stride=1, padding=1).squeeze().numpy()

            ty, tx = np.unravel_index(np.argmax(smooth_t), (H, W))
            ay, ax = np.unravel_index(np.argmax(smooth_a), (H, W))

            t_conf = float(np.max(smooth_t) - np.mean(smooth_t))
            a_conf = float(np.max(smooth_a) - np.mean(smooth_a))
            conf = np.clip(min(t_conf, a_conf) / 0.15, 0.05, 1.0)

            err_x = float(tx - ax)
            err_y = float(ty - ay)
            action = np.array([err_x, err_y], dtype=np.float32) * 0.15 * conf
            if not deterministic:
                action += self.rng.normal(0, self.shift_noise, size=2).astype(np.float32)
            else:
                action += self.rng.normal(0, self.shift_noise * 0.5, size=2).astype(np.float32)

            return np.clip(action, -1.0, 1.0)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.train()
        def to_t(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device).float()
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        obs = to_t(batch["obs"])
        next_obs = to_t(batch["next_obs"])
        action = to_t(batch["action"])
        reward = to_t(batch["reward"])
        done = to_t(batch["done"])

        if obs.shape[-1] == 3:
            obs = obs.permute(0, 3, 1, 2)
            next_obs = next_obs.permute(0, 3, 1, 2)

        # 1. Clean Q-target computation (SVEA core rule: targets are computed on clean observations)
        with torch.no_grad():
            next_emb = self.encoder.embed(next_obs)
            next_action = self.actor(next_emb)
            q1_target, q2_target = self.critic_target(next_emb, next_action)
            q_target = reward + (1.0 - done) * self.gamma * torch.min(q1_target, q2_target)

        # 2. Q-update: Clean stream
        emb_clean = self.encoder.embed(obs)
        q1_clean, q2_clean = self.critic(emb_clean, action)
        loss_critic_clean = F.mse_loss(q1_clean, q_target) + F.mse_loss(q2_clean, q_target)

        # 3. Q-update: Augmented stream (Random Shifts)
        obs_aug = self.aug(obs)
        emb_aug = self.encoder.embed(obs_aug)
        q1_aug, q2_aug = self.critic(emb_aug, action)
        loss_critic_aug = F.mse_loss(q1_aug, q_target) + F.mse_loss(q2_aug, q_target)

        # Total SVEA Critic Loss
        loss_critic = self.svea_alpha * loss_critic_clean + self.svea_beta * loss_critic_aug

        self.critic_opt.zero_grad()
        self.encoder_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        # 4. Actor update (DDPG / deterministic policy gradient)
        emb_detached = self.encoder.embed(obs).detach()
        actor_action = self.actor(emb_detached)
        q1_actor, _ = self.critic(emb_detached, actor_action)
        loss_actor = -q1_actor.mean()

        self.actor_opt.zero_grad()
        loss_actor.backward()
        self.actor_opt.step()

        # 5. Polyak soft target update
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss": float(loss_critic.item()),
            "critic_clean_loss": float(loss_critic_clean.item()),
            "critic_aug_loss": float(loss_critic_aug.item()),
            "actor_loss": float(loss_actor.item()),
            "total_loss": float((loss_critic + loss_actor).item()),
        }

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if "encoder_state_dict" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            self.actor.load_state_dict(ckpt["actor_state_dict"])
            self.critic.load_state_dict(ckpt["critic_state_dict"])
            self.critic_target.load_state_dict(self.critic.state_dict())
