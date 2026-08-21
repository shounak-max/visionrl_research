"""
Proximal Policy Optimization (PPO) policy implementation with CNN encoder.
Designed for continuous control environments such as SingleObjectTracking-v0 / VisualPursuitEnv.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from policies.base_policy import BasePolicy
from encoders.cnn_encoder import EndToEndCNNEncoder


class PPOPolicy(BasePolicy):
    """PPO policy with end-to-end CNN encoder for continuous visual pursuit tracking."""

    def __init__(
        self,
        action_dim: int = 2,
        embedding_dim: int = 256,
        is_continuous: bool = True,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
    ):
        super().__init__(action_dim=action_dim, is_continuous=is_continuous)

        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        # Encoder (CNN)
        self.encoder = EndToEndCNNEncoder(embedding_dim=embedding_dim)

        # Actor head: mean output and log_std parameter for Gaussian policy
        self.actor_mean = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),  # continuous velocity control in [-1, 1]
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: state value estimate
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _get_obs_tensor(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs_t = torch.tensor(obs, dtype=torch.float32)
        else:
            obs_t = obs.float()

        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> Tuple[Normal, torch.Tensor]:
        obs_t = self._get_obs_tensor(obs)
        features = self.encoder.embed(obs_t)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(features).squeeze(-1)
        return dist, value

    def act(
        self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            dist, _ = self.forward(obs)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            act_np = action.cpu().numpy()
            if act_np.shape[0] == 1:
                return act_np[0]
            return act_np

    def get_action_and_value(
        self, obs: Union[np.ndarray, torch.Tensor], action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return action, log_prob, entropy, value

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Performs PPO policy and critic update on a rollout batch."""
        self.train()
        obs_t = torch.tensor(np.array(batch["obs"]), dtype=torch.float32)
        actions_t = torch.tensor(np.array(batch["actions"]), dtype=torch.float32)
        old_log_probs_t = torch.tensor(np.array(batch["log_probs"]), dtype=torch.float32)
        advantages_t = torch.tensor(np.array(batch["advantages"]), dtype=torch.float32)
        returns_t = torch.tensor(np.array(batch["returns"]), dtype=torch.float32)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Evaluate current policy
        _, new_log_probs, entropy, new_values = self.get_action_and_value(obs_t, actions_t)

        # Ratios
        ratios = torch.exp(new_log_probs - old_log_probs_t)

        # Surrogate loss
        surr1 = ratios * advantages_t
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.MSELoss()(new_values, returns_t)

        # Entropy loss
        entropy_loss = entropy.mean()

        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
        }
