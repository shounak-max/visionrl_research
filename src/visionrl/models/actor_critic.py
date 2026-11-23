"""Actor and Critic network definitions."""

from typing import Tuple

import torch
import torch.nn as nn

from visionrl.models.encoder import weight_init


def _noisy_action(mu: torch.Tensor, std: float, clip: float) -> torch.Tensor:
    """Add Gaussian exploration noise with clipping to bounded action range [-1, 1]."""
    noise = (torch.randn_like(mu) * std).clamp(-clip, clip)
    return (mu + noise).clamp(-1.0, 1.0)


class Actor(nn.Module):
    """Deterministic policy network mapping latent representations to continuous actions in [-1, 1]."""

    def __init__(self, repr_dim: int, act_dim: int, feature_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.apply(weight_init)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy(self.trunk(h)))


class Critic(nn.Module):
    """Twin Q-network estimating action values from representation and action."""

    def __init__(self, repr_dim: int, act_dim: int, feature_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        def make_q() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim + act_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()
        self.apply(weight_init)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([self.trunk(h), a], -1)
        return self.q1(z), self.q2(z)
