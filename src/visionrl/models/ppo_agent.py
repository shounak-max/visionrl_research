"""Discrete action PPO agent for discrete control benchmarking."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from visionrl.config import get_device
from visionrl.models.encoder import Encoder


class PPOAgent(nn.Module):
    """Discrete action policy network and value head."""

    def __init__(self, n_actions: int, in_ch: int, img: int, feature_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(in_ch, img)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.repr_dim, feature_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(feature_dim, n_actions)
        self.v = nn.Linear(feature_dim, 1)
        self.to(get_device())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(self.encoder(x))
        return self.pi(z), self.v(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_u8: np.ndarray, rule: str = "greedy") -> int:
        """rule is explicit: 'greedy' (argmax) or 'sampled'.
        Evaluate and REPORT both (the action rule decides which policy is measured).
        """
        device = next(self.parameters()).device
        logits, _ = self(torch.as_tensor(obs_u8, dtype=torch.float32, device=device).unsqueeze(0))
        if rule == "greedy":
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())
