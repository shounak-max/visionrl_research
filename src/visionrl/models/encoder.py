"""Convolutional encoder and weight initialization."""

import numpy as np
import torch
import torch.nn as nn


def weight_init(m: nn.Module) -> None:
    """Orthogonal weight initialization for Linear and Conv2d modules."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        g = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, g)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    """DrQ-v2 convolutional encoder.
    `spatial` exposes the feature map used by the semantic probe.
    Input obs: float tensor in [0, 255].
    """

    def __init__(self, in_ch: int, img: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            self.repr_dim = int(np.prod(self.convs(torch.zeros(1, in_ch, img, img)).shape[1:]))
        self.apply(weight_init)

    def spatial(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: float in [0, 255] -> spatial conv features."""
        return self.convs(obs / 255.0 - 0.5)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Flattened representation vector."""
        return self.spatial(obs).flatten(1)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias for forward representation embedding."""
        return self(obs)

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is on correct device and dtype."""
        device = next(self.parameters()).device
        return torch.as_tensor(obs, dtype=torch.float32, device=device)
