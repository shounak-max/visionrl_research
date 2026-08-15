"""
End-to-End CNN Convolutional Encoder (DrQ-v2 / SVEA style architecture).
"""

import torch
import torch.nn as nn

from encoders.base_encoder import BaseEncoder


class EndToEndCNNEncoder(BaseEncoder):
    """End-to-end standard 4-layer convolutional trunk used in DrQ-v2 and SVEA.

    Output embedding dimension: 512.
    Exposes spatial feature maps [B, 32, H_spatial, W_spatial] before pooling.
    """

    def __init__(self, embedding_dim: int = 512, in_channels: int = 3):
        super().__init__(embedding_dim=embedding_dim, encoder_id="end2end_cnn")

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, embedding_dim)

    def embed_spatial(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_obs(obs)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        spatial = self.embed_spatial(obs)
        pooled = self.avgpool(spatial)
        flattened = torch.flatten(pooled, 1)
        return self.fc(flattened)
