"""
ResNet-18 perception encoders (Frozen ImageNet trunk and partial fine-tuning regime).
"""

import torch
import torch.nn as nn
import torchvision.models as models

from encoders.base_encoder import BaseEncoder


class FrozenResNet18Encoder(BaseEncoder):
    """Frozen ImageNet pre-trained ResNet-18 encoder.

    Extracts 512-dim global embeddings and spatial feature maps [B, 512, H_feat, W_feat].
    Per Stooke et al. (ATC-style) decoupled representation learning.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__(embedding_dim=embedding_dim, encoder_id="frozen_resnet18")
        # Load torchvision ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Extract features trunk up to layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def embed_spatial(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_obs(obs)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
        return x

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_obs(obs)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            pooled = self.avgpool(x)
            flattened = torch.flatten(pooled, 1)
        return flattened


class PartialFineTuneResNet18Encoder(FrozenResNet18Encoder):
    """Partially fine-tuned ResNet-18 encoder per Jones et al. (2025).

    Freezes early layers (conv1 through layer2) while keeping top layers (layer3, layer4) trainable.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__(embedding_dim=embedding_dim, pretrained=pretrained)
        self.encoder_id = "partial_resnet18"

        # Unfreeze layer3 and layer4
        for param in self.layer3.parameters():
            param.requires_grad = True
        for param in self.layer4.parameters():
            param.requires_grad = True
        self.train()

    def embed_spatial(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_obs(obs)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
        return x

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_obs(obs)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
        # Trainable top blocks
        x = self.layer3(x)
        x = self.layer4(x)
        pooled = self.avgpool(x)
        return torch.flatten(pooled, 1)
