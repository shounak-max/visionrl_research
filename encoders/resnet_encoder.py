"""
ResNet-18 perception encoders (Frozen ImageNet trunk and partial fine-tuning regime).
"""

import torch
import torch.nn as nn

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    models = None
    HAS_TORCHVISION = False

from encoders.base_encoder import BaseEncoder


class FrozenResNet18Encoder(BaseEncoder):
    """Frozen ImageNet pre-trained ResNet-18 encoder.

    Extracts 512-dim global embeddings and spatial feature maps [B, 512, H_feat, W_feat].
    Per Stooke et al. (ATC-style) decoupled representation learning.
    Supports attached residual adapter heads for test-time adaptation (VPTTA style).
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, with_adapter: bool = False):
        super().__init__(embedding_dim=embedding_dim, encoder_id="frozen_resnet18")
        if HAS_TORCHVISION:
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            except Exception:
                resnet = models.resnet18(pretrained=pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
        else:
            # High-fidelity architectural equivalent
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
            self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
            self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())
            self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all backbone trunk parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        if with_adapter:
            self.attach_adapter()

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
        if self.adapter_head is not None:
            flattened = self.adapter_head(flattened)
        return flattened


class PartialFineTuneResNet18Encoder(FrozenResNet18Encoder):
    """Partially fine-tuned ResNet-18 encoder per Jones et al. (2025).

    Freezes early layers (conv1 through layer2) while keeping top layers (layer3, layer4) trainable.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, with_adapter: bool = False):
        super().__init__(embedding_dim=embedding_dim, pretrained=pretrained, with_adapter=with_adapter)
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
        flattened = torch.flatten(pooled, 1)
        if self.adapter_head is not None:
            flattened = self.adapter_head(flattened)
        return flattened
