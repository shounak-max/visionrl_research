"""Visual data augmentations for pixel-based RL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShift(nn.Module):
    """DrQ-v2 random shift augmentation (replicate pad, then per-sample random crop via grid_sample)."""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        if self.pad <= 0:
            return x
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device, dtype=x.dtype) * (2.0 / (h + 2 * self.pad))
        return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


def random_overlay(x: torch.Tensor) -> torch.Tensor:
    """SVEA strong augmentation: blend each sample with a smooth random field.
    (The original SVEA overlays natural images from Places365; that dataset is not available offline,
    so a smoothed random field is used instead. State this substitution in the paper.)
    x: float [N, C, H, W] in [0, 255].
    """
    n, c, h, w = x.shape
    low = torch.rand(n, 3, max(h // 8, 2), max(w // 8, 2), device=x.device)
    field = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False) * 255.0
    field = field.repeat(1, c // 3, 1, 1)
    alpha = 0.2 + 0.3 * torch.rand(n, 1, 1, 1, device=x.device)
    return (1.0 - alpha) * x + alpha * field
