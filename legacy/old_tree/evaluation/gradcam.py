"""
Grad-CAM and Spatial Attention Heatmap Engine for Visual Reinforcement Learning.

Provides gradient-weighted activation mapping (Selvaraju et al., 2017) and spatial attention
heatmaps for policy actors, value critics, and perception encoders under distribution shifts.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for Visual RL networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.clone().detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone().detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_scalar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes normalized Grad-CAM heatmap [B, H, W] for the given input tensor."""
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # Ensure input tensor requires grad
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)

        if target_scalar is not None:
            score = target_scalar
        elif target_fn is not None:
            score = target_fn(output)
        else:
            # Default target: L2 norm or sum of outputs
            score = output.norm(p=2, dim=-1).sum() if output.dim() > 1 else output.sum()

        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check target_layer placement.")

        # Global average pooling of gradients: [B, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(-2, -1), keepdim=True)

        # Weighted combination of activation maps: [B, H_feat, W_feat]
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)

        # Upsample to input resolution
        _, _, H, W = input_tensor.shape
        cam = F.interpolate(
            cam.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)

        # Per-sample normalization to [0, 1]
        b_min = cam.amin(dim=(-2, -1), keepdim=True)
        b_max = cam.amax(dim=(-2, -1), keepdim=True)
        norm_cam = (cam - b_min) / (b_max - b_min + 1e-8)

        return norm_cam.detach()


def compute_spatial_attention_map(feature_maps: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Computes non-parametric spatial activation attention map [B, H, W] from convolutional features."""
    # Mean activation magnitude across channels
    att = torch.mean(torch.abs(feature_maps), dim=1, keepdim=True)
    att = F.interpolate(att, size=target_hw, mode="bilinear", align_corners=False).squeeze(1)
    b_min = att.amin(dim=(-2, -1), keepdim=True)
    b_max = att.amax(dim=(-2, -1), keepdim=True)
    return (att - b_min) / (b_max - b_min + 1e-8)


def overlay_heatmap(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlays a 2D float heatmap [H, W] in [0, 1] onto an RGB image [H, W, 3] in [0, 1] or [0, 255].
    
    Returns:
        np.ndarray [H, W, 3] float32 in [0, 1].
    """
    if img_rgb.dtype == np.uint8 or img_rgb.max() > 1.0:
        img_rgb = img_rgb.astype(np.float32) / 255.0
    else:
        img_rgb = img_rgb.astype(np.float32)

    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
    except Exception:
        cmap = cm.get_cmap(colormap)
        
    heatmap_colored = cmap(heatmap)[:, :, :3].astype(np.float32)  # Drop alpha channel -> [H, W, 3]

    overlay = (1.0 - alpha) * img_rgb + alpha * heatmap_colored
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)
