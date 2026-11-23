"""
Distance Engine for computing representation distances between clean and perturbed observations.

Includes:
- Classical drep: E[||phi(s) - phi(f(s))||_2^2]
- Semantic-weighted d_tilde_rep: E[||Ws * (phi(s) - phi(f(s)))||_2^2] (Phase 1)
  Explicit pooling-order decision: pre-pooling spatial feature map weighting.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceEngine:
    """Distance Engine for computing representation distances."""

    @staticmethod
    def d_rep(
        phi_clean: torch.Tensor,
        phi_perturbed: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Computes classical representation distance: d_rep(s, f(s)) = ||phi(s) - phi(f(s))||_2^2.

        Args:
            phi_clean: Clean global embeddings [B, D].
            phi_perturbed: Perturbed global embeddings [B, D].
            normalize: If True, normalizes distance by embedding dimension D.

        Returns:
            Tensor [B] containing element-wise squared L2 distances.
        """
        if not isinstance(phi_clean, torch.Tensor):
            phi_clean = torch.tensor(phi_clean, dtype=torch.float32)
        if not isinstance(phi_perturbed, torch.Tensor):
            phi_perturbed = torch.tensor(phi_perturbed, dtype=torch.float32)

        diff = phi_clean - phi_perturbed
        sq_dist = torch.sum(diff**2, dim=-1)

        if normalize:
            sq_dist = sq_dist / float(phi_clean.shape[-1])

        return sq_dist

    @staticmethod
    def d_tilde_rep(
        feat_clean_spatial: torch.Tensor,
        feat_perturbed_spatial: torch.Tensor,
        Ws: torch.Tensor,
        pooling_order: str = "pre_pooling",
    ) -> torch.Tensor:
        """Computes semantic-weighted representation distance d_tilde_rep.

        Pooling-order Decision (Pre-Pooling Spatial Feature Map Weighting):
        -------------------------------------------------------------------
        Ws is computed at observation resolution [B, H, W] and interpolated to spatial feature map resolution [B, 1, H', W'].
        Spatial feature activations F(s) [B, C, H', W'] are elementwise multiplied by Ws pre-GAP.
        Global average pooling is performed AFTER Hadamard weighting:
            phi_W(s) = GAP( Ws * F(s) )
            d_tilde_rep = || phi_W(s) - phi_W(f(s)) ||_2^2

        Lemma Property:
            Background/appearance noise in unmasked regions (where Ws -> 0) is zeroed out before spatial averaging,
            preventing background perturbations from polluting the pooled embedding vector.

        Regression Test Equivalence:
            When Ws is all-ones (Ws = 1.0), d_tilde_rep reduces exactly to classical d_rep over global average pooled embeddings.
        """
        if not isinstance(feat_clean_spatial, torch.Tensor):
            feat_clean_spatial = torch.tensor(feat_clean_spatial, dtype=torch.float32)
        if not isinstance(feat_perturbed_spatial, torch.Tensor):
            feat_perturbed_spatial = torch.tensor(feat_perturbed_spatial, dtype=torch.float32)
        if not isinstance(Ws, torch.Tensor):
            Ws = torch.tensor(Ws, dtype=torch.float32)

        # Handle dimension shapes
        if Ws.ndim == 2:
            Ws = Ws.unsqueeze(0)  # [1, H, W]
        if Ws.ndim == 3:
            Ws = Ws.unsqueeze(1)  # [B, 1, H, W]

        # Handle cases where feat tensors are already 2D pooled embeddings [B, D]
        if feat_clean_spatial.ndim == 2 and feat_perturbed_spatial.ndim == 2:
            B, D = feat_clean_spatial.shape
            Ws_flat = F.adaptive_avg_pool2d(Ws, (1, 1)).view(B, 1)
            diff = (feat_clean_spatial - feat_perturbed_spatial) * Ws_flat
            return torch.sum(diff**2, dim=-1)

        # Spatial feature tensor shapes [B, C, H', W']
        B, C, H_feat, W_feat = feat_clean_spatial.shape

        # Downsample / interpolate Ws from [B, 1, H, W] -> [B, 1, H', W']
        Ws_resized = F.interpolate(
            Ws, size=(H_feat, W_feat), mode="bilinear", align_corners=False
        )

        if pooling_order == "pre_pooling":
            # Apply normalized spatial feature map weighting pre-GAP
            sum_w = torch.sum(Ws_resized, dim=(-2, -1), keepdim=True) + 1e-8
            pooled_clean = (
                torch.sum(feat_clean_spatial * Ws_resized, dim=(-2, -1), keepdim=True)
                / sum_w
            ).squeeze(-1).squeeze(-1)
            pooled_pert = (
                torch.sum(feat_perturbed_spatial * Ws_resized, dim=(-2, -1), keepdim=True)
                / sum_w
            ).squeeze(-1).squeeze(-1)

            # Compute squared L2 distance over weighted pooled embeddings
            diff = pooled_clean - pooled_pert
            sq_dist = torch.sum(diff**2, dim=-1)
            return sq_dist

        elif pooling_order == "direct_spatial":
            # Direct spatial sum of weighted feature differences normalized by total weight
            diff_sq = torch.sum(
                (feat_clean_spatial - feat_perturbed_spatial) ** 2, dim=1, keepdim=True
            )
            weighted_diff_sq = diff_sq * Ws_resized
            sum_w = torch.sum(Ws_resized, dim=(-3, -2, -1)) + 1e-8
            return torch.sum(weighted_diff_sq, dim=(-3, -2, -1)) / sum_w

        else:
            raise ValueError(f"Unknown pooling_order: '{pooling_order}'")
