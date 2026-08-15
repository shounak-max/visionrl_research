"""
Unit tests for Phase 1: Semantic-Weighted Distance Metric (d_tilde_rep) and SemanticWeighters.
"""

import unittest
import torch
import numpy as np

from semantic_weighting import (
    make_semantic_weighter,
    SegmentationWeighter,
    TLDAWeighter,
    SaliencyWeighter,
)
from encoders import FrozenResNet18Encoder, EndToEndCNNEncoder
from evaluation import DistanceEngine


class TestPhase1Semantic(unittest.TestCase):

    def test_semantic_weighters_output_bounds(self):
        obs_clean = torch.rand(2, 64, 64, 3)
        obs_pert = torch.rand(2, 64, 64, 3)
        gt_mask = torch.zeros(2, 64, 64)
        gt_mask[:, 10:20, 10:20] = 1.0

        weighters = [
            SegmentationWeighter(background_floor=0.1),
            TLDAWeighter(background_floor=0.1),
            SaliencyWeighter(background_floor=0.1),
        ]

        for w in weighters:
            ws = w.compute_weights(obs_clean, obs_pert, ground_truth_mask=gt_mask)
            self.assertEqual(ws.shape, (2, 64, 64), f"Failed shape for {w.weighter_id}")
            self.assertTrue(torch.all(ws >= 0.0), f"Lower bound failed for {w.weighter_id}")
            self.assertTrue(torch.all(ws <= 1.0), f"Upper bound failed for {w.weighter_id}")

    def test_d_tilde_rep_regression_to_d_rep(self):
        """Regression test: d_tilde_rep must reduce to classical d_rep when Ws is all-ones (Ws = 1.0)."""
        encoder = EndToEndCNNEncoder(embedding_dim=512)
        obs_clean = torch.randn(3, 64, 64, 3)
        obs_pert = torch.randn(3, 64, 64, 3)

        # Compute pre-GAP spatial feature maps
        feat_clean_spatial = encoder.embed_spatial(obs_clean)
        feat_pert_spatial = encoder.embed_spatial(obs_pert)

        # Compute classical d_rep over global average pooled embeddings
        pooled_clean = torch.mean(feat_clean_spatial, dim=(-2, -1))
        pooled_pert = torch.mean(feat_pert_spatial, dim=(-2, -1))
        d_rep_vals = DistanceEngine.d_rep(pooled_clean, pooled_pert)

        # Compute d_tilde_rep with all-ones weight map Ws = 1.0
        ones_ws = torch.ones(3, 64, 64)
        d_tilde_vals = DistanceEngine.d_tilde_rep(
            feat_clean_spatial, feat_pert_spatial, ones_ws, pooling_order="pre_pooling"
        )

        np.testing.assert_array_almost_equal(
            d_rep_vals.detach().numpy(),
            d_tilde_vals.detach().numpy(),
            decimal=5,
            err_msg="d_tilde_rep failed to reduce to d_rep when Ws is all-ones!",
        )

    def test_background_suppression_effect(self):
        """Verifies that downweighting background shift reduces representation distance relative to clean/hazard shift."""
        encoder = EndToEndCNNEncoder(embedding_dim=512)

        clean_obs = torch.zeros(1, 64, 64, 3)
        # Background shift only (outer region)
        bg_shift_obs = clean_obs.clone()
        bg_shift_obs[:, 40:, 40:, :] = 1.0

        # Spatial feature maps
        feat_clean = encoder.embed_spatial(clean_obs)
        feat_bg_shift = encoder.embed_spatial(bg_shift_obs)

        # Mask focused ONLY on upper left foreground (0s in background shift area)
        fg_mask = torch.zeros(1, 64, 64)
        fg_mask[:, :20, :20] = 1.0
        fg_mask = fg_mask + 0.05 * (1.0 - fg_mask)  # background floor 0.05

        ones_mask = torch.ones(1, 64, 64)

        d_unweighted = DistanceEngine.d_tilde_rep(feat_clean, feat_bg_shift, ones_mask)
        d_weighted = DistanceEngine.d_tilde_rep(feat_clean, feat_bg_shift, fg_mask)

        # Weighted metric should significantly downweight background shift distance
        self.assertLess(
            float(d_weighted.item()),
            float(d_unweighted.item()),
            "Semantic weighting failed to downweight background shift!",
        )


if __name__ == "__main__":
    unittest.main()
