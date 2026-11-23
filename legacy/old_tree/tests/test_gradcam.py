"""
Unit tests for Grad-CAM and Spatial Attention Engine in Visual RL.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from evaluation.gradcam import GradCAM, compute_spatial_attention_map, overlay_heatmap
from encoders.cnn_encoder import EndToEndCNNEncoder


class TestGradCAM(unittest.TestCase):
    """Tests Grad-CAM hook capture, normalization, and heatmap overlay operations."""

    def setUp(self):
        torch.manual_seed(42)
        self.encoder = EndToEndCNNEncoder(embedding_dim=512)
        self.gradcam = GradCAM(model=self.encoder, target_layer=self.encoder.conv4)

    def tearDown(self):
        self.gradcam.remove_hooks()

    def test_gradcam_shape_and_range(self):
        """Verifies that Grad-CAM produces spatial heatmaps in [0, 1] matching input H, W."""
        dummy_obs = torch.randn(2, 3, 64, 64, requires_grad=True)
        cam = self.gradcam.compute_cam(dummy_obs)

        self.assertEqual(cam.shape, (2, 64, 64))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)

    def test_spatial_attention_map(self):
        """Tests non-parametric spatial activation attention pooling."""
        feature_maps = torch.randn(2, 32, 16, 16)
        att_map = compute_spatial_attention_map(feature_maps, target_hw=(64, 64))

        self.assertEqual(att_map.shape, (2, 64, 64))
        self.assertGreaterEqual(float(att_map.min()), 0.0)
        self.assertLessEqual(float(att_map.max()), 1.0 + 1e-6)

    def test_overlay_heatmap_blending(self):
        """Tests blending of heatmap with RGB image."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        heatmap = np.random.rand(64, 64).astype(np.float32)

        blended = overlay_heatmap(img, heatmap, alpha=0.5, colormap="jet")
        self.assertEqual(blended.shape, (64, 64, 3))
        self.assertEqual(blended.dtype, np.float32)
        self.assertGreaterEqual(blended.min(), 0.0)
        self.assertLessEqual(blended.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
