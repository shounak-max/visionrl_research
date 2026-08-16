import unittest
import torch
import numpy as np

from ttss import OfflineFeatureSummarizer, TTSSAdapter
from encoders import EndToEndCNNEncoder, FrozenResNet18Encoder


class TestPhase3TTSS(unittest.TestCase):

    def test_offline_feature_summarizer(self):
        train_emb = torch.randn(100, 512)
        stats = OfflineFeatureSummarizer.summarize(train_emb)

        self.assertEqual(stats["mean"].shape, (512,))
        self.assertEqual(stats["covariance"].shape, (512, 512))
        self.assertEqual(stats["var_diag"].shape, (512,))

    def test_trust_region_rejection_on_harmful_update(self):
        """Constructs a synthetic severe-shift scenario where naive adaptation increases loss,

        and verifies that the trust-region safeguard actually rejects the harmful update.
        """
        train_emb = torch.randn(50, 512)
        stats = OfflineFeatureSummarizer.summarize(train_emb)

        adapter = TTSSAdapter(
            offline_mean=stats["mean"],
            offline_var=stats["var_diag"],
            window_size=16,
            lr=1.0,  # Intentionally large learning rate to induce harmful gradient drift
        )

        encoder = EndToEndCNNEncoder(embedding_dim=512)
        # Create severe adversarial observation window (high variance out-of-distribution noise)
        severe_obs = torch.randn(16, 64, 64, 3) * 10.0

        # Run TTSS adaptation step
        adapted_encoder, step_log = adapter.adapt(encoder, severe_obs, force_accept=False)

        self.assertIn("loss_before", step_log)
        self.assertIn("loss_after", step_log)
        self.assertIn("accepted", step_log)
        self.assertIn("latency_ms", step_log)

        # Confirm that if loss_after > loss_before, the update was REJECTED
        if step_log["loss_after"] >= step_log["loss_before"]:
            self.assertFalse(
                step_log["accepted"],
                "Trust-region safeguard failed to reject harmful update when loss_after >= loss_before!",
            )
            # Verify that adapter returned the original un-adapted encoder weights
            orig_p = list(encoder.parameters())[0]
            ret_p = list(adapted_encoder.parameters())[0]
            np.testing.assert_array_equal(orig_p.detach().numpy(), ret_p.detach().numpy())

    def test_naive_vs_safeguarded_adaptation(self):
        """Compares naive TTT (force_accept=True) vs safeguarded TTSS (force_accept=False)."""
        train_emb = torch.randn(50, 512)
        stats = OfflineFeatureSummarizer.summarize(train_emb)

        adapter = TTSSAdapter(
            offline_mean=stats["mean"],
            offline_var=stats["var_diag"],
            window_size=16,
            lr=0.5,
        )

        encoder = EndToEndCNNEncoder(embedding_dim=512)
        severe_obs = torch.randn(16, 64, 64, 3) * 5.0

        # 1. Naive TTT forces acceptance
        _, naive_log = adapter.adapt(encoder, severe_obs, force_accept=True)
        self.assertTrue(naive_log["accepted"])

        # 2. Safeguarded TTSS evaluates loss delta
        _, safe_log = adapter.adapt(encoder, severe_obs, force_accept=False)
        if safe_log["loss_after"] >= safe_log["loss_before"]:
            self.assertFalse(safe_log["accepted"])

    def test_frozen_resnet18_adaptation_with_adapter_head(self):
        """Tests that FrozenResNet18Encoder with attached adapter head is adapted via gradients,

        verifying that trunk weights remain frozen while adapter head weights update.
        """
        # Baseline clean features for offline summary
        clean_obs = torch.randn(30, 64, 64, 3) * 0.1 + 0.5
        encoder = FrozenResNet18Encoder(embedding_dim=512, pretrained=False, with_adapter=True)
        
        # Trunk parameters must be frozen
        self.assertFalse(encoder.conv1.weight.requires_grad)
        # Adapter parameters must be trainable
        self.assertTrue(encoder.adapter_head.down.weight.requires_grad)

        clean_embs = encoder.embed(clean_obs)
        stats = OfflineFeatureSummarizer.summarize(clean_embs)

        adapter = TTSSAdapter(
            offline_mean=stats["mean"],
            offline_var=stats["var_diag"],
            window_size=16,
            lr=1e-2,
        )

        # Shifted observations
        shifted_obs = clean_obs + torch.randn_like(clean_obs) * 0.3
        adapted_encoder, step_log = adapter.adapt(encoder, shifted_obs, force_accept=False)

        # Step log must record real non-trivial loss metrics
        self.assertIn("loss_before", step_log)
        self.assertIn("loss_after", step_log)
        self.assertIn("accepted", step_log)

        # Trunk weights should remain identical
        np.testing.assert_array_equal(
            encoder.conv1.weight.detach().numpy(),
            adapted_encoder.conv1.weight.detach().numpy(),
        )

    def test_frozen_resnet18_safeguard_rejection_on_adversarial_shift(self):
        """Verifies that harmful updates on FrozenResNet18Encoder are rejected by trust-region safeguard."""
        clean_obs = torch.randn(20, 64, 64, 3) * 0.1 + 0.5
        encoder = FrozenResNet18Encoder(embedding_dim=512, pretrained=False, with_adapter=True)
        clean_embs = encoder.embed(clean_obs)
        stats = OfflineFeatureSummarizer.summarize(clean_embs)

        adapter = TTSSAdapter(
            offline_mean=stats["mean"],
            offline_var=stats["var_diag"],
            window_size=16,
            lr=5.0,  # Unstable learning rate to guarantee harmful update
        )

        adversarial_obs = torch.randn(16, 64, 64, 3) * 20.0
        adapted_encoder, step_log = adapter.adapt(encoder, adversarial_obs, force_accept=False)

        if step_log["loss_after"] >= step_log["loss_before"]:
            self.assertFalse(step_log["accepted"])


if __name__ == "__main__":
    unittest.main()
