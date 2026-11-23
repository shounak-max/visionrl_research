import unittest
import torch
import numpy as np
from policies.taco import TACOPolicy


class TestTACOPolicy(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 512
        self.action_dim = 2
        self.policy = TACOPolicy(
            action_dim=self.action_dim,
            embedding_dim=self.embedding_dim,
            is_continuous=True,
            beta=0.1,
        )

    def test_initialization(self):
        self.assertEqual(self.policy.beta, 0.1)
        self.assertEqual(self.policy.proj_dim, 128)
        self.assertIsNotNone(self.policy.projection_head)
        self.assertIsNotNone(self.policy.transition_model)

    def test_contrastive_loss(self):
        batch_size = 8
        proj_dim = 128
        z_anchor = torch.randn(batch_size, proj_dim)
        z_pos = torch.randn(batch_size, proj_dim)
        all_emb = torch.randn(batch_size, proj_dim)
        loss = self.policy.temporal_contrastive_loss(z_anchor, z_pos, all_emb)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_update_step(self):
        batch_size = 4
        batch = {
            "obs": torch.randn(batch_size, 64, 64, 3),
            "action": torch.randn(batch_size, self.action_dim),
            "reward": torch.randn(batch_size, 1),
            "next_obs": torch.randn(batch_size, 64, 64, 3),
            "done": torch.zeros(batch_size, 1),
        }
        metrics = self.policy.update(batch)
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("contrastive_loss", metrics)
        self.assertIn("total_loss", metrics)
        self.assertTrue(np.isfinite(metrics["total_loss"]))


if __name__ == "__main__":
    unittest.main()
