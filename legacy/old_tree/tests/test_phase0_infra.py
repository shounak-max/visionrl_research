"""
Unit tests for Phase 0 Infrastructure & Baseline Reproduction.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from provenance import compute_config_hash, stamp_provenance, verify_provenance
from envs import make_env, PerturbationConfig, VisualPursuitEnv, ProcgenCoinRunEnv, ProcgenOODEnv
from policies import VisualPursuitExpertPolicy, CoinRunBaselinePolicy
from encoders import make_encoder, FrozenResNet18Encoder, PartialFineTuneResNet18Encoder, EndToEndCNNEncoder
from rollout_collector import RolloutCollector, save_rollout_batch, load_rollout_batch
from evaluation import DistanceEngine, EvaluationEngine


class TestPhase0Infra(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_provenance_utility(self):
        config = {"env": "coinrun", "seed": 42, "model": "resnet18"}
        hash1 = compute_config_hash(config)
        self.assertEqual(len(hash1), 16)

        record = stamp_provenance(config, seed=42, env_id="coinrun", perturbation_tag="clean")
        self.assertTrue(verify_provenance(record))
        self.assertTrue(verify_provenance(record, expected_config=config))

        # Tamper check
        record_tampered = dict(record)
        record_tampered["config_hash"] = "invalid_hash"
        self.assertFalse(verify_provenance(record_tampered))

    def test_environments(self):
        # 1. Visual pursuit
        vp_env = VisualPursuitEnv(seed=42)
        obs, info = vp_env.reset()
        self.assertEqual(obs.shape, (64, 64, 3))
        self.assertIn("ground_truth_mask", info)
        self.assertEqual(info["ground_truth_mask"].shape, (64, 64))

        next_obs, reward, term, trunc, info2 = vp_env.step([0.1, -0.2])
        self.assertEqual(next_obs.shape, (64, 64, 3))
        vp_env.close()

        # 2. Procgen CoinRun
        cr_env = ProcgenCoinRunEnv(seed=42)
        obs_cr, info_cr = cr_env.reset()
        self.assertEqual(obs_cr.shape, (64, 64, 3))
        self.assertIn("ground_truth_mask", info_cr)
        cr_env.close()

        # 3. ProcgenOOD
        ood_env = ProcgenOODEnv(seed=42)
        obs_ood, info_ood = ood_env.reset()
        self.assertEqual(obs_ood.shape, (64, 64, 3))
        ood_env.close()

    def test_encoders_tensor_shapes(self):
        batch_obs = torch.randn(4, 64, 64, 3)

        encoders = [
            FrozenResNet18Encoder(embedding_dim=512),
            PartialFineTuneResNet18Encoder(embedding_dim=512),
            EndToEndCNNEncoder(embedding_dim=512),
        ]

        for enc in encoders:
            emb = enc.embed(batch_obs)
            self.assertEqual(emb.shape, (4, 512), f"Failed for {enc.encoder_id}")

            spatial = enc.embed_spatial(batch_obs)
            self.assertEqual(spatial.ndim, 4, f"Spatial failed for {enc.encoder_id}")
            self.assertEqual(spatial.shape[0], 4)

    def test_rollout_collector_and_caching(self):
        policy = VisualPursuitExpertPolicy()
        p_cfg = PerturbationConfig(axis="photometric", intensity=0.3)
        batch = RolloutCollector.collect(
            env_id="visual_pursuit",
            policy=policy,
            perturbation_config=p_cfg,
            n_episodes=2,
            seed=42,
        )

        self.assertGreater(len(batch), 0)
        self.assertEqual(batch.observations.shape[1:], (64, 64, 3))
        self.assertIsNotNone(batch.ground_truth_mask)

        # Cache roundtrip test
        filepath = save_rollout_batch(batch, cache_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(filepath))

        loaded_batch = load_rollout_batch(filepath)
        self.assertEqual(len(loaded_batch), len(batch))
        np.testing.assert_array_almost_equal(loaded_batch.observations, batch.observations)

    def test_distance_and_correlation_engines(self):
        # Distance test
        phi1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        phi2 = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        dists = DistanceEngine.d_rep(phi1, phi2)
        np.testing.assert_array_almost_equal(dists.numpy(), [4.0, 9.0])

        # Correlation test
        d_metrics = [1.0, 2.0, 3.0, 4.0, 5.0]
        p_drops = [1.5, 2.8, 3.1, 4.0, 5.2]
        res = EvaluationEngine.correlate(d_metrics, p_drops, metric_name="d_rep")
        self.assertGreater(res.spearman_rho, 0.9)
        self.assertGreater(res.pearson_r, 0.9)


if __name__ == "__main__":
    unittest.main()
