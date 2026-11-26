"""
Unit tests for scores/new_scores.py — PAC-Drift (Score 10) and GCM-Drift (Score 11).

Required tests:
  - test_pac_drift_zero_on_identical_frames: PAC-Drift == 0 when clean == shifted
  - test_gcm_drift_symmetry: d(A, B) == d(B, A)
  - test_gcm_drift_zero_on_identical: GCM-Drift == 0 when Sigma_0 == Sigma_s
  - test_gcm_drift_affine_invariance: d(W@A@W.T, W@B@W.T) == d(A, B)
  - test_pac_lambda0_vs_lambda1: lambda=0 <= lambda=1
"""

import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import torch.nn as nn

from scores.new_scores import GCMDriftScore, PACDriftScore


# ---------------------------------------------------------------------------
# Minimal stub policy for testing (avoids needing a full DrQv2Policy)
# ---------------------------------------------------------------------------

class _StubEncoder(nn.Module):
    """Tiny encoder: maps [B, C, H, W] -> [B, embed_dim]."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 8):
        super().__init__()
        self.embedding_dim = embed_dim
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, embed_dim)

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize to [B, C, H, W] float in [0, 1]."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] in [1, 3] and obs.shape[1] not in [1, 3]:
            obs = obs.permute(0, 3, 1, 2)
        if obs.max() > 1.5:
            obs = obs / 255.0
        return obs

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(obs))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class _StubPolicy(nn.Module):
    """Minimal policy with encoder + actor for testing."""

    def __init__(self, embed_dim: int = 8, action_dim: int = 2):
        super().__init__()
        self.encoder = _StubEncoder(embed_dim=embed_dim)
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 4),
            nn.Tanh(),
            nn.Linear(4, action_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.actor(z)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_policy():
    policy = _StubPolicy(embed_dim=8, action_dim=2)
    policy.eval()
    return policy


def _make_frames(n: int = 4, h: int = 16, w: int = 16, c: int = 3, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.rand(n, c, h, w, generator=rng)


# ---------------------------------------------------------------------------
# PAC-Drift Tests
# ---------------------------------------------------------------------------

class TestPACDrift:

    def test_pac_drift_zero_on_identical_frames(self, stub_policy):
        """PAC-Drift must be 0 (< 1e-6) when clean_frames == shifted_frames.

        When dz = encoder(shifted) - encoder(clean) = 0, every term is 0.
        """
        scorer = PACDriftScore()
        frames = _make_frames(n=4, seed=42)
        # Pass the SAME frames as both clean and shifted
        score = scorer.compute(frames, frames.clone(), stub_policy, lambda_val=1.0)
        assert score < 1e-6, (
            f"PAC-Drift on identical frames should be < 1e-6, got {score:.2e}"
        )

    def test_pac_lambda0_vs_lambda1(self, stub_policy):
        """lambda=0 gives score <= lambda=1 (Hessian term is non-negative additive).

        When lambda=0, only first-order term contributes.
        When lambda=1, we also add 0.5 * ||dz^T H dz||_2 >= 0.
        """
        scorer = PACDriftScore()
        clean = _make_frames(n=4, seed=1)
        shifted = _make_frames(n=4, seed=2)  # genuinely different frames

        score_lam0 = scorer.compute(clean, shifted, stub_policy, lambda_val=0.0)
        score_lam1 = scorer.compute(clean, shifted, stub_policy, lambda_val=1.0)

        # lambda=0 score should be <= lambda=1 score (second term adds non-negative value)
        # Allow tiny floating-point tolerance
        assert score_lam0 <= score_lam1 + 1e-9, (
            f"lambda=0 score ({score_lam0:.6f}) should be <= lambda=1 score ({score_lam1:.6f})"
        )

    def test_pac_drift_nonzero_on_different_frames(self, stub_policy):
        """PAC-Drift should be > 0 for genuinely different frames (sanity check)."""
        scorer = PACDriftScore()
        clean = _make_frames(n=4, seed=10)
        # Shifted frames with large difference to guarantee non-zero dz
        shifted = torch.ones_like(clean)  # all-ones vs random

        score = scorer.compute(clean, shifted, stub_policy, lambda_val=1.0)
        # With a non-trivial dz and non-trivial Jacobian, score must be strictly positive
        assert score > 0.0, f"PAC-Drift must be > 0 on distinct frames, got {score}"


# ---------------------------------------------------------------------------
# GCM-Drift Tests
# ---------------------------------------------------------------------------

class TestGCMDrift:

    def test_gcm_drift_zero_on_identical(self, stub_policy):
        """GCM-Drift == 0 when Sigma_0 == Sigma_s (clean == shifted frames)."""
        scorer = GCMDriftScore()
        frames = _make_frames(n=16, seed=99)  # need enough samples for covariance
        score = scorer.compute(frames, frames.clone(), stub_policy)
        assert score < 1e-5, (
            f"GCM-Drift on identical frames should be ~0, got {score:.2e}"
        )

    def test_gcm_drift_symmetry(self, stub_policy):
        """d(A, B) == d(B, A) — AIRM is symmetric."""
        scorer = GCMDriftScore()
        clean = _make_frames(n=16, seed=3)
        shifted = _make_frames(n=16, seed=7)

        score_AB = scorer.compute(clean, shifted, stub_policy)
        score_BA = scorer.compute(shifted, clean, stub_policy)

        assert abs(score_AB - score_BA) < 1e-5, (
            f"GCM-Drift symmetry violated: d(A,B)={score_AB:.6f}, d(B,A)={score_BA:.6f}"
        )

    def test_gcm_drift_affine_invariance(self):
        """d(W@A@W.T, W@B@W.T) == d(A, B) for invertible W.

        Tests the AIRM property directly on covariance matrices.
        """
        scorer = GCMDriftScore()
        torch.manual_seed(42)
        D = 6

        # Create two random SPD matrices A, B
        def random_spd(d: int, seed: int) -> torch.Tensor:
            torch.manual_seed(seed)
            X = torch.randn(d, d)
            return X @ X.T + 1e-3 * torch.eye(d)

        A = random_spd(D, 10)
        B = random_spd(D, 20)

        # Create an invertible W (random well-conditioned matrix)
        torch.manual_seed(5)
        W = torch.randn(D, D)
        # Make W well-conditioned by adding a multiple of identity
        W = W + 2.0 * torch.eye(D)

        A_transformed = W @ A @ W.T
        B_transformed = W @ B @ W.T

        d_original = scorer.airm_distance(A, B)
        d_transformed = scorer.airm_distance(A_transformed, B_transformed)

        assert abs(d_original - d_transformed) < 1e-4, (
            f"AIRM affine invariance violated: d(A,B)={d_original:.6f}, "
            f"d(WAW^T,WBW^T)={d_transformed:.6f}, diff={abs(d_original - d_transformed):.2e}"
        )

    def test_gcm_drift_nonnegative(self, stub_policy):
        """GCM-Drift must be non-negative."""
        scorer = GCMDriftScore()
        clean = _make_frames(n=16, seed=11)
        shifted = _make_frames(n=16, seed=22)
        score = scorer.compute(clean, shifted, stub_policy)
        assert score >= 0.0, f"GCM-Drift must be non-negative, got {score}"


# ---------------------------------------------------------------------------
# Integration Tests with Full DrQv2Policy
# ---------------------------------------------------------------------------

class TestRealPolicyIntegration:

    def test_pac_drift_with_drq_v2_policy(self):
        """Test PAC-Drift on full DrQv2 policy architecture."""
        from visionrl.config import make_cfg
        from visionrl.models.drq_agent import DrQAgent
        cfg = make_cfg("smoke", "pursuit")
        agent = DrQAgent(cfg, "drqv2", 0)
        agent.encoder.eval()
        agent.actor.eval()

        clean = torch.rand(2, 9, 32, 32)
        shifted = torch.rand(2, 9, 32, 32)

        scorer = PACDriftScore()
        ident_score = scorer.compute(clean, clean, agent)
        assert ident_score < 1e-6, f"Expected near zero, got {ident_score}"

        score_l0 = scorer.compute(clean, shifted, agent, lambda_val=0.0)
        score_l1 = scorer.compute(clean, shifted, agent, lambda_val=1.0)
        assert score_l0 > 0.0
        assert score_l1 > score_l0

        # Also verify compute_pac_drift with Actor.policy wrapped in torch.tanh has l1 > l0
        from visionrl.probes.metrics import compute_pac_drift

        class _TanhPolicy(nn.Module):
            def __init__(self, pol):
                super().__init__()
                self.pol = pol

            def forward(self, z):
                return torch.tanh(self.pol(z))

        with torch.no_grad():
            z0 = agent.actor.trunk(agent.encoder(clean))
            zs = agent.actor.trunk(agent.encoder(shifted))
        wrapped = _TanhPolicy(agent.actor.policy)
        pac_l0 = compute_pac_drift(wrapped, z0, zs, lambda_curv=0.0)
        pac_l1 = compute_pac_drift(wrapped, z0, zs, lambda_curv=1.0)
        assert np.all(pac_l1 > pac_l0), f"Expected pac_l1 > pac_l0 with tanh, got {pac_l0} vs {pac_l1}"

    def test_encoder_preprocess_obs_hwc(self):
        """Encoder.preprocess_obs must permute [N, H, W, C] and [H, W, C] to [N, C, H, W]."""
        from visionrl.models.encoder import Encoder

        enc = Encoder(in_ch=9, img=32)
        nhwc = torch.randint(0, 256, (2, 32, 32, 9), dtype=torch.uint8)
        out_4d = enc.preprocess_obs(nhwc)
        assert out_4d.shape == (2, 9, 32, 32)
        assert out_4d.dtype == torch.float32

        hwc = torch.randint(0, 256, (32, 32, 9), dtype=torch.uint8)
        out_3d = enc.preprocess_obs(hwc)
        assert out_3d.shape == (1, 9, 32, 32)
        assert out_3d.dtype == torch.float32

        # End-to-end with PACDriftScore on [N, H, W, C]
        from visionrl.config import make_cfg
        from visionrl.models.drq_agent import DrQAgent

        cfg = make_cfg("smoke", "pursuit")
        agent = DrQAgent(cfg, "drqv2", 0)
        agent.encoder.eval()
        agent.actor.eval()
        clean_hwc = torch.rand(2, 32, 32, 9) * 255.0
        shifted_hwc = torch.rand(2, 32, 32, 9) * 255.0
        score = PACDriftScore().compute(clean_hwc, shifted_hwc, agent, lambda_val=1.0)
        assert score > 0.0

    def test_gcm_drift_with_drq_v2_policy(self):
        """Test GCM-Drift on full DrQv2 policy architecture."""
        from visionrl.config import make_cfg
        from visionrl.models.drq_agent import DrQAgent
        cfg = make_cfg("smoke", "pursuit")
        agent = DrQAgent(cfg, "drqv2", 0)
        agent.encoder.eval()

        frames = torch.rand(8, 9, 32, 32)
        scorer = GCMDriftScore()
        ident_score = scorer.compute(frames, frames, agent)
        assert ident_score < 1e-5, f"Expected near zero, got {ident_score}"

