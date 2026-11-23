"""Unit tests for representation probes and test-time adaptation."""

import numpy as np
import pytest
import torch

from visionrl.config import make_cfg
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes import (
    AdaptedPolicy,
    FeatureBank,
    RefBank,
    centroid_distance,
    clean_feature_stats,
    collect_clean,
    compute_pac_drift,
    gcm_airm_distance,
    grassmann_distance,
    knn_distance,
    score_episode,
    semantic_pool,
)


def test_probe_metrics():
    rng = np.random.default_rng(0)
    Fs, Fc = rng.normal(size=(5, 8)), rng.normal(size=(6, 8))
    d_cent = centroid_distance(Fs, Fc)
    assert d_cent > 0

    sp = torch.randn(2, 4, 8, 8)
    w = torch.ones(2, 1, 8, 8)
    pool = semantic_pool(sp, w)
    assert pool.shape == (2, 4)
    assert torch.allclose(pool, sp.mean((2, 3)))

    X = rng.normal(size=(20, 6))
    assert grassmann_distance(X, X, 3) < 1e-6

    P = rng.normal(size=(4, 3))
    R = rng.normal(size=(10, 3))
    knn_d = knn_distance(P, R, 2)
    assert len(knn_d) == 4


def test_feature_bank_and_tta():
    cfg = make_cfg("smoke", "pursuit")
    agent = DrQAgent(cfg, "drqv2", 0)
    bank = FeatureBank(agent, cfg)
    recs = collect_clean(agent, cfg, "ref", 2)
    ref = RefBank(bank, recs, cfg)

    # Score an episode
    r0, knn_p, feats = score_episode(bank, ref, recs[0], "noise", 0.5, cfg)
    assert "knn_policy" in r0
    assert "maha_policy" in r0
    assert "sem_knn_uniform" in r0

    # TTA
    mu, var, cs = clean_feature_stats(agent, recs, cfg)
    pol_ttt = AdaptedPolicy(agent, cfg, "ttt", (mu, var), cs)
    obs = np.zeros((3 * cfg.n_stack, cfg.img, cfg.img), dtype=np.uint8)
    for _ in range(4):
        pol_ttt.act(obs)
    assert pol_ttt.accepted > 0


def test_pac_drift_properties():
    # Linear policy mapping z to action for analytic verification
    policy = torch.nn.Sequential(
        torch.nn.Linear(8, 2, bias=False)
    )
    z0 = torch.randn(5, 8)
    
    # 1. Identity at zs == z0
    pac_zero = compute_pac_drift(policy, z0, z0)
    assert np.allclose(pac_zero, 0.0)
    
    # 2. Non-zero drift under displacement
    zs = z0 + 0.1 * torch.randn(5, 8)
    pac_val = compute_pac_drift(policy, z0, zs)
    assert (pac_val > 0.0).all()


def test_gcm_airm_distance_properties():
    rng = np.random.default_rng(42)
    d = 4
    A_raw = rng.normal(size=(20, d))
    B_raw = rng.normal(size=(20, d))
    covA = np.cov(A_raw.T) + 1e-3 * np.eye(d)
    covB = np.cov(B_raw.T) + 1e-3 * np.eye(d)

    # 1. Identity: d(A, A) == 0
    assert np.isclose(gcm_airm_distance(covA, covA), 0.0, atol=1e-5)

    # 2. Symmetry: d(A, B) == d(B, A)
    d_ab = gcm_airm_distance(covA, covB)
    d_ba = gcm_airm_distance(covB, covA)
    assert np.isclose(d_ab, d_ba, atol=1e-5)

    # 3. Inversion invariance: d(A^-1, B^-1) == d(A, B)
    invA, invB = np.linalg.inv(covA), np.linalg.inv(covB)
    d_inv = gcm_airm_distance(invA, invB)
    assert np.isclose(d_ab, d_inv, atol=1e-4)
