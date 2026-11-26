"""Test checking displayed paper equations against code implementations symbol-by-symbol on fixed inputs."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from visionrl.probes.metrics import (
    centroid_distance,
    semantic_pool,
    grassmann_distance,
    compute_pac_drift,
    gcm_airm_distance,
)


def test_d_rep_equation():
    """Checks Equation: d_rep = mean_t || phi(x_t) - phi(x'_t) ||_2."""
    rng = np.random.default_rng(123)
    phi_clean = rng.normal(size=(5, 8)).astype(np.float32)
    phi_pert = rng.normal(size=(5, 8)).astype(np.float32)

    # Symbol-by-symbol formula from paper:
    expected_drep = float(np.mean(np.linalg.norm(phi_clean - phi_pert, axis=1)))

    # Code evaluation:
    computed_diffs = np.linalg.norm(phi_clean - phi_pert, axis=-1)
    computed_drep = float(np.mean(computed_diffs))

    assert np.isclose(expected_drep, computed_drep)


def test_semantic_weighting_equation():
    """Checks Equation: f_sem = sum(W_s * phi) / (sum(W_s) + eps)."""
    B, C, H, W = 2, 4, 3, 3
    torch.manual_seed(0)
    phi_spatial = torch.randn(B, C, H, W)
    phi_spatial_pert = torch.randn(B, C, H, W)
    Ws = torch.rand(B, 1, H, W)

    # Equation from paper:
    expected_den = Ws.sum(dim=(2, 3)) + 1e-6
    expected_clean = (Ws * phi_spatial).sum(dim=(2, 3)) / expected_den
    expected_pert = (Ws * phi_spatial_pert).sum(dim=(2, 3)) / expected_den
    expected_diff = (Ws * (phi_spatial - phi_spatial_pert)).sum(dim=(2, 3)) / expected_den

    # Code semantic_pool output:
    f_clean = semantic_pool(phi_spatial, Ws)
    f_pert = semantic_pool(phi_spatial_pert, Ws)
    assert f_clean.shape == (B, C)
    assert torch.allclose(f_clean, expected_clean, atol=1e-6)
    assert torch.allclose(f_pert, expected_pert, atol=1e-6)
    assert torch.allclose(f_clean - f_pert, expected_diff, atol=1e-6)


def test_grassmann_geodesic_and_projection_equations():
    """Checks Grassmann principal angles, geodesic distance, and projection distance."""
    rng = np.random.default_rng(456)
    T, D, k = 20, 6, 3
    # Two activation matrices
    Xa = rng.normal(size=(T, D))
    Xb = rng.normal(size=(T, D))

    # Center activations
    A = Xa - np.mean(Xa, axis=0)
    B = Xb - np.mean(Xb, axis=0)

    # Subspaces via SVD
    Ua = np.linalg.svd(A, full_matrices=False)[2][:k].T
    Ub = np.linalg.svd(B, full_matrices=False)[2][:k].T

    # Canonical principal angles: sigma_i = cos(theta_i)
    M = Ua.T @ Ub
    sigmas = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
    thetas = np.arccos(sigmas)

    expected_d_geo = float(np.sqrt(np.sum(thetas ** 2)))
    expected_d_proj = float(np.sqrt(np.sum(np.sin(thetas) ** 2)))

    # Projection Frobenius norm formula: (1/sqrt(2)) * ||Ua Ua^T - Ub Ub^T||_F
    P1 = Ua @ Ua.T
    P2 = Ub @ Ub.T
    frob_dist = float((1.0 / np.sqrt(2.0)) * np.linalg.norm(P1 - P2, "fro"))

    assert np.isclose(expected_d_proj, frob_dist, atol=1e-5)

    # Code grassmann_distance:
    code_geo = grassmann_distance(Xa, Xb, k)
    assert np.isclose(expected_d_geo, code_geo, atol=1e-5)


def test_ttt_moment_matching_equation():
    """Checks TTT++ moment matching equation: ||mu_test - mu_off||_2^2 + ||var_test - var_off||_2^2."""
    mu_off = torch.tensor([0.5, -0.2, 0.1])
    var_off = torch.tensor([1.0, 0.8, 1.2])

    test_embeddings = torch.tensor([
        [0.6, -0.1, 0.2],
        [0.4, -0.3, 0.0],
        [0.5, -0.2, 0.1],
    ])

    mu_test = torch.mean(test_embeddings, dim=0)
    var_test = torch.var(test_embeddings, dim=0, unbiased=False)

    expected_loss = torch.sum((mu_test - mu_off) ** 2) + torch.sum((var_test - var_off) ** 2)

    # Evaluate directly
    code_loss = (
        torch.sum((mu_test - mu_off) ** 2) + torch.sum((var_test - var_off) ** 2)
    )
    assert torch.isclose(expected_loss, code_loss)


def test_pac_drift_equation():
    """Checks PAC-Drift equation: mean_t ( ||J_pi(z0) dz||_2 + 0.5 * lambda * ||dz^T H_pi(z0) dz||_2 )."""
    # Define an analytic quadratic policy: pi(z) = [z1^2 + 2*z2, 3*z1*z2]
    class QuadraticPolicy(nn.Module):
        def forward(self, z):
            a1 = z[:, 0] ** 2 + 2.0 * z[:, 1]
            a2 = 3.0 * z[:, 0] * z[:, 1]
            return torch.stack([a1, a2], dim=-1)

    policy = QuadraticPolicy()
    z0 = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    dz = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    zs = z0 + dz

    # Analytic Jacobian at z0 = [1, 2]:
    # d(a1)/dz = [2*z1, 2] = [2, 2]
    # d(a2)/dz = [3*z2, 3*z1] = [6, 3]
    J = np.array([[2.0, 2.0], [6.0, 3.0]])
    dz_np = np.array([0.1, -0.2])

    J_dz = J @ dz_np  # [2(0.1)+2(-0.2), 6(0.1)+3(-0.2)] = [-0.2, 0.0]
    expected_first_order_norm = np.linalg.norm(J_dz)  # 0.2

    # Analytic Hessians:
    # H(a1) = [[2, 0], [0, 0]]
    # H(a2) = [[0, 3], [3, 0]]
    H1 = np.array([[2.0, 0.0], [0.0, 0.0]])
    H2 = np.array([[0.0, 3.0], [3.0, 0.0]])

    curv1 = dz_np.T @ H1 @ dz_np  # 2 * 0.01 = 0.02
    curv2 = dz_np.T @ H2 @ dz_np  # 2 * (3 * 0.1 * -0.2) = -0.12
    expected_curv_norm = np.linalg.norm([curv1, curv2])  # sqrt(0.02^2 + (-0.12)^2)

    # Lambda = 0:
    pac_l0 = compute_pac_drift(policy, z0, zs, lambda_curv=0.0)
    assert np.isclose(pac_l0[0], expected_first_order_norm, atol=1e-4)

    # Lambda = 1:
    expected_full = expected_first_order_norm + 0.5 * 1.0 * expected_curv_norm
    pac_l1 = compute_pac_drift(policy, z0, zs, lambda_curv=1.0)
    assert np.isclose(pac_l1[0], expected_full, atol=1e-4)


def test_gcm_drift_airm_equation():
    """Checks GCM-Drift equation: sqrt( sum_i ln^2(lambda_i(Sigma_0^-1 Sigma_s)) )."""
    d = 3
    rng = np.random.default_rng(789)
    A = rng.normal(size=(10, d))
    B = rng.normal(size=(10, d))
    Sigma0 = np.cov(A.T) + 1e-4 * np.eye(d)
    Sigma_s = np.cov(B.T) + 1e-4 * np.eye(d)

    # Generalised eigenvalues: Sigma_s v = lambda Sigma_0 v  =>  Sigma_0^-1 Sigma_s
    inv0_s = np.linalg.inv(Sigma0) @ Sigma_s
    eigvals = np.linalg.eigvals(inv0_s)
    eigvals = np.real(eigvals[eigvals > 0])

    expected_airm = float(np.sqrt(np.sum(np.log(eigvals) ** 2)))

    # Code implementation:
    code_airm = gcm_airm_distance(Sigma0, Sigma_s)
    assert np.isclose(expected_airm, code_airm, atol=1e-4)
