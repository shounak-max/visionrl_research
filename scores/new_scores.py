"""
New OOD Scores for Visual RL Generalization Bounds.

Score 10 — PAC-Drift:
    PAC-Drift = mean over frames of (
        ||J_pi(z0) dz||_2 + 0.5 * lambda * ||dz^T H_pi(z0) dz||_2
    )
    Where:
    - z0 = encoder(clean_frame), dz = encoder(shifted_frame) - encoder(clean_frame)
    - J_pi(z0) = Jacobian of policy action output w.r.t. z at z0
    - H_pi(z0) dz = Hessian-vector product per action dimension
    - lambda = 0 → first-order only; lambda = 1 → full curvature term

Score 11 — GCM-Drift (Geodesic Covariance Matrix distance, AIRM):
    GCM-Drift = sqrt( sum of ln^2(lambda_i) )
    Where lambda_i are generalised eigenvalues of Sigma_s @ inv(Sigma_0)
    - Uses affine-invariant Riemannian metric (AIRM)
    - Ridge 1e-5 added to both covariance matrices before inversion
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility: Jacobian and HVP via torch.autograd
# ---------------------------------------------------------------------------

def _get_policy_features(policy: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Get the actor output from an embedding z.

    Handles both DrQv2Policy (which has an .actor attribute) and plain
    nn.Sequential / nn.Module (direct call).
    """
    if hasattr(policy, "actor"):
        return policy.actor(z)
    return policy(z)


def _jacobian_per_frame(actor_fn, z0: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian J [action_dim x embed_dim] for a single embedding z0.

    Uses torch.func.jacrev when available (PyTorch >= 2.0), falls back to
    torch.autograd.functional.jacobian.

    Args:
        actor_fn: callable(z) -> action tensor of shape [action_dim]
        z0: embedding tensor of shape [embed_dim], requires_grad not needed here.

    Returns:
        J: tensor of shape [action_dim, embed_dim]
    """
    try:
        import torch.func as func  # PyTorch >= 2.0
        # jacrev returns [action_dim, embed_dim] for scalar z0 input
        J = func.jacrev(actor_fn)(z0)
        return J  # [action_dim, embed_dim]
    except (ImportError, AttributeError):
        pass

    # Fallback: torch.autograd.functional.jacobian
    z0_param = z0.detach().requires_grad_(True)
    J = torch.autograd.functional.jacobian(actor_fn, z0_param)
    # J: [action_dim, embed_dim]
    return J


def _hvp_per_frame(actor_fn, z0: torch.Tensor, v: torch.Tensor, action_idx: int) -> torch.Tensor:
    """Compute Hessian-vector product H_pi(z0)[action_idx] @ v.

    H_pi(z0) dz for a single action dimension is a vector of shape [embed_dim].

    Args:
        actor_fn: callable(z) -> action tensor [action_dim]
        z0: embedding [embed_dim]
        v: direction vector [embed_dim] = dz
        action_idx: which action dimension to compute Hessian for

    Returns:
        hvp: tensor of shape [embed_dim], the Hessian-vector product
    """
    def scalar_fn(z):
        return actor_fn(z)[action_idx]

    # torch.autograd.functional.hvp computes (grad, hvp)
    _, hvp = torch.autograd.functional.hvp(scalar_fn, z0, v)
    return hvp


# ---------------------------------------------------------------------------
# Score 10: PAC-Drift
# ---------------------------------------------------------------------------

class PACDriftScore:
    """PAC-Drift OOD score (Score 10).

    PAC-Drift = mean over frames of (
        ||J_pi(z0) dz||_2 + 0.5 * lambda_val * ||dz^T H_pi(z0) dz||_2
    )

    The first term captures first-order action deviation due to latent shift dz.
    The second term captures the second-order curvature correction.
    """

    def compute(
        self,
        clean_frames: torch.Tensor,
        shifted_frames: torch.Tensor,
        policy: nn.Module,
        lambda_val: float = 1.0,
    ) -> float:
        """Compute PAC-Drift score.

        Args:
            clean_frames: Tensor [N, C, H, W] or [N, H, W, C] of clean observations.
            shifted_frames: Tensor [N, C, H, W] or [N, H, W, C] of shifted observations.
            policy: A policy module. Must have either:
                - .encoder (BaseEncoder) + .actor (nn.Module)
                - Or be directly callable as a feature extractor; in this case
                  we call policy.actor on the embedding if .actor exists.
            lambda_val: Trade-off for curvature term. 0 = first-order only,
                1 = full curvature. Default: 1.0.

        Returns:
            Scalar float: mean PAC-Drift over all frames.
        """
        # Put policy in eval mode; we re-enable gradients explicitly
        policy.eval()

        encoder = getattr(policy, "encoder", None)
        if encoder is None:
            raise AttributeError(
                "policy must have an .encoder attribute (e.g. DrQv2Policy.encoder)"
            )

        # Preprocess frames through encoder's preprocess (handles HWC->CHW, uint8 scale)
        with torch.no_grad():
            z0_batch = encoder.embed(encoder.preprocess_obs(clean_frames))   # [N, D]
            zs_batch = encoder.embed(encoder.preprocess_obs(shifted_frames)) # [N, D]
            dz_batch = zs_batch - z0_batch                                   # [N, D]

        N = z0_batch.shape[0]
        frame_scores = []

        for i in range(N):
            z0 = z0_batch[i].detach()   # [D]
            dz = dz_batch[i].detach()   # [D]

            # Define actor function (embedding → action vector)
            def actor_fn(z):
                return _get_policy_features(policy, z.unsqueeze(0)).squeeze(0)

            # --- First-order term: ||J_pi(z0) dz||_2 ---
            # J: [action_dim, D]
            J = _jacobian_per_frame(actor_fn, z0.requires_grad_(False))
            # J_dz: [action_dim]
            J_dz = J @ dz
            first_order_norm = J_dz.norm(p=2)

            # --- Second-order term: ||dz^T H_pi(z0) dz||_2 (over action dims) ---
            if lambda_val != 0.0:
                # Action dim
                action_dim = J.shape[0]
                hvp_norms = []
                for a in range(action_dim):
                    # hvp_a: [D], the HVP for action dimension a
                    hvp_a = _hvp_per_frame(actor_fn, z0, dz, a)
                    # dz^T H_a dz = scalar
                    scalar_a = (dz * hvp_a).sum()
                    hvp_norms.append(scalar_a)
                # ||dz^T H_pi(z0) dz||_2 across action dims
                hvp_vec = torch.stack(hvp_norms)  # [action_dim]
                second_order_norm = hvp_vec.norm(p=2)
            else:
                second_order_norm = torch.tensor(0.0, dtype=z0.dtype)

            frame_score = first_order_norm + 0.5 * lambda_val * second_order_norm
            frame_scores.append(frame_score.item())

        return float(np.mean(frame_scores))


# ---------------------------------------------------------------------------
# Score 11: GCM-Drift (Affine-Invariant Riemannian Metric on covariance matrices)
# ---------------------------------------------------------------------------

class GCMDriftScore:
    """GCM-Drift OOD score (Score 11) — Geodesic Covariance Matrix distance.

    Uses the Affine-Invariant Riemannian Metric (AIRM):
        d_AIRM(A, B) = sqrt( sum_i ln^2(lambda_i(A^{-1} B)) )
    where lambda_i are generalised eigenvalues of B w.r.t. A.

    GCM-Drift = d_AIRM(Sigma_0, Sigma_s)
    where:
    - Sigma_0 = covariance of policy features from clean frames + ridge 1e-5
    - Sigma_s = covariance of policy features from shifted frames + ridge 1e-5

    AIRM is symmetric: d(A,B) = d(B,A).
    AIRM is affine-invariant: d(W A W^T, W B W^T) = d(A, B) for invertible W.
    """

    _RIDGE: float = 1e-5

    @staticmethod
    def _compute_covariance(features: torch.Tensor) -> torch.Tensor:
        """Compute (unbiased) covariance matrix of [N, D] feature matrix.

        Returns [D, D] covariance matrix with ridge regularisation.
        """
        N, D = features.shape
        mean = features.mean(dim=0, keepdim=True)        # [1, D]
        centered = features - mean                        # [N, D]
        cov = (centered.T @ centered) / max(N - 1, 1)    # [D, D]
        # Ridge regularisation
        cov = cov + GCMDriftScore._RIDGE * torch.eye(D, dtype=features.dtype, device=features.device)
        return cov

    @staticmethod
    def airm_distance(A: torch.Tensor, B: torch.Tensor) -> float:
        """Compute AIRM distance d(A, B) = sqrt(sum ln^2(eig(A^{-1} B))).

        Args:
            A: [D, D] SPD matrix.
            B: [D, D] SPD matrix.

        Returns:
            Scalar float distance.
        """
        A = A.double()
        B = B.double()

        # AIRM: d_AIRM(A, B) = sqrt(sum_i ln^2(lambda_i(A^{-1} B)))
        # Using Cholesky decomposition A = L L^T:
        # Eigenvalues of A^{-1} B are identical to eigenvalues of L^{-1} B L^{-T},
        # which is symmetric positive definite and can be diagonalised via eigvalsh.
        try:
            L = torch.linalg.cholesky(A)
            inv_L = torch.linalg.inv(L)
            C = inv_L @ B @ inv_L.T
            C = (C + C.T) / 2.0  # enforce numerical symmetry
            eigvals = torch.linalg.eigvalsh(C)
            eigvals = torch.clamp(eigvals, min=1e-15)
            dist = torch.sqrt((torch.log(eigvals) ** 2).sum())
            return float(dist.item())
        except Exception:
            # Fallback: generalised eigenvalues via standard eig of pinv(A) @ B
            Ainv = torch.linalg.pinv(A)
            M = Ainv @ B
            eigvals = torch.linalg.eigvals(M).real
            eigvals = torch.clamp(eigvals, min=1e-15)
            dist = torch.sqrt((torch.log(eigvals) ** 2).sum())
            return float(dist.item())

    def compute(
        self,
        clean_frames: torch.Tensor,
        shifted_frames: torch.Tensor,
        policy: nn.Module,
        lambda_val: float = 1.0,
    ) -> float:
        """Compute GCM-Drift score.

        Args:
            clean_frames: Tensor [N, C, H, W] or [N, H, W, C] of clean observations.
            shifted_frames: Tensor [N, C, H, W] or [N, H, W, C] of shifted observations.
            policy: A policy module with .encoder and .actor attributes.
            lambda_val: Unused for GCM-Drift (kept for API consistency).

        Returns:
            Scalar float: AIRM geodesic distance between feature covariances.
        """
        policy.eval()

        encoder = getattr(policy, "encoder", None)
        if encoder is None:
            raise AttributeError(
                "policy must have an .encoder attribute (e.g. DrQv2Policy.encoder)"
            )

        with torch.no_grad():
            # Extract policy features (post-actor embeddings)
            z0_batch = encoder.embed(encoder.preprocess_obs(clean_frames))   # [N, D]
            zs_batch = encoder.embed(encoder.preprocess_obs(shifted_frames)) # [N, D]

            # Get policy "features" = actor pre-activation or final embedding
            # We use encoder embeddings as the policy features (z space)
            # This is the latent representation the actor operates on.
            feats0 = z0_batch   # [N, D]
            feats_s = zs_batch  # [N, D]

        # Covariance matrices with ridge regularisation
        Sigma_0 = self._compute_covariance(feats0)   # [D, D]
        Sigma_s = self._compute_covariance(feats_s)  # [D, D]

        return self.airm_distance(Sigma_0, Sigma_s)
