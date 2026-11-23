"""Representation probe metrics: Centroid distance, Semantic pooling, kNN, and Grassmann geodesic distance."""

import numpy as np
import torch

from visionrl.config import get_device


def centroid_distance(Fs: np.ndarray, Fc: np.ndarray) -> float:
    """Eq. (1): D_cent = || mean_i f(o_i^shifted) - mean_j f(o_j^clean) ||_2 for feature matrices (n, d)."""
    return float(np.linalg.norm(np.asarray(Fs).mean(0) - np.asarray(Fc).mean(0)))


def semantic_pool(sp: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. (2): semantic pooling of a feature map.
    sp: (B, C, h, w) features, w: (B, 1, h, w) weights in [0, 1].
    f_sem = sum_uv w(u,v) f(:,u,v) / (sum_uv w(u,v) + eps).
    The pixel-space entity mask W_s is resampled to the feature-map resolution by area averaging,
    so w is the fraction of each feature cell covered by agent or target pixels.
    """
    return (sp * w).sum((2, 3)) / (w.sum((2, 3)) + eps)


def knn_distance(X: np.ndarray, R: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    """Mean L2 distance from each row of X to its k nearest rows of the reference bank R (numpy in, numpy out)."""
    device = get_device()
    Rt = torch.as_tensor(R, dtype=torch.float32, device=device)
    k = min(k, len(Rt))
    out = []
    for i in range(0, len(X), chunk):
        d = torch.cdist(torch.as_tensor(X[i : i + chunk], dtype=torch.float32, device=device), Rt)
        out.append(d.topk(k, dim=1, largest=False).values.mean(1).cpu().numpy())
    return np.concatenate(out)


def grassmann_distance(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Geodesic distance on the Grassmannian manifold between the top-k principal subspaces of two activation sets.
    d = sqrt(sum_i theta_i^2), where theta_i are the principal angles. Returns 0 for identical subspaces.
    """
    A = np.asarray(Xa, float) - np.mean(Xa, 0)
    B = np.asarray(Xb, float) - np.mean(Xb, 0)
    k = int(min(k, A.shape[0], B.shape[0], A.shape[1]))
    Ua = np.linalg.svd(A, full_matrices=False)[2][:k].T
    Ub = np.linalg.svd(B, full_matrices=False)[2][:k].T
    s = np.clip(np.linalg.svd(Ua.T @ Ub, compute_uv=False), -1.0, 1.0)
    return float(np.sqrt((np.arccos(s) ** 2).sum()))


def compute_pac_drift(
    policy_fn: torch.nn.Module,
    z0: torch.Tensor,
    zs: torch.Tensor,
    lambda_curv: float = 1.0,
    act_dim: int = 2,
) -> np.ndarray:
    """Policy-Action Curvature Drift (PAC-Drift).
    Evaluates first-order Jacobian pullback metric drift ||J_pi(z0) Delta z||_2
    and second-order directional curvature C_pi(z0; Delta z) via Hessian-Vector Products.
    
    z0: (B, d) clean latent representation
    zs: (B, d) shifted latent representation
    lambda_curv: weighting of the second-order curvature term
    """
    device = z0.device if isinstance(z0, torch.Tensor) else get_device()
    z0_t = torch.as_tensor(z0, dtype=torch.float32, device=device)
    zs_t = torch.as_tensor(zs, dtype=torch.float32, device=device)
    delta_z = zs_t - z0_t
    B = z0_t.shape[0]

    if B == 0:
        return np.array([], dtype=np.float32)

    # If delta_z is completely zero (e.g. clean input), return exact zeros
    if torch.allclose(delta_z, torch.zeros_like(delta_z)):
        return np.zeros(B, dtype=np.float32)

    z0_var = z0_t.detach().clone().requires_grad_(True)
    mu0 = policy_fn(z0_var)
    act_dim = mu0.shape[-1]

    first_order_dev = []
    second_order_curv = []

    for k in range(act_dim):
        grad_k = torch.autograd.grad(
            outputs=mu0[:, k].sum(),
            inputs=z0_var,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Directional derivative along delta_z
        j_dot_dz = (grad_k * delta_z).sum(dim=-1)
        first_order_dev.append(j_dot_dz ** 2)

        # Hessian-vector product
        hvp_tuple = torch.autograd.grad(
            outputs=j_dot_dz.sum(),
            inputs=z0_var,
            retain_graph=(k < act_dim - 1),
            allow_unused=True,
        )
        hvp = hvp_tuple[0]

        if hvp is not None:
            curv_k = (delta_z * hvp).sum(dim=-1).abs()
        else:
            curv_k = torch.zeros(B, device=device)
        second_order_curv.append(curv_k ** 2)

    fo_norm = torch.sqrt(torch.stack(first_order_dev, dim=-1).sum(dim=-1))
    so_norm = torch.sqrt(torch.stack(second_order_curv, dim=-1).sum(dim=-1))
    pac = fo_norm + 0.5 * lambda_curv * so_norm
    return pac.detach().cpu().numpy()


def gcm_airm_distance(cov0: np.ndarray, covs: np.ndarray, eps: float = 1e-5) -> float:
    """Geodesic Covariance Manifold Drift (GCM-Drift) via Affine-Invariant Riemannian Metric (AIRM).
    delta_AIRM(Sigma_0, Sigma_s) = sqrt(sum_i ln^2 lambda_i(Sigma_0^-1 Sigma_s))
    where lambda_i are generalized eigenvalues satisfying Sigma_s v_i = lambda_i Sigma_0 v_i.
    """
    from scipy.linalg import eigh
    d = cov0.shape[0]
    C0 = np.asarray(cov0, dtype=np.float64) + eps * np.eye(d)
    Cs = np.asarray(covs, dtype=np.float64) + eps * np.eye(d)

    # Symmetrize
    C0 = 0.5 * (C0 + C0.T)
    Cs = 0.5 * (Cs + Cs.T)

    vals = eigh(Cs, C0, eigvals_only=True)
    vals = np.maximum(vals, 1e-12)
    return float(np.sqrt(np.sum(np.log(vals) ** 2)))
