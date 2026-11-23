"""
Geodesic Distance metrics on Grassmann Manifolds Gr(k, D).

Implements canonical principal angles, Riemannian geodesic distance, and projection metric distance
per Huang et al. (2016) and Bendokat et al. (2020) Grassmann handbook.
"""

from typing import Tuple, Union
import numpy as np
import torch

from geometry.grassmann import GrassmannPoint


class GeodesicDistance:
    """Computes principal angle geodesic metrics between Grassmann points."""

    @staticmethod
    def principal_angles(
        p1: GrassmannPoint, p2: GrassmannPoint
    ) -> torch.Tensor:
        """Computes principal canonical angles theta_i in [0, pi/2] between two subspaces.

        Args:
            p1: GrassmannPoint representing subspace U1 [D, k1].
            p2: GrassmannPoint representing subspace U2 [D, k2].

        Returns:
            Tensor [k] of principal angles in radians sorted ascendingly.
        """
        U1 = p1.subspace  # [D, k1]
        U2 = p2.subspace  # [D, k2]

        k = min(p1.subspace_dim, p2.subspace_dim)
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]

        # Inner product matrix M = U1_k^T U2_k  [k, k] in double precision
        M = torch.matmul(U1_k.double().T, U2_k.double())

        # Singular values sigma_i = cos(theta_i)
        S = torch.linalg.svdvals(M)
        S_clamped = torch.clamp(S, 0.0, 1.0)
        # Singular values within numerical float precision of 1.0 correspond to angle 0.0
        S_clamped[S_clamped > (1.0 - 1e-5)] = 1.0

        # Principal angles theta_i = arccos(sigma_i)
        angles = torch.acos(S_clamped).float()
        return torch.sort(angles)[0]

    @staticmethod
    def compute(
        p1: GrassmannPoint,
        p2: GrassmannPoint,
        metric_type: str = "geodesic",
    ) -> float:
        """Computes subspace distance between p1 and p2.

        Supported metric_types:
        - 'geodesic': Riemannian geodesic distance sqrt(sum theta_i^2).
        - 'projection': Projection metric distance sqrt(sum sin^2 theta_i).
        - 'max_angle': Maximum principal angle max(theta_i).

        Returns:
            float scalar distance.
        """
        angles = GeodesicDistance.principal_angles(p1, p2)

        if metric_type == "geodesic":
            dist = torch.sqrt(torch.sum(angles**2))
        elif metric_type == "projection":
            dist = torch.sqrt(torch.sum(torch.sin(angles) ** 2))
        elif metric_type == "max_angle":
            dist = torch.max(angles)
        else:
            raise ValueError(f"Unknown metric_type: '{metric_type}'")

        return float(dist.item())
