"""
Grassmann Manifold Subspace Mapper.

Maps trajectory activation matrices [T, D] to points on the Grassmann manifold Gr(k, D)
by computing SVD and extracting the top-k left singular vectors as the orthonormal subspace representative.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
import torch


@dataclass
class GrassmannPoint:
    """Represents a point on the Grassmann manifold Gr(k, D).

    Attributes:
        subspace: Orthonormal basis matrix Uk [D, k] where Uk^T Uk = I_k.
        subspace_dim: Dimension k of the subspace.
        feature_dim: Dimension D of the embedding space.
        singular_values: Top-k singular values of the activation matrix.
    """

    subspace: torch.Tensor  # [D, k]
    subspace_dim: int  # k
    feature_dim: int  # D
    singular_values: torch.Tensor  # [k]

    def to_numpy(self) -> np.ndarray:
        return self.subspace.detach().cpu().numpy()


class GrassmannMapper:
    """Maps trajectory activations to Grassmann manifold points."""

    @staticmethod
    def to_subspace(
        activations: Union[torch.Tensor, np.ndarray],
        subspace_dim: int = 10,
        center: bool = True,
    ) -> GrassmannPoint:
        """Extracts top-k orthonormal basis vectors from activation matrix [T, D].

        Args:
            activations: Trajectory activation tensor [T, D].
            subspace_dim: Dimension k of subspace (e.g. 5, 10, 20).
            center: If True, subtracts mean trajectory embedding before SVD.

        Returns:
            GrassmannPoint containing orthonormal subspace basis Uk [D, k].
        """
        if not isinstance(activations, torch.Tensor):
            act_t = torch.tensor(activations, dtype=torch.float32)
        else:
            act_t = activations.float()

        T, D = act_t.shape
        k = min(subspace_dim, T, D)

        if center:
            act_t = act_t - torch.mean(act_t, dim=0, keepdim=True)

        # Compute SVD of activation matrix A [T, D] -> A = U S V^T
        # Left singular vectors of A^T [D, T] are V (the V in A = U S V^T or V_right in A^T = V S U^T)
        # Using PyTorch linalg.svd on A^T [D, T] gives U_left [D, min(D,T)]
        U_left, S, Vh = torch.linalg.svd(act_t.T, full_matrices=False)

        # Top k orthonormal basis vectors Uk [D, k]
        Uk = U_left[:, :k]
        top_S = S[:k]

        # Orthonormalization check Uk^T Uk = I_k
        gram = torch.matmul(Uk.T, Uk)
        identity = torch.eye(k, device=Uk.device)
        if not torch.allclose(gram, identity, atol=1e-4):
            # Q, R re-orthonormalization as fallback safeguard
            Uk, _ = torch.linalg.qr(Uk)

        return GrassmannPoint(
            subspace=Uk,
            subspace_dim=k,
            feature_dim=D,
            singular_values=top_S,
        )
