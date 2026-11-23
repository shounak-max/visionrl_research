"""
Offline Feature Summarizer for TTSS / TTT++ test-time adaptation.

Computes and caches training distribution feature mean vector mu_train [D]
and covariance matrix Sigma_train [D, D].
"""

import os
from typing import Dict, Optional, Tuple, Union
import torch

from provenance import stamp_provenance, verify_provenance


class OfflineFeatureSummarizer:
    """Offline feature mean and covariance statistic generator."""

    @staticmethod
    def summarize(
        embeddings: Union[torch.Tensor, torch.Tensor],
        config_hash: str = "offline_stats",
    ) -> Dict[str, torch.Tensor]:
        """Computes mean vector mu [D] and covariance matrix Sigma [D, D] of training embeddings [N, D]."""
        if not isinstance(embeddings, torch.Tensor):
            emb_t = torch.tensor(embeddings, dtype=torch.float32)
        else:
            emb_t = embeddings.float()

        N, D = emb_t.shape
        mu = torch.mean(emb_t, dim=0)

        # Center embeddings
        emb_centered = emb_t - mu.unsqueeze(0)

        # Covariance matrix Sigma = (1 / (N - 1)) * emb_centered^T * emb_centered
        if N > 1:
            cov = torch.matmul(emb_centered.T, emb_centered) / float(N - 1)
        else:
            cov = torch.eye(D)

        return {
            "mean": mu,
            "covariance": cov,
            "var_diag": torch.diagonal(cov),
            "num_samples": torch.tensor(N),
            "feature_dim": torch.tensor(D),
        }

    @staticmethod
    def save_stats(
        stats: Dict[str, torch.Tensor],
        filepath: str = "embeddings/offline_feature_stats.pt",
        seed: int = 42,
    ) -> str:
        """Saves feature statistics with provenance record."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        prov = stamp_provenance(
            config={"stats": "offline_summary"},
            seed=seed,
            env_id="training",
            perturbation_tag="clean",
        )
        data_to_save = dict(stats)
        data_to_save["provenance"] = prov
        torch.save(data_to_save, filepath)
        return filepath

    @staticmethod
    def load_stats(
        filepath: str = "embeddings/offline_feature_stats.pt",
    ) -> Dict[str, torch.Tensor]:
        """Loads offline feature statistics."""
        data = torch.load(filepath, weights_only=False)
        return data
