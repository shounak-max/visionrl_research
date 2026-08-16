"""
TTSS Adapter: Test-Time Self-Supervised Adaptation with TTT++ Moment Matching and Trust-Region Rejection.

Reimplements TTT++ (Liu et al., 2021) online moment matching to regularize test-time training,
combined with an explicit Trust-Region Rejection Safeguard to prevent adaptation updates from
deteriorating performance under severe distribution shifts.
"""

import copy
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.base_encoder import BaseEncoder


class TTSSAdapter:
    """TTSS Test-Time Adapter with trust-region rejection logic."""

    def __init__(
        self,
        offline_mean: torch.Tensor,
        offline_var: torch.Tensor,
        window_size: int = 16,
        lr: float = 1e-4,
        moment_weight: float = 1.0,
    ):
        self.offline_mean = offline_mean.detach()
        self.offline_var = offline_var.detach()
        self.window_size = window_size
        self.lr = lr
        self.moment_weight = moment_weight
        self.logs: List[Dict[str, float]] = []

    def local_consistency_loss(
        self, encoder: BaseEncoder, obs_window: torch.Tensor
    ) -> torch.Tensor:
        """Computes temporal self-supervised consistency loss over rolling window of frames [k, H, W, C]."""
        embeddings = encoder.embed(obs_window)  # [k, D]
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Temporal smoothness loss: distance between consecutive frame embeddings
        diffs = embeddings[1:] - embeddings[:-1]
        smooth_loss = torch.mean(diffs**2)
        return smooth_loss

    def moment_matching_loss(
        self, encoder: BaseEncoder, obs_window: torch.Tensor
    ) -> torch.Tensor:
        """Computes TTT++ moment matching loss pulling test moments to stored offline training moments."""
        embeddings = encoder.embed(obs_window)  # [k, D]
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        test_mean = torch.mean(embeddings, dim=0)
        test_var = torch.var(embeddings, dim=0, unbiased=False)

        mean_loss = torch.sum((test_mean - self.offline_mean.to(embeddings.device)) ** 2)
        var_loss = torch.sum((test_var - self.offline_var.to(embeddings.device)) ** 2)

        return mean_loss + var_loss

    def adapt(
        self,
        encoder: BaseEncoder,
        obs_stream: torch.Tensor,
        force_accept: bool = False,
    ) -> Tuple[BaseEncoder, Dict[str, float]]:
        """Executes a single test-time adaptation step over recent obs_stream window [k, H, W, C].

        Returns:
            (adapted_or_retained_encoder, step_log_dict)
        """
        start_time = time.perf_counter()

        # Keep rolling window of last k frames
        if obs_stream.shape[0] > self.window_size:
            obs_window = obs_stream[-self.window_size :]
        else:
            obs_window = obs_stream

        if obs_window.shape[0] < 4:
            return encoder, {
                "loss_before": 0.0,
                "loss_after": 0.0,
                "accepted": True,
                "latency_ms": 0.1,
            }

        # 1. Compute loss BEFORE candidate adaptation step
        loss_local_before = self.local_consistency_loss(encoder, obs_window)
        loss_moment_before = self.moment_matching_loss(encoder, obs_window)
        loss_total_before = loss_local_before + self.moment_weight * loss_moment_before

        # 2. Perform candidate self-supervised update on cloned encoder
        candidate_encoder = copy.deepcopy(encoder)
        
        # If the candidate encoder has zero trainable parameters, attach an adapter head if supported
        if not any(p.requires_grad for p in candidate_encoder.parameters()):
            if hasattr(candidate_encoder, "attach_adapter"):
                candidate_encoder.attach_adapter()
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                step_log = {
                    "loss_before": float(loss_total_before.item()),
                    "loss_after": float(loss_total_before.item()),
                    "accepted": False,
                    "status": "no_trainable_parameters",
                    "latency_ms": float(elapsed_ms),
                }
                self.logs.append(step_log)
                return encoder, step_log

        candidate_encoder.train()
        trainable_params = [p for p in candidate_encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.lr,
            momentum=0.5,
        )

        for _ in range(2):
            optimizer.zero_grad()
            c_loss_local = self.local_consistency_loss(candidate_encoder, obs_window)
            c_loss_moment = self.moment_matching_loss(candidate_encoder, obs_window)
            c_total_loss = c_loss_local + self.moment_weight * c_loss_moment

            c_total_loss.backward()
            optimizer.step()

        candidate_encoder.eval()

        # 3. Compute loss AFTER candidate adaptation step
        with torch.no_grad():
            loss_local_after = self.local_consistency_loss(candidate_encoder, obs_window)
            loss_moment_after = self.moment_matching_loss(candidate_encoder, obs_window)
            loss_total_after = (
                loss_local_after + self.moment_weight * loss_moment_after
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. TRUST-REGION REJECTION SAFEGUARD
        # Accept update ONLY if candidate loss is strictly improved (or if force_accept)
        accepted = (loss_total_after.item() < loss_total_before.item()) or force_accept

        step_log = {
            "loss_before": float(loss_total_before.item()),
            "loss_after": float(loss_total_after.item()),
            "accepted": bool(accepted),
            "latency_ms": float(elapsed_ms),
        }
        self.logs.append(step_log)

        if accepted:
            return candidate_encoder, step_log
        else:
            # Reject candidate update, rollback to pre-update encoder!
            return encoder, step_log
