"""Feature extraction bank and clean reference bank for rollout-free forecasting."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from visionrl.config import Cfg, get_device, seed_for
from visionrl.environments import make_env, make_stacks, rollout
from visionrl.models.encoder import Encoder
from visionrl.probes.metrics import (
    centroid_distance,
    compute_pac_drift,
    gcm_airm_distance,
    knn_distance,
    semantic_pool,
)
from visionrl.shifts.observation_shifts import apply_shift_episode


class FeatureBank:
    """Everything computed from visual observations alone, for one trained agent."""

    def __init__(self, agent: Any, cfg: Cfg):
        self.agent, self.cfg = agent, cfg
        device = get_device()
        self.device = device
        st = torch.get_rng_state()
        cuda_st = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(123)
        self.rand_enc = Encoder(3 * cfg.n_stack, cfg.img).to(device).eval()   # untrained control encoder
        torch.set_rng_state(st)
        if cuda_st is not None:
            torch.cuda.set_rng_state_all(cuda_st)

    @torch.no_grad()
    def extract(self, stacks_u8: np.ndarray, masks: np.ndarray, bs: int = 64) -> Dict[str, np.ndarray]:
        a = self.agent
        keys = ["policy", "sem", "semu", "rand", "pix", "act", "q_min", "q_gap"]
        out = {k: [] for k in keys}

        for i in range(0, len(stacks_u8), bs):
            x = torch.as_tensor(stacks_u8[i : i + bs], dtype=torch.float32, device=self.device)
            sp = a.encoder.spatial(x)
            h = sp.flatten(1)
            z = a.actor.trunk(h)
            mu = torch.tanh(a.actor.policy(z))
            q1, q2 = a.critic(h, mu)

            m = masks[i : i + bs]
            fg = torch.as_tensor(((m == 1) | (m == 2)).astype(np.float32), device=self.device).unsqueeze(1)
            w = F.adaptive_avg_pool2d(fg, sp.shape[-2:])

            out["policy"].append(z.cpu().numpy())
            out["sem"].append(semantic_pool(sp, w).cpu().numpy())
            out["semu"].append(semantic_pool(sp, torch.ones_like(w)).cpu().numpy())
            out["rand"].append(F.adaptive_avg_pool2d(self.rand_enc.spatial(x), 4).flatten(1).cpu().numpy())
            out["pix"].append(F.adaptive_avg_pool2d(x / 255.0, 8).flatten(1).cpu().numpy())
            out["act"].append(mu.cpu().numpy())
            out["q_min"].append(torch.min(q1, q2).squeeze(1).cpu().numpy())
            out["q_gap"].append((q1 - q2).abs().squeeze(1).cpu().numpy())

        return {k: np.concatenate(v) for k, v in out.items()}


class RefBank:
    """Clean reference features for one agent, built on reference seeds disjoint from evaluation seeds."""

    def __init__(self, bank: FeatureBank, ref_recs: List[Dict[str, Any]], cfg: Cfg):
        feats = []
        for r in ref_recs:
            st = make_stacks(r["clean_u8"], cfg.n_stack)[:-1]
            feats.append(bank.extract(st, r["masks"][:-1]))
        cat = {k: np.concatenate([f[k] for f in feats]) for k in feats[0]}
        rng = np.random.default_rng(0)
        n = len(cat["policy"])
        keep = rng.choice(n, min(n, cfg.ref_max_rows), replace=False)
        self.F = {k: v[keep] for k, v in cat.items()}

        P = self.F["policy"].astype(np.float64)
        self.mu = P.mean(0)
        cov = np.cov(P.T) + (1e-3 * np.trace(np.cov(P.T)) / P.shape[1] + 1e-8) * np.eye(P.shape[1])
        self.cov = cov
        self.prec = np.linalg.inv(cov)

    def knn(self, name: str, X: np.ndarray, k: int) -> np.ndarray:
        return knn_distance(X, self.F[name], k)

    def maha(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec, d), 0.0))


def collect_clean(agent: Any, cfg: Cfg, split: str, n: int, env: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Collect clean recorded episodes with ground-truth masks for post-hoc injection."""
    env = env or make_env(cfg)
    return [rollout(env, agent.act, seed_for(split, i), cfg, record=True) for i in range(n)]


def score_episode(
    bank: FeatureBank,
    ref: RefBank,
    rec: Dict[str, Any],
    kind: str,
    sev: float,
    cfg: Cfg,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Failure-forecast scores for ONE clean recorded episode re-rendered under (kind, sev).
    No shifted rollout is used: the policy never acts under the shift here, only observes post-hoc shifted frames.
    """
    n, T = cfg.n_stack, cfg.ep_len
    if sev > 0:
        view, ms = apply_shift_episode(rec["clean_u8"], rec["masks"], kind, sev, rec["seed"], cfg.img)
    else:
        view, ms = rec["clean_u8"], rec["masks"]

    fs = bank.extract(make_stacks(view, n)[:-1], ms[:-1])
    if "_cf" not in rec:
        rec["_cf"] = bank.extract(make_stacks(rec["clean_u8"], n)[:-1], rec["masks"][:-1])
    fc = rec["_cf"]

    idx = np.arange(0, T, max(1, cfg.frame_stride))
    k = cfg.knn_k
    knn_pol = ref.knn("policy", fs["policy"], k)

    # Novel Extension: Policy-Action Curvature Drift (PAC-Drift)
    if hasattr(bank.agent.actor, "policy"):
        _raw_policy = bank.agent.actor.policy

        class _TanhPolicy(torch.nn.Module):
            def __init__(self, policy: torch.nn.Module):
                super().__init__()
                self.policy = policy

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return torch.tanh(self.policy(z))

        policy_submodule = _TanhPolicy(_raw_policy)
    else:
        policy_submodule = bank.agent.actor
    pac_scores_l0 = compute_pac_drift(
        policy_submodule,
        fc["policy"][idx],
        fs["policy"][idx],
        lambda_curv=0.0,
    )
    pac_scores_l1 = compute_pac_drift(
        policy_submodule,
        fc["policy"][idx],
        fs["policy"][idx],
        lambda_curv=1.0,
    )

    # Novel Extension: Geodesic Covariance Manifold Drift (GCM-Drift)
    if len(fs["policy"][idx]) > 2 and hasattr(ref, "cov"):
        cov_s = np.cov(fs["policy"][idx].astype(np.float64).T)
        gcm_val = gcm_airm_distance(ref.cov, cov_s)
    else:
        gcm_val = 0.0

    row = dict(
        knn_policy=float(knn_pol[idx].mean()),
        pac_drift_l0=float(pac_scores_l0.mean()) if len(pac_scores_l0) else 0.0,
        pac_drift_l1=float(pac_scores_l1.mean()) if len(pac_scores_l1) else 0.0,
        pac_drift=float(pac_scores_l1.mean()) if len(pac_scores_l1) else 0.0,
        gcm_drift=float(gcm_val),
        maha_policy=float(ref.maha(fs["policy"])[idx].mean()),
        cent_policy=centroid_distance(fs["policy"][idx], ref.mu[None, :]),
        sem_knn_policy=float(ref.knn("sem", fs["sem"], k)[idx].mean()),
        sem_knn_uniform=float(ref.knn("semu", fs["semu"], k)[idx].mean()),   # isolates mask weighting from layer choice
        knn_random=float(ref.knn("rand", fs["rand"], k)[idx].mean()),
        knn_pixel=float(ref.knn("pix", fs["pix"], k)[idx].mean()),
        act_disagree=float(np.linalg.norm(fs["act"] - fc["act"], axis=1)[idx].mean()),
        q_drop=float((fc["q_min"] - fs["q_min"])[idx].mean()),
        critic_gap=float(fs["q_gap"][idx].mean()),
    )
    return row, knn_pol, fs["policy"][idx]
