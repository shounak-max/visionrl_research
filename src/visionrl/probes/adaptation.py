"""Test-time adaptation (TTA) with safeguarded rollback (TTSS) and paired statistical comparisons."""

import collections
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from visionrl.config import Cfg, get_device, seed_for
from visionrl.environments import from_u8, is_collapse, make_env, make_stacks, rollout
from visionrl.forecasting.metrics import holm, mcnemar_exact, paired_boot_diff, paired_signflip_p
from visionrl.shifts.observation_shifts import Shift


def clean_feature_stats(agent: Any, ref_recs: List[Dict[str, Any]], cfg: Cfg, n: int = 256) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Per-channel mean and variance of the encoder feature map on clean reference observations."""
    device = get_device()
    st = np.concatenate([make_stacks(r["clean_u8"], cfg.n_stack)[:-1] for r in ref_recs])
    rng = np.random.default_rng(0)
    st = st[rng.choice(len(st), min(n, len(st)), replace=False)]
    with torch.no_grad():
        sp = agent.encoder.spatial(torch.as_tensor(st, dtype=torch.float32, device=device))
        return sp.mean((0, 2, 3)), sp.var((0, 2, 3), unbiased=False), st


class AdaptedPolicy:
    """mode 'none' : no adaptation (same code path, so that arms differ in nothing but the adaptation rule).
       mode 'ttt'  : naive test-time training. Every step, one gradient step on a feature moment-matching loss.
       mode 'ttss' : safeguarded. The same step is kept only if (i) loss on recent-observation buffer decreases
                     and (ii) clean-reference actions move by at most accept_delta; otherwise rolled back.
       ttt and ttss share learning rate, loss weight and buffer; the safeguard is the ONLY difference.
       act() always runs the encoder that was adapted (self.enc), and counts accepted and rejected updates.
    """

    def __init__(self, base: Any, cfg: Cfg, mode: str, stats: Tuple[torch.Tensor, torch.Tensor], clean_stacks: np.ndarray):
        assert mode in ("none", "ttt", "ttss"), f"Invalid mode: {mode}"
        self.base, self.cfg, self.mode = base, cfg, mode
        self.device = get_device()
        self.enc = copy.deepcopy(base.encoder)
        self.actor = base.actor
        self.enc0 = copy.deepcopy(base.encoder.state_dict())
        self.mu_c, self.var_c = stats
        cs = clean_stacks[: cfg.tta_clean_batch]
        self.cx = torch.as_tensor(cs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.a_base = base.actor(base.encoder(self.cx))
        self.reset_episode()

    def reset_episode(self) -> None:
        self.enc.load_state_dict(self.enc0)
        self.opt = torch.optim.Adam(self.enc.parameters(), lr=self.cfg.tta_lr)
        self.buf = collections.deque(maxlen=self.cfg.tta_buffer)
        self.accepted = 0
        self.rejected = 0

    def _mm(self, x: torch.Tensor) -> torch.Tensor:
        sp = self.enc.spatial(x)
        return F.mse_loss(sp.mean((0, 2, 3)), self.mu_c) + F.mse_loss(sp.var((0, 2, 3), unbiased=False), self.var_c)

    def _adapt(self) -> None:
        xb = torch.cat(list(self.buf), 0)
        with torch.no_grad():
            before = float(self._mm(xb))
        snap = {k: v.clone() for k, v in self.enc.state_dict().items()}
        opt_state = copy.deepcopy(self.opt.state_dict())
        loss = self.cfg.tta_mm_weight * self._mm(xb)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.mode == "ttt":
            self.accepted += 1
            return

        with torch.no_grad():
            after = float(self._mm(xb))
            dev = float((self.actor(self.enc(self.cx)) - self.a_base).norm(dim=1).mean())

        if after < before and dev <= self.cfg.tta_accept_delta:
            self.accepted += 1
        else:
            self.enc.load_state_dict(snap)
            self.opt.load_state_dict(opt_state)
            self.rejected += 1

    def act(self, obs_u8: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs_u8, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.buf.append(x)
        if self.mode != "none" and len(self.buf) >= 2:
            self._adapt()
        with torch.no_grad():
            return self.actor(self.enc(x)).squeeze(0).cpu().numpy()


def run_tta(
    agent: Any,
    agent_id: str,
    cfg: Cfg,
    ref_recs: List[Dict[str, Any]],
    floor: float,
    arms: Tuple[str, ...] = ("none", "ttt", "ttss"),
) -> pd.DataFrame:
    """Paired arms: every arm is evaluated on identical (kind, severity, episode seed) triples."""
    mu, var, clean_stacks = clean_feature_stats(agent, ref_recs, cfg)
    env = make_env(cfg)
    rows = []
    seeds = [seed_for("test", i) for i in range(cfg.tta_eps)]
    pols = {a: AdaptedPolicy(agent, cfg, a, (mu, var), clean_stacks) for a in arms}

    for kind in cfg.tta_kinds:
        for sev in cfg.tta_severities:
            for i, s in enumerate(seeds):
                sh = Shift(kind, sev, s, cfg.img)
                for a, pol in pols.items():
                    pol.reset_episode()
                    r = rollout(env, pol.act, s, cfg, shift=sh)
                    rows.append(dict(
                        agent_id=agent_id,
                        kind=kind,
                        sev=sev,
                        seed=s,
                        seed_idx=i,
                        arm=a,
                        success=r["success"],
                        mean_cle=r["mean_cle"],
                        collapse=is_collapse(r["mean_cle"], cfg, floor),
                        accepted=pol.accepted,
                        rejected=pol.rejected,
                    ))
    return pd.DataFrame(rows)


def compare_arms(
    t: pd.DataFrame,
    contrasts: Tuple[Tuple[str, str], ...] = (("ttt", "none"), ("ttss", "none"), ("ttss", "ttt")),
    B: int = 2000,
    n_perm: int = 5000,
) -> pd.DataFrame:
    """Paired comparisons with exact McNemar tests (success, collapse), paired-bootstrap CIs for mean CLE,
    paired sign-flip permutation tests for mean CLE, and Holm correction over all contrasts and outcomes.
    """
    rows = []
    for (kind, sev), g in t.groupby(["kind", "sev"]):
        # pair on (agent, seed): with several agents the same seed occurs once per agent
        wide = {a: g[g.arm == a].set_index(["agent_id", "seed"]).sort_index() for a in g.arm.unique()}
        for a, b in contrasts:
            if a not in wide or b not in wide:
                continue
            A, Bm = wide[a], wide[b]
            assert A.index.equals(Bm.index), "arms must share identical (agent, episode seed) pairs"
            dcle, ci = paired_boot_diff(A.mean_cle.values, Bm.mean_cle.values, B)
            p_s, b01, b10 = mcnemar_exact(A.success.values, Bm.success.values)
            p_c, c01, c10 = mcnemar_exact(A.collapse.values, Bm.collapse.values)
            rows.append(dict(
                kind=kind,
                sev=sev,
                contrast=f"{a} vs {b}",
                n=len(A),
                d_mean_cle=dcle,
                ci_lo=ci[0],
                ci_hi=ci[1],
                success_a=A.success.mean(),
                success_b=Bm.success.mean(),
                p_success=p_s,
                collapse_a=A.collapse.mean(),
                collapse_b=Bm.collapse.mean(),
                p_collapse=p_c,
                accepted_mean_a=A.accepted.mean(),
                p_cle=paired_signflip_p(A.mean_cle.values, Bm.mean_cle.values, n_perm),
            ))
    r = pd.DataFrame(rows)
    if len(r):
        m_ = len(r)
        adj = holm(np.r_[r.p_success.values, r.p_collapse.values, r.p_cle.values])
        r["p_success_holm"], r["p_collapse_holm"], r["p_cle_holm"] = adj[:m_], adj[m_:2 * m_], adj[2 * m_:]
    return r


def arm_config_table(cfg: Cfg) -> pd.DataFrame:
    base = dict(lr=cfg.tta_lr, mm_weight=cfg.tta_mm_weight, buffer=cfg.tta_buffer, accept_delta=cfg.tta_accept_delta)
    return pd.DataFrame([
        dict(arm="none", adapts=False, safeguard=False, **base),
        dict(arm="ttt", adapts=True, safeguard=False, **base),
        dict(arm="ttss", adapts=True, safeguard=True, **base),
    ]).set_index("arm")
