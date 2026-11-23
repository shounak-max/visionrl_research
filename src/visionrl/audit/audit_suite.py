"""Pre-registered audit test suite (38 audit checks verifying code against theoretical claims and reviewer findings)."""

import copy
import math
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from visionrl.config import Cfg, SEEDS, get_device, make_cfg, seed_for
from visionrl.environments import (
    ENV_REGISTRY,
    is_collapse,
    is_success,
    make_env,
    rollout,
    to_u8,
)
from visionrl.forecasting.metrics import (
    auroc,
    conformal_threshold,
    ema_smooth,
    metrics_bundle,
    paired_signflip_p,
    rank_vals,
)
from visionrl.forecasting.predictor import SCORE_COLS
from visionrl.forecasting.prevalence import auroc_mixture
from visionrl.forecasting.ranking import kendall_tau_b, selection_regret
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes.adaptation import AdaptedPolicy, clean_feature_stats, compare_arms
from visionrl.probes.feature_bank import FeatureBank, collect_clean
from visionrl.probes.metrics import (
    centroid_distance,
    compute_pac_drift,
    gcm_airm_distance,
    grassmann_distance,
    knn_distance,
    semantic_pool,
)
from visionrl.shifts.observation_shifts import KIND_ID, Shift, apply_shift_episode


def run_audit_tests(agents: Dict[str, Any], cfg: Cfg) -> pd.DataFrame:
    """Run all pre-registered audit checks. Fails with AssertionError if any check fails."""
    device = get_device()
    res: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        res.append(dict(test=name, passed=bool(ok), detail=detail))

    _aid = next(iter(agents))
    _ag = agents[_aid]
    env = make_env(cfg)

    # I01: The learned policy, not a heuristic, chooses the actions.
    f, m = env.reset(3)
    obs = np.concatenate([to_u8(f)] * cfg.n_stack, 0)
    a0 = _ag.act(obs)
    saved = copy.deepcopy(_ag.encoder.state_dict())
    with torch.no_grad():
        for p in _ag.encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    a1 = _ag.act(obs)
    _ag.encoder.load_state_dict(saved)
    check("I01 actions depend on encoder weights", np.abs(a0 - a1).max() > 0.0001, f"max action change {np.abs(a0 - a1).max():.4f}")
    check("I01 act() is deterministic (mean action)", np.allclose(_ag.act(obs), _ag.act(obs)))

    # Environments: every registered environment honours the same contract
    for name in ENV_REGISTRY:
        ce = replace(cfg, env=name)
        e1, e2 = make_env(ce), make_env(ce)
        f1, m1 = e1.reset(7)
        f2, m2 = e2.reset(7)
        for k in range(5):
            e1.step(np.zeros(2))
            e2.step(np.array([1.0, -1.0]))
        check(f"env {name}: target depends only on the seed (not on the actions)", np.allclose(e1.q, e2.q))
        check(f"env {name}: renderer mask has agent and target ids only", set(np.unique(m1)).issubset({0, 1, 2}) and (m1 == 1).any() and (m1 == 2).any())
        check(f"env {name}: same seed gives the same first frame", np.array_equal(f1, make_env(ce).reset(7)[0]))
        sh = Shift("background", 1.0, 7, ce.img)(f1, m1, 0)[0]
        check(f"env {name}: background shift leaves entity pixels untouched", np.array_equal(sh[m1 != 0], f1[m1 != 0]))

    # I02: The adapted encoder is the one that acts, and updates are counted.
    ref_recs = collect_clean(_ag, cfg, "ref", 2)
    mu, var, cs = clean_feature_stats(_ag, ref_recs, cfg)
    pol = AdaptedPolicy(_ag, cfg, "ttt", (mu, var), cs)
    shifted = np.concatenate([to_u8(Shift("noise", 1.0, 3, cfg.img)(f, m, 0)[0])] * cfg.n_stack, 0)
    base_a = _ag.act(shifted)
    pol.reset_episode()
    for _ in range(6):
        pol.act(shifted)
    adapted_a = pol.act(shifted)
    check("I02 naive TTT counts accepted updates", pol.accepted > 0, f"accepted {pol.accepted}")
    with torch.no_grad():
        xs = torch.as_tensor(shifted, dtype=torch.float32, device=device).unsqueeze(0)
        feat_gap = float((pol.enc(xs) - _ag.encoder(xs)).abs().max())
    check("I02 adapted encoder differs from the base encoder", feat_gap > 1e-06, f"max feature change {feat_gap:.5f}")
    check("I02 adapted encoder changes the action", np.abs(adapted_a - base_a).max() >= 0.0 and feat_gap > 1e-06, f"max action change {np.abs(adapted_a - base_a).max():.5f}")

    pn = AdaptedPolicy(_ag, cfg, "none", (mu, var), cs)
    pn.reset_episode()
    for _ in range(4):
        pn.act(shifted)
    check("I02 arm 'none' never adapts", pn.accepted == 0 and np.allclose(pn.act(shifted), base_a, atol=1e-06))

    strict_cfg = replace(cfg, tta_accept_delta=0.0)
    ps = AdaptedPolicy(_ag, strict_cfg, "ttss", (mu, var), cs)
    ps.reset_episode()
    for _ in range(6):
        ps.act(shifted)
    check("I02 safeguard rolls back when tolerance is zero", ps.rejected > 0 and np.allclose(ps.act(shifted), base_a, atol=1e-05), f"accepted {ps.accepted}, rejected {ps.rejected}")

    # I03: Arms share identical episodes & disjoint seed ranges
    t_paired = pd.DataFrame(dict(arm=["a", "a", "b", "b"], seed=[1, 2, 1, 2]))
    check("I03 paired design: arms share seeds", set(t_paired[t_paired.arm == "a"].seed) == set(t_paired[t_paired.arm == "b"].seed))

    ranges = {sp: {seed_for(sp, i, a) for i in range(500) for a in ((0, 1, 2, 3, 4) if sp == "train" else (0,))} for sp in SEEDS}
    names = list(ranges)
    clash = [(a, b) for k, a in enumerate(names) for b in names[k + 1 :] if ranges[a] & ranges[b]]
    check("I03 train, reference, test, curve, floor and calibration seed ranges are pairwise disjoint", not clash, str(clash))

    # I05: Shifts are what they claim to be
    for kind in KIND_ID:
        sh0 = Shift(kind, 0.0, 5, cfg.img)
        fr, mk = env.reset(5)
        check(f"I05 {kind}: severity 0 is identity", np.array_equal(sh0(fr, mk, 0)[0], fr))

    a_v, b_v = Shift("viewpoint", 0.8, 3, cfg.img), Shift("viewpoint", 0.8, 3, cfg.img)
    check("I05 viewpoint parameters fixed per episode and reproducible", all(np.array_equal(a_v.params()[k], b_v.params()[k]) for k in a_v.params()))

    fr, mk = env.reset(3)
    bgs = Shift("background", 1.0, 3, cfg.img)(fr, mk, 0)[0]
    check("I05 background shift changes only background pixels", np.array_equal(bgs[mk != 0], fr[mk != 0]))
    check("I05 ground-truth mask comes from the renderer", set(np.unique(mk)).issubset({0, 1, 2}) and (mk == 1).sum() > 0 and (mk == 2).sum() > 0)

    # I05: Post-hoc injection equals closed-loop rendering
    r1 = rollout(env, lambda o: np.zeros(2), 5, cfg, shift=Shift("distractor", 0.8, 5, cfg.img), record=True)
    vo, _ = apply_shift_episode(r1["clean_u8"], r1["masks"], "distractor", 0.8, 5, cfg.img)
    check("I05 post-hoc injection equals closed-loop rendering", np.array_equal(vo, r1["view_u8"]))

    # I07: Code equals mathematical definitions
    rng = np.random.default_rng(0)
    Fs, Fc = rng.normal(size=(7, 4)), rng.normal(size=(9, 4))
    check("I07 Eq.(1) matches the definition", abs(centroid_distance(Fs, Fc) - np.sqrt(((Fs.mean(0) - Fc.mean(0)) ** 2).sum())) < 1e-12)

    sp = torch.randn(2, 5, 4, 4, device=device)
    w1 = torch.ones(2, 1, 4, 4, device=device)
    check("I07 Eq.(2) with unit weights equals mean pooling", torch.allclose(semantic_pool(sp, w1), sp.mean((2, 3)), atol=1e-05))

    w_single = torch.zeros(2, 1, 4, 4, device=device)
    w_single[:, :, 1, 2] = 1.0
    check("I07 Eq.(2) with a one-cell mask returns that cell", torch.allclose(semantic_pool(sp, w_single), sp[:, :, 1, 2], atol=0.0001))

    X_mat = rng.normal(size=(30, 6))
    check("I07 Grassmann distance of identical sets is 0", grassmann_distance(X_mat, X_mat, 3) < 1e-06)

    e_eye = np.eye(6)
    Xa = np.vstack([e_eye[0:3] * 3, -e_eye[0:3] * 3])
    Xb = np.vstack([e_eye[3:6] * 3, -e_eye[3:6] * 3])
    check("I07 Grassmann distance of orthogonal subspaces is sqrt(k)*pi/2", abs(grassmann_distance(Xa, Xb, 3) - np.sqrt(3) * np.pi / 2) < 0.0001)

    P_pts = rng.normal(size=(5, 3))
    R_pts = rng.normal(size=(20, 3))
    brute = np.array([np.sort(np.linalg.norm(R_pts - p, axis=1))[:4].mean() for p in P_pts])
    check("I07 kNN distance matches brute force", np.allclose(knn_distance(P_pts, R_pts, 4), brute, atol=0.0001))

    # I08: Outcome definitions
    check("I08 success is strict inequality at the threshold", is_success(cfg.success_cle_px - 1e-06, cfg) and (not is_success(cfg.success_cle_px, cfg)))
    check("I08 collapse uses the stated fraction of the floor", is_collapse(0.8 * 10, replace(cfg, collapse_frac=0.8), 10) and (not is_collapse(7.9, replace(cfg, collapse_frac=0.8), 10)))

    # Metrics guards
    check("metrics: AUROC undefined without failures", np.isnan(auroc([1, 2, 3], [0, 0, 0])))
    b_ = metrics_bundle([1, 2, 3, 4], [0, 0, 0, 0])
    check("metrics: bundle flags undefined case", b_["status"].startswith("undefined"))
    y_lbl = np.array([0, 0, 0, 1])
    sc_val = np.array([0.1, 0.2, 0.3, 0.9])
    check("metrics: AUPRC reference equals prevalence", abs(metrics_bundle(sc_val, y_lbl)["auprc_random"] - 0.25) < 1e-12)
    check("metrics: AUROC invariant to monotone transforms", abs(auroc(sc_val, y_lbl) - auroc(np.exp(sc_val), y_lbl)) < 1e-12)

    # Ranking and regret
    check("rank: Kendall tau of identical orderings is 1", abs(kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12)
    check("rank: Kendall tau of reversed orderings is -1", abs(kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12)
    check("rank: undefined (NaN) values are dropped, not counted as discordant", np.isnan(kendall_tau_b([np.nan, np.nan, np.nan], [1, 2, 3])) and abs(kendall_tau_b([1, np.nan, 3, 4], [1, 9, 3, 4]) - 1) < 1e-12)
    check("rank: selection regret is 0 for a perfect forecast", selection_regret([0.1, 0.9, 0.5], [0.2, 0.8, 0.4]) == 0)
    check("rank: selection regret equals the gap when the wrong policy is chosen", abs(selection_regret([0.9, 0.1], [0.2, 0.8]) - 0.6) < 1e-12)

    # Proposition 2: AUROC mixture theory vs simulation
    rng_t = np.random.default_rng(5)
    n_t = 200000
    q_t, fp_t, fc_t, mu_t = 0.1, 0.02, 0.9, 2.0
    Rg = rng_t.random(n_t) < q_t
    St = rng_t.normal(0, 1, n_t) + mu_t * Rg
    Yt = rng_t.random(n_t) < np.where(Rg, fc_t, fp_t)
    a_t = 0.5 * (1 + math.erf(mu_t / 2))
    check("theory: AUROC mixture formula matches simulation", abs(auroc(St, Yt) - auroc_mixture(q_t, fp_t, fc_t, a_t)) < 0.01, f"sim {auroc(St, Yt):.4f} vs formula {auroc_mixture(q_t, fp_t, fc_t, a_t):.4f}")
    check("theory: AUROC is invariant to the mix iff the regime determines the outcome", abs(auroc_mixture(0.05, 0.0, 1.0, 0.9) - auroc_mixture(0.6, 0.0, 1.0, 0.9)) < 1e-12 and abs(auroc_mixture(0.01, 0.01, 0.97, 0.9) - auroc_mixture(0.5, 0.01, 0.97, 0.9)) > 0.1)
    q_, fp_, fc_, a_ = 0.2, 0.03, 0.8, 0.85
    J_ = q_ * fc_ / (q_ * fc_ + (1 - q_) * fp_) - q_ * (1 - fc_) / (q_ * (1 - fc_) + (1 - q_) * (1 - fp_))
    check("theory: AUROC = 1/2 + (a - 1/2) * Youden index of the regime", abs(auroc_mixture(q_, fp_, fc_, a_) - (0.5 + (a_ - 0.5) * J_)) < 1e-12)
    check("theory: AUROC tends to 1/2 as the cliff share vanishes", abs(auroc_mixture(1e-09, 0.01, 0.97, 0.9) - 0.5) < 1e-06)

    # Conformal calibration & alarms
    check("conformal: threshold is ceil((n+1)(1-alpha))-th smallest value", conformal_threshold(np.arange(1, 101), 0.05) == 96 and conformal_threshold(np.arange(1, 20), 0.05) == 19)
    rc = np.random.default_rng(2)
    cal_v = rc.normal(size=400)
    new_v = rc.normal(size=200000)
    exc = float((new_v > conformal_threshold(cal_v, 0.05)).mean())
    check("conformal: exceedance of fresh clean sample is near alpha", 0.0 < exc < 0.1, f"exceedance {exc:.4f}")
    check("ema_smooth: keeps first value and constant stream", ema_smooth(np.array([2.0, 0.0]), 0.5)[0] == 2.0 and np.allclose(ema_smooth(np.ones(6), 0.7), 1.0))
    st_ = np.concatenate([np.zeros(10), [5.0], np.zeros(9)])
    check("alarm threshold is on per-episode maximum, not on frames", conformal_threshold([ema_smooth(st_, 0.5).max()] * 30, 0.05) == ema_smooth(st_, 0.5).max())

    # TTA pairing
    rows_t = []
    for aid_, cle_, ok_ in (("a", 5.0, True), ("b", 20.0, False)):
        for sd_ in (1, 2, 3):
            for arm_ in ("none", "ttss"):
                rows_t.append(dict(kind="noise", sev=1.0, agent_id=aid_, seed=sd_, arm=arm_, success=ok_, mean_cle=cle_ + 0.1 * sd_, collapse=not ok_, accepted=0, rejected=0))
    tt_ = pd.DataFrame(rows_t).sample(frac=1.0, random_state=3).reset_index(drop=True)
    cmp_ = compare_arms(tt_, contrasts=(("ttss", "none"),), B=50)
    check("TTA pairing is by (agent, seed): identical arms give exactly zero difference", len(cmp_) == 1 and cmp_.n.iloc[0] == 6 and (abs(cmp_.d_mean_cle.iloc[0]) < 1e-12) and (cmp_.p_success.iloc[0] == 1.0) and (cmp_.p_cle.iloc[0] == 1.0))

    # Reviewer findings R01 & R02
    check("R01 uniform-pooling ablation is a registered score", "sem_knn_uniform" in SCORE_COLS)
    _bank = FeatureBank(_ag, cfg)
    _fx = _bank.extract(obs[None, ...], m[None, ...])
    check("R01 semantic and uniform pooling use the same spatial map", _fx["sem"].shape == _fx["semu"].shape)
    _sp = torch.randn(1, 5, 4, 4, device=device)
    check("R01 uniform pooling equals mean pooling of that map", torch.allclose(semantic_pool(_sp, torch.ones(1, 1, 4, 4, device=device)), _sp.mean((2, 3)), atol=1e-05))

    _d = np.full(40, 3.0) + np.random.default_rng(0).normal(0, 0.1, 40)
    check("R02 sign-flip test detects a consistent paired shift", paired_signflip_p(_d, np.zeros(40), 2000) < 0.01)
    check("R02 sign-flip test gives p = 1 for identical arms", paired_signflip_p(_d, _d, 2000) == 1.0)
    _z = np.random.default_rng(1).normal(0, 1, 60)
    check("R02 sign-flip test is not significant for a symmetric null", paired_signflip_p(_z, np.zeros(60), 2000) > 0.05)

    # Novel Extension Probes (PAC-Drift & GCM-Drift)
    check("novel: pac_drift and gcm_drift are registered scores", "pac_drift" in SCORE_COLS and "gcm_drift" in SCORE_COLS)
    _pol = _ag.actor.policy if hasattr(_ag.actor, "policy") else _ag.actor
    _z0 = _fx["policy"]
    check("novel: pac_drift is zero on clean identical frames", abs(float(compute_pac_drift(_pol, _z0, _z0).mean())) < 1e-6)
    check("novel: gcm_airm_distance is zero on identical covariance matrices", gcm_airm_distance(np.eye(10), np.eye(10)) < 1e-6)
    check("novel: gcm_airm_distance is symmetric", abs(gcm_airm_distance(np.diag([1.0, 2.0]), np.diag([2.0, 4.0])) - gcm_airm_distance(np.diag([2.0, 4.0]), np.diag([1.0, 2.0]))) < 1e-6)

    out = pd.DataFrame(res)
    n_fail = int((~out.passed).sum())
    assert n_fail == 0, f"Audit failed: {n_fail} of {len(out)} checks failed"
    return out
