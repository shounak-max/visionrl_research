"""Configuration, pre-registration rules, and seed definitions for visionrl."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Cfg:
    mode: str = "smoke"
    env: str = "pursuit"             # pursuit | reacher
    # ---- environment -------------------------------------------------------
    img: int = 32
    ep_len: int = 20
    n_stack: int = 3
    max_step_px: float = 2.0
    target_speed: float = 0.8
    agent_radius: float = 2.0
    link_frac: float = 0.30          # reacher: link length as a fraction of the image width
    max_dtheta: float = 0.35         # reacher: largest joint-angle change per step (radians)
    # Success and collapse are defined ONCE, here, and used everywhere.
    success_cle_px: float = 5.0      # success iff episode-mean CLE < this (strict)
    collapse_frac: float = 0.8       # floor collapse iff episode-mean CLE >= frac * random-policy CLE
    # ---- agent training ----------------------------------------------------
    train_steps: int = 800
    seed_steps: int = 200
    batch: int = 32
    feature_dim: int = 32
    hidden: int = 64
    lr: float = 3e-4
    discount: float = 0.99
    n_step: int = 3
    tau: float = 0.01
    update_every: int = 2
    std_start: float = 1.0
    std_end: float = 0.1
    std_decay_steps: int = 500
    std_clip: float = 0.3
    aug_pad: int = 3
    eval_every_eps: int = 10
    n_eval_eps: int = 5
    svea_alpha: float = 0.5
    svea_beta: float = 0.5
    train_gate_success: float = 0.0  # training gate on clean success (0 disables)
    # ---- study grid --------------------------------------------------------
    agents: tuple = ("drqv2", "svea", "noaug")
    agent_seeds: tuple = (0,)
    shift_kinds: tuple = ("background", "noise", "distractor", "viewpoint", "color_gain")
    severities: tuple = (0.0, 0.5, 1.0)
    eps_per_cell: int = 6
    # ---- forecasting probes ------------------------------------------------
    knn_k: int = 5
    ref_eps: int = 8
    ref_max_rows: int = 2000
    frame_stride: int = 1
    clean_fpr: float = 0.05
    ema: float = 0.5
    grassmann_ks: tuple = (3, 5)
    boot_B: int = 200
    cal_clean_eps: int = 10          # clean-only episodes used ONLY to calibrate thresholds (label-free and cheap)
    # ---- test-time adaptation ----------------------------------------------
    tta_kinds: tuple = ("background", "noise")
    tta_severities: tuple = (1.0,)
    tta_eps: int = 6
    tta_lr: float = 3e-4
    tta_mm_weight: float = 1.0
    tta_accept_delta: float = 0.40
    tta_buffer: int = 8
    tta_clean_batch: int = 32
    tta_agents: int = 99             # run the (slow, secondary) TTA study on the first N agents only
    # ---- additions driven by the ICVGIP reviews ---------------------------------------------------
    extra_recipes: tuple = ()        # e.g. ("taco",): training recipes in addition to cfg.agents
    taco_dim: int = 64
    taco_weight: float = 1.0
    taco_temp: float = 0.1
    latency_reps: int = 30           # repetitions for the latency protocol (median and p90 are reported)
    tta_perm: int = 5000             # sign-flip permutations for the paired mean-CLE test


def _make_cfg(mode: str, env: str = "pursuit") -> Cfg:
    if mode == "smoke":
        return Cfg(mode="smoke", env=env)
    if mode == "sandbox":      # sized for ONE CPU core: small images, short episodes, 3 recipes x 2 seeds
        return Cfg(
            mode="sandbox", env=env, img=24, ep_len=24, max_step_px=2.0, target_speed=0.6, success_cle_px=3.5,
            train_steps=16000, seed_steps=480, batch=32, feature_dim=32, hidden=64, lr=3e-4, std_decay_steps=8000,
            aug_pad=3, eval_every_eps=40, n_eval_eps=8, train_gate_success=0.4, agent_seeds=(0, 1),
            severities=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), eps_per_cell=24, ref_eps=16, ref_max_rows=1500, frame_stride=2,
            grassmann_ks=(3, 5), boot_B=300, cal_clean_eps=40, tta_kinds=("noise", "background"), tta_severities=(0.8,), tta_eps=8,
            tta_agents=2, tta_lr=1e-3
        )
    if mode == "pilot":
        return Cfg(
            mode="pilot", env=env, img=48, ep_len=50, train_steps=20000, seed_steps=1000, batch=128,
            feature_dim=50, hidden=256, lr=1e-4, std_decay_steps=15000, aug_pad=4,
            eval_every_eps=20, n_eval_eps=10, train_gate_success=0.5,
            agent_seeds=(0, 1, 2), severities=tuple(np.round(np.linspace(0, 1, 11), 2)),
            eps_per_cell=40, ref_eps=30, frame_stride=5, grassmann_ks=(5, 10, 20),
            boot_B=1000, cal_clean_eps=100, tta_eps=30, tta_severities=(0.6, 0.8, 1.0), tta_agents=3, tta_lr=1e-3
        )
    if mode == "full":
        return Cfg(
            mode="full", env=env, img=64, ep_len=100, train_steps=200000, seed_steps=4000, batch=256,
            feature_dim=50, hidden=1024, lr=1e-4, std_decay_steps=100000, aug_pad=4,
            eval_every_eps=50, n_eval_eps=20, train_gate_success=0.7,
            agent_seeds=(0, 1, 2, 3, 4), severities=tuple(np.round(np.linspace(0, 1, 11), 2)),
            eps_per_cell=100, ref_eps=60, ref_max_rows=4000, frame_stride=5,
            grassmann_ks=(5, 10, 20), boot_B=2000, cal_clean_eps=200, tta_eps=100,
            tta_severities=(0.6, 0.8, 1.0), tta_agents=5, tta_lr=3e-4, tta_accept_delta=0.40
        )
    raise ValueError(f"Unknown mode: {mode}")


def make_cfg(mode: str, env: str = "pursuit") -> Cfg:
    cfg = _make_cfg(mode, env)
    if env == "reacher" and mode != "smoke":      # inverse kinematics from pixels needs a larger budget
        cfg = replace(cfg, train_steps=3 * cfg.train_steps, std_decay_steps=3 * cfg.std_decay_steps, seed_steps=2 * cfg.seed_steps)
    return cfg


# Seed ranges. Disjoint by construction so that training, calibration and test never share an episode.
SEEDS: Dict[str, int] = dict(train=10_000, ref=20_000, test=30_000, eval_curve=40_000, floor=50_000, cal=60_000)


def seed_for(split: str, i: int, agent_seed: int = 0) -> int:
    return SEEDS[split] + 1000 * int(agent_seed) * (split == "train") + int(i)


# Pre-registered decision rules
PREREG = dict(
    version="1.1",
    h1_min_cliffness=0.5, h1_min_total_drop=0.3, h1_min_share=0.5,        # abrupt failure is common
    h2_max_median_abs_err=0.15, h2_require_beats=("knn_pixel", "knn_random"), h2_ci_excludes_zero=True,   # forecast error in severity units
    h3_min_tau=0.5, h3_max_regret=0.10,                                   # policy ranking and selection
    h4_min_tau_gap=0.3,                                                   # policy-conditional beats policy-agnostic
    h5_max_auroc_error=0.05,                                              # prevalence-portable summary
    h6_alpha=0.05,                                                        # secondary: safeguarded TTA versus naive TTA
    h7_max_false_alarm_factor=2.0,                                        # clean false alarms within 2x nominal
    h_novel_max_err=0.10, h_novel_min_tau=0.65, h_novel_max_regret=0.05, # Novel PAC-Drift extension
    min_envs_supported=2,                                                 # H2 to H5 are claimed as general only if they hold in this many environments
    proposed="knn_policy", novel="pac_drift", agnostic=("knn_pixel", "knn_random"), policy_conditional=("knn_policy", "pac_drift", "act_disagree"),
)
PREREG_HASH = hashlib.sha256(json.dumps(PREREG, sort_keys=True).encode()).hexdigest()[:16]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
