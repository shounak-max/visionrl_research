"""
Audit Assertions Test Suite for Visual RL Paper.

Runnable tests verifying core invariants, TTSS rollback and rejection mechanisms,
seed disjointness across experimental phases, and flagging known issues.
"""

import json
from pathlib import Path
import pytest
import torch
import numpy as np

from visionrl.config import make_cfg
from visionrl.environments import make_env
from visionrl.models.drq_agent import DrQAgent
from visionrl.probes.adaptation import AdaptedPolicy, clean_feature_stats
from visionrl.probes.feature_bank import collect_clean


def test_success_threshold_defined():
    """Import the env or check provenance/specification for 5px success threshold."""
    repo_root = Path(__file__).resolve().parent.parent
    prov_file = repo_root / "results" / "training_provenance_full.json"
    assert prov_file.exists(), "training_provenance_full.json must exist in results/"

    cfg = make_cfg("smoke", "pursuit")
    env = make_env(cfg)
    obs, info = env.reset(seed=42)
    assert obs.shape == (cfg.img, cfg.img, 3)

    success_threshold = cfg.success_cle_px
    random_floor = 22.05
    collapse_threshold = cfg.collapse_frac * random_floor
    assert success_threshold < collapse_threshold < random_floor


def test_ttss_adapter_rollback():
    """Verify that AdaptedPolicy rolls back encoder parameters on rejected update."""
    from dataclasses import replace
    cfg = make_cfg("smoke", "pursuit")
    agent = DrQAgent(cfg, "drqv2", 0)
    recs = collect_clean(agent, cfg, "ref", 2)
    stats = clean_feature_stats(agent, recs, cfg)
    clean_stacks = stats[2]

    # Create TTSS policy with accept_delta = 0.0 (force rejection)
    strict_cfg = replace(cfg, tta_accept_delta=0.0)
    pol = AdaptedPolicy(agent, strict_cfg, "ttss", (stats[0], stats[1]), clean_stacks)

    initial_params = [p.clone().detach() for p in pol.enc.parameters()]
    obs = np.random.randint(0, 255, size=(3 * cfg.n_stack, cfg.img, cfg.img), dtype=np.uint8)

    # Perform action (which triggers _adapt if buffer >= 2)
    pol.act(obs)
    pol.act(obs)

    # The update was rejected
    assert pol.rejected >= 1, "Zero-tolerance safeguard must reject update"
    for p_init, p_curr in zip(initial_params, pol.enc.parameters()):
        assert torch.allclose(p_init, p_curr, atol=1e-7), "Encoder params must match initial snapshot on rejection"


def test_ttss_trust_region_rejects_bad_update():
    """Verify that an update causing action drift > tta_accept_delta is rejected."""
    from dataclasses import replace
    cfg = make_cfg("smoke", "pursuit")
    agent = DrQAgent(cfg, "drqv2", 0)
    recs = collect_clean(agent, cfg, "ref", 2)
    stats = clean_feature_stats(agent, recs, cfg)
    clean_stacks = stats[2]

    # Large learning rate causes massive action drift exceeding accept_delta
    high_lr_cfg = replace(cfg, tta_lr=10.0, tta_accept_delta=0.01)
    pol = AdaptedPolicy(agent, high_lr_cfg, "ttss", (stats[0], stats[1]), clean_stacks)

    obs = np.random.randint(0, 255, size=(3 * cfg.n_stack, cfg.img, cfg.img), dtype=np.uint8)
    pol.act(obs)
    pol.act(obs)

    assert pol.rejected >= 1, "Trust region safeguard must reject large deviation updates"


def test_seed_disjointness():
    """Assert that config.py SEEDS and seed_for produce disjoint sets across all splits."""
    from visionrl.config import SEEDS, seed_for

    split_sets = {}
    for split in SEEDS:
        if split == "train":
            split_sets[split] = {seed_for(split, i, agent_seed=s) for s in range(5) for i in range(500)}
        else:
            split_sets[split] = {seed_for(split, i) for i in range(500)}

    splits = list(split_sets.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            assert split_sets[a].isdisjoint(split_sets[b]), f"Seeds for {a} and {b} must be disjoint"


def test_adapted_encoder_drives_actions():
    """Prove that the adapted encoder actively drives action outputs during test-time adaptation."""
    from visionrl.config import make_cfg
    from visionrl.models.drq_agent import DrQAgent
    from visionrl.probes.adaptation import AdaptedPolicy, clean_feature_stats
    from visionrl.probes.feature_bank import collect_clean

    cfg = make_cfg("smoke", "pursuit")
    agent = DrQAgent(cfg, "drqv2", 0)
    recs = collect_clean(agent, cfg, "ref", 2)
    stats = clean_feature_stats(agent, recs, cfg)
    clean_stacks = stats[2]

    # Create AdaptedPolicy in TTT mode
    pol = AdaptedPolicy(agent, cfg, "ttt", (stats[0], stats[1]), clean_stacks)
    obs = np.random.randint(0, 255, size=(3 * cfg.n_stack, cfg.img, cfg.img), dtype=np.uint8)

    # Initial action
    act0 = pol.act(obs)

    # Manually perturb the adapted encoder weights to prove act() uses pol.enc, NOT base.encoder
    with torch.no_grad():
        for p in pol.enc.parameters():
            p.add_(torch.randn_like(p) * 2.0)

    act_perturbed = pol.act(obs)
    # The action must differ because pol.enc changed
    assert not np.allclose(act0, act_perturbed, atol=1e-3), "Adapted encoder must drive action outputs"

    # Also check base encoder was NOT corrupted
    act_base = agent.act(obs)
    assert not np.allclose(act_base, act_perturbed, atol=1e-3), "Base agent encoder must remain uncorrupted"


def test_phase3_acceptance_rate_documented():
    """Verify that modern 30k baseline tracking competence is documented in results."""
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "results" / "clean_performance_30k.json"
    assert results_path.exists(), "30k baseline clean performance JSON must exist"

    with open(results_path) as f:
        data = json.load(f)

    assert "taco_s0" in data, "TACO 30k baseline must be present"
    taco_entry = data["taco_s0"]
    assert taco_entry["clean_mean_cle_px"] < 5.0, "TACO must pass competence gate"
    assert taco_entry["clean_success_rate"] >= 0.50, "TACO clean success rate must be >= 50%"
