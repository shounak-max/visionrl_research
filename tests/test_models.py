"""Unit tests for models, augmentations, replay buffer, and agent updates."""

import numpy as np
import pytest
import torch

from visionrl.config import make_cfg
from visionrl.models import (
    Actor,
    Critic,
    DrQAgent,
    Encoder,
    EpisodeReplay,
    PPOAgent,
    RandomShift,
    random_overlay,
)


def test_encoder_and_actor_critic():
    in_ch, img = 9, 32
    enc = Encoder(in_ch, img)
    x = torch.rand(4, in_ch, img, img) * 255.0
    sp = enc.spatial(x)
    assert sp.shape[0] == 4
    h = enc(x)
    assert h.shape == (4, enc.repr_dim)

    actor = Actor(enc.repr_dim, 2, 32, 64)
    act = actor(h)
    assert act.shape == (4, 2)
    assert (act >= -1.0).all() and (act <= 1.0).all()

    critic = Critic(enc.repr_dim, 2, 32, 64)
    q1, q2 = critic(h, act)
    assert q1.shape == (4, 1) and q2.shape == (4, 1)


def test_augmentations():
    x = torch.rand(2, 9, 32, 32) * 255.0
    shift = RandomShift(pad=3)
    xs = shift(x)
    assert xs.shape == x.shape

    overlay = random_overlay(x)
    assert overlay.shape == x.shape


def test_drq_agent_recipes():
    cfg = make_cfg("smoke", "pursuit")
    for variant in ("drqv2", "svea", "noaug", "taco"):
        ag = DrQAgent(cfg, variant=variant, seed=0)
        obs_u8 = np.zeros((3 * cfg.n_stack, cfg.img, cfg.img), dtype=np.uint8)
        act = ag.act(obs_u8)
        assert act.shape == (2,)

        # Mock update
        B = 4
        rnd = lambda *s: torch.rand(*s, device=ag.device)
        batch = (
            rnd(B, 3 * cfg.n_stack, cfg.img, cfg.img) * 255.0,
            rnd(B, 2) * 2 - 1,
            rnd(B, 1),
            torch.full((B, 1), cfg.discount ** cfg.n_step, device=ag.device),
            rnd(B, 3 * cfg.n_stack, cfg.img, cfg.img) * 255.0,
        )
        aseq = rnd(B, cfg.n_step * 2) if variant == "taco" else None
        lc, la = ag.update(batch, 0.2, aseq=aseq)
        assert np.isfinite(lc) and np.isfinite(la)


def test_ppo_agent():
    cfg = make_cfg("smoke", "pursuit")
    ppo = PPOAgent(5, 3 * cfg.n_stack, cfg.img)
    obs_u8 = np.zeros((3 * cfg.n_stack, cfg.img, cfg.img), dtype=np.uint8)
    a_greedy = ppo.act(obs_u8, rule="greedy")
    a_sampled = ppo.act(obs_u8, rule="sampled")
    assert 0 <= a_greedy < 5
    assert 0 <= a_sampled < 5
