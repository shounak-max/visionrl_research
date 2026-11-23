"""Unit tests for environment contracts and outcome definitions."""

import numpy as np
import pytest

from visionrl.config import make_cfg
from visionrl.environments import (
    DiscretePursuit,
    PursuitEnv,
    ReacherEnv,
    is_collapse,
    is_success,
    make_env,
    rollout,
)


def test_pursuit_contract():
    cfg = make_cfg("smoke", "pursuit")
    env = PursuitEnv(cfg)
    f, m = env.reset(42)
    assert f.shape == (cfg.img, cfg.img, 3)
    assert m.shape == (cfg.img, cfg.img)
    assert set(np.unique(m)).issubset({0, 1, 2})
    assert (m == 1).any() and (m == 2).any()

    # Step
    img, mask, r, cle, done = env.step(np.array([0.5, -0.5]))
    assert img.shape == f.shape
    assert isinstance(r, float)
    assert isinstance(cle, float)
    assert isinstance(done, bool)


def test_reacher_contract():
    cfg = make_cfg("smoke", "reacher")
    env = ReacherEnv(cfg)
    f, m = env.reset(42)
    assert f.shape == (cfg.img, cfg.img, 3)
    assert m.shape == (cfg.img, cfg.img)
    assert set(np.unique(m)).issubset({0, 1, 2})
    assert (m == 1).any() and (m == 2).any()

    img, mask, r, cle, done = env.step(np.array([0.2, -0.2]))
    assert img.shape == f.shape
    assert isinstance(r, float)
    assert isinstance(cle, float)


def test_discrete_pursuit():
    cfg = make_cfg("smoke", "pursuit")
    denv = DiscretePursuit(cfg)
    obs = denv.reset(10)
    assert obs.shape == (3 * cfg.n_stack, cfg.img, cfg.img)
    o2, r, done, info = denv.step(1)
    assert o2.shape == obs.shape
    assert "cle" in info


def test_outcome_definitions():
    cfg = make_cfg("smoke", "pursuit")
    assert is_success(cfg.success_cle_px - 0.01, cfg) is True
    assert is_success(cfg.success_cle_px, cfg) is False
    assert is_collapse(0.8 * 10.0, cfg, 10.0) is True
    assert is_collapse(7.9, cfg, 10.0) is False
