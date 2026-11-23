"""Unit tests for observation shifts and post-hoc injection equivalence."""

import numpy as np
import pytest

from visionrl.config import make_cfg
from visionrl.environments import PursuitEnv, rollout
from visionrl.shifts import KIND_ID, Shift, apply_shift_episode


def test_shifts_identity():
    cfg = make_cfg("smoke", "pursuit")
    env = PursuitEnv(cfg)
    f, m = env.reset(123)

    for kind in KIND_ID:
        sh = Shift(kind, 0.0, 123, cfg.img)
        sf, sm = sh(f, m, 0)
        assert np.array_equal(sf, f), f"Severity 0 failed for {kind}"
        assert np.array_equal(sm, m)


def test_post_hoc_equals_closed_loop():
    cfg = make_cfg("smoke", "pursuit")
    env = PursuitEnv(cfg)
    r1 = rollout(env, lambda o: np.zeros(2), 5, cfg, shift=Shift("distractor", 0.8, 5, cfg.img), record=True)
    vo, _ = apply_shift_episode(r1["clean_u8"], r1["masks"], "distractor", 0.8, 5, cfg.img)
    assert np.array_equal(vo, r1["view_u8"])


def test_background_shift_preserves_foreground():
    cfg = make_cfg("smoke", "pursuit")
    env = PursuitEnv(cfg)
    f, m = env.reset(7)
    sh = Shift("background", 1.0, 7, cfg.img)
    sf, _ = sh(f, m, 0)
    assert np.array_equal(sf[m != 0], f[m != 0])
