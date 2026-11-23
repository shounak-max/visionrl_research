"""Integration test running all 38 pre-registered audit checks."""

import pytest

from visionrl.audit.audit_suite import run_audit_tests
from visionrl.config import make_cfg
from visionrl.models.drq_agent import DrQAgent


def test_full_audit_suite():
    cfg = make_cfg("smoke", "pursuit")
    agent = DrQAgent(cfg, "drqv2", seed=0)
    agents = {"drqv2_s0": agent}

    audit_df = run_audit_tests(agents, cfg)
    assert len(audit_df) >= 38
    assert audit_df["passed"].all(), f"Failed tests: {audit_df[~audit_df['passed']]['test'].tolist()}"
