"""Experiments package initialization."""

from visionrl.experiments.claim_registry import ClaimRegistry, config_tables, git_commit
from visionrl.experiments.hypotheses import cross_env_summary, evaluate_hypotheses, paired_error_gain
from visionrl.experiments.latency import measure_latency
from visionrl.experiments.runner import run_pipeline
from visionrl.experiments.smoke_recipe import smoke_extra_recipe
from visionrl.experiments.study import run_study, study_one_agent
from visionrl.experiments.train import (
    TrainingPaused,
    eval_cell,
    eval_discrete,
    random_floor_cle,
    train_agent,
    train_ppo,
    training_gate,
)

__all__ = [
    "eval_cell",
    "random_floor_cle",
    "TrainingPaused",
    "train_agent",
    "training_gate",
    "train_ppo",
    "eval_discrete",
    "study_one_agent",
    "run_study",
    "measure_latency",
    "smoke_extra_recipe",
    "ClaimRegistry",
    "config_tables",
    "git_commit",
    "paired_error_gain",
    "evaluate_hypotheses",
    "cross_env_summary",
    "run_pipeline",
]
