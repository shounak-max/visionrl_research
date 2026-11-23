"""Environment module initialization and registry."""

from typing import Any, Dict

from visionrl.config import Cfg
from visionrl.environments.base import from_u8, is_collapse, is_success, make_stacks, rollout, to_u8
from visionrl.environments.discrete_pursuit import DiscretePursuit
from visionrl.environments.procgen_adapter import ProcgenCoinRun, procgen_available
from visionrl.environments.pursuit import PursuitEnv
from visionrl.environments.reacher import ReacherEnv

ENV_REGISTRY: Dict[str, Any] = {
    "pursuit": PursuitEnv,
    "reacher": ReacherEnv,
}


def make_env(cfg: Cfg) -> Any:
    """Instantiate the registered environment specified in cfg.env."""
    if cfg.env not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {cfg.env}. Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[cfg.env](cfg)


__all__ = [
    "PursuitEnv",
    "ReacherEnv",
    "DiscretePursuit",
    "ProcgenCoinRun",
    "procgen_available",
    "ENV_REGISTRY",
    "make_env",
    "is_success",
    "is_collapse",
    "to_u8",
    "from_u8",
    "make_stacks",
    "rollout",
]
