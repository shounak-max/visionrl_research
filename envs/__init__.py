"""
Environment package initialization.
"""

from envs.base_env import BaseVisualEnvironment, PerturbationConfig
from envs.procgen_coinrun import ProcgenCoinRunEnv
from envs.procgen_ood import ProcgenOODEnv, get_procgen_ood_configs
from envs.visual_pursuit import VisualPursuitEnv


def make_env(
    env_id: str,
    seed: int = 42,
    perturbation_config: PerturbationConfig = None,
) -> BaseVisualEnvironment:
    """Factory function to instantiate visual RL research environments.

    Supported env_ids:
    - 'coinrun' or 'procgen:procgen-coinrun-v0' -> ProcgenCoinRunEnv
    - 'procgen_ood' -> ProcgenOODEnv
    - 'visual_pursuit', 'tracking', 'SingleObjectTracking-v0' -> VisualPursuitEnv
    """
    if env_id in ["coinrun", "procgen:procgen-coinrun-v0"]:
        return ProcgenCoinRunEnv(seed=seed, perturbation_config=perturbation_config)
    elif env_id == "procgen_ood":
        return ProcgenOODEnv(seed=seed, perturbation_config=perturbation_config)
    elif env_id in ["visual_pursuit", "tracking", "SingleObjectTracking-v0"]:
        return VisualPursuitEnv(seed=seed, perturbation_config=perturbation_config)
    else:
        raise ValueError(f"Unknown env_id: '{env_id}'")
