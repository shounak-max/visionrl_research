"""
Rollout collector and dataset caching module.

Collects trajectory rollouts across environments, attaches provenance records,
and manages dataset caching in PyTorch .pt / HDF5 shards.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from envs import make_env, BaseVisualEnvironment, PerturbationConfig
from policies.base_policy import BasePolicy
from provenance import stamp_provenance, verify_provenance


@dataclass
class RolloutBatch:
    """Trajectory rollout dataset data contract."""

    observations: np.ndarray  # [T, H, W, C] float32
    actions: np.ndarray  # [T, action_dim] or [T]
    rewards: np.ndarray  # [T] float32
    next_observations: np.ndarray  # [T, H, W, C] float32
    dones: np.ndarray  # [T] bool
    episode_ids: np.ndarray  # [T] int32
    perturbation_tag: str  # e.g. 'clean', 'bg_shift_0.5'
    ground_truth_mask: Optional[np.ndarray] = None  # [T, H, W] float32 or None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)


class RolloutCollector:
    """Rollout collector for visual RL research environments."""

    @staticmethod
    def collect(
        env_id: str,
        policy: BasePolicy,
        perturbation_config: Optional[PerturbationConfig] = None,
        n_episodes: int = 5,
        seed: int = 42,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> RolloutBatch:
        """Executes n_episodes of policy rollouts under perturbation_config.

        Returns RolloutBatch containing all trajectory transitions and provenance metadata.
        """
        if perturbation_config is None:
            perturbation_config = PerturbationConfig()

        if config_dict is None:
            config_dict = {
                "env_id": env_id,
                "n_episodes": n_episodes,
                "seed": seed,
                "perturbation": perturbation_config.to_tag(),
            }

        env = make_env(
            env_id=env_id, seed=seed, perturbation_config=perturbation_config
        )

        all_obs = []
        all_next_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_ep_ids = []
        all_masks = []

        has_mask = False

        for ep_idx in range(n_episodes):
            ep_seed = seed + ep_idx * 1000
            obs, info = env.reset(seed=ep_seed)

            mask = info.get("ground_truth_mask", None)
            if mask is not None:
                has_mask = True

            done = False
            truncated = False

            while not (done or truncated):
                action = policy.act(obs)
                next_obs, reward, done, truncated, next_info = env.step(action)

                all_obs.append(obs)
                all_next_obs.append(next_obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_dones.append(done or truncated)
                all_ep_ids.append(ep_idx)

                if mask is not None:
                    all_masks.append(mask)

                obs = next_obs
                mask = next_info.get("ground_truth_mask", mask)

        env.close()

        obs_arr = np.array(all_obs, dtype=np.float32)
        next_obs_arr = np.array(all_next_obs, dtype=np.float32)
        act_arr = np.array(all_actions)
        rew_arr = np.array(all_rewards, dtype=np.float32)
        done_arr = np.array(all_dones, dtype=bool)
        ep_arr = np.array(all_ep_ids, dtype=np.int32)
        mask_arr = np.array(all_masks, dtype=np.float32) if has_mask else None

        provenance_rec = stamp_provenance(
            config=config_dict,
            seed=seed,
            env_id=env_id,
            perturbation_tag=perturbation_config.to_tag(),
            extra_metadata={"n_episodes": n_episodes, "total_steps": len(obs_arr)},
        )

        return RolloutBatch(
            observations=obs_arr,
            actions=act_arr,
            rewards=rew_arr,
            next_observations=next_obs_arr,
            dones=done_arr,
            episode_ids=ep_arr,
            perturbation_tag=perturbation_config.to_tag(),
            ground_truth_mask=mask_arr,
            provenance=provenance_rec,
        )


def save_rollout_batch(batch: RolloutBatch, cache_dir: str = "rollout_cache") -> str:
    """Saves RolloutBatch to PyTorch binary file in cache_dir with provenance stamp."""
    os.makedirs(cache_dir, exist_ok=True)
    env_id = batch.provenance.get("env_id", "env")
    p_tag = batch.perturbation_tag
    seed = batch.provenance.get("seed", 42)
    cfg_hash = batch.provenance.get("config_hash", "00000000")

    filename = f"rollout_{env_id}_{p_tag}_seed{seed}_{cfg_hash}.pt"
    filepath = os.path.join(cache_dir, filename)

    torch.save(
        {
            "observations": torch.tensor(batch.observations),
            "actions": torch.tensor(batch.actions),
            "rewards": torch.tensor(batch.rewards),
            "next_observations": torch.tensor(batch.next_observations),
            "dones": torch.tensor(batch.dones),
            "episode_ids": torch.tensor(batch.episode_ids),
            "perturbation_tag": batch.perturbation_tag,
            "ground_truth_mask": torch.tensor(batch.ground_truth_mask)
            if batch.ground_truth_mask is not None
            else None,
            "provenance": batch.provenance,
        },
        filepath,
    )
    return filepath


def load_rollout_batch(filepath: str) -> RolloutBatch:
    """Loads RolloutBatch from cached binary file and verifies provenance."""
    data = torch.load(filepath, weights_only=False)
    provenance = data.get("provenance", {})

    if not verify_provenance(provenance):
        print(f"[Warning] Provenance check failed or missing for '{filepath}'.")

    return RolloutBatch(
        observations=data["observations"].numpy(),
        actions=data["actions"].numpy(),
        rewards=data["rewards"].numpy(),
        next_observations=data["next_observations"].numpy(),
        dones=data["dones"].numpy(),
        episode_ids=data["episode_ids"].numpy(),
        perturbation_tag=data["perturbation_tag"],
        ground_truth_mask=data["ground_truth_mask"].numpy()
        if data["ground_truth_mask"] is not None
        else None,
        provenance=provenance,
    )
