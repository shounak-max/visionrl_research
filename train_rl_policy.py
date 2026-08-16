"""
RL Policy Training Pipeline for Visual RL Generalization Research.

Trains EndToEndCNNEncoder via DrQ-v2 on VisualPursuit / Procgen environments,
ensuring that representation geometry comparisons in Phase 2 evaluate
genuinely trained task representations rather than random PyTorch initialization.
"""

import argparse
import os
import sys
import numpy as np
import torch

from envs import VisualPursuitEnv, ProcgenOODEnv, PerturbationConfig
from policies.drq_v2 import DrQv2Policy, ReplayBuffer


def train_policy(
    env_name: str = "visual_pursuit",
    total_steps: int = 500,
    batch_size: int = 32,
    checkpoint_path: str = "checkpoints/drq2_cnn_encoder.pt",
    seed: int = 42,
) -> DrQv2Policy:
    """Trains DrQ-v2 policy end-to-end and saves learned encoder weights."""
    print("=" * 80)
    print(f"   TRAINING DrQ-v2 RL POLICY ON '{env_name}' (Steps: {total_steps})")
    print("=" * 80)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_name == "visual_pursuit":
        env = VisualPursuitEnv(seed=seed)
        action_dim = 2
        is_continuous = True
    else:
        env = ProcgenOODEnv(seed=seed, perturbation_config=PerturbationConfig(axis="clean"))
        action_dim = env.action_space_dim
        is_continuous = False

    policy = DrQv2Policy(
        action_dim=action_dim,
        embedding_dim=512,
        is_continuous=is_continuous,
        lr=1e-3,
    )
    buffer = ReplayBuffer(capacity=5000, obs_shape=env.image_shape, action_dim=action_dim)

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    episodes = 0

    print("Collecting transitions and performing gradient updates...")
    for step in range(total_steps):
        action = policy.act(obs, deterministic=False)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        # Store transition
        act_arr = np.array(action, dtype=np.float32) if is_continuous else np.array([action], dtype=np.float32)
        buffer.add(obs, act_arr, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward

        if done:
            obs, _ = env.reset(seed=seed + episodes)
            episodes += 1
            ep_reward = 0.0

        # Perform DrQ-v2 policy and encoder update
        if buffer.size >= batch_size:
            batch = buffer.sample(batch_size=batch_size)
            metrics = policy.update(batch)
            if (step + 1) % 100 == 0:
                print(
                    f"Step {step + 1}/{total_steps} | "
                    f"Critic Loss: {metrics['critic_loss']:.4f} | "
                    f"Actor Loss: {metrics['actor_loss']:.4f}"
                )

    env.close()

    # Save trained checkpoint
    policy.save_checkpoint(checkpoint_path)
    print(f"\n[SUCCESS] Trained DrQ-v2 encoder checkpoint saved to: '{checkpoint_path}'")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DrQ-v2 visual RL policy.")
    parser.add_argument("--env", type=str, default="visual_pursuit", choices=["visual_pursuit", "procgen_ood"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/drq2_cnn_encoder.pt")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_policy(
        env_name=args.env,
        total_steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    )
