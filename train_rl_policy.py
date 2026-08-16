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
    env_name: str = "procgen_ood",
    total_steps: int = 1500,
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
        lr=5e-4,
    )
    buffer = ReplayBuffer(capacity=5000, obs_shape=env.image_shape, action_dim=action_dim if is_continuous else 1)

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    episodes = 0
    loss_history = []

    print(f"Environment initialized: {env.backend_mode if hasattr(env, 'backend_mode') else env_name}")
    print("Warm-up & training loop running...")

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
            loss_history.append(metrics["total_loss"])
            if (step + 1) % 250 == 0:
                recent_loss = float(np.mean(loss_history[-100:]))
                print(
                    f"Step {step + 1:4d}/{total_steps} | "
                    f"Critic Loss: {metrics['critic_loss']:8.4f} | "
                    f"Actor Loss: {metrics['actor_loss']:8.4f} | "
                    f"Avg Loss (last 100): {recent_loss:8.4f}"
                )

    env.close()

    if loss_history:
        init_loss = float(np.mean(loss_history[:50])) if len(loss_history) >= 50 else loss_history[0]
        final_loss = float(np.mean(loss_history[-50:]))
        print(f"\n[Training Summary] Initial Loss: {init_loss:.4f} -> Final Loss: {final_loss:.4f} (Delta: {final_loss - init_loss:.4f})")

    # Save trained checkpoint
    policy.save_checkpoint(checkpoint_path)
    print(f"[SUCCESS] Trained DrQ-v2 encoder checkpoint saved to: '{checkpoint_path}'")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DrQ-v2 visual RL policy.")
    parser.add_argument("--env", type=str, default="procgen_ood", choices=["visual_pursuit", "procgen_ood"])
    parser.add_argument("--steps", type=int, default=1500)
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
