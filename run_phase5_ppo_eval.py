"""
Phase 5 PPO Policy Evaluation Runner on SingleObjectTracking-v0 (VisualPursuitEnv).

Evaluates PPO Policy (with End-to-End CNN Encoder) against a Random Policy baseline across 5 seeds.
Calculates per-seed:
1. Mean Closest-Labeled-Error (CLE, in pixels).
2. Success Rate (fraction of timesteps with tracking_error_px <= 5.0 px).
3. Statistical significance testing (Two-sample Welch's t-test & Mann-Whitney U test).

Saves canonical outputs to:
- paper_artifacts/paper_table1_singleobjtrack.json
- paper_artifacts/paper_headline_results_combined.json (updated with singleobjtrack table)
"""

import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch

from envs import make_env, PerturbationConfig
from policies import PPOPolicy, RandomPolicy


def train_ppo_seed(
    env_id: str = "SingleObjectTracking-v0",
    seed: int = 42,
    train_episodes: int = 25,
    max_steps_per_ep: int = 100,
) -> PPOPolicy:
    """Trains a PPO agent for a single seed on SingleObjectTracking-v0."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(env_id=env_id, seed=seed)
    policy = PPOPolicy(action_dim=2, embedding_dim=256, lr=5e-4)

    gamma = policy.gamma
    gae_lambda = policy.gae_lambda

    for ep in range(train_episodes):
        obs, _ = env.reset(seed=seed * 100 + ep)
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_log_probs = []
        ep_values = []
        ep_dones = []

        done = False
        step_count = 0
        while not done and step_count < max_steps_per_ep:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_t, log_prob_t, _, value_t = policy.get_action_and_value(obs_t)

            action = action_t.squeeze(0).cpu().detach().numpy()
            log_prob = float(log_prob_t.item())
            value = float(value_t.item())

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_values.append(value)
            ep_dones.append(done)

            obs = next_obs
            step_count += 1

        # Compute GAE Advantages & Returns for rollout batch
        ep_len = len(ep_rewards)
        advantages = np.zeros(ep_len, dtype=np.float32)
        returns = np.zeros(ep_len, dtype=np.float32)

        last_gae = 0.0
        # Bootstrap last value if truncated
        next_value = 0.0
        for t in reversed(range(ep_len)):
            if t == ep_len - 1:
                next_non_terminal = 1.0 - float(ep_dones[t])
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(ep_dones[t])
                next_val = ep_values[t + 1]

            delta = ep_rewards[t] + gamma * next_val * next_non_terminal - ep_values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            returns[t] = advantages[t] + ep_values[t]

        batch = {
            "obs": ep_obs,
            "actions": ep_actions,
            "log_probs": ep_log_probs,
            "advantages": list(advantages),
            "returns": list(returns),
        }

        # Perform 4 optimization epochs per rollout
        for _ in range(4):
            policy.update(batch)

    env.close()
    return policy


def evaluate_policy_seed(
    policy: Any,
    env_id: str = "SingleObjectTracking-v0",
    seed: int = 42,
    n_episodes: int = 20,
    success_threshold_px: float = 5.0,
) -> Tuple[float, float]:
    """Evaluates a policy on a given seed across n_episodes.
    
    Returns (seed_mean_cle_px, seed_success_rate).
    """
    env = make_env(env_id=env_id, seed=seed)
    ep_cle_list = []
    ep_success_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        step_errors = [info["tracking_error_px"]]
        done = False

        while not done:
            action = policy.act(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            step_errors.append(info["tracking_error_px"])
            done = term or trunc

        mean_cle = float(np.mean(step_errors))
        success_rate = float(np.mean(np.array(step_errors) <= success_threshold_px))

        ep_cle_list.append(mean_cle)
        ep_success_list.append(success_rate)

    env.close()
    return float(np.mean(ep_cle_list)), float(np.mean(ep_success_list))


def run_phase5_ppo_evaluation(
    seeds: List[int] = [42, 43, 44, 45, 46],
    n_test_episodes: int = 20,
) -> Dict[str, Any]:
    """Runs training & evaluation across 5 seeds for PPO Policy and Random Policy."""
    print("=" * 80)
    print("   PHASE 5 EVALUATION: PPO POLICY vs. RANDOM POLICY ON SingleObjectTracking-v0")
    print(f"   Seeds ({len(seeds)}): {seeds}")
    print("=" * 80)

    ppo_cles = []
    ppo_successes = []

    rand_cles = []
    rand_successes = []

    os.makedirs("checkpoints", exist_ok=True)

    for i, s in enumerate(seeds):
        print(f"\n--- Seed [{i+1}/{len(seeds)}] : {s} ---")

        # 1. Train PPO policy
        print(f"Training PPO Policy (seed={s})...")
        ppo_policy = train_ppo_seed(env_id="SingleObjectTracking-v0", seed=s, train_episodes=30)
        ckpt_path = f"checkpoints/ppo_singleobjtrack_seed{s}.pt"
        torch.save(ppo_policy.state_dict(), ckpt_path)

        # 2. Evaluate PPO policy
        ppo_cle, ppo_succ = evaluate_policy_seed(
            ppo_policy, env_id="SingleObjectTracking-v0", seed=s + 100, n_episodes=n_test_episodes
        )
        ppo_cles.append(ppo_cle)
        ppo_successes.append(ppo_succ)
        print(f"  PPO Policy      -> Mean CLE: {ppo_cle:6.2f} px | Success Rate: {ppo_succ * 100:5.1f}%")

        # 3. Evaluate Random policy
        rand_policy = RandomPolicy(action_dim=2, is_continuous=True, seed=s)
        rand_cle, rand_succ = evaluate_policy_seed(
            rand_policy, env_id="SingleObjectTracking-v0", seed=s + 100, n_episodes=n_test_episodes
        )
        rand_cles.append(rand_cle)
        rand_successes.append(rand_succ)
        print(f"  Random Policy   -> Mean CLE: {rand_cle:6.2f} px | Success Rate: {rand_succ * 100:5.1f}%")

    # Aggregate Statistics
    ppo_cle_mean, ppo_cle_std = float(np.mean(ppo_cles)), float(np.std(ppo_cles, ddof=1))
    ppo_succ_mean, ppo_succ_std = float(np.mean(ppo_successes)), float(np.std(ppo_successes, ddof=1))

    rand_cle_mean, rand_cle_std = float(np.mean(rand_cles)), float(np.std(rand_cles, ddof=1))
    rand_succ_mean, rand_succ_std = float(np.mean(rand_successes)), float(np.std(rand_successes, ddof=1))

    # Statistical Significance Testing (Welch's t-test & Mann-Whitney U test)
    t_stat_cle, p_val_cle_t = stats.ttest_ind(ppo_cles, rand_cles, equal_var=False)
    mwu_stat_cle, p_val_cle_u = stats.mannwhitneyu(ppo_cles, rand_cles, alternative="two-sided")

    t_stat_succ, p_val_succ_t = stats.ttest_ind(ppo_successes, rand_successes, equal_var=False)
    mwu_stat_succ, p_val_succ_u = stats.mannwhitneyu(ppo_successes, rand_successes, alternative="two-sided")

    summary = {
        "metadata": {
            "environment": "SingleObjectTracking-v0",
            "num_seeds": len(seeds),
            "seeds": seeds,
            "episodes_per_seed": n_test_episodes,
        },
        "ppo_policy": {
            "mean_cle_px": ppo_cle_mean,
            "std_cle_px": ppo_cle_std,
            "success_rate": ppo_succ_mean,
            "std_success_rate": ppo_succ_std,
            "per_seed_cle_px": [float(x) for x in ppo_cles],
            "per_seed_success_rate": [float(x) for x in ppo_successes],
        },
        "random_policy": {
            "mean_cle_px": rand_cle_mean,
            "std_cle_px": rand_cle_std,
            "success_rate": rand_succ_mean,
            "std_success_rate": rand_succ_std,
            "per_seed_cle_px": [float(x) for x in rand_cles],
            "per_seed_success_rate": [float(x) for x in rand_successes],
        },
        "statistical_tests": {
            "cle_t_test": {
                "t_statistic": float(t_stat_cle),
                "p_value": float(p_val_cle_t),
            },
            "cle_mann_whitney_u": {
                "u_statistic": float(mwu_stat_cle),
                "p_value": float(p_val_cle_u),
            },
            "success_rate_t_test": {
                "t_statistic": float(t_stat_succ),
                "p_value": float(p_val_succ_t),
            },
            "success_rate_mann_whitney_u": {
                "u_statistic": float(mwu_stat_succ),
                "p_value": float(p_val_succ_u),
            },
        },
    }

    print("\n" + "=" * 80)
    print("      FINAL AGGREGATED BENCHMARK RESULTS (SingleObjectTracking-v0)")
    print("=" * 80)
    print(f"PPO Policy    -> Mean CLE: {ppo_cle_mean:6.2f} ± {ppo_cle_std:5.2f} px | Success Rate: {ppo_succ_mean*100:5.1f} ± {ppo_succ_std*100:4.1f}%")
    print(f"Random Policy -> Mean CLE: {rand_cle_mean:6.2f} ± {rand_cle_std:5.2f} px | Success Rate: {rand_succ_mean*100:5.1f} ± {rand_succ_std*100:4.1f}%")
    print("-" * 80)
    print(f"CLE T-Test           : t = {t_stat_cle:7.4f}, p = {p_val_cle_t:.6f}")
    print(f"CLE Mann-Whitney U   : U = {mwu_stat_cle:7.4f}, p = {p_val_cle_u:.6f}")
    print(f"Success Rate T-Test  : t = {t_stat_succ:7.4f}, p = {p_val_succ_t:.6f}")
    print("=" * 80)

    # Save canonical results artifact in paper_artifacts/
    os.makedirs("paper_artifacts", exist_ok=True)
    out_path = "paper_artifacts/paper_table1_singleobjtrack.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCanonical artifact saved to: '{out_path}'")

    # Update combined headline results
    combined_path = "paper_artifacts/paper_headline_results_combined.json"
    if os.path.exists(combined_path):
        with open(combined_path, "r") as f:
            combined_data = json.load(f)
        combined_data["table1_singleobjtrack"] = summary
        with open(combined_path, "w") as f:
            json.dump(combined_data, f, indent=2)
        print(f"Updated combined headline results: '{combined_path}'")

    # Cleanup root-level duplicate JSONs per Step 3 instructions
    for duplicate_json in ["phase0_sanity_check_results.json", "phase2_geometry_results.json", "phase3_ttss_results.json"]:
        if os.path.exists(duplicate_json):
            os.remove(duplicate_json)
            print(f"Cleaned up stale root-level duplicate JSON: '{duplicate_json}'")

    return summary


if __name__ == "__main__":
    run_phase5_ppo_evaluation()
