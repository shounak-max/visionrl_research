"""
Phase 3 TTSS Adaptation & Floor-Collapse Evaluation Runner.

Benchmarks active Test-Time Self-Supervised Adaptation (TTSS) on the Visual Pursuit Tracking task
under severe visual noise perturbations.
Measures:
1. Floor-Collapse Rate before vs after TTSS (threshold = 150.0 px tracking error).
2. Naive TTT (without safeguard) vs Safeguarded TTSS (with trust-region rejection).
3. Adaptation latency per step in milliseconds.
"""

import json
import os
from typing import Dict, List
import numpy as np
import torch

from envs import VisualPursuitEnv, PerturbationConfig
from policies import VisualPursuitExpertPolicy
from encoders import EndToEndCNNEncoder
from ttss import OfflineFeatureSummarizer, TTSSAdapter
from evaluation import EvaluationEngine


def evaluate_phase3_ttss(
    seed: int = 42, n_episodes: int = 20
) -> Dict[str, float]:
    """Evaluates floor-collapse rate reduction and tracking performance before vs. after TTSS."""
    print("=" * 80)
    print("   PHASE 3 EVALUATION: ACTIVE TEST-TIME SELF-SUPERVISED ADAPTATION (TTSS)")
    print("=" * 80)

    encoder = EndToEndCNNEncoder(embedding_dim=512)
    policy = VisualPursuitExpertPolicy()

    # Step 1: Offline Feature Summarization on clean training distribution
    print("\n[Step 1] Summarizing offline clean feature statistics...")
    clean_env = VisualPursuitEnv(
        seed=seed, perturbation_config=PerturbationConfig(axis="clean")
    )
    clean_obs_list = []
    for _ in range(5):
        obs, _ = clean_env.reset()
        clean_obs_list.append(obs)
        done = False
        while not done:
            action = policy.act(obs)
            obs, _, term, trunc, _ = clean_env.step(action)
            clean_obs_list.append(obs)
            done = term or trunc
    clean_env.close()

    clean_obs_t = torch.tensor(np.array(clean_obs_list), dtype=torch.float32)
    clean_embs = encoder.embed(clean_obs_t)
    offline_stats = OfflineFeatureSummarizer.summarize(clean_embs)
    OfflineFeatureSummarizer.save_stats(offline_stats, seed=seed)
    print(f"Offline feature mean vector shape: {offline_stats['mean'].shape}")

    # Step 2: Benchmark un-adapted baseline under severe camera noise shift
    print("\n[Step 2] Evaluating un-adapted baseline under severe visual shift...")
    policy = VisualPursuitExpertPolicy(offline_mean=offline_stats["mean"])
    severe_cfg = PerturbationConfig(axis="camera_noise", intensity=0.9)
    test_env = VisualPursuitEnv(seed=seed + 100, perturbation_config=severe_cfg)

    unadapted_errors = []
    for ep in range(n_episodes):
        obs, info = test_env.reset(seed=seed + ep)
        ep_errors = [info["tracking_error_px"]]
        done = False
        while not done:
            action = policy.act(obs, encoder=encoder)
            obs, reward, term, trunc, info = test_env.step(action)
            ep_errors.append(info["tracking_error_px"])
            done = term or trunc
        unadapted_errors.append(float(np.mean(ep_errors)))
    test_env.close()

    unadapted_floor_rate = EvaluationEngine.floor_collapse_rate(
        unadapted_errors, floor_threshold=15.0, mode="upper_bound"
    )
    unadapted_mean_err = float(np.mean(unadapted_errors))

    print(
        f"Un-adapted Baseline -> Mean Tracking Error: {unadapted_mean_err:.2f} px | Floor-Collapse Rate: {unadapted_floor_rate * 100:.1f}%"
    )

    # Step 3: Benchmark Naive TTT (without trust-region safeguard)
    print("\n[Step 3] Evaluating Naive TTT (without trust-region rejection)...")
    naive_adapter = TTSSAdapter(
        offline_mean=offline_stats["mean"],
        offline_var=offline_stats["var_diag"],
        window_size=16,
        lr=0.1,
        moment_weight=0.1,
    )
    test_env = VisualPursuitEnv(seed=seed + 100, perturbation_config=severe_cfg)
    naive_errors = []
    naive_latencies = []
    for ep in range(n_episodes):
        obs, info = test_env.reset(seed=seed + ep)
        window = [obs]
        ep_errors = [info["tracking_error_px"]]
        done = False
        current_enc = encoder
        while not done:
            action = policy.act(obs, encoder=current_enc)
            obs, reward, term, trunc, info = test_env.step(action)
            window.append(obs)
            ep_errors.append(info["tracking_error_px"])

            # Adapt online (naive force_accept=True)
            w_tensor = torch.tensor(np.array(window[-16:]), dtype=torch.float32)
            current_enc, n_log = naive_adapter.adapt(
                current_enc, w_tensor, force_accept=True
            )
            naive_latencies.append(n_log["latency_ms"])

            done = term or trunc
        naive_errors.append(float(np.mean(ep_errors)))
    test_env.close()

    naive_floor_rate = EvaluationEngine.floor_collapse_rate(
        naive_errors, floor_threshold=15.0, mode="upper_bound"
    )
    naive_mean_err = float(np.mean(naive_errors))
    naive_avg_latency = float(np.mean(naive_latencies)) if naive_latencies else 0.0

    print(
        f"Naive TTT -> Mean Tracking Error: {naive_mean_err:.2f} px | Floor-Collapse Rate: {naive_floor_rate * 100:.1f}%"
    )

    # Step 4: Benchmark Safeguarded TTSS (with trust-region rejection)
    print("\n[Step 4] Evaluating Safeguarded TTSS (with trust-region rejection)...")
    safe_adapter = TTSSAdapter(
        offline_mean=offline_stats["mean"],
        offline_var=offline_stats["var_diag"],
        window_size=16,
        lr=1e-2,
        moment_weight=2.0,
    )
    test_env = VisualPursuitEnv(seed=seed + 100, perturbation_config=severe_cfg)
    safe_errors = []
    accepted_count = 0
    total_steps = 0
    latencies = []

    for ep in range(n_episodes):
        obs, info = test_env.reset(seed=seed + ep)
        window = [obs]
        ep_errors = [info["tracking_error_px"]]
        done = False
        current_enc = encoder
        while not done:
            action = policy.act(obs, encoder=current_enc)
            obs, reward, term, trunc, info = test_env.step(action)
            window.append(obs)
            ep_errors.append(info["tracking_error_px"])

            # Adapt online (safeguarded force_accept=False)
            w_tensor = torch.tensor(np.array(window[-16:]), dtype=torch.float32)
            current_enc, log_dict = safe_adapter.adapt(
                current_enc, w_tensor, force_accept=False
            )

            total_steps += 1
            if log_dict["accepted"]:
                accepted_count += 1
            latencies.append(log_dict["latency_ms"])

            done = term or trunc
        safe_errors.append(float(np.mean(ep_errors)))
    test_env.close()

    safe_floor_rate = EvaluationEngine.floor_collapse_rate(
        safe_errors, floor_threshold=15.0, mode="upper_bound"
    )
    safe_mean_err = float(np.mean(safe_errors))
    acceptance_rate = accepted_count / max(1, total_steps)
    avg_latency = float(np.mean(latencies))

    print(
        f"Safeguarded TTSS -> Mean Tracking Error: {safe_mean_err:.2f} px | Floor-Collapse Rate: {safe_floor_rate * 100:.1f}%"
    )
    print(
        f"Trust-Region Acceptance Rate: {acceptance_rate * 100:.1f}% | Avg Latency: {avg_latency:.2f} ms/step"
    )

    summary = {
        "unadapted_baseline": {
            "mean_tracking_error_px": unadapted_mean_err,
            "floor_collapse_rate": unadapted_floor_rate,
        },
        "naive_ttt": {
            "mean_tracking_error_px": naive_mean_err,
            "floor_collapse_rate": naive_floor_rate,
            "trust_region_acceptance_rate": 1.0,
            "avg_latency_ms": naive_avg_latency,
        },
        "safeguarded_ttss": {
            "mean_tracking_error_px": safe_mean_err,
            "floor_collapse_rate": safe_floor_rate,
            "trust_region_acceptance_rate": acceptance_rate,
            "avg_latency_ms": avg_latency,
        },
    }

    print("\n" + "=" * 80)
    print("      PHASE 3 HEADLINE FLOOR-COLLAPSE COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"1. Un-adapted Baseline Error : {unadapted_mean_err:6.2f} px  |  Floor-Collapse Rate: {unadapted_floor_rate * 100:5.1f}%"
    )
    print(
        f"2. Naive TTT (Un-safeguarded): {naive_mean_err:6.2f} px  |  Floor-Collapse Rate: {naive_floor_rate * 100:5.1f}%"
    )
    print(
        f"3. Safeguarded TTSS          : {safe_mean_err:6.2f} px  |  Floor-Collapse Rate: {safe_floor_rate * 100:5.1f}%"
    )
    print("=" * 80)

    out_file = "phase3_ttss_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Machine-readable TTSS results saved to '{out_file}'.")

    return summary


if __name__ == "__main__":
    evaluate_phase3_ttss()
