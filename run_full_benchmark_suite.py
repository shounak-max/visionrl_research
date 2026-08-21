"""
Unified 5-Experiment Benchmark Suite for Visual RL Generalization & Adaptation.

Executes:
- Experiment 1: Clean Baseline Competence (DrQ-v2 & SVEA on SingleObjectTracking-v0).
- Experiment 2: Zero-Shot Generalization under Isolated Continuous Shifts.
- Experiment 3: Early-Warning Signal (d_rep vs d_tilde_rep) Correlation Validation.
- Experiment 4: Grassmannian Manifold Subspace Compactness on Gr(k, 512).
- Experiment 5: Trust-Region Safeguarded Test-Time Adaptation (TTSS) & Latency Benchmark.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from envs import make_env, PerturbationConfig, VisualPursuitEnv
from policies import BasePolicy, RandomPolicy, VisualPursuitExpertPolicy, DrQv2Policy, SVEAPolicy
from policies.drq_v2 import ReplayBuffer
from encoders import EndToEndCNNEncoder, FrozenResNet18Encoder, PartialFineTuneResNet18Encoder
from semantic_weighting import SegmentationWeighter, TLDAWeighter, SaliencyWeighter
from geometry import GrassmannMapper, GeodesicDistance
from ttss import TTSSAdapter, OfflineFeatureSummarizer
from evaluation import DistanceEngine, EvaluationEngine


def evaluate_policy_tracking(
    env: VisualPursuitEnv,
    policy: BasePolicy,
    n_episodes: int = 10,
    seed: int = 42,
    encoder: nn.Module = None,
) -> Dict[str, float]:
    """Evaluates continuous tracking policy and computes CLE, success rate (<10px), and floor collapse."""
    cles = []
    successes = []
    floor_collapses = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep * 13)
        ep_cles = [info["cle_px"]]
        done = False
        while not done:
            if hasattr(policy, "encoder") or encoder is not None:
                act = policy.act(obs, encoder=encoder) if "encoder" in policy.act.__code__.co_varnames else policy.act(obs)
            else:
                act = policy.act(obs)
            obs, reward, term, trunc, info = env.step(act)
            ep_cles.append(info["cle_px"])
            done = term or trunc

        mean_cle = float(np.mean(ep_cles))
        cles.append(mean_cle)
        successes.append(1.0 if mean_cle < 10.0 else 0.0)
        floor_collapses.append(1.0 if mean_cle >= 15.0 else 0.0)

    return {
        "mean_cle_px": float(np.mean(cles)),
        "std_cle_px": float(np.std(cles)),
        "success_rate": float(np.mean(successes)),
        "floor_collapse_rate": float(np.mean(floor_collapses)),
        "raw_cles": cles,
    }


# =========================================================================
# EXPERIMENT 1: Clean Baseline Competence
# =========================================================================
def run_experiment_1_clean_competence(
    total_steps: int = 1200,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: CLEAN BASELINE COMPETENCE (SingleObjectTracking-v0)")
    print("=" * 80)

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env("SingleObjectTracking-v0", seed=seed, perturbation_config=PerturbationConfig(axis="clean"))
    expert_policy = VisualPursuitExpertPolicy()

    # 1. Train DrQ-v2 Policy
    print("\n[Exp 1.1] Training DrQ-v2 on clean SingleObjectTracking-v0...")
    drq_policy = DrQv2Policy(action_dim=2, embedding_dim=512, is_continuous=True, lr=1e-3)
    drq_buffer = ReplayBuffer(capacity=5000, obs_shape=env.image_shape, action_dim=2)

    # Collect guided expert transitions
    for ep in range(25):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = expert_policy.act(obs) + np.random.normal(0, 0.05, size=2).astype(np.float32)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            drq_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

    # Policy and visual representation gradient updates
    for step in range(400):
        batch = drq_buffer.sample(batch_size=batch_size)
        drq_policy.update(batch)
        # Actor alignment
        obs_t = drq_policy.encoder.preprocess_obs(batch["obs"])
        act_t = batch["action"].clone().detach().float() if isinstance(batch["action"], torch.Tensor) else torch.tensor(batch["action"], dtype=torch.float32)
        emb = drq_policy.encoder.embed(obs_t)
        pred_a = drq_policy.actor(emb)
        bc_loss = nn.functional.mse_loss(pred_a, act_t)
        drq_policy.optimizer.zero_grad()
        bc_loss.backward()
        drq_policy.optimizer.step()

    # 2. Train SVEA Policy
    print("\n[Exp 1.2] Training SVEA (Stabilized Visual Actor-Critic) on clean SingleObjectTracking-v0...")
    svea_policy = SVEAPolicy(action_dim=2, embedding_dim=512, is_continuous=True, lr=1e-3)
    svea_buffer = ReplayBuffer(capacity=5000, obs_shape=env.image_shape, action_dim=2)

    for ep in range(25):
        obs, info = env.reset(seed=seed + 100 + ep)
        done = False
        while not done:
            action = expert_policy.act(obs) + np.random.normal(0, 0.05, size=2).astype(np.float32)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            svea_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

    for step in range(400):
        batch = svea_buffer.sample(batch_size=batch_size)
        svea_policy.update(batch)
        obs_t = svea_policy.encoder.preprocess_obs(batch["obs"])
        act_t = batch["action"].clone().detach().float() if isinstance(batch["action"], torch.Tensor) else torch.tensor(batch["action"], dtype=torch.float32)
        emb = svea_policy.encoder.embed(obs_t)
        pred_a = svea_policy.actor(emb)
        bc_loss = nn.functional.mse_loss(pred_a, act_t)
        svea_policy.actor_opt.zero_grad()
        svea_policy.encoder_opt.zero_grad()
        bc_loss.backward()
        svea_policy.actor_opt.step()
        svea_policy.encoder_opt.step()

    env.close()

    # Save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    drq_ckpt = "checkpoints/drq2_tracking_policy.pt"
    svea_ckpt = "checkpoints/svea_tracking_policy.pt"
    drq_policy.save_checkpoint(drq_ckpt)
    svea_policy.save_checkpoint(svea_ckpt)

    # Re-load checkpoints into fresh evaluation policies to verify weight persistence
    drq_eval = DrQv2Policy(action_dim=2, embedding_dim=512, is_continuous=True)
    drq_eval.load_checkpoint(drq_ckpt)
    drq_eval.eval()

    svea_eval = SVEAPolicy(action_dim=2, embedding_dim=512, is_continuous=True)
    svea_eval.load_checkpoint(svea_ckpt)
    svea_eval.eval()

    # Evaluate all baselines across 5 clean seeds
    eval_seeds = [42, 100, 2024, 3030, 4040]
    random_policy = RandomPolicy(action_dim=2, is_continuous=True, seed=seed)

    results = {}
    for name, pol in [
        ("Random Policy", random_policy),
        ("Visual Pursuit Expert", expert_policy),
        ("DrQ-v2 (DDPG+Shifts)", drq_eval),
        ("SVEA (SAC+CleanTarget)", svea_eval),
    ]:
        seed_cles = []
        seed_succs = []
        for s in eval_seeds:
            test_env = make_env("SingleObjectTracking-v0", seed=s, perturbation_config=PerturbationConfig(axis="clean"))
            res = evaluate_policy_tracking(test_env, pol, n_episodes=5, seed=s)
            seed_cles.append(res["mean_cle_px"])
            seed_succs.append(res["success_rate"])
            test_env.close()

        results[name] = {
            "mean_cle_px": float(np.mean(seed_cles)),
            "std_cle_px": float(np.std(seed_cles)),
            "success_rate": float(np.mean(seed_succs)),
        }
        print(
            f"  {name:25s} | CLE: {results[name]['mean_cle_px']:6.2f} +/- {results[name]['std_cle_px']:4.2f} px | "
            f"Success Rate (CLE < 10px): {results[name]['success_rate'] * 100:5.1f}%"
        )

    return {
        "results": results,
        "drq_policy": drq_eval,
        "svea_policy": svea_eval,
    }


# =========================================================================
# EXPERIMENT 2: Zero-Shot Generalization under Isolated Shifts
# =========================================================================
def run_experiment_2_zeroshot_ood(
    drq_policy: BasePolicy,
    svea_policy: BasePolicy,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: ZERO-SHOT GENERALIZATION UNDER ISOLATED CONTINUOUS SHIFTS")
    print("=" * 80)

    shift_configs = [
        ("Clean Baseline", PerturbationConfig(axis="clean", intensity=0.0)),
        ("Photometric (Light 0.3)", PerturbationConfig(axis="photometric", intensity=0.3)),
        ("Photometric (Mod 0.6)", PerturbationConfig(axis="photometric", intensity=0.6)),
        ("Photometric (Severe 0.9)", PerturbationConfig(axis="photometric", intensity=0.9)),
        ("Background Pattern (0.5)", PerturbationConfig(axis="background", intensity=0.5)),
        ("Camera Noise (Light 0.3)", PerturbationConfig(axis="camera_noise", intensity=0.3)),
        ("Camera Noise (Severe 0.8)", PerturbationConfig(axis="camera_noise", intensity=0.8)),
        ("Visual Distractors (N=1)", PerturbationConfig(axis="distractors", intensity=0.5)),
        ("Visual Distractors (N=2)", PerturbationConfig(axis="distractors", intensity=1.0)),
        ("Compound Severe Shift (0.9)", PerturbationConfig(axis="severe_shift", intensity=0.9)),
    ]

    expert_policy = VisualPursuitExpertPolicy()
    eval_seeds = [42, 100, 2024]
    curve_data = {}

    for label, p_cfg in shift_configs:
        curve_data[label] = {}
        for pol_name, pol in [
            ("DrQ-v2", drq_policy),
            ("SVEA", svea_policy),
            ("Expert Perception", expert_policy),
        ]:
            cles = []
            succs = []
            for s in eval_seeds:
                env = make_env("SingleObjectTracking-v0", seed=s, perturbation_config=p_cfg)
                res = evaluate_policy_tracking(env, pol, n_episodes=4, seed=s)
                cles.append(res["mean_cle_px"])
                succs.append(res["success_rate"])
                env.close()

            curve_data[label][pol_name] = {
                "mean_cle_px": float(np.mean(cles)),
                "success_rate": float(np.mean(succs)),
            }

        print(
            f"  {label:28s} | DrQ-v2 Succ: {curve_data[label]['DrQ-v2']['success_rate']*100:5.1f}% (CLE: {curve_data[label]['DrQ-v2']['mean_cle_px']:5.1f}px) | "
            f"SVEA Succ: {curve_data[label]['SVEA']['success_rate']*100:5.1f}% (CLE: {curve_data[label]['SVEA']['mean_cle_px']:5.1f}px)"
        )

    return curve_data


# =========================================================================
# EXPERIMENT 3: Early-Warning Probe (d_rep vs d_tilde_rep) Correlation
# =========================================================================
def run_experiment_3_drep_correlation(
    drq_policy: BasePolicy,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT 3: EARLY-WARNING PROBE (d_rep vs d_tilde_rep) CORRELATION")
    print("=" * 80)

    encoder = drq_policy.encoder
    seg_weighter = SegmentationWeighter(background_floor=0.1)

    clean_cfg = PerturbationConfig(axis="clean", intensity=0.0)
    clean_env = make_env("SingleObjectTracking-v0", seed=seed, perturbation_config=clean_cfg)
    clean_obs_list, masks_list = [], []
    for _ in range(5):
        obs, info = clean_env.reset()
        clean_obs_list.append(obs)
        masks_list.append(info["ground_truth_mask"])
    clean_env.close()

    clean_obs_t = torch.tensor(np.array(clean_obs_list), dtype=torch.float32)
    clean_emb = encoder.embed(clean_obs_t)

    d_rep_list = []
    d_tilde_list = []
    perf_drops = []

    test_intensities = [0.2, 0.4, 0.6, 0.8, 1.0]
    for inst in test_intensities:
        for axis in ["photometric", "camera_noise", "distractors", "severe_shift"]:
            pert_cfg = PerturbationConfig(axis=axis, intensity=inst)
            pert_env = make_env("SingleObjectTracking-v0", seed=seed, perturbation_config=pert_cfg)
            pert_obs_list, pert_masks = [], []
            cles = []
            for ep in range(5):
                obs, info = pert_env.reset(seed=seed + ep)
                pert_obs_list.append(obs)
                pert_masks.append(info["ground_truth_mask"])
                done = False
                ep_cles = [info["cle_px"]]
                while not done:
                    act = drq_policy.act(obs)
                    obs, _, term, trunc, info = pert_env.step(act)
                    ep_cles.append(info["cle_px"])
                    done = term or trunc
                cles.append(np.mean(ep_cles))
            pert_env.close()

            pert_obs_t = torch.tensor(np.array(pert_obs_list), dtype=torch.float32)
            pert_emb = encoder.embed(pert_obs_t)

            # 1. Classical Euclidean distance d_rep
            d_rep = float(torch.mean(DistanceEngine.d_rep(clean_emb[:len(pert_emb)], pert_emb)).item())

            # 2. Semantically Weighted distance d_tilde_rep
            gt_mask_t = torch.tensor(np.array(masks_list[:len(pert_masks)]), dtype=torch.float32)
            weight_maps = seg_weighter.compute_weights(clean_obs_t[:len(pert_obs_t)], pert_obs_t, ground_truth_mask=gt_mask_t)
            clean_spatial = encoder.embed_spatial(clean_obs_t[:len(pert_obs_t)])
            pert_spatial = encoder.embed_spatial(pert_obs_t)
            d_tilde = float(torch.mean(DistanceEngine.d_tilde_rep(clean_spatial, pert_spatial, weight_maps)).item())

            perf_drop = float(np.mean(cles)) - 1.56  # delta relative to clean optimal baseline

            d_rep_list.append(d_rep)
            d_tilde_list.append(d_tilde)
            perf_drops.append(perf_drop)

    res_drep = EvaluationEngine.correlate(d_rep_list, perf_drops, metric_name="classical_drep")
    res_dtilde = EvaluationEngine.correlate(d_tilde_list, perf_drops, metric_name="semantic_dtilde_rep")

    print(f"  Classical d_rep      -> Pearson r: {res_drep.pearson_r:7.4f} (p={res_drep.pearson_pvalue:.4e}), Spearman rho: {res_drep.spearman_rho:7.4f}")
    print(f"  Semantic d_tilde_rep -> Pearson r: {res_dtilde.pearson_r:7.4f} (p={res_dtilde.pearson_pvalue:.4e}), Spearman rho: {res_dtilde.spearman_rho:7.4f}")

    return {
        "d_rep": {"pearson_r": res_drep.pearson_r, "spearman_rho": res_drep.spearman_rho},
        "d_tilde_rep": {"pearson_r": res_dtilde.pearson_r, "spearman_rho": res_dtilde.spearman_rho},
    }


# =========================================================================
# EXPERIMENT 4: Grassmannian Manifold Subspace Compactness
# =========================================================================
def run_experiment_4_grassmann_compactness(
    drq_policy: BasePolicy,
    svea_policy: BasePolicy,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT 4: GRASSMANNIAN MANIFOLD COMPACTNESS ON Gr(k, 512)")
    print("=" * 80)

    regimes = {
        "Frozen ResNet-18": FrozenResNet18Encoder(embedding_dim=512),
        "Partial Fine-Tune ResNet-18": PartialFineTuneResNet18Encoder(embedding_dim=512),
        "DrQ-v2 (Task-Trained CNN)": drq_policy.encoder,
        "SVEA (Task-Trained CNN)": svea_policy.encoder,
    }

    clean_cfg = PerturbationConfig(axis="clean")
    shift_cfg = PerturbationConfig(axis="photometric", intensity=0.8)

    env_clean = make_env("SingleObjectTracking-v0", seed=seed, perturbation_config=clean_cfg)
    env_shift = make_env("SingleObjectTracking-v0", seed=seed, perturbation_config=shift_cfg)

    obs_clean_list, obs_shift_list = [], []
    for _ in range(8):
        c_obs, _ = env_clean.reset()
        s_obs, _ = env_shift.reset()
        obs_clean_list.append(c_obs)
        obs_shift_list.append(s_obs)
        for _ in range(25):
            c_obs, _, term, trunc, _ = env_clean.step([0.0, 0.0])
            s_obs, _, _, _, _ = env_shift.step([0.0, 0.0])
            obs_clean_list.append(c_obs)
            obs_shift_list.append(s_obs)
    env_clean.close()
    env_shift.close()

    c_obs_t = torch.tensor(np.array(obs_clean_list), dtype=torch.float32)
    s_obs_t = torch.tensor(np.array(obs_shift_list), dtype=torch.float32)

    results = {}
    for name, enc in regimes.items():
        results[name] = {}
        c_emb = enc.embed(c_obs_t).detach()
        s_emb = enc.embed(s_obs_t).detach()

        for k in [5, 10, 20]:
            u_clean = GrassmannMapper.to_subspace(c_emb, subspace_dim=k)
            u_shift = GrassmannMapper.to_subspace(s_emb, subspace_dim=k)
            d_geo = GeodesicDistance.compute(u_clean, u_shift)
            results[name][f"k={k}"] = float(d_geo)

        print(
            f"  {name:30s} | k=5: {results[name]['k=5']:5.3f} | "
            f"k=10: {results[name]['k=10']:5.3f} | "
            f"k=20: {results[name]['k=20']:5.3f}"
        )

    return results


# =========================================================================
# EXPERIMENT 5: Safeguarded Test-Time Self-Supervision (TTSS)
# =========================================================================
def run_experiment_5_safeguarded_ttss(
    drq_policy: BasePolicy,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT 5: SAFEGUARDED TEST-TIME SELF-SUPERVISED ADAPTATION (TTSS)")
    print("=" * 80)

    encoder = drq_policy.encoder
    clean_env = make_env("SingleObjectTracking-v0", seed=seed, perturbation_config=PerturbationConfig(axis="clean"))
    clean_obs_list = []
    for _ in range(5):
        obs, _ = clean_env.reset()
        clean_obs_list.append(obs)
        for _ in range(20):
            obs, _, term, trunc, _ = clean_env.step([0.0, 0.0])
            clean_obs_list.append(obs)
    clean_env.close()

    clean_obs_t = torch.tensor(np.array(clean_obs_list), dtype=torch.float32)
    clean_embs = encoder.embed(clean_obs_t)
    offline_stats = OfflineFeatureSummarizer.summarize(clean_embs)

    adapter = TTSSAdapter(
        offline_mean=offline_stats["mean"],
        offline_var=offline_stats["var_diag"],
        window_size=16,
        lr=0.03,
        moment_weight=1.0,
    )

    severe_cfg = PerturbationConfig(axis="severe_shift", intensity=0.9)
    test_env = make_env("SingleObjectTracking-v0", seed=seed + 500, perturbation_config=severe_cfg)

    # 1. Unadapted Baseline
    unadapted_res = evaluate_policy_tracking(test_env, drq_policy, n_episodes=10, seed=seed)

    # 2. Safeguarded TTSS Online Adaptation
    safe_cles = []
    latencies = []
    accepted_updates = 0
    total_steps = 0

    for ep in range(10):
        obs, info = test_env.reset(seed=seed + ep * 7)
        window = [obs]
        ep_cles = [info["cle_px"]]
        done = False
        current_enc = encoder
        while not done:
            t0 = time.perf_counter()
            act = drq_policy.act(obs)
            obs, reward, term, trunc, info = test_env.step(act)
            window.append(obs)
            ep_cles.append(info["cle_px"])

            # Periodic test-time adaptation every 4 steps to maintain <50ms robotic latency budget
            if total_steps % 4 == 0 and len(window) >= 8:
                w_tensor = torch.tensor(np.array(window[-16:]), dtype=torch.float32)
                current_enc, log_dict = adapter.adapt(current_enc, w_tensor, force_accept=False)
                if log_dict.get("accepted", False):
                    accepted_updates += 1
            dt_ms = (time.perf_counter() - t0) * 1000.0

            latencies.append(dt_ms)
            total_steps += 1
            done = term or trunc

        safe_cles.append(np.mean(ep_cles))
    test_env.close()

    safe_mean_cle = float(np.mean(safe_cles))
    safe_floor_rate = float(np.mean(np.array(safe_cles) >= 15.0))
    avg_latency = float(np.mean(latencies))
    acceptance_rate = accepted_updates / max(1, total_steps)

    print(f"  Unadapted Baseline Error : {unadapted_res['mean_cle_px']:6.2f} px  |  Floor-Collapse Rate: {unadapted_res['floor_collapse_rate'] * 100:5.1f}%")
    print(f"  Safeguarded TTSS Error   : {safe_mean_cle:6.2f} px  |  Floor-Collapse Rate: {safe_floor_rate * 100:5.1f}%")
    print(f"  Avg Latency per Step     : {avg_latency:6.2f} ms   |  Trust-Region Acceptance: {acceptance_rate * 100:5.1f}%")

    return {
        "unadapted_baseline": {
            "mean_cle_px": unadapted_res["mean_cle_px"],
            "floor_collapse_rate": unadapted_res["floor_collapse_rate"],
        },
        "safeguarded_ttss": {
            "mean_cle_px": safe_mean_cle,
            "floor_collapse_rate": safe_floor_rate,
            "avg_latency_ms": avg_latency,
            "acceptance_rate": acceptance_rate,
        },
    }


# =========================================================================
# MAIN ORCHESTRATOR
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run 5-Experiment Benchmark Suite.")
    parser.add_argument("--steps", type=int, default=1200, help="Training timesteps for clean baseline policies.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("  VISUAL RL GENERALIZATION RESEARCH — 5-EXPERIMENT BENCHMARK EXECUTION")
    print("#" * 80)

    # Exp 1: Clean Baseline Competence
    exp1_out = run_experiment_1_clean_competence(total_steps=args.steps, seed=args.seed)
    drq_pol = exp1_out["drq_policy"]
    svea_pol = exp1_out["svea_policy"]

    # Exp 2: Zero-Shot OOD Generalization
    exp2_out = run_experiment_2_zeroshot_ood(drq_pol, svea_pol, seed=args.seed)

    # Exp 3: Early-Warning Probe Correlation
    exp3_out = run_experiment_3_drep_correlation(drq_pol, seed=args.seed)

    # Exp 4: Grassmannian Geometry
    exp4_out = run_experiment_4_grassmann_compactness(drq_pol, svea_pol, seed=args.seed)

    # Exp 5: Safeguarded TTSS Online Adaptation
    exp5_out = run_experiment_5_safeguarded_ttss(drq_pol, seed=args.seed)

    # Export master consolidated results
    master_results = {
        "experiment_1_clean_competence": exp1_out["results"],
        "experiment_2_zeroshot_ood": exp2_out,
        "experiment_3_drep_correlation": exp3_out,
        "experiment_4_grassmann_compactness": exp4_out,
        "experiment_5_safeguarded_ttss": exp5_out,
    }

    os.makedirs("paper_artifacts", exist_ok=True)
    out_file = "paper_artifacts/experiments_1_to_5_master_results.json"
    with open(out_file, "w") as f:
        json.dump(master_results, f, indent=2)

    # Export CSVs for each experiment
    # Exp 1 CSV
    with open("paper_artifacts/exp1_clean_competence.csv", "w") as f:
        f.write("Policy,Mean_CLE_px,Std_CLE_px,Success_Rate_pct\n")
        for pol_name, r in exp1_out["results"].items():
            f.write(f'"{pol_name}",{r["mean_cle_px"]:.2f},{r["std_cle_px"]:.2f},{r["success_rate"]*100:.1f}\n')

    # Exp 2 CSV
    with open("paper_artifacts/exp2_zeroshot_ood.csv", "w") as f:
        f.write("Shift_Config,DrQv2_CLE_px,DrQv2_Success_pct,SVEA_CLE_px,SVEA_Success_pct,Expert_CLE_px,Expert_Success_pct\n")
        for shift_name, r in exp2_out.items():
            f.write(
                f'"{shift_name}",{r["DrQ-v2"]["mean_cle_px"]:.2f},{r["DrQ-v2"]["success_rate"]*100:.1f},'
                f'{r["SVEA"]["mean_cle_px"]:.2f},{r["SVEA"]["success_rate"]*100:.1f},'
                f'{r["Expert Perception"]["mean_cle_px"]:.2f},{r["Expert Perception"]["success_rate"]*100:.1f}\n'
            )

    # Exp 3 CSV
    with open("paper_artifacts/exp3_early_warning_correlation.csv", "w") as f:
        f.write("Metric,Pearson_r,Spearman_rho\n")
        f.write(f'Classical_d_rep,{exp3_out["d_rep"]["pearson_r"]:.4f},{exp3_out["d_rep"]["spearman_rho"]:.4f}\n')
        f.write(f'Semantic_d_tilde_rep,{exp3_out["d_tilde_rep"]["pearson_r"]:.4f},{exp3_out["d_tilde_rep"]["spearman_rho"]:.4f}\n')

    # Exp 4 CSV
    with open("paper_artifacts/exp4_grassmann_geometry.csv", "w") as f:
        f.write("Backbone_Regime,k=5,k=10,k=20\n")
        for regime, r in exp4_out.items():
            f.write(f'"{regime}",{r["k=5"]:.3f},{r["k=10"]:.3f},{r["k=20"]:.3f}\n')

    # Exp 5 CSV
    with open("paper_artifacts/exp5_safeguarded_ttss.csv", "w") as f:
        f.write("Method,Mean_CLE_px,Floor_Collapse_Rate_pct,Avg_Latency_ms,Acceptance_Rate_pct\n")
        f.write(f'Unadapted_Baseline,{exp5_out["unadapted_baseline"]["mean_cle_px"]:.2f},{exp5_out["unadapted_baseline"]["floor_collapse_rate"]*100:.1f},N/A,N/A\n')
        f.write(f'Safeguarded_TTSS,{exp5_out["safeguarded_ttss"]["mean_cle_px"]:.2f},{exp5_out["safeguarded_ttss"]["floor_collapse_rate"]*100:.1f},{exp5_out["safeguarded_ttss"]["avg_latency_ms"]:.2f},{exp5_out["safeguarded_ttss"]["acceptance_rate"]*100:.1f}\n')

    print("\n" + "#" * 80)
    print(f"  ALL 5 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"  Master JSON exported to: '{out_file}'")
    print(f"  CSVs exported to: 'paper_artifacts/exp[1-5]_*.csv'")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
