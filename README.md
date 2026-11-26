# VisionRL: Forecasting Competence Boundaries in Visual RL Without Rollouts

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()
[![Invariants Verified](https://img.shields.io/badge/invariants-100%25%20verified-brightgreen.svg)]()

Official open-source repository and reproduction suite for **"Forecasting Competence Boundaries in Visual Reinforcement Learning Without Rollouts"**.

---

## 📌 Overview

Deploying vision-based reinforcement learning policies in safety-critical robotics requires anticipating failure before executing actions. However, measuring out-of-distribution (OOD) vulnerability via physical trial rollouts risks catastrophic hardware damage.

**VisionRL** formalizes and benchmarks **rollout-free competence boundary forecasting**: predicting the continuous performance degradation curve and locating the exact critical failure threshold ($s^*$) where tracking collapses, **using exclusively unlabelled static observations and pre-computed latent statistics**—completely offline without environment execution or generative rollouts.

```
                              Rollout-Free Competence Forecasting
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
 [ Latent Geometry ]                [ Manifold Curvature ]               [ Policy-Conditional ]
   D_kNN (Eq. 3)                      PAC-Drift & GCM-Drift                Q-Drop (τ=0.430)
   AUROC = 0.823 (AUPRC = 0.966)      Jacobian Pullback & AIRM             Action Disagreement (τ=0.200)
   Sem-kNN Uniform (AUROC = 0.877)    L1 Curvature > L0 Pullback           30k Benchmark (n=1,530)
```

---

## 🔬 Matched 30,000-Step Baseline Benchmark

Head-to-head empirical comparison across continuous control baselines trained to **30,000 steps** under identical conditions (Visual Pursuit, $48 \times 48 \times 3$, batch 128, lr $3 \times 10^{-4}$, seed 0, evaluated across 20 held-out evaluation seeds):

| Metric Family | Metric Name | DrQ-v2 (`30k`) | SVEA (`30k`) | TACO (`30k`) | Key Insight |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Clean Control** | **Clean Mean CLE** (lower is better) | 5.72 px | 5.43 px | **4.57 px** | **TACO crosses competence gate (<5.0 px)** |
| | **Clean Success Rate** ($\% < 5$ px) | 20.0% | 20.0% | **55.0%** | **TACO achieves 2.75× higher success** |
| | **Clean Return** | -11.91 | -11.30 | **-9.52** | **TACO achieves highest task reward** |
| **Representation Drift** | **kNN Latent Drift ($D_{\text{kNN}}$)** | 3.524 | **3.116** | 3.388 | SVEA maintains tightest latent clustering |
| | **Mahalanobis Distance ($D_{\text{maha}}$)** | 14.909 | **13.597** | 14.402 | SVEA yields lowest covariance drift |
| | **Centroid Displacement ($D_{\text{cent}}$)** | 3.448 | 3.636 | **3.379** | TACO yields lowest mean centroid shift |
| | **Semantic Entity kNN ($D_{\text{sem}}$)** | 2.211 | **1.254** | 2.544 | SVEA focuses most sharply on target entity |
| **Manifold Geometry** | **Grassmann Distance ($\delta_{\text{Grassmann}}$)** | 2.498 | **1.995** | 2.160 | SVEA yields lowest principal subspace rotation |
| | **GCM-Drift ($\delta_{\text{AIRM}}$)** | 49.908 | **48.703** | 49.903 | SVEA yields lowest AIRM covariance drift |
| | **PAC-Drift $\lambda=0$** (1st-order pullback) | **1.224** | 1.347 | 1.370 | DrQ-v2 yields lowest local action gradient drift |
| | **PAC-Drift $\lambda=1$** (Hessian curvature) | 3.461 | **2.906** | 3.631 | SVEA yields lowest curvature-aware pullback drift |
| **Value & Policy Impact**| **Action Disagreement ($\|\Delta a\|_2$)** | **0.770** | 0.772 | 0.800 | DrQ-v2 / SVEA produce smoothest actions |
| | **Expected Q-Drop ($Q_c - Q_s$)** | 4.189 | 4.167 | **3.647** | **TACO value predictions collapse least** |
| | **Twin-Critic Gap ($|Q_1 - Q_2|$)** | 0.475 | **0.368** | 0.485 | SVEA yields highest twin-critic agreement |

---

## 📁 Repository Structure

```
visionrl_research/
├── src/visionrl/               # Core library
│   ├── models/                 # DrQ-v2, SVEA, TACO, PPO, Encoders, Replay Buffer
│   ├── environments/           # Visual Pursuit, Reacher, Discrete Pursuit
│   ├── probes/                 # FeatureBank, RefBank, PAC-Drift, GCM-Drift, Grassmann
│   ├── forecasting/            # Conformal thresholding, ranking, critical boundary
│   ├── shifts/                 # Post-hoc visual corruptions (noise, background, etc.)
│   ├── experiments/            # Training routines, evaluation cells, hypothesis tester
│   └── audit/                  # Invariant assertion verification suite
├── scripts/                    # Reproduction & evaluation entrypoints
│   ├── train_30k_baselines.py  # Train DrQ-v2 & SVEA to 30,000 steps on CUDA
│   ├── train_taco_baseline.py  # Train TACO baseline to 30,000 steps on CUDA
│   ├── measure_30k_clean.py    # Measure clean tracking performance across 20 held-out seeds
│   ├── evaluate_engineered_metrics.py # Comprehensive evaluation of all representation metrics
│   ├── diagnose_tta.py         # Test-time adaptation diagnostic suite
│   ├── test_tta_lr_scaling.py  # Trust-region action deviation scaling test
│   └── check_consistency.py   # Invariants & consistency checker (0 tolerance)
├── checkpoints/                # Trained 30,000-step model weights
│   ├── drqv2_30k_s0.pt         # Full state: encoder, actor, critic, critic_target (7.28 MB)
│   ├── svea_30k_s0.pt          # Full state: encoder, actor, critic, critic_target (7.28 MB)
│   └── taco_30k_s0.pt          # Full state: encoder, actor, critic, critic_target, aux (6.68 MB)
├── results/                    # Authoritative CSV/JSON experiment logs & metrics
├── tests/                      # 48 Pytest unit tests (100% passing)
├── legacy/                     # Quarantined exploratory runner scripts & old tree
└── pyproject.toml              # Build & dependency specification
```

---

## 🚀 Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/anonymous/visionrl.git
cd visionrl

# Install in editable mode
pip install -e .
```

Requirements: Python $\ge 3.9$, PyTorch $\ge 2.0$, NumPy, SciPy, scikit-learn, Pandas, Matplotlib, Pytest.

---

## ⚡ Reproduction Guide

Every empirical result and table can be reproduced with a single command:

### 1. Run Unit Tests (48/48 Passing)
```bash
pytest tests -v
```

### 2. Verify Data-Manuscript Consistency (0 Warnings, 0 Failures)
```bash
python scripts/check_consistency.py
```

### 3. Evaluate Clean Performance of 30k Models (20 Held-Out Seeds)
```bash
python scripts/measure_30k_clean.py
# Outputs: results/clean_performance_30k.json
```

### 4. Evaluate All Engineered Representation & Drift Metrics
```bash
python scripts/evaluate_engineered_metrics.py
# Outputs: results/engineered_metrics_comparison_30k.json
#          results/engineered_metrics_comparison_30k.csv
```

### 5. Retrain 30,000-Step Baselines from Scratch (CUDA)
```bash
# Train DrQ-v2 and SVEA to 30k steps with full state serialization
python scripts/train_30k_baselines.py

# Train TACO baseline to 30k steps
python scripts/train_taco_baseline.py
```

---

## 📊 Pre-Registered Hypotheses & Evidence (30k Benchmark)

| ID | Hypothesis Statement | Verdict | Evidence |
| :---: | :--- | :---: | :--- |
| **$H_0$** | Agents solve the clean task above the pre-training gate | **SUPPORTED** | Minimum clean success = 0.20, gate $\ge 0.20$ (TACO 0.55, DrQ-v2 0.20, SVEA 0.20) |
| **$H_1$** | Visual RL failure is predominantly abrupt (cliff-like) | **NOT SUPPORTED** | 7/15 policy $\times$ shift curves show cliffness $\ge 0.5$ (`all_agents_s_star_30k.csv`) |
| **$H_2$** | Policy $k$NN forecasts failure better than pixel/random baselines | **SUPPORTED** | `knn_policy` AUROC = 0.823; +0.062 over `knn_pixel` (0.761), +0.114 over `knn_random` (0.709) |
| **$H_3$** | Policy $k$NN accurately ranks policies by true robustness | **NOT SUPPORTED** | `knn_policy` Kendall $\tau = 0.267$, selection regret = 0.113 |
| **$H_4$** | Policy-conditional signals rank policies better than agnostic ones | **SUPPORTED** | `q_drop` $\tau = 0.430$, `act_disagree` $\tau = 0.200$ vs `knn_pixel` $\tau = -0.242$, `knn_random` $\tau = 0.046$ |

---

## 📜 Citation

```bibtex
@article{visionrl2026forecasting,
  title={Forecasting Competence Boundaries in Visual Reinforcement Learning Without Rollouts},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
