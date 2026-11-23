# Authoritative Context & Experiment Log (`visionrl_research`)

**Status:** Authoritative Evidence Base Synchronized with `notebooks/notebook_revised.ipynb` and `results/`.  
**Notice:** All previous preliminary heuristic runs (Phase 0-3 heuristic scripts) have been quarantined to `legacy/heuristic_scripts/`.

---

## 1. Environment & Policy Infrastructure

- **Environment:** Visual Pursuit (continuous 2D tracking reticle control, $48 \times 48 \times 3$ RGB, 3-frame stack).
  - Target speed: $0.8$\,px/step, max reticle step: $2.0$\,px.
  - Operational success criterion: Episode-mean Centre-of-target Localization Error (CLE) $< 5.0$\,px.
  - Random-policy baseline floor: $22.05$\,px (Collapse threshold: $\ge 0.8 \times \text{floor} = 17.64$\,px).
- **Trained Policies:** 6 deep RL policies trained for 50,000 environment steps each:
  - 3 seeds of DrQ-v2 (seeds 0, 1, 2)
  - 3 seeds of SVEA (seeds 0, 1, 2) with decoupled critic loss (clean + augmented) and actor loss (clean only)
  - Pre-training validation gate ($H_0$): All 6 policies pass with clean success $\ge 0.70$ (mean CLE $< 4.8$\,px).
- **Modern Baseline:** TACO (Temporal Action-Contrastive RL) implemented, trained, and evaluated on Visual Pursuit (`results/taco_summary.json`, `results/taco_checkpoint_s0.pt`).

---

## 2. Benchmark Evaluation Volume

- **Closed-Loop Episodes:** 6,600 episodes (6 policies $\times$ 5 shift families $\times$ 11 severity tiers $\times$ 20 test episodes per cell).
- **Shift Modalities:**
  - `background`: moving sinusoidal chromatic grating
  - `noise`: additive independent Gaussian pixel noise
  - `distractor`: visual clutter / moving secondary distractors
  - `viewpoint`: affine perspective scale and translation jitter
  - `color_gain`: photometric channel gain
  - Severity $s \in [0.0, 1.0]$, where $s = 0.0$ is strictly verified as exact identity.
- **Critical Severity ($s^*$):** Defined as the first severity at which policy success drops below $50\%$ of nominal clean success.
  - Exactly 9 of 30 curves are censored at $s^* = 1.0$ (viewpoint shifts across all 6 policies, and color gain shifts across SVEA policies).

---

## 3. Core Forecasting Results (Strict Holdout Protocol)

Evaluated under strict leave-one-kind-out and disjoint test seeds protocol ($N = 3,060$ points, failure prevalence $0.592$):

| Candidate Score | AUROC | AUPRC | TPR@5%FPR | Clean FAR | Kendall $\tau$ | Selection Regret |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k$NN Policy Features (Ours)** | **0.912** | **0.949** | **0.726** | 0.117 | 0.347 | 0.028 |
| **Action Disagreement** | 0.884 | 0.928 | 0.656 | 0.000 | **0.581** | 0.032 |
| **$Q$-Value Drop** | 0.833 | 0.911 | 0.735 | 0.000 | 0.513 | **0.023** |
| **Raw Pixel $k$NN** | 0.834 | 0.854 | 0.260 | 0.017 | 0.375 | 0.032 |
| **Mahalanobis Latent Distance** | 0.825 | 0.880 | 0.442 | 0.017 | 0.052 | 0.045 |
| **Random Projection $k$NN** | 0.806 | 0.833 | 0.207 | 0.017 | 0.132 | 0.053 |
| **Semantic $k$NN Features** | 0.792 | 0.874 | 0.509 | 0.000 | 0.375 | 0.035 |
| **Critic Temporal Gap** | 0.790 | 0.855 | 0.470 | 0.017 | 0.245 | 0.052 |
| **Centroid / Lyu Bound** | 0.638 | 0.702 | 0.120 | 0.033 | 0.120 | 0.068 |

- **Forecast Error in Severity Units:**
  - Policy $k$NN isotonic calibration: Mean Absolute Error = $0.083$ (Median = $0.017$).
  - $Q$-value drop isotonic calibration: Mean Absolute Error = $0.070$.
- **Prevalence Mixture Portability (Proposition 2):**
  - Analytical AUROC mixture formula verified over 105 mixture regimes with Mean Absolute Error = $0.010$.

---

## 4. Test-Time Adaptation Diagnostic (Negative Finding)

- **Paired Arms:** 40 seed-matched episodes per condition comparing `none`, `ttt`, and `ttss`.
- **Arm Parameters:** Isolated single-factor comparison (identical lr $0.001$, buffer size 8, moment weight $1.0$, differing only in safeguard update rejection).
- **Empirical Status:**
  - Update rejection is active under severe noise (49.0 rejected, 0.0 accepted).
  - However, TTSS achieves **no statistically significant improvement** over unadapted baseline (Holm-adjusted $p = 1.0$ across all conditions).
  - The manuscript reframes test-time adaptation as a diagnostic negative result: self-supervised moment matching fails to prevent policy failure under severe distribution shifts.

---

## 5. Pre-Specified Hypotheses Status

- **H0 (Pre-Training Validation Gate):** SUPPORTED (clean success $\ge 0.70$ vs gate $0.20$).
- **H1 (Abrupt Cliff-Like Failure):** NOT SUPPORTED (only 11 of 25 curves exhibit cliffness $\ge 0.5$; failure is predominantly gradual).
- **H2 (Latent $k$NN Gain vs Baselines):** SUPPORTED (gain vs pixel $0.147$ [$0.132, 0.167$], gain vs random $0.262$ [$0.246, 0.286$]).
- **H3 (Latent $k$NN Policy Ranking Target):** NOT SUPPORTED ($\tau = 0.347 < 0.60$).
- **H4 (Conditional Ranking Advantage):** SUPPORTED (best conditional $\tau = 0.581$ vs mean agnostic $\tau = 0.254$).
- **H5 (Prevalence Mixture Portability):** SUPPORTED (mixture MAE $0.010$ across 105 cases).
- **H6 (TTA Efficacy):** NOT SUPPORTED (Holm $p = 1.0$; no statistically significant recovery).
- **H7 (Conformal Clean False Alarm Calibration):** NOT SUPPORTED (clean false alarm rate $0.117 > 0.05$).
