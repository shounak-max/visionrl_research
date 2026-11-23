# SCOPE.md: Scope Decision & Reconciliation Audit

**Repository:** `shounak-max/visionrl_research`  
**Date:** November 23, 2026  
**Status:** Task 1 Proposal — Awaiting User Confirmation Before Proceeding to Task 2

---

## 1. Executive Summary: The Two Incompatible Studies

An audit of the repository reveals that the repository currently contains code, artifacts, and text from **two fundamentally distinct studies**:

| Dimension | Earlier Study (Currently in Repo Files) | Actual Evidence Base (In Notebook) |
| :--- | :--- | :--- |
| **Primary Research Goal** | Representation distance correlation ($d_\mathrm{rep}$) and Grassmannian manifold shape spaces | Rollout-free competence boundary forecasting ($s^*$ critical severity) |
| **Benchmark Domain** | Visual Pursuit ($64 \times 64$) + Procgen CoinRun (offline transitions) | Visual Pursuit ($48 \times 48$) with 5 controlled physical shifts |
| **Policy Portfolio** | 1 checkpoint (`drq2_cnn_encoder.pt`), training logged for 5,000 steps (seeds 42, 43) | 6 fully trained policies: DrQ-v2 (seeds 0,1,2) and SVEA (seeds 0,1,2) trained for full 50,000 steps |
| **Evaluation Volume** | 30 test episodes per adaptation condition; 400 offline CoinRun transitions | **6,600 closed-loop episodes** (6 policies $\times$ 5 shift types $\times$ 11 severities $\times$ 20 episodes) |
| **Scoring Framework** | $d_\mathrm{rep}$, $\tilde{d}_\mathrm{rep}$ (segmentation, TLDA, saliency), Grassmann geodesic | **9 rollout-free scores** ($k$NN policy, $k$NN pixel, $k$NN random, Mahalanobis, Centroid/Lyu bound, Action Disagreement, $Q$-drop, Critic Gap, Semantic $k$NN) |
| **Decision Targets** | Pearson $r$ / Spearman $\rho$ correlation with return drop | Held-out AUROC, TPR@5%FPR, critical severity $s^*$ forecast error (mean/median), Kendall $\tau$-b, selection regret, clean false alarms |
| **Theoretical Bounds** | Lyu et al. Lipschitz bound, Grassmann geodesic projection | Identifiability (Proposition 1), Portability & AUROC Mixture (Proposition 2) |
| **Adaptation Study** | TTSS with 100% acceptance rate (untriggered trust-region safeguard) | Paired TTA (40 episodes per contrast), Holm correction, reporting true failure to stabilize |

---

## 2. Inventory: Notebook vs. Repository

### A. What Exists ONLY in the Notebook (`notebooks/notebook_revised.ipynb`)
1. **$48 \times 48$ Visual Pursuit Environment:** Exact 2D continuous visual tracking with 5 parameterized shifts:
   - `background`: moving procedural sinusoidal color grating.
   - `noise`: additive independent Gaussian pixel noise.
   - `distractor`: moving secondary distractors / visual clutter.
   - `viewpoint`: affine perspective scale and translation shift.
   - `color_gain`: channel-wise color gain / photometric distortion.
   - Severity parameter $\alpha \in \{0.0, 0.1, 0.2, \dots, 1.0\}$ where $\alpha = 0.0$ is exact identity.
2. **6 Fully-Trained Policy Artifacts & Learning Curves:**
   - 3 seeds of DrQ-v2 (seeds 0, 1, 2) trained for 50,000 environment steps.
   - 3 seeds of SVEA (seeds 0, 1, 2) with decoupled critic/actor losses trained for 50,000 environment steps.
   - Pre-training validation gate ($H_0$): all 6 policies verified with clean success $\ge 0.70$ (mean CLE $< 5.0$\,px).
3. **6,600 Closed-Loop Episode Dataset & Ground Truth Success Curves:**
   - Evaluated across 11 severities per shift type with critical severity $s^*$ defined at success threshold $\text{CLE} < 5.0$\,px.
   - Censoring protocol: 9 of 30 curves censored at $s^* = 1.0$ (viewpoint shift and severe SVEA color gain).
4. **9 Candidate Rollout-Free Predictors:**
   - Feature $k$NN (`knn_policy`), pixel $k$NN (`knn_pixel`), random projection $k$NN (`knn_random`), Mahalanobis distance (`maha_policy`), Centroid/Lyu distance (`cent_policy`), Action Disagreement (`act_disagree`), $Q$-value drop (`q_drop`), Critic gap (`critic_gap`), and Semantic feature $k$NN (`sem_knn_policy`).
5. **Rigorous Validation Protocols:**
   - Conformal clean-only calibration (controlling clean false alarm rate $\le 0.05$).
   - Strict leave-one-kind-out isotonic calibration (fit on 4 shift kinds, evaluate forecast on the 5th held-out kind).
   - Paired cluster bootstrap resampling clustered by policy seed with Holm-Bonferroni correction.
   - AUROC mixture formula verification over 105 mixture test cases (mean error $0.010$).
6. **Pre-specified Hypotheses ($H_0$ through $H_7$):**
   - Mechanically evaluated: $H_0, H_2, H_4, H_5$ supported; $H_1, H_3, H_6, H_7$ not supported.

### B. What Exists ONLY in the Repository (`visionrl_research`)
1. **Procgen CoinRun Offline Infrastructure:**
   - `envs/procgen_coinrun.py` and `envs/procgen_ood.py` (Procgen Gym wrapper; fails on Windows due to missing C-API `gym` binary).
   - `paper_artifacts/raw_phase1_episode_level_20rows.csv` and Table 1 correlation grid across 400 transitions.
2. **Grassmann Manifold Geometric Analysis:**
   - `geometry/grassmann.py` and `geometry/distance.py` (canonical principal angles, geodesic distance, projection metric).
   - Subspace dimension sweeps ($k \in \{5, 10, 20\}$) comparing Frozen ResNet-18, Partial Fine-Tuning, and DrQ-v2 CNN.
3. **TTSS Adapter with Trust-Region Safeguard:**
   - `ttss/adapter.py`: TTT++ moment matching + temporal consistency loss + candidate update rollback safeguard.
   - Evaluated on $64 \times 64$ Visual Pursuit under compound `severe_shift` (30 test episodes per condition).
4. **Grad-CAM Visualizer:**
   - `evaluation/gradcam.py` and `run_gradcam_visualizer.py` (spatial saliency heatmap blending).
5. **Remote GPU Deployment Scripts:**
   - `deploy_remote_gpu.py` (SSH/SCP automation for Tesla K80 cluster).

### C. What Overlaps
1. **Core Problem Definition:** Pre-rollout anticipation of visual RL degradation without online rollouts.
2. **Visual Pursuit Tracking Task:** Reticle velocity control $(\Delta x, \Delta y)$, Centre-of-target Localization Error (CLE), $5.0$\,px operational success threshold, and $\approx 22.05$\,px random-policy floor.
3. **DrQ-v2 Base Trunk Architecture:** Convolutional encoder + Actor-Critic architecture with random shift data augmentation.
4. **Tested Scores 10 & 11:** PAC-Drift (Jacobian/Hessian autograd) and GCM-Drift (AIRM covariance distance) implemented in `scores/new_scores.py`.

---

## 3. Scope Decision & Recommendation

### Proposed Scope: **The $s^*$ Boundary Forecasting Paper (Evidence Base: Notebook)**

We propose defining the paper's scope strictly as:
> **"Forecasting Competence Boundaries in Visual RL Without Rollouts"**  
> Grounded in the complete 6,600-episode Visual Pursuit ($48 \times 48$) benchmark, evaluating rollout-free forecasting of the critical failure severity $s^*$ across 5 shift families, comparing 11 predictor scores (the 9 existing scores + PAC-Drift and GCM-Drift), and reporting pre-specified hypotheses honestly.

### Disposition of Repository Components:
1. **Grassmann Manifold Analysis:**  
   *Move to Appendix or Drop entirely.* Reviewer ykVG noted that Grassmann analysis was unexplained and disconnected from the core forecasting claim. In the $s^*$ boundary forecasting setting, trajectory subspaces do not predict boundary severities. We recommend dropping Grassmann analysis from the main text or placing it as an exploratory representational analysis in Appendix C.
2. **Procgen CoinRun Offline Transitions:**  
   *Drop from Main Paper (or Retain strictly as motivating counter-example in Appendix).* The C-API Procgen gym is not runnable in standard environments and reviewer G1nX specifically objected to claims regarding Procgen masks. Scoping the paper cleanly to continuous visual control eliminates unbacked Procgen forecasting claims.
3. **Test-Time Adaptation (TTSS):**  
   *Report as a Secondary Empirical Study with Honest Framing.* Rename from "Safeguarded TTSS" to **"Regularized Self-Supervised Adaptation"**. Report the notebook's 40-episode paired comparisons (Holm-corrected), honestly disclosing that while self-supervised moment matching stabilizes certain shifts, it fails to achieve statistically significant success improvements ($p = 1.0$), and the trust-region safeguard rejected updates on only a subset of runs.

---

## 4. Work Plan for Subsequent Tasks (Upon User Confirmation)

1. **Task 2 (Port Notebook to Modules):**
   - Port `PursuitEnv` ($48 \times 48$), the 5 parameterized observation shifts ($\alpha \in [0, 1]$), the 9 scoring functions, isotonic calibration, and paired bootstrap statistical routines into clean `src/` and `scripts/` modules with comprehensive pytest suites.
2. **Task 3 (Training Provenance):**
   - Implement genuinely distinct DrQ-v2 and SVEA recipes (`policies/svea.py` with decoupled augmented critic loss and clean actor loss).
   - Generate and store complete 50,000-step training logs and checkpoints across all 6 policies.
   - Programmatically emit empirical learning curves and config tables.
3. **Task 4 (Evaluate Scores 10 & 11):**
   - Run PAC-Drift ($\lambda=0$ and $\lambda=1$) and GCM-Drift through the unified 6,600-episode pipeline.
   - Test whether curvature ($\lambda=1$) outperforms action disagreement and $\lambda=0$.
4. **Task 5 (Adaptation Claims):**
   - Rerun TTA paired evaluations with $\ge 40$ episodes per contrast and report 95% bootstrap intervals.
5. **Task 6 (Consistency & Cleanup):**
   - Delete stale root JSONs, correct `CONTEXT.md`, unify `method.tex`, and enforce CI consistency check.
6. **Tasks 7 & 8 (Baselines & Manuscript):**
   - Finalize paper text, limitations, and `CHANGELOG.md`.

---
*Please review this proposed scope. Once confirmed, we will immediately commence Task 2.*
