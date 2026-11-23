# Changelog: Mapping Reviewer Comments to Paper Changes

This document maps every critique from Area Chairs and Reviewers (G1nX, ykVG, ZjEg) to the precise code, data, and manuscript changes implemented.

---

### Area Chair Rejection Reasons
| AC Critique | Root Problem | Fix & Evidence |
| :--- | :--- | :--- |
| **"Code, claims and tables did not match"** | Hand-typed numbers in LaTeX tables conflicted with experiment logs and artifacts. | Built `scripts/generate_paper_tables.py`, `scripts/generate_paper_figures.py`, and `scripts/check_consistency.py`. Every number in `paper/tables/` is directly read from `results/` CSV/JSON files. Zero warnings and zero failures enforced. |
| **"Novelty unclear"** | Unclear distinction between representation distance bounds and practical OOD boundary forecasting. | Added explicit theoretical and empirical contrast (`paper/sections/related_work.tex`) against Lyu et al. (JAIR 2024), kNN/Mahalanobis OOD detection, Majumdar et al. (CoRL 2025), Römer et al. (NeurIPS 2025), Farid et al. (RSS 2022), and AIRM. Formalized Proposition 1 (identifiability limit of policy-agnostic scores) and Proposition 2 (AUROC mixture formula). |
| **"Baselines old"** | Relied primarily on older DrQ-v2 baselines. | Implemented and executed modern Task-aligned Contrastive RL (TACO) on Visual Pursuit with verified contrastive loss, learning curve, checkpoint, and evaluation (`results/taco_summary.json`). Documented simulator differentiability requirements for D.Va. |

---

### Reviewer G1nX (Rating: 4)
| Comment | Location in Paper | Code / Artifact Reference | Details of Resolution |
| :--- | :--- | :--- | :--- |
| **"Evaluated policies were heuristic, not learned"** | Section 4 (`sections/results.tex`), Section 3 (`sections/method.tex`) | `legacy/heuristic_scripts/`, `notebooks/notebook_revised.ipynb`, `results/training_provenance_full.json` | Quarantined the old Phase 1–3 runner scripts (which used `VisualPursuitExpertPolicy`, a hand-coded proportional controller) to `legacy/heuristic_scripts/`. All paper evaluations are strictly derived from the 6,600-episode learned policy benchmark (DrQ-v2 and SVEA across 3 seeds each, 50k steps). |
| **"Adapted encoder never used in test-time adaptation"** | Section 4.5 (`sections/results.tex`), Section 3.4 (`sections/method.tex`) | `ttss/adapter.py`, `tests/test_audit_assertions.py::test_adapted_encoder_drives_actions` | Verified that TTSSAdapter directly drives actions through the adapted encoder during online deployment. Unit tests confirm active weight modification and rollback when candidate updates deteriorate loss. |
| **"Procgen masks / interventions not as stated"** | Section 3 (`sections/method.tex`), Section 4 (`sections/results.tex`) | `envs/procgen_coinrun.py` | Clarified in `envs/procgen_coinrun.py` that the red-channel mask is an ad-hoc RGB threshold heuristic, not simulator ground truth. Completely excised all claims of "ground-truth segmentation" on Procgen and Procgen forecasting from the paper. |
| **"An equation differs from the code"** | Section 3.2 (`sections/method.tex`), Section 4 (`sections/results.tex`) | `scores/new_scores.py`, `tests/test_symbolic_equations.py` | Audited all mathematical formulations against PyTorch implementations. Symbolically verified Grassmann geodesic distances, projection metrics, and TTT++ moment matching against analytic tests. |

---

### Reviewer ykVG (Rating: 5)
| Comment | Location in Paper | Code / Artifact Reference | Details of Resolution |
| :--- | :--- | :--- | :--- |
| **"Novelty unclear"** | Section 2 (`sections/related_work.tex`), Section 1 (`sections/introduction.tex`) | `paper/sections/method.tex`, `paper/sections/results.tex` | Positioned the paper around rollout-free forecasting of continuous critical severity $s^*$, Proposition 1 (identifiability), Proposition 2 (AUROC mixture formula), and selection regret. Differentiated from predictive red-teaming (which requires rollouts) and FIPER (which predicts scalar returns). |
| **"Grassmann analysis unexplained"** | Section 4.6 (`sections/results.tex`) | `notebooks/notebook_revised.ipynb` (Cell 35) | Clarified that Grassmann distance versus success drop across $k \in \{5, 10, 20\}$ ($\rho = 0.719, 0.707, 0.814, N=330$) is a descriptive trajectory subspace diagnostic, not claimed to predict competence boundaries. |
| **"Method details missing"** | Section 3 (`sections/method.tex`) | `paper/sections/method.tex`, `ttss/adapter.py` | Added explicit mathematical definitions for all 9 candidate scoring rules, leave-one-kind-out isotonic calibration, TTT++ moment matching, and trust-region safeguard update gating. |
| **"Table 1 says 100% clean success while Table 3 says 18.9%"** | Section 4.1 (`sections/results.tex`), Section 4.5 (`sections/results.tex`) | `scripts/check_consistency.py`, `results/hypotheses_verdicts.csv` | Standardized definitions across all sections: operational clean success is mean Centre-of-target Localization Error (CLE) $< 5.0$\,px (verified by pre-training gate $H_0 \ge 0.70$ on learned policies), relative to the $22.05$\,px random-policy floor. Conflicting heuristic tables replaced with verified notebook tables. |

---

### Reviewer ZjEg (Rating: 5)
| Comment | Location in Paper | Code / Artifact Reference | Details of Resolution |
| :--- | :--- | :--- | :--- |
| **"Use recent policies (TACO, D.Va)"** | Section 4.6 (`sections/results.tex`), Section 5 (`sections/conclusion.tex`) | `scripts/train_taco_baseline.py`, `results/taco_summary.json` | Executed a real TACO baseline run on Visual Pursuit ($1,000$ steps), verifying operational execution, contrastive loss, checkpoint, and evaluation (`results/taco_summary.json`). Documented that D.Va requires a differentiable simulator backend, whereas Visual Pursuit closed-form dynamics are non-differentiable through simulator step functions. |
| **"Add learning curves and configuration tables"** | Section 4.1 (`sections/results.tex`), Table 1 (`paper/tables/hyperparams.tex`) | `scripts/generate_paper_figures.py`, `results/training_provenance_full.json`, `paper/figures/fig_learning_curves.pdf` | Emitted empirical learning curve plots from verified DrQ-v2 and SVEA training runs across 50,000 steps for all 3 seeds. Generated complete LaTeX hyperparameter table directly from verified configuration parameters. |
| **"Explain the CLE threshold and environment names"** | Section 3.1 (`sections/method.tex`), Section 4.1 (`sections/results.tex`) | `paper/sections/method.tex`, `envs/visual_pursuit.py` | Clarified that CLE represents Centre-of-target Localization Error (Euclidean distance in pixels between the 2D agent reticle and moving target center). Defined the $5.0$\,px operational success threshold relative to the unguided random-policy floor of $22.05$\,px. Detailed Visual Pursuit mechanics and the 5 physical shift modalities. |

---

### Known Audit Problems Resolved
1. **Quarantine Heuristic Scripts:** Quarantined `run_phase0_sanity_check.py`, `run_phase1_semantic_eval.py`, `run_phase2_geometry_eval.py`, `run_phase3_ttss_eval.py`, and `run_phase4_unified_eval.py` into `legacy/heuristic_scripts/`. All paper evaluations are strictly derived from the 6,600-episode learned policy pipeline.
2. **Hardcoded Acceptance Rate Eliminated:** The hard-coded literal `trust_region_acceptance_rate = 1.0` in the old script has been excised. The paper reports the real notebook TTA data (`tta_paired.csv`), where the safeguard actively rejects updates under noise (49.0 rejected, 0.0 accepted).
3. **TTSS Confound Eliminated & Honest Negative Verdict:** The paper no longer claims an adaptation benefit. TTSS is honestly reframed as a diagnostic study showing negative results on paired comparisons (Holm $p = 1.0$, $H_6$ NOT SUPPORTED).
4. **All Consistency Warnings Resolved:** `scripts/check_consistency.py` enforces zero warnings and zero discrepancies, verifying all 20 data-manuscript invariants with exit code 0.
5. **Modern Baseline Execution:** Executed real TACO training on Visual Pursuit, logging contrastive loss, evaluation metrics, and saving checkpoint.
6. **Procgen Claims Excised:** All claims of "ground-truth segmentation" on Procgen and Procgen forecasting have been completely excised.
7. **Stale Files Cleaned:** Updated `CONTEXT.md` to match `results/` exactly, quarantined `md/main.md` with an explicit SUPERSEDED banner, and verified zero stale root result JSONs.
8. **End-to-End TeX Compilation:** Compiled `paper/main.tex` into `paper/main.pdf` (248.7 KB) using bundled `tools/tectonic.exe` with zero errors.
