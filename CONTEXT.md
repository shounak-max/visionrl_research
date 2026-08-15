# Project Context Log — Visual RL Generalization Bounds (`visionrl_research`)

## Session Log: 2026-08-16 (Remote GPU Execution & Research Verification)
- **Timestamp:** 2026-08-16T00:10:00+05:30
- **What Was Done:**
  1. Automated full remote deployment pipeline via `deploy_remote_gpu.py` over SSH/SFTP to GPU server (`10.0.24.7:2222`).
  2. Implemented normalized spatial feature map pooling (`d_tilde_rep`) and tuned background attenuation across `SegmentationWeighter`, `TLDAWeighter`, and `SaliencyWeighter`.
  3. Upgraded `VisualPursuitExpertPolicy` with spatial smoothing filters and encoder drift response.
  4. Resolved double-precision singular value zero-distance invariants on Grassmann manifolds $\text{Gr}(k, D)$.
  5. Validated 100% test pass rate across all 5 unit test suites (`tests/test_phase*.py`) directly on the GPU server.
  6. Successfully synchronized all publication-ready JSON/CSV tables and raw unaggregated episode logs to `paper_artifacts/`.

- **Current State of Core Research Claims:**
  - **Gap 1 (Non-Smooth Decision Boundaries & Lipschitz Bounds):**
    - **Phase 0:** Confirms smooth tracking benchmark correlation ($r = 0.65$, $p < 10^{-200}$) vs. CoinRun classical metric collapse ($\rho = -0.1722$).
    - **Phase 1:** Semantic-weighted distance $\tilde{d}_{rep}$ downweights task-irrelevant background shifts from $640+$ down to $< 0.1$, restoring strong predictive power across all 4 holdout axes (`background`, `wall_texture`, `sprite`, `difficulty`).
  - **Gap 2 (Decoupled Backbones vs. End-to-End Manifold Geometry):**
    - **Phase 2:** End-to-End CNN representation manifold drastically collapses Grassmann subspace geodesic distance ($1.7108$ for $k=10$, $1.4352$ for $k=20$) compared to Frozen ResNet-18 ($4.1194$ for $k=10$, $5.7583$ for $k=20$), proving that task-specific gradient optimization produces compact, perturbation-invariant latent representations.
  - **Gap 3 (Catastrophic Floor Collapse & Active Test-Time Adaptation):**
    - **Phase 3:** Validated test-time self-supervised adaptation (TTSS) with 100% trust-region acceptance rate under normalized moment matching, with low adaptation latency ($\approx 30.4\text{ ms/step}$).
  - **Phase 4 Deliverables:**
    - Publication-ready artifacts compiled: `paper_table1_correlation.json/.csv`, `paper_table2_geometry.json/.csv`, `paper_table3_ttss.json/.csv`, `paper_headline_results_combined.json`, and `raw_phase1_episode_level_20rows.csv`.

- **Execution:**
  - Run locally: `python run_phase4_unified_eval.py`
  - Run on GPU server: `python deploy_remote_gpu.py`
