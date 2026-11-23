# Paper Revision Report

## What Was Run
- Visual Pursuit environment ($64\times 64$, continuous tracking, success = mean CLE < 5\,px)
- DrQ-v2 and SVEA-style policies, 3 seeds each, 50k steps
- Phase 1: Semantic weighting of $d_\mathrm{rep}$ (Procgen CoinRun, 400 episodes across 4 holdout axes: background, wall texture, sprite, difficulty)
- Phase 2: Grassmann manifold analysis across 3 encoder regimes (Frozen ResNet-18, Partial Fine-Tune ResNet-18, End-to-End CNN) for subspace dimensions $k \in \{5, 10, 20\}$
- Phase 3: TTSS adaptation under severe shift (30 test episodes per condition: Unadapted Baseline, Naive TTT, Safeguarded TTSS)

## What Was NOT Run (and Removed from Paper)
- Reacher environment: not run, removed from all paper text
- TACO policy: implementation deferred due to lack of GPU hardware (benchmarked CPU speed at ~10.9 steps/s, requiring ~3.8h per seed without OOD sweeps); excluded from experimental results and reported as future work
- D.Va baseline: requires differentiable simulator; excluded because Visual Pursuit closed-form dynamics are non-differentiable through PyTorch autograd graph
- Procgen-based competence boundary forecasting: not run as stated in paper title (only Phase 1 semantic weighting grid evaluated on CoinRun transitions)

## Hypotheses
- **H1: Classical representation distance $d_\mathrm{rep}$ linearly forecasts performance degradation.**
  - *Status:* **PARTIALLY HELD (Domain-Dependent).**
  - Held strongly on smooth continuous-control Visual Pursuit (Pearson $r = -0.8853$, Spearman $\rho = -0.6592$, $N = 2500$, $p < 10^{-200}$).
  - Failed completely on combinatorial transitions in Procgen CoinRun (Spearman $\rho = -0.1143$, $p = 0.4545$, $N = 45$).
- **H2: Semantic spatial weighting ($\tilde{d}_\mathrm{rep}$) restores correlation on combinatorial shifts.**
  - *Status:* **FAILED / MARGINAL.**
  - Ground-truth segmentation mask achieved modest recovery ($\rho = 0.1075$, $p = 0.0315$ vs baseline $\rho = 0.0926$, $p = 0.0643$).
  - Model-based weighters failed to achieve statistical significance (TLDA Lipschitz $\rho = 0.0205$, $p = 0.6825$; Critic Saliency $\rho = 0.0272$, $p = 0.5877$).
  - All weighters showed $\rho = 0.0$ on background shift due to policy invariance.
- **H3: End-to-end RL training structures representation manifolds into more compact subspaces under shift.**
  - *Status:* **HELD.**
  - End-to-End CNN achieved lower Grassmann geodesic distance across all $k \in \{5, 10, 20\}$ ($1.9255, 2.2297, 1.5402$) compared to Frozen ResNet-18 ($2.9735, 4.2021, 5.7208$).
- **H4: Test-Time Self-Supervised Adaptation (TTSS) reduces catastrophic floor collapse under severe shift.**
  - *Status:* **HELD.**
  - Safeguarded TTSS reduced catastrophic floor collapse from $40.0\%$ to $20.0\%$ ($50\%$ relative reduction) and improved mean CLE from $14.49$\,px to $13.12$\,px.
- **H5: The trust-region safeguard actively rejects harmful candidate updates.**
  - *Status:* **FAILED / UNTRIGGERED EMPIRICALLY.**
  - Both naive TTT and safeguarded TTSS recorded `trust_region_acceptance_rate = 1.0` ($100\%$). The safeguard accepted all updates because SGD consistently reduced local window loss, acting as a no-op during this benchmark.

## Known Issues
1. TTSS `trust_region_acceptance_rate = 1.0` for both naive and safeguarded — safeguard effectively did not reject any updates.
2. Discrepancy: CONTEXT.md says `floor_collapse_rate = 26.7%` but JSON says `20.0%` — JSON is source of truth.
3. Discrepancy: CONTEXT.md says Grassmann $k=10$ E2E CNN = 2.4133 but JSON says 2.2297 — JSON is source of truth.

## Open Items
- Run TACO (3 seeds, 50k steps) on GPU hardware when remote GPU credentials are configured.
- Investigate why trust-region acceptance = 100% and explore validation on holdout validation buffers or curvature-based rejection criteria.
- Add second environment (e.g. robotic reacher or manipulator) to substantiate generalizability claims.
