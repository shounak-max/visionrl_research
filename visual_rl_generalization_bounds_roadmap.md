# Research Roadmap: Robust Generalization Bounds in Visual RL
### Closing the Gap Between Smooth Representation Bounds and Non-Smooth Procedural Environments

---

## 0. Framing

The proposal identifies three gaps that all trace back to one root cause: **Euclidean/L2 representation-distance metrics assume smooth, task-irrelevant visual drift, but procedural benchmarks like Procgen CoinRun violate that assumption at decision boundaries.** This is the same failure mode you've already observed empirically in `vision-rl` — the pooled Spearman correlation (ρ = −0.549) holds up for appearance-based augmentation pairs but weakens for geometric/viewpoint shifts. This proposal is effectively the theoretical generalization of that empirical finding, so the roadmap below is written to let the two projects share infrastructure where possible (same rollout harness, same embedding extraction pipeline, same correlation-analysis code), while keeping this as a separate, higher-ceiling contribution aimed at a theory-adjacent venue.

Below: literature grounding for each gap, then a phase-by-phase execution plan with concrete deliverables, baselines, and a suggested timeline.

---

## 1. Literature Landscape

### 1.1 Gap 1 — Lipschitz violation / non-smooth representation bounds

- **[Understanding What Affects the Generalization Gap in Visual RL](https://consensus.app/papers/details/fbdcd4a6208358139c3f5014dfe7d142/?utm_source=claude_desktop)** (Lyu et al., 2024, JAIR) is the closest prior theoretical anchor: it proves that minimizing representation distance between train/test environments bounds the generalization gap, validated on DMC-GB. Your proposal's Eq. (1) is essentially this bound — the contribution is showing where it *breaks* (Procgen) and why.
- **[Task-Aware Lipschitz Data Augmentation (TLDA)](https://consensus.app/papers/details/d0fbdcd32a425f71825fba8f0b7df37b/?utm_source=claude_desktop)** (Yuan et al., 2022) is directly relevant to Phase 1: it already identifies task-correlated pixels via *large local Lipschitz constants* and augments only task-irrelevant ones. Your semantic weight matrix `Ws` is a complementary idea — TLDA uses Lipschitz sensitivity as the mask, you propose using semantic segmentation. Worth explicitly differentiating from, or fusing with, TLDA in the related-work section.
- **[SEQA — Saliency-Extracted Q-value by Augmentation](https://consensus.app/papers/details/ac31f8bb3b105b9dbee446fbc3f773f4/?utm_source=claude_desktop)** (Han et al., 2025) trains a mask decoder with critic losses to isolate salient pixels, evaluated on DMControl-GB. This is close prior art for "auxiliary-signal-derived weighting of embeddings" and should be a baseline for `d̃rep`.
- **[On the Generalization of Representations in RL](https://consensus.app/papers/details/43088372b28a5d7ba34bbbbbdda7fde0/?utm_source=claude_desktop)** (Le Lan et al., 2022) gives an alternative generalization bound based on *effective dimension* rather than Lipschitz continuity — a useful fallback framework if the locally-Lipschitz partitioning proves hard to formalize cleanly for discrete transitions.
- **[ProcgenOOD](https://consensus.app/papers/details/60a79bbe61025ce1b072f6c819d756be/?utm_source=claude_desktop)** (Corder et al., 2026) is a near-perfect experimental substrate for Phase 1/4: it extends Procgen with *configurable holdout variables* (background, wall texture, sprites, difficulty) and shows ID/OOD rankings diverge — exactly the controlled perturbation axis you need to isolate "background shift vs. hazard shift" and test whether `d̃rep` tracks the right one.
- **[Leveraging Procedural Generation to Benchmark RL (Procgen)](https://consensus.app/papers/details/bf64daed1b435f5c8a2287315bd01864/?utm_source=claude_desktop)** (Cobbe et al., 2019) — the base benchmark citation.
- **[Automatic Data Augmentation for Generalization in Deep RL](https://consensus.app/papers/details/063fbebb46675604a56e2330ec287a15/?utm_source=claude_desktop)** (Raileanu et al., 2020) — UCB-DrAC — useful as an additional Procgen generalization baseline beyond DrQ-v2/SVEA.

### 1.2 Gap 2 — Decoupled backbones and representational geometry

- **[Decoupling Representation Learning from RL (ATC)](https://consensus.app/papers/details/25f78eb8283b5d61b2d4ed11767a174f/?utm_source=claude_desktop)** (Stooke et al., 2020) is the foundational decoupling paper — frozen contrastively-pretrained encoders match/beat end-to-end RL.
- **[SECANT](https://arxiv.org/abs/2106.09678)** (2021) — self-expert cloning that explicitly decouples robust representation learning from policy optimization for zero-shot visual generalization; a strong architecture-level baseline to compare your ResNet-18 decoupling against.
- **[Pre-trained Visual Representations Generalize Where it Matters in Model-Based RL](https://consensus.app/papers/details/bb80c135f9755123ab618fb7ab7994fa/?utm_source=claude_desktop)** (Jones et al., 2025) directly tests PVMs under severe visual domain shift and finds partial fine-tuning best preserves performance — relevant for deciding whether your ResNet-18 should be fully frozen or partially fine-tuned during Phase 2 analysis.
- **[Building Deep Networks on Grassmann Manifolds](https://consensus.app/papers/details/cb74d8e7723851c9a23c8ed1ec5f2970/?utm_source=claude_desktop)** (Huang et al., 2016) and **[A Grassmann Manifold Handbook](https://consensus.app/papers/details/d57c734f64915f35878f15c7c1112451/?utm_source=claude_desktop)** (Bendokat et al., 2020) — the mathematical toolkit (projection metric, geodesics, Fréchet mean, parallel transport) you'll need to actually compute geodesic angles between perturbation trajectories.
- **[Geometry-Aware Similarity Metrics for Neural Representations (MSA)](https://consensus.app/papers/details/e5f308dd25465eccad2dfb2e56850c06/?utm_source=claude_desktop)** (Cayco Gajic et al., 2026) is very close prior art — it uses *intrinsic* Riemannian geometry (rather than extrinsic/Euclidean comparison) to distinguish representational solutions in deep networks. This is essentially the tool you'd want to adapt for comparing decoupled vs. end-to-end CNN manifolds; cite as the closest methodological precedent and either build on it or explicitly differentiate (they haven't applied it to RL policy generalization).
- **[Mapping Representations in RL via Semantic Alignment for Zero-Shot Stitching](https://arxiv.org/abs/2503.01881)** (2025) — learns latent-space mappings between agents trained under different visual/task variations without fine-tuning; relevant precedent for "how much does representational geometry constrain cross-shift transfer."

### 1.3 Gap 3 — Catastrophic collapse and active test-time adaptation

- **[A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts](https://consensus.app/papers/details/20f437bf541352debef5a3a1828c6bf3/?utm_source=claude_desktop)** (Liang et al., 2023) — use this to scope which TTA family (batch/online/domain) fits a single-episode, non-i.i.d. RL rollout, since most TTA literature assumes i.i.d. test batches, which policy rollouts are not.
- **[Test-Time Training with Self-Supervision (TTT)](https://consensus.app/papers/details/610cbe87fab9515882db624291916104/?utm_source=claude_desktop)** (Sun et al., 2019) and **[TTT++](https://consensus.app/papers/details/3a41e8fbeed85d29a57b5c4252d688a2/?utm_source=claude_desktop)** (Liu et al., 2021) — the core "single-sample self-supervised update before prediction" mechanism your Phase 3 TTSS pipeline is modeled on; TTT++ is particularly relevant since it explicitly studies **when TTT fails/deteriorates under severe shift** — directly relevant to avoiding a second collapse mode inside your adaptation mechanism itself.
- Note: none of the strong TTA hits returned were RL-policy-specific (most are classification/VLM-centric) — **this is itself evidence of the gap you're proposing to fill**, i.e., there is no established "test-time self-supervised adaptation before rollout execution" method benchmarked on visual RL control. Worth stating explicitly in the related-work section as a positioning point rather than a limitation of the search.

---

## 2. Phase-by-Phase Execution Plan

### Phase 0 (Weeks 1–2): Infrastructure and reproduction
- Stand up Procgen CoinRun + ProcgenOOD (for controlled holdout axes) and a smooth-tracking task (reuse your `vision-rl` tracking harness) as the two benchmark poles.
- Reproduce the two reference correlations from the proposal (r = −0.8099 smooth tracking; ρ = −0.1094 CoinRun) as a sanity baseline before touching the new metric — this is your falsifiable starting point.
- Reuse/extend the `vision-rl` embedding-extraction and Spearman-correlation pipeline rather than rebuilding it; this project and `vision-rl` should share a rollout/embedding cache format.
- **Deliverable:** reproduced baseline correlations + shared data pipeline.

### Phase 1 (Weeks 3–8): Locally-Lipschitz Semantic Bound
1. Implement the semantic weight matrix `Ws` two ways in parallel (don't commit to one prematurely):
   - **(a) Segmentation-derived:** off-the-shelf foreground/hazard segmentation (or Procgen's ground-truth asset masks, since you have simulator access — cheaper and more reliable than learned segmentation for a first pass).
   - **(b) Learned saliency-derived:** adapt SEQA's mask-decoder-with-critic-loss approach as a baseline comparison for `Ws`.
2. Compute `d̃rep` vs. classical `drep` correlation with policy performance drop across ProcgenOOD's four holdout axes (background, difficulty, wall texture, sprites) — this directly tests the proposal's core claim that background shift (high `drep`, should be low `d̃rep`) and hazard shift (low `drep`, should be high `d̃rep`) are being mis-weighted by the classical metric.
3. Compare against TLDA's Lipschitz-sensitivity masking and UCB-DrAC as baselines — if `d̃rep` doesn't outperform simply reusing TLDA's existing sensitivity mask, that's an important negative result to know early.
4. Formalize the locally-Lipschitz partitioning argument (state space split into smooth manifolds separated by singular decision boundaries) as a short lemma with a clean proof sketch — this is the part most likely to draw reviewer scrutiny at a theory-adjacent venue, so budget real time for it rather than treating it as a formality after the empirical result.
- **Deliverable:** `d̃rep` implementation + correlation results across Procgen/ProcgenOOD + draft lemma.
- **Risk:** if `d̃rep` only recovers correlation for semantic/hazard shifts but not viewpoint/geometric shifts (the same weak spot you saw in `vision-rl`), that's not a failure — it's a scoping result. Say so explicitly rather than overclaiming a universal fix.

### Phase 2 (Weeks 6–12, overlapping with Phase 1): Geodesic Shape Space Analysis
1. Extract activation manifolds from (a) frozen pre-trained ResNet-18 and (b) end-to-end scratch CNN encoders, under the same augmentation trajectories used in Phase 1.
2. Map to Grassmann manifold representations (subspace per augmentation trajectory) using the Huang et al. / Bendokat et al. toolkit; compute geodesic (principal-angle) distances between trajectories.
3. Where feasible, adapt the MSA framework's intrinsic-geometry comparison instead of building the Grassmann pipeline fully from scratch — check whether their released code supports RL activation tensors directly before reimplementing.
4. Test the two concrete hypotheses from the proposal: (i) pre-trained backbones yield more compact/perturbation-invariant manifolds; (ii) DA trajectories deform shape spaces in backbone-dependent ways that could guide complementary augmentation ensemble design.
5. Cross-reference against Jones et al. (2025)'s finding that *partial* fine-tuning of PVMs preserves performance best under severe shift — run the geometry analysis at 3 fine-tuning regimes (frozen / partial / full) rather than just frozen-vs-scratch, since that's a cheap extension that meaningfully strengthens the contribution.
- **Deliverable:** geodesic-distance comparison plots across backbone/fine-tuning regimes + qualitative manifold visualization (e.g., PCA/t-SNE on tangent-space projections for intuition, backed by the actual geodesic numbers).

### Phase 3 (Weeks 10–16): Active Test-Time Self-Supervision (TTSS)
1. Scope which TTA family actually applies: policy rollouts are sequential and non-i.i.d., so batch-based TTA (BN recalibration, entropy minimization over a batch) doesn't transfer directly — you need single-sample, online-stream methods. TTT / TTT++ are the right starting family; MT3's meta-learned adaptation policy is a stretch goal if time allows.
2. Design the local consistency self-supervised loss for the single-forward-pass update (candidates: temporal contrastive consistency reused from your embedding pipeline, or a lightweight reconstruction/prediction loss over the most recent k frames — avoid anything requiring a batch of test samples).
3. **Explicitly test for TTT++'s documented failure mode** — self-supervised test-time updates *deteriorating* performance under severe shift — before claiming TTSS prevents collapse. Build in a rollback/trust-region safeguard (e.g., only accept the adapted encoder if local consistency loss actually decreases) so Phase 3's fix doesn't introduce a second collapse mode.
4. Benchmark against the "Tracking Floor" collapse scenario from the proposal (−153.16 px floor, 100% of evaluations) — success criterion is variance restoration (agent no longer floors uniformly), not just average error reduction, since the proposal's own diagnosis is that floor-collapse invalidates correlation-based ranking metrics in the first place.
- **Deliverable:** TTSS pipeline + before/after variance and floor-collapse-rate comparison on the tracking + Procgen tasks.

### Phase 4 (Weeks 14–18): Unified Evaluation + Writeup
- Run the full Table 1 evaluation grid (Procgen CoinRun / Visual Pursuit / Multi-Stage Navigation × DrQ-v2, SVEA, decoupled ResNet-18) with `d̃rep` vs. `drep` predictive correlation as the headline comparison.
- Compile the Phase 1–3 results into the paper structure: Sections 2 (gaps, already drafted) → 3 (methodology, now empirically grounded) → 4 (results) → 5 (impact).
- Target venue: given the theoretical framing (generalization bounds, manifold geometry) plus the empirical RL benchmark, this reads best for a **learning-theory-adjacent RL venue** — consider **NeurIPS/ICML workshop track first** (faster iteration, lower bar to get feedback on the Lipschitz-partitioning lemma) before a full ICLR/NeurIPS submission; alternatively a robotics-adjacent venue (**RA-L/CoRL**) if you lean the framing toward the TTSS deployment angle in Phase 3 rather than the pure theory. This is a different track from the ICVGIP submission — keep the empirical `vision-rl` result (ρ = −0.549) as a *cited prior finding* motivating this proposal, not a section reused verbatim.

---

## 3. Suggested Timeline Summary

| Phase | Weeks | Core Output |
|---|---|---|
| 0 — Infra & reproduction | 1–2 | Shared pipeline, reproduced baseline correlations |
| 1 — Semantic Lipschitz bound | 3–8 | `d̃rep` metric + lemma draft |
| 2 — Geodesic shape space | 6–12 | Backbone geometry comparison |
| 3 — Active TTSS | 10–16 | Collapse-mitigation pipeline |
| 4 — Unified eval + writeup | 14–18 | Full paper draft |

Phases 1 and 2 overlap deliberately — Phase 2's geometry analysis reuses Phase 1's augmentation trajectories, so running them in parallel avoids redundant rollout collection.

## 4. Key Risks to Track Early
1. **Phase 1's lemma is the reviewer-critical piece.** Empirical correlation gains are necessary but not sufficient for a theory-flavored venue — don't defer the proof sketch to the end.
2. **Phase 3 needs a safeguard against TTT++'s known failure mode** or the "fix" for catastrophic collapse could itself collapse under severe shift.
3. **Scope Phase 1's claims to the shift types where it actually works** (likely semantic/hazard-boundary shifts, mirroring your `vision-rl` finding that geometric/viewpoint shifts remain weak) — this matches your existing instinct from `vision-rl` to scope contributions honestly rather than overclaim, and reviewers respond well to that framing.
