# Architecture Specification
### System design for the visual RL generalization bounds research plan

Companion to `visual_rl_generalization_bounds_roadmap.md`. This document specifies each component in the pipeline diagram: its responsibility, interface, data contract, and implementation notes, followed by cross-cutting infrastructure decisions.

---

## 1. Design principles

- **Shared substrate with `vision-rl`.** The rollout collector, embedding cache format, and correlation-analysis code should be one shared library imported by both projects, not duplicated. This project extends that substrate with semantic weighting, geometry analysis, and test-time adaptation — it does not replace it.
- **Every module is independently testable.** Each box in the diagram is a component with a narrow, typed interface, so Phase 1 (semantic weighting) can be validated before Phase 2 (geometry) exists, and Phase 3 (TTSS) can be developed against a mocked policy without waiting on Phase 1/2 to finish.
- **Offline analysis is decoupled from online training.** Semantic weighting and geometry analysis operate on *cached* embeddings/rollouts, not live environment steps — this keeps iteration cheap (no re-running Procgen for every metric variant) and matches how `vision-rl`'s pipeline already works.
- **Test-time adaptation is the one component that must run inside the rollout loop**, and it is the only component with a hard latency budget — flagged explicitly in §5.

---

## 2. Component specifications

### 2.1 Environments and rollout collection

**Responsibility:** produce (observation, action, reward, next-observation, ground-truth-mask) tuples across the three environment families, under a controlled set of augmentation/holdout perturbations.

| Environment | Role | Perturbation axes |
|---|---|---|
| Procgen CoinRun | Primary non-smooth benchmark | Background, wall texture, sprite, difficulty (via ProcgenOOD holdout config) |
| ProcgenOOD | Controlled OOD variant of the above | Explicit train/test split per axis, interpolation or extrapolation mode |
| Visual pursuit (tracking) | Smooth-continuous control pole | Camera noise, photometric shift (reuse `vision-rl` harness) |

**Interface:**
```
RolloutCollector.collect(
    env_id: str,
    policy: Policy,
    perturbation_config: PerturbationConfig,
    n_episodes: int,
    seed: int,
) -> RolloutBatch
```
`RolloutBatch` fields: `observations [T,H,W,C]`, `actions [T]`, `rewards [T]`, `episode_ids [T]`, `perturbation_tag: str`, `ground_truth_mask [T,H,W]` (present for Procgen via asset layer, `None` otherwise).

**Output artifact:** parquet/HDF5 shards keyed by `(env_id, perturbation_tag, seed)`, written to a shared `rollout_cache/` directory. This is the same cache format `vision-rl` already uses — extend, don't fork.

**Implementation notes:** Procgen exposes asset/entity positions through its C API; use this for ground-truth hazard/foreground masks rather than a learned segmentation model wherever possible — cheaper and removes a confound (you don't want segmentation-model error conflated with the `d̃rep` hypothesis test in Phase 1).

---

### 2.2 Perception layer — encoders

Two encoder variants run in parallel throughout, since the whole point of Phase 2 is comparing them.

| Encoder | Config | Frozen? |
|---|---|---|
| A — Frozen ResNet-18 | ImageNet-pretrained, decoupled per Stooke et al. (ATC-style) | Yes, plus a partial-fine-tune variant per Jones et al. (2025) findings |
| B — End-to-end CNN | Standard DrQ-v2/SVEA convolutional trunk | No — trained jointly with the policy |

**Interface:**
```
Encoder.embed(observations: Tensor[B,H,W,C]) -> Tensor[B,D]
```
`D` = 512 for ResNet-18 (post-GAP), matches whatever the end-to-end CNN's penultimate layer width is (typically 50–100 for DrQ-v2-style trunks — project both to a common `D` via a linear head only if Phase 2's manifold comparison requires matched dimensionality; keep native dimensionality for Phase 1's distance computation, since `drep`/`d̃rep` don't require it).

**Output artifact:** embedding tensors cached alongside the rollout shard they were computed from — `embeddings/{env_id}/{perturbation_tag}/{encoder_id}.pt`.

---

### 2.3 Semantic weighting module (Phase 1)

**Responsibility:** compute the weight matrix `Ws` and the resulting `d̃rep`.

**Interface:**
```
SemanticWeighter.compute_weights(
    obs_clean: Tensor[B,H,W,C],
    obs_perturbed: Tensor[B,H,W,C],
    ground_truth_mask: Tensor[B,H,W] | None,
) -> Tensor[B,H,W]   # Ws

DistanceEngine.d_tilde_rep(
    phi_clean: Tensor[B,D],
    phi_perturbed: Tensor[B,D],
    Ws: Tensor[B,H,W],
) -> Tensor[B]
```

**Two backends, run side by side per the roadmap:**
- `SegmentationWeighter` — uses `ground_truth_mask` directly when available (Procgen), falls back to an off-the-shelf segmentation model otherwise.
- `SaliencyWeighter` — SEQA-style mask decoder trained with critic-loss supervision; needs a training loop of its own (small — a few conv layers), so this is the one sub-component with actual gradient-based training rather than pure computation.
- Also implement `TLDAWeighter` as a third variant purely as a baseline comparator (reuses the published Lipschitz-sensitivity masking) — not a novel contribution here, but necessary to show `d̃rep` beats it.

**Note on the mask-to-embedding-space mapping:** `Ws` is computed in pixel space (`H×W`) but must weight a distance in embedding space (`D`-dim). This requires either (a) projecting `Ws` through the encoder's spatial feature maps before global pooling (works cleanly for CNN encoders that expose spatial activations pre-GAP), or (b) computing `d̃rep` at the last spatial feature layer rather than the pooled embedding. Decide this explicitly early in Phase 1 — it's a modeling choice, not an implementation detail, and changes what the lemma in the roadmap actually proves.

---

### 2.4 Geodesic geometry module (Phase 2)

**Responsibility:** map encoder activations under an augmentation trajectory to points/subspaces on the Grassmann manifold and compute geodesic (principal-angle) distances between trajectories.

**Interface:**
```
GrassmannMapper.to_subspace(
    activations: Tensor[T,D],   # a trajectory of T embeddings under one augmentation
    subspace_dim: int,
) -> GrassmannPoint

GeodesicDistance.compute(
    p1: GrassmannPoint,
    p2: GrassmannPoint,
) -> float   # principal-angle / projection-metric distance
```

**Implementation notes:**
- Use SVD of the trajectory's activation matrix to get an orthonormal basis (the subspace representative), per the standard Grassmann-manifold construction in Bendokat et al.'s handbook.
- `subspace_dim` is a hyperparameter — sweep a small range (e.g. 5, 10, 20) rather than fixing it arbitrarily; report sensitivity.
- Run this at three encoder fine-tuning regimes (frozen / partial / full), per the roadmap's extension of the frozen-vs-scratch comparison.
- If the MSA (Cayco Gajic et al.) codebase is available, prefer adapting it over a from-scratch Grassmann implementation — check license/API compatibility before committing to either path.

**Output artifact:** a distance matrix per (encoder variant, fine-tune regime, perturbation-pair) — feeds directly into the evaluation engine (§2.6), bypassing the distance engine, since it's a separate analytical claim (representational compactness) rather than a generalization-gap predictor.

---

### 2.5 Policy training and TTSS test-time adapter (Phase 3)

**Policy interface** (standard — DrQ-v2/SVEA/PPO, unchanged from their reference implementations):
```
Policy.act(obs: Tensor[H,W,C]) -> Action
Policy.update(batch: TransitionBatch) -> Metrics
```

**TTSS adapter interface:**
```
TTSSAdapter.adapt(
    encoder: Encoder,
    obs_stream: Tensor[k,H,W,C],   # most recent k frames
) -> Encoder   # returns adapted (or unchanged) encoder
```

**Internal logic (the safeguard is not optional — build it in from the start, not as a follow-up):**
```
def adapt(self, encoder, obs_stream):
    consistency_loss_before = self.local_consistency_loss(encoder, obs_stream)
    candidate_encoder = self.self_supervised_step(encoder, obs_stream)
    consistency_loss_after = self.local_consistency_loss(candidate_encoder, obs_stream)
    if consistency_loss_after < consistency_loss_before:
        return candidate_encoder
    return encoder  # reject the update, trust-region rollback
```
This directly implements the roadmap's Phase 3 risk mitigation (TTT++'s documented failure mode) as a first-class part of the interface, not an afterthought.

**Self-supervised loss candidates** (pick one for v1, keep interface loss-agnostic so swapping is cheap):
- Temporal contrastive consistency (reuse `vision-rl`'s embedding pipeline — same contrastive setup, different application point).
- Lightweight k-frame reconstruction/prediction loss.

**Output artifact:** per-episode logs of `(consistency_loss_before, consistency_loss_after, adaptation_accepted: bool, tracking_error or success_rate)` — this is what lets you measure floor-collapse-rate specifically, not just average error.

---

### 2.6 Evaluation and correlation engine

**Responsibility:** the single place where all upstream signals get turned into the paper's actual claims.

**Interface:**
```
EvaluationEngine.correlate(
    distance_metric: Tensor[N],       # drep or d̃rep per rollout pair
    performance_drop: Tensor[N],      # policy performance delta
) -> SpearmanResult  # rho, p-value, per-holdout-axis breakdown

EvaluationEngine.floor_collapse_rate(
    episode_logs: list[EpisodeLog],
    floor_threshold: float,
) -> float
```

**Required breakdowns** (don't just report a pooled number — the roadmap's whole thesis depends on the per-axis split):
- Correlation split by ProcgenOOD holdout axis (background / texture / sprite / difficulty), matching the granularity you already use in `vision-rl` for appearance vs. geometric shift.
- Correlation for `drep` vs. `d̃rep` side by side, same axes, same environments — this is the headline comparison table.
- Floor-collapse rate before/after TTSS, on the tracking task specifically (where the −153.16 px floor was observed).

**Output artifact:** the results tables and plots that go directly into the paper's Section 4.

---

## 3. Data contracts (cross-cutting)

| Artifact | Format | Written by | Read by |
|---|---|---|---|
| Rollout shard | Parquet/HDF5 | Rollout collector | Encoders, semantic weighter |
| Embedding cache | `.pt` tensor files | Encoders | Semantic weighter, geometry mapper, distance engine |
| Ws weight maps | `.pt` tensor files | Semantic weighter | Distance engine |
| Grassmann distance matrix | `.npz` | Geometry module | Evaluation engine |
| Episode adaptation log | JSONL | TTSS adapter | Evaluation engine |
| Correlation results | JSON + CSV table | Evaluation engine | Paper writeup |

Keeping these as explicit typed artifacts (rather than passing tensors in-process) is what lets Phase 1 and Phase 2 run in parallel on the same cached rollouts without re-collecting data, per the roadmap's Phase 0/1/2 overlap plan.

---

## 4. Suggested repo layout

```
visual-rl-genbounds/
├── envs/                  # env wrappers, ProcgenOOD config, tracking harness (shared w/ vision-rl)
├── encoders/               # frozen ResNet-18, end-to-end CNN, fine-tune regime configs
├── semantic_weighting/      # SegmentationWeighter, SaliencyWeighter (SEQA-style), TLDAWeighter baseline
├── geometry/                # GrassmannMapper, GeodesicDistance, optional MSA adapter
├── ttss/                    # TTSSAdapter, consistency losses, trust-region logic
├── policies/                # DrQ-v2 / SVEA / PPO reference implementations (likely vendored, not rewritten)
├── evaluation/               # correlation engine, floor-collapse analysis, plotting
├── rollout_cache/            # shared cache dir (symlink to vision-rl's if colocated)
└── configs/                  # experiment configs per phase
```

## 5. Cross-cutting concerns

- **Compute budget.** Phases 1–2 are offline/analytical and can run on cached data — cheap. Phase 3 (TTSS) requires live rollouts and is the only GPU-hour-heavy component; scope its Procgen/tracking episode counts conservatively before the "full GPU scale-up run" mentioned as a `vision-rl` next step, since both projects will compete for the same compute.
- **Latency budget for TTSS.** The self-supervised update happens *before* each action or short action-chunk during rollout — it must be cheap enough not to distort the control loop's effective frame rate. Budget and measure this explicitly (e.g. wall-clock ms per adaptation step) rather than assuming it's negligible; if it isn't, that becomes a limitation to report honestly rather than something to discover at deadline time.
- **Reproducibility.** Every artifact in §3 should carry the seed and config hash that produced it — given how many moving pieces there are (encoder variant × fine-tune regime × weighting backend × environment × holdout axis), silent config drift between runs is the most likely source of an irreproducible headline number.
- **Baseline parity.** TLDA, SEQA, and SECANT should run through the *same* rollout cache and evaluation engine as your own methods — implement them as alternative `SemanticWeighter`/encoder backends behind the same interface rather than separate scripts, so the comparison table is apples-to-apples by construction rather than by careful post-hoc alignment.
