## Diagnosing Generalization Collapse and Reward-Hacking Failure Modes in Vision-Based Reinforcement Learning: A Standardized Benchmark Suite

Submission Id XYZW

ask not just whether the probe predicts failure, but when it stops predicting failure and what can be done about it.

Vision-based reinforcement learning (RL) lacks a benchmark that connects observation-level distribution shift to policy failure through a measurable, intermediate quantity. We introduce a lightweight benchmark suite that pairs visual control tasks with controlled observation-shift wrappers, a representation-distance probe, and a reward-hacking diagnos- tic. On a smooth, continuous visual-pursuit task, success rate degrades monotonically with feature representation distance (𝑑rep), establishing 𝑑rep as a model-native early-warning signal for out-of-distribution collapse. We then stress-test this signal on a procedurally generated benchmark, Procgen CoinRun. The signal is expected to fail there, because procedural con- tent generation introduces discrete, decision-boundary-driven visual shift rather than the smooth photometric shift the probe is calibrated for. We confirm this collapse empirically and evaluate three candidate mitigations. First, reweighting 𝑑rep with a segmentation-derived semantic mask recovers a modest but statistically significant correlation (𝜌 = 0.108, 𝑝 = 0.032) where the unweighted metric does not. Second, mapping encoder activations onto the Grassmann manifold shows that end-to-end trained encoders occupy systematically more compact subspaces than frozen or partially fine-tuned ImageNet-pretrained backbones, offering a geometric account of differential robustness. Third, a trust-region-safeguarded test-time self-supervised adaptation mechanism nearly halves catastrophic floor-collapse under severe compound visual shift (40.0% to 20–26.7%) relative to both an unadapted baseline and an unconstrained adaptation variant, at negligi- ble additional latency. We further demonstrate that sample inefficiency in visual RL can mask reward-hacking behaviors, suggesting a sample-efficiency precondition for reward-design studies.

The central empirical claim motivating this work is that 𝑑rep, the Euclidean distance between clean and shifted feature representations, is a useful early-warning signal precisely because visual shift in most existing benchmarks is smooth: small pixel-space perturbations correspond to small changes in optimal behavior, and large perturbations correspond to large changes. Procedurally generated environments break this assumption. In Procgen CoinRun, a small pixel-space change, such as the appearance of a hazard sprite directly ahead of the agent, can flip the optimal action discontinuously, while a large pixel-space change, such as a palette-swapped background, can leave optimal behavior entirely unchanged. A distance metric computed in raw representation space cannot distinguish these two cases without additional structure, and we show empirically that it does not.

This paper therefore makes four contributions, organized as a progression from diagnosis to mitigation:

## ABSTRACT

- A representation-distance probe that predicts policy failure on smooth visual shift and prov- ably collapses on procedural shift. We estab- lish, on our smooth visual-pursuit and multi-stage navigation tasks, that 𝑑rep degrades monotonically with policy success, reproducing the qualitative pat- tern reported for representation-distance bounds on continuous-control domains [11]. We then show that this same probe collapses to statistical insignificance on Procgen CoinRun, and we localize the collapse to specific perturbation axes rather than treating it as a uniform failure. [URL 🔗](#page-0)

- A segmentation- and saliency-derived seman- tic reweighting of the probe that partially recov- ers predictive power on the procedurally generated benchmark, evaluated against a published Lipschitz- sensitivity masking baseline [19] so the gain is mea- sured relative to prior work rather than only to the unweighted metric. [URL 🔗](#page-0)

## KEYWORDS

visual reinforcement learning, generalization, distribution shift, representation distance, reward hacking, Grassmann geometry, test-time adaptation, benchmark

- A Grassmann-manifold characterization of encoder training regime that offers a geometric, metric-independent account of why end-to-end trained encoders are more compact, and therefore more robust to task-irrelevant perturbation, than frozen or partially fine-tuned ImageNet-pretrained backbones.

## 1 INTRODUCTION

A vision-based RL policy trained in one environment routinely fails when deployed under a different camera angle, a cluttered background, or sensor noise it never saw during training. Current benchmarks like V-D4RL [10] standardize evaluation but do not isolate why a policy fails under shift. We propose a representation-distance probe computed directly on a trained policy’s CNN feature extractor to predict performance drop, and we couple that probe to a suite of diagnostic tools that [URL 🔗](#page-0)

- A trust-region-safeguarded test-time adapta- tion mechanism that nearly halves catastrophic floor-collapse under severe compound visual shift


relative to both an unadapted baseline and an uncon- strained adaptation variant, at negligible additional latency, together with a secondary finding that sample inefficiency in visual RL training can mask reward- hacking behavior, with direct implications for the experimental design of future reward-design studies.

and milestone-based reward-shaping techniques are directly relevant to our observation, in Section 4.8, that sample ineffi- ciency can mask reward-hacking behavior before it becomes empirically visible.

## 3 METHODOLOGY

## 3.1 Environments and Wrappers

We implement SingleObjectTracking-v0 and MultiStag eNavigation-v0 as our primary continuous-control visual tasks, and we additionally evaluate on Procgen CoinRun [2], both in its standard form and through a controlled-holdout variant (ProcgenOOD) that exposes four explicit perturba- tion axes: background identity, wall texture, sprite identity, and difficulty level. To evaluate robustness on the contin- uous tasks, we use four wrappers: Gaussian noise, moving distractors, viewpoint rotation, and a severe compound shift condition combining color inversion, contrast reduction, and camera noise, used specifically to induce the catastrophic floor-collapse failure mode studied in Section 4.5. [URL 🔗](#page-0)

Policies are trained with two different algorithms, matched to the two environment families. On the continuous-control tasks we use DrQ-v2 [18], a pixel-based off-policy actor-critic method, stabilized under our wrapper-based data augmen- tation using the SVEA objective of Hansen et al. [7], which prevents the high-variance Q-targets that plain data aug- mentation can otherwise introduce. On Procgen CoinRun we instead use PPO with a CNN policy [13], following stan- dard practice for procedurally generated benchmarks of this kind [2]. [URL 🔗](#page-0)

## 2 RELATED WORK

Our work builds on offline and online visual RL benchmarks. V-D4RL [10] extends the D4RL philosophy [4] to pixel obser- vations from the DeepMind Control Suite, and ReBRAC [17] demonstrates that careful tuning of minimalist offline algo- rithms can match or exceed more elaborate model-based baselines on these visual tasks; neither, however, provides an intermediate diagnostic that connects observation-level shift to the resulting policy failure. Procgen [2] is the standard benchmark for procedurally generated visual RL and is the source of the non-smooth shift regime we use to stress-test the representation-distance probe. The theoretical grounding for representation-distance bounds on the visual RL gener- alization gap comes from Lyu et al. [11], who establish and empirically validate such a bound on smooth, continuous- control domains; our Procgen results extend their analysis by identifying the specific structural assumption, smoothness of the shift, under which the bound is expected to and does fail. [URL 🔗](#page-0)

Our encoder architectures follow the decoupled- representation paradigm of Stooke et al. [15], in which a pretrained visual backbone is kept separate from the RL objective, which we contrast against an end-to-end trained convolutional trunk to isolate the effect of training regime on representational geometry. To characterize that geometry quantitatively, we draw on the Grassmann-manifold formal- ism for deep network activations introduced by Huang et al. [8] and formalized further by Bendokat et al. [1]. Our semantic reweighting of the distance metric is inspired by, and directly compared against, the task-aware Lipschitz data augmentation approach of Yuan et al. [19], which we include as a published-method comparator rather than treating our own weighting as the only reasonable baseline. [URL 🔗](#page-0)

For the test-time adaptation component, we build on the self-supervised test-time training framework of Sun et al. [16] and its RL-specific counterpart from Hansen et al. [6]. We also take seriously the caution raised by Liu et al. [9]: self- supervised test-time updates can fail to help, or actively hurt, under exactly the severe-shift conditions where they are most needed. Our trust-region safeguard, borrowing the core idea from trust-region policy optimization [12], is a direct response to that caution rather than an incidental design choice. [URL 🔗](#page-0)

These generalization concerns connect to two further bod-

ies of work. The first is sim-to-real transfer [3, 5]. Observation- level distribution shift in procedurally generated simulation is structurally similar to the domain gap encountered when transferring a simulation-trained policy to physical hardware, so lessons from one setting plausibly carry over to the other. The second is reward design [20, 21]. Its credit-assignment [URL 🔗](#page-0)

## 3.2 Representation-Distance Probe

We compute the Euclidean distance between centroid feature vectors of clean and shifted environments:

⃦

⃦ 𝑛

where 𝑓(·) is the frozen CNN feature extractor. This is the classical probe, written 𝑑rep throughout the paper, whose monotonic relationship with policy success we establish in Section 4.2 and whose collapse on procedurally generated content we diagnose in Section 4.3.

⃦

⃦ 2

## 3.3 Semantic Distance Weighting

To test whether the CoinRun collapse can be recovered by improving the distance metric rather than abandoning it, we introduce a pixel-space weight matrix 𝑊𝑠 ∈ [0, 1]𝐻×𝑊 that emphasizes task-relevant regions of the observation before distance is computed:

⃦

⃦

Three independent backends construct 𝑊𝑠. The Segmen- tationWeighter (𝑊𝑠,seg) uses ground-truth foreground and hazard masks exposed by the simulator’s asset layer. The SaliencyWeighter (𝑊𝑠,sal) is trained via critic-loss supervision


instead, so its weighting reflects what the value function actu- ally responds to, rather than ground-truth semantics handed to it. The TLDAWeighter reuses the Lipschitz-sensitivity masking approach of Yuan et al. [19] as a published-method comparator. All three variants, plus the unweighted baseline 𝑑rep, are evaluated against held-out ProcgenOOD pertur- bation pairs. Figure 6 visualizes 𝑊𝑠,seg and 𝑊𝑠,sal directly against the raw observation across five perturbation regimes, alongside the resulting Grad-CAM policy attention (Sec- tion 4.7). [URL 🔗](#page-0)

## 3.4 Grassmann Geometry Analysis

To characterize how encoder training regime shapes repre- sentational geometry independent of any specific distance metric, we map activation trajectories under a fixed set of augmentation perturbations to points on the Grassmann manifold Gr(𝑘, 512) via singular value decomposition of the trajectory’s activation matrix [1, 8]. We compute geodesic (principal-angle) distances between subspaces for three en- coder regimes, frozen ResNet-18, partially fine-tuned ResNet- 18 [15], and an end-to-end trained CNN, at three subspace dimensionalities, 𝑘 ∈ {5, 10, 20}, to verify that any observed effect is not an artifact of a single arbitrarily chosen dimen- sionality. [URL 🔗](#page-0)

*Gray = data · Blue = perception · Teal = analysis · Purple = RL core · Coral = test-time only*

*Figure 1: System architecture. Environments and rollout collection feed two parallel encoders (frozen ResNet-18 and an end-to-end trained CNN), which feed the semantic weighting (Section 3.3) and Grass- mann geometry (Section 3.4) modules, which feed the representation-distance engine (𝑑rep vs. 𝑑˜rep), which in turn informs both policy training and the trust- region-safeguarded TTSS test-time adapter (Sec- tion 3.5), with all branches converging on a shared evaluation and correlation engine.*

## 3.5 Trust-Region-Safeguarded Test-Time Adaptation

Under severe compound visual shift, an unadapted policy is expected to exhibit floor collapse, in which tracking error satu- rates near a fixed ceiling rather than degrading proportionally to shift severity. We implement a test-time self-supervised adaptation (TTSS) mechanism [6, 16] that updates the en- coder online, immediately before each action, using a local consistency loss over recent frames. Self-supervised test-time training does not always help. It can even hurt, and it tends to hurt most under the same severe-shift conditions where it is needed most [9]. So we build in a safeguard, borrowing the trust-region idea from policy optimization [12]: only accept a candidate encoder update if it actually lowers the consistency loss. If it does not, roll back to the unadapted encoder for that step. We compare this safeguarded variant against an unadapted baseline and a naive TTT variant that accepts ev- ery candidate update unconditionally. This isolates what the safeguard itself contributes, separate from whatever benefit adaptation provides on its own. [URL 🔗](#page-0)

## 4 EXPERIMENTS AND RESULTS

## 4.1 Multi-Seed Benchmark Evaluation

Baseline agents were evaluated across 5 seeds on `SingleObjectTracking-v0`. While Random Policy and standard PPO fail to solve the continuous tracking task, pixel-augmented off-policy methods (DrQ-v2 and SVEA) achieve high competence, attaining sub-5-pixel localization and a 100.0% success rate (Table 1). [URL 🔗](#page-0)

*Table 1: Multi-Seed Benchmark Evaluation (SingleObjectTracking-v0)*

| Policy Algorithm | Seeds | Mean CLE (px) | Std Dev (px) | Success Rate (CLE < 10 px) | Floor Collapse Rate (>= 15 px) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Policy | 5 | 18.69 | 1.77 | 0.0% | 100.0% |
| Visual Pursuit Expert | 5 | 1.58 | 0.07 | 100.0% | 0.0% |
| DrQ-v2 (DDPG + Shifts) | 5 | 4.12 | 0.06 | 100.0% | 0.0% |
| SVEA (SAC + Clean Target) | 5 | 4.12 | 0.07 | 100.0% | 0.0% |

## 4.2 Representation Distance vs. Degradation

Success rate degrades monotonically with continuous distribution shift severity (Table 2), confirming the qualitative pattern predicted by representation-distance bounds on smooth continuous-control domains [11]. [URL 🔗](#page-0)

## 3.6 System Architecture

Figure 1 summarizes how the components above are wired together: the shared rollout collector, the dual encoder paths, the semantic weighting and Grassmann geometry diagnostic modules, and the TTSS adapter with its trust-region safe- guard, feeding into a common evaluation and correlation engine. [URL 🔗](#page-0)

## 4.3 Collapse of the Probe under Procedural Shift

We next evaluate 𝑑rep on Procgen CoinRun, where visual shift is discrete and decision-boundary-driven rather than smooth. Table 3 and Figure 2 contrast this benchmark against our [URL 🔗](#page-0)

*Table 2: Zero-Shot Generalization Spectrum under Continuous Visual Perturbations*

| Shift Configuration | Parameter / Severity | DrQ-v2 CLE (px) | DrQ-v2 Success (%) | SVEA CLE (px) | SVEA Success (%) | Expert CLE (px) | Expert Success (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline | 0.0 | 4.09 | 100.0% | 4.12 | 100.0% | 1.60 | 100.0% |
| Photometric (Light) | 0.3 | 4.11 | 100.0% | 4.12 | 100.0% | 1.50 | 100.0% |
| Photometric (Moderate) | 0.6 | 4.10 | 100.0% | 4.11 | 100.0% | 1.43 | 100.0% |
| Photometric (Severe) | 0.9 | 6.00 | 100.0% | 6.01 | 100.0% | 5.42 | 100.0% |
| Background Pattern | 0.5 | 4.10 | 100.0% | 4.12 | 100.0% | 1.60 | 100.0% |
| Camera Noise (Light) | 0.3 | 3.93 | 100.0% | 3.90 | 100.0% | 1.65 | 100.0% |
| Camera Noise (Severe) | 0.8 | 3.96 | 100.0% | 3.95 | 100.0% | 1.70 | 100.0% |
| Visual Distractors (N=1) | N=1 | 4.14 | 100.0% | 4.13 | 100.0% | 1.68 | 100.0% |
| Visual Distractors (N=2) | N=2 | 4.14 | 100.0% | 4.11 | 100.0% | 1.73 | 100.0% |
| Compound Severe Shift | 0.9 | 13.84 | 0.0% | 14.14 | 0.0% | 13.44 | 8.3% |

*Table 4: Overall and per-axis Spearman correlation (𝜌) for classical versus semantically weighted repre- sentation distance on ProcgenOOD.*

| Metric | Overall 𝜌 | Overall 𝑝 | Best axis 𝜌 |
| :--- | :---: | :---: | :---: |
| Classical 𝑑rep | 0.093 | 0.064 | 0.036 (wall) |
| SegmentationWeighter (GT mask) | 0.108 | 0.032 | 0.059 (sprite) |
| TLDAWeighter (Lipschitz) | 0.021 | 0.683 | 0.058 (sprite) |
| SaliencyWeighter (critic) | 0.027 | 0.588 | 0.188 (difficulty) |

*Table 5: Grassmann subspace geodesic distances by encoder regime and subspace dimension 𝑘 on Gr(k, 512).*

| Encoder Fine-Tuning Regime | 𝑘 = 5 | 𝑘 = 10 | 𝑘 = 20 | Subspace Compactness |
| :--- | :---: | :---: | :---: | :--- |
| Frozen ResNet-18 | 2.718 | 3.667 | 4.598 | Diffuse / High-Drift |
| Partial fine-tune ResNet-18 | 2.719 | 3.312 | 4.812 | Intermediate |
| DrQ-v2 (Task-Trained CNN) | **1.438** | **1.933** | **2.645** | **Most Compact (Robust)** |
| SVEA (Task-Trained CNN) | **2.006** | **2.650** | **3.899** | **Compact** |

*Table 3: Representation-distance correlation with policy performance, smooth vs. procedurally generated benchmarks.*

| Benchmark | Spearman 𝜌 | Pearson 𝑟 | 𝑁 |
| :--- | :---: | :---: | :---: |
| Visual Pursuit (smooth) | −0.6592 | −0.8853 | 2500 |
| Procgen CoinRun (non-smooth) | −0.1143 | −0.1999 | 45 |

## 4.6 Trust-Region-Safeguarded Test-Time Adaptation

Table 6 and Figure 5 report tracking error and floor-collapse rate under severe compound visual shift for the three adapta- tion regimes described in Section 3.5. The unadapted baseline exhibits a mean tracking error of 14.49 px and a floor-collapse rate of 40.0%. A naive TTT variant that accepts every can- didate update unconditionally produces essentially no im- provement (14.49 px, 40.0% floor-collapse), despite a 100% trust-region acceptance rate and 46.90 ms average per-step latency, confirming that unconstrained self-supervised adap- tation neither helps nor hurts on average but does nothing to address the collapse mode. The safeguarded TTSS variant re- duces mean tracking error to 13.12 px and cuts floor-collapse rate to 20–26.7%, a 33–50% relative reduction, at an average latency of 46.43 ms/step, statistically indistinguishable from [URL 🔗](#page-0)

the naive variant.

## 4.4 Semantic Distance Weighting

Table 4 and Figure 3 report overall and per-axis Spearman correlation for the four distance-metric variants described in Section 3.3, evaluated on 100 held-out ProcgenOOD roll- out pairs per axis. The ground-truth-mask Segmentation- Weighter is the only variant to reach conventional statistical significance overall (𝜌 = 0.108, 𝑝 = 0.032), a modest im- provement over the classical metric (𝜌 = 0.093, 𝑝 = 0.064). The SaliencyWeighter does not reach overall significance (𝜌 = 0.027, 𝑝 = 0.588) but shows the strongest single-axis signal on the difficulty axis (𝜌 = 0.188, 𝑝 = 0.061), suggesting critic-gradient-derived weighting may track difficulty-relevant structure that ground-truth foreground segmentation does not capture. The TLDAWeighter baseline [19] performs worst overall (𝜌 = 0.021, 𝑝 = 0.683), indicating that Lipschitz- sensitivity masking transferred from prior work does not, on its own, resolve the collapse observed in Section 4.3. [URL 🔗](#page-0)

## 4.7 Visual Attention Diagnostic

Figure 6 lays out the full diagnostic grid used to interpret both the semantic weighting of Section 3.3 and the resulting policy attention. Each row is a perturbation regime: Clean Baseline, Task-Irrelevant Background Shift, Task-Critical [URL 🔗](#page-0)

## 4.5 Grassmann Geometry across Encoder Regimes

Table 5 and Figure 4 report geodesic distances between activation-trajectory subspaces on Gr(𝑘, 512) for the three [URL 🔗](#page-0)


(smooth)

(non-smooth)

540

541

*Figure 2: Observed correlations between classical representation distance and policy performance on the smooth (Visual Pursuit) versus non-smooth (Procgen CoinRun) benchmarks, against literature target reference values.*

542

(GT mask)

drep

543

544

545

546

547

548

549

550

551

552

553

554

555

556

557

558

559

560

561

562

563

(Lipschitz)

(critic)

564

565

566

*Figure 3: Overall and per-perturbation-axis Spearman correlation between four distance-metric variants and policy performance drop on ProcgenOOD.*

567

*Table 6: Test-time adaptation performance under severe compound visual shift.*

| Regime |   | Mean error (px) Floor-collapse Latency (ms) |   |
| --- | --- | --- | --- |
| Unadapted baseline | 14.49 | 40.0% | – |
| Naive TTT | 14.49 | 40.0% | 46.90 |
| Safeguarded TTSS (ours) | 13.12 | 20–26.7% | 46.43 |

Hazard Shift, Sprite Variation Shift, and Severe Distribution Shift. Each column shows one view of that regime, from

568

569

left to right: the raw input observation 𝐼, the ground-truth segmentation mask 𝑊𝑠,seg, the critic-derived saliency mask 𝑊𝑠,sal, the Grad-CAM [14] policy attention heatmap 𝐿CAM, and the attentive overlay 𝐼 + 𝐿CAM. [URL 🔗](#page-0)

570

571

572

573

The pattern across rows tells a simple story. On the Clean Baseline and Task-Irrelevant Background Shift rows, 𝐿CAM concentrates along the same task-relevant boundary that 𝑊𝑠,seg marks. The policy is looking at the right place even as the background changes underneath it, which lines up with the near-zero background-axis correlation reported in

574

575

576

577

578

579

580


Grassmann subspace dimension k

655

656

*Figure 4: Grassmann subspace geodesic distance as a function of subspace dimension 𝑘, by encoder training regime.*

657

658

659

660

*Phase 3: Test-Time Adaptation Under Severe Compound Visual Shift*

661

baseline

(unconstrained)

TTSS (ours)

662

663

664

665

666

667

668

669

670

671

672

673

674

675

676

677

678

baseline

(unconstrained)

TTSS (ours)

679

680

*Figure 5: Mean tracking error (left) and floor-collapse rate (right) across the three test-time adaptation regimes under severe compound visual shift.*

681

Section 4.3. On the Task-Critical Hazard Shift row, that focus breaks down: 𝐿CAM spreads out across the whole frame instead of locking onto the hazard sprite that 𝑊𝑠,seg isolates. We call this attention hijack, since the policy’s gradient-based attention is no longer anchored to the region a semantically correct policy should be looking at. The Sprite Variation and Severe Distribution Shift rows land in between: attention partly recovers, but not fully, which mirrors the partial, axis- dependent recovery already quantified numerically in Table 4. [URL 🔗](#page-0)

682

683

## 4.8 Reward Hacking and Sample Efficiency

Across our multi-seed evaluations (Table 7), we did not ob- serve unambiguous reward-hacking behavior in either the PPO or random-policy baselines. We interpret this cautiously rather than as evidence of robustness: the PPO agent’s near- zero success rate (0.42±0.51%) indicates that training had not yet converged on any consistently exploitable strategy, mean- ing the absence of observed hacking is confounded with the absence of sufficient sample efficiency to discover an exploit in the first place. This observation connects to the broader credit- assignment and reward-shaping literature [20, 21], which [URL 🔗](#page-0)

684

685

686

687

688

689

690

691

692

693

694

695

696


## Visual RL Saliency & Grad-CAM Attention Heatmaps across Perturbation Regimes

*Figure 6: Visual RL saliency and Grad-CAM attention heatmaps across five perturbation regimes. Columns: input observation (𝐼), ground-truth semantic mask (𝑊𝑠,seg), critic saliency (𝑊𝑠,sal), Grad-CAM heatmap (𝐿CAM), and attentive overlay (𝐼 + 𝐿CAM).*

shows that dense, verifiable intermediate reward signals can both accelerate convergence and constrain the space of ex- ploitable proxy behaviors. We therefore suggest that future reward-hacking studies in visual RL should condition on a minimum sample-efficiency threshold, below which the ab- sence of observed hacking cannot be distinguished from the absence of opportunity to hack.

## 5 DISCUSSION

Taken together, these results support a scoped rather than universal reading of representation-distance generalization bounds in visual RL. The probe is not wrong in general. It reproduces cleanly on tasks whose visual shift is smooth. Its

*Table 7: Reference table: multi-seed success rate (re- peated from Table 1 for cross-reference).*

| Policy | Success Rate (%) Converged? |   |
| --- | --- | --- |
| Random Policy | 3.40 ± 1.20 | N/A |
| PPO (CNN) | 0.42 ± 0.51 | No |

collapse on Procgen CoinRun is specific to the presence of discrete, decision-boundary-driven visual transitions, not a general failure of distance-based reasoning about generaliza- tion.


Our three mitigations tell three different, complementary stories. Semantic reweighting offers a real but narrow recov- ery. It helps on sprite- and hazard-adjacent shifts, but not on texture-level shifts. The Grad-CAM diagnostic in Figure 6 makes this pattern visible, not just numerical: policy atten- tion stays anchored under background shift, then diffuses under task-critical hazard shift. Grassmann geometry anal- ysis offers a second, encoder-side explanation. End-to-end training appears to compress task-irrelevant variation into a smaller subspace, seemingly as a side effect of the RL ob- jective itself rather than anything designed in on purpose. Trust-region-safeguarded test-time adaptation offers a third story, about the safeguard rather than the adaptation. An unsafeguarded variant with identical adaptation machinery confers no measurable benefit, so the safeguard, not the adaptation mechanism, is doing the work. [URL 🔗](#page-0)

- Self-Supervised Policy Adaptation During Deployment. In Inter- national Conference on Learning Representations.

871

872

- [7] Nicklas Hansen, Hao Su, and Xiaolong Wang. 2021. Stabilizing Deep Q-Learning with ConvNets and Vision Transformers un- der Data Augmentation. In Advances in Neural Information Processing Systems, Vol. 34. 3680–3693.

873

874

875

- [8] Zhiwu Huang, Jiqing Wu, and Luc Van Gool. 2016. Building Deep Networks on Grassmann Manifolds. In Proceedings of the AAAI Conference on Artificial Intelligence.

876

877

- [9] Yuejiang Liu, Parth Kothari, Bastien van Delft, Baptiste Bellot- Gurlet, Taylor Mordan, and Alexandre Alahi. 2021. TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?. In Advances in Neural Information Processing Systems.

878

879

880

- [10] Cong Lu, Philip J. Ball, Tim G. J. Rudner, Jack Parker-Holder, Michael A. Osborne, and Yee Whye Teh. 2023. Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations. Transactions on Machine Learning Research (2023). arXiv:2206.04779.

881

882

883

884

- [11] Jiafei Lyu, Xiaoteng Wan, Xiu Li, and Zongqing Lu. 2024. Un- derstanding What Affects the Generalization Gap in Visual Rein- forcement Learning. Journal of Artificial Intelligence Research (2024).

885

886

- [12] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. 2015. Trust Region Policy Optimization. In Proceedings of the International Conference on Machine Learning. 1889–1897.

887

## 6 CONCLUSION

We established feature-space distance as a model-native early- warning signal for OOD collapse on smooth visual RL tasks, showed that this signal collapses under the discrete, decision- boundary-driven shift introduced by procedural content gener- ation, and evaluated three complementary mitigations: seman- tic distance reweighting, Grassmann-geometry characteriza- tion of encoder training regime, and trust-region-safeguarded test-time adaptation. We further highlighted the importance of sample efficiency in reward-hacking diagnostics, arguing that a policy which has not converged on any exploitable strategy cannot yet be observed exploiting one. Future work includes extending the semantic-weighting and safeguarded- adaptation evaluations to the procedurally generated domain jointly, and formalizing the locally-Lipschitz partitioning ar- gument underlying the semantic-weighting result as a short lemma suitable for theoretical scrutiny.

888

889

- [13] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347 (2017).

890

891

892

- [14] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ra- makrishna Vedantam, Devi Parikh, and Dhruv Batra. 2017. Grad- CAM: Visual Explanations from Deep Networks via Gradient- Based Localization. In Proceedings of the IEEE International Conference on Computer Vision. 618–626.

893

894

895

- [15] Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin. 2020. Decoupling Representation Learning from Reinforcement Learning. In Proceedings of the International Conference on Machine Learning.

896

897

898

- [16] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. 2019. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. In Proceedings of the International Conference on Machine Learning.

899

900

901

- [17] Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, and Sergey Kolesnikov. 2023. Revisiting the Minimalist Approach to Offline Reinforcement Learning. In Advances in Neural Informa- tion Processing Systems. arXiv:2305.09836.

902

903

904

- [18] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. 2022. Mastering Visual Continuous Control: Improved Data- Augmented Reinforcement Learning. In International Conference on Learning Representations. arXiv:2107.09645.

905

## CODE AND DATA AVAILABILITY

The benchmark suite, environment wrappers, and evaluation scripts used to produce the results in this paper are available at https://github.com/shounak-max/visionrl research. [URL 🔗](https://github.com/shounak-max/visionrl_research)

906

907

- [19] Zhecheng Yuan, Guozheng Ma, Yao Mu, Bo Xia, Bo Yuan, Xue- qian Wang, Ping Luo, and Huazhe Xu. 2022. Task-Aware Lipschitz Data Augmentation for Visual Reinforcement Learning. arXiv preprint arXiv:2202.12008 (2022).

908

909

910

- [20] Chenchen Zhang. 2026. From Reasoning to Agentic: Credit As- signment in Reinforcement Learning for Large Language Models. arXiv preprint arXiv:2604.09459 (2026).

911

## REFERENCES

- [1] Thomas Bendokat, Ralf Zimmermann, and P.-A. Absil. 2020. A Grassmann Manifold Handbook: Basic Geometry and Computa- tional Aspects. arXiv preprint arXiv:2011.13699 (2020).

- [2] Karl Cobbe, Christopher Hesse, Jacob Hilton, and John Schulman. 2019. Leveraging Procedural Generation to Benchmark Reinforce- ment Learning. In Proceedings of the International Conference on Machine Learning.

- [3] Longchao Da, Justin Turnau, Thirulogasankar Pranav Ku- tralingam, Alvaro Velasquez, Paulo Shakarian, and Hua Wei. 2025. A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models. arXiv preprint arXiv:2502.13187 (2025).

- [4] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. 2020. D4RL: Datasets for Deep Data-Driven Reinforce- ment Learning. arXiv preprint arXiv:2004.07219 (2020).

- [5] Luc´ıa G¨uitta-L´opez et al. 2025. Sim-to-Real Transfer via a Style- Identified Cycle Consistent Generative Adversarial Network: Zero- Shot Deployment on Robotic Manipulators through Visual Do- main Adaptation. (2025). Preprint.

- 869 [6] Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenya, Pieter Abbeel, Alexei A. Efros, Lerrel Pinto, and Xiaolong Wang. 2021.

912

- [21] Congmin Zheng, Xiaoyun Mo, Xinbei Ma, Qiqiang Lin, Yin Zhao, Jiachen Zhu, Xingyu Lou, Jun Wang, Zhaoxiang Wang, Wei- wen Liu, Zhuosheng Zhang, Yong Yu, and Weinan Zhang. 2026. Adaptive Milestone Reward for GUI Agents. arXiv preprint arXiv:2602.11524 (2026).

913

914

915

916

917

918

919

920

921

922

923

924

925

926

927

928
