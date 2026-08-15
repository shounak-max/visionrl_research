"""
Phase 4 Unified Evaluation Pipeline Runner.

Runs end-to-end evaluation across:
1. Phase 1 Semantic-Weighted Generalization Bounds (d_tilde_rep vs d_rep)
2. Phase 2 Geodesic Geometry on Grassmann Manifolds (Frozen vs Partial vs End-to-End CNN)
3. Phase 3 Active Test-Time Self-Supervised Adaptation (Floor-Collapse Rate & Latency)

Outputs publication-ready JSON and CSV tables formatted for paper insertion.
"""

import json
import os
import sys

from run_phase1_semantic_eval import evaluate_phase1_grid
from run_phase2_geometry_eval import evaluate_phase2_geometry
from run_phase3_ttss_eval import evaluate_phase3_ttss
from evaluation import UnifiedEvaluationEngine


def main():
    print("#" * 80)
    print("      VISUAL RL GENERALIZATION BOUNDS — MASTER UNIFIED EVALUATION PIPELINE")
    print("#" * 80)

    seed = 42

    # 1. Run Phase 1 Semantic Weighting Grid
    print("\n>>> EXECUTION PHASE 1: Semantic Distance Metrics (d_tilde_rep) <<<")
    phase1_results = evaluate_phase1_grid(seed=seed, n_episodes=5)

    # 2. Run Phase 2 Geodesic Geometry Grid
    print("\n>>> EXECUTION PHASE 2: Grassmann Geodesic Geometry Analysis <<<")
    phase2_results = evaluate_phase2_geometry(seed=seed, n_episodes=5)

    # 3. Run Phase 3 TTSS Adaptation Grid
    print("\n>>> EXECUTION PHASE 3: Active Test-Time Self-Supervised Adaptation <<<")
    phase3_results = evaluate_phase3_ttss(seed=seed, n_episodes=15)

    # 4. Generate Master Paper Tables & Export Artifacts
    print("\n>>> EXECUTION PHASE 4: Generating Paper Tables & Artifact Exports <<<")
    paths = UnifiedEvaluationEngine.export_headline_results(
        phase1_results, phase2_results, phase3_results, output_dir="paper_artifacts"
    )

    print("\n" + "=" * 80)
    print("                    MASTER PUBLICATION ARTIFACTS EXPORTED")
    print("=" * 80)
    for name, filepath in paths.items():
        print(f"- {name:<20}: {filepath}")
    print("=" * 80)
    print("[COMPLETE] Visual RL Generalization Bounds Codebase Execution Finished Successfully.")


if __name__ == "__main__":
    main()
