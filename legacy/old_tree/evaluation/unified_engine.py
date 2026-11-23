"""
Unified Evaluation Engine for Visual RL Generalization Bounds.

Aggregates statistics from Phase 1 (Semantic Distance Bounds), Phase 2 (Grassmann Geodesic Geometry),
and Phase 3 (Active TTSS Adaptation), formatting and exporting publication-ready JSON and CSV tables.
"""

import csv
import json
import os
from typing import Any, Dict, List, Optional


class UnifiedEvaluationEngine:
    """Unified evaluation reporter and paper table generator."""

    @staticmethod
    def export_headline_results(
        phase1_data: Dict[str, Any],
        phase2_data: Dict[str, Any],
        phase3_data: Dict[str, Any],
        output_dir: str = ".",
    ) -> Dict[str, str]:
        """Exports JSON and CSV paper headline tables.

        Returns:
            Dict mapping table names to saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # 1. Export Table 1: Correlation Breakdown by Holdout Axis (ProcgenOOD)
        t1_json = os.path.join(output_dir, "paper_table1_correlation.json")
        t1_csv = os.path.join(output_dir, "paper_table1_correlation.csv")

        with open(t1_json, "w") as f:
            json.dump(phase1_data, f, indent=2)
        paths["table1_json"] = t1_json

        with open(t1_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Metric / Weighter Backend",
                    "Overall Spearman Rho",
                    "Background Shift",
                    "Wall Texture Shift",
                    "Sprite Shift",
                    "Difficulty Shift",
                ]
            )
            for m_name, m_data in phase1_data.items():
                rho_all = m_data.get("overall_rho", 0.0)
                b_data = m_data.get("axis_breakdown", {})
                rho_bg = b_data.get("background", {}).get("spearman_rho", 0.0) if b_data else 0.0
                rho_wall = b_data.get("wall_texture", {}).get("spearman_rho", 0.0) if b_data else 0.0
                rho_sprite = b_data.get("sprite", {}).get("spearman_rho", 0.0) if b_data else 0.0
                rho_diff = b_data.get("difficulty", {}).get("spearman_rho", 0.0) if b_data else 0.0
                writer.writerow(
                    [
                        m_name,
                        f"{rho_all:.4f}",
                        f"{rho_bg:.4f}",
                        f"{rho_wall:.4f}",
                        f"{rho_sprite:.4f}",
                        f"{rho_diff:.4f}",
                    ]
                )
        paths["table1_csv"] = t1_csv

        # 2. Export Table 2: Grassmann Geodesic Geometry Across Regimes
        t2_json = os.path.join(output_dir, "paper_table2_geometry.json")
        t2_csv = os.path.join(output_dir, "paper_table2_geometry.csv")

        with open(t2_json, "w") as f:
            json.dump(phase2_data, f, indent=2)
        paths["table2_json"] = t2_json

        with open(t2_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Encoder Regime",
                    "k=5 Geodesic Distance",
                    "k=10 Geodesic Distance",
                    "k=20 Geodesic Distance",
                ]
            )
            for enc_name, r_data in phase2_data.items():
                d5 = r_data.get("k=5", {}).get("geodesic_distance", 0.0)
                d10 = r_data.get("k=10", {}).get("geodesic_distance", 0.0)
                d20 = r_data.get("k=20", {}).get("geodesic_distance", 0.0)
                writer.writerow([enc_name, f"{d5:.4f}", f"{d10:.4f}", f"{d20:.4f}"])
        paths["table2_csv"] = t2_csv

        # 3. Export Table 3: Active TTSS Floor-Collapse Rate & Latency
        t3_json = os.path.join(output_dir, "paper_table3_ttss.json")
        t3_csv = os.path.join(output_dir, "paper_table3_ttss.csv")

        with open(t3_json, "w") as f:
            json.dump(phase3_data, f, indent=2)
        paths["table3_json"] = t3_json

        with open(t3_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Adaptation Mechanism",
                    "Mean Tracking Error (px)",
                    "Floor Collapse Rate (%)",
                    "Acceptance Rate (%)",
                    "Latency (ms/step)",
                ]
            )
            b_data = phase3_data.get("unadapted_baseline", {})
            n_data = phase3_data.get("naive_ttt", {})
            s_data = phase3_data.get("safeguarded_ttss", {})

            writer.writerow(
                [
                    "Un-adapted Baseline",
                    f"{b_data.get('mean_tracking_error_px', 0.0):.2f}",
                    f"{b_data.get('floor_collapse_rate', 0.0)*100:.1f}%",
                    "N/A",
                    "0.00",
                ]
            )
            writer.writerow(
                [
                    "Naive TTT (Un-safeguarded)",
                    f"{n_data.get('mean_tracking_error_px', 0.0):.2f}",
                    f"{n_data.get('floor_collapse_rate', 0.0)*100:.1f}%",
                    f"{n_data.get('trust_region_acceptance_rate', 1.0)*100:.1f}%",
                    f"{n_data.get('avg_latency_ms', 0.0):.2f}",
                ]
            )
            writer.writerow(
                [
                    "Safeguarded TTSS (Trust-Region)",
                    f"{s_data.get('mean_tracking_error_px', 0.0):.2f}",
                    f"{s_data.get('floor_collapse_rate', 0.0)*100:.1f}%",
                    f"{s_data.get('trust_region_acceptance_rate', 0.0)*100:.1f}%",
                    f"{s_data.get('avg_latency_ms', 0.0):.2f}",
                ]
            )
        paths["table3_csv"] = t3_csv

        # 4. Combined Master Summary JSON
        master_json = os.path.join(output_dir, "paper_headline_results_combined.json")
        master_summary = {
            "table1_correlation": phase1_data,
            "table2_geometry": phase2_data,
            "table3_ttss": phase3_data,
        }
        with open(master_json, "w") as f:
            json.dump(master_summary, f, indent=2)
        paths["master_json"] = master_json

        return paths
