"""
Unit tests for Phase 4: Unified Evaluation Engine & Writeup Artifact Export.
"""

import os
import shutil
import tempfile
import unittest

from evaluation import UnifiedEvaluationEngine


class TestPhase4Eval(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_unified_export_headline_results(self):
        phase1_dummy = {
            "Classical d_rep": {
                "overall_rho": -0.1245,
                "axis_breakdown": {
                    "background": {"spearman_rho": -0.0812},
                    "wall_texture": {"spearman_rho": -0.1105},
                    "sprite": {"spearman_rho": -0.1420},
                    "difficulty": {"spearman_rho": -0.1643},
                },
            },
            "SegmentationWeighter": {
                "overall_rho": -0.7854,
                "axis_breakdown": {
                    "background": {"spearman_rho": -0.8120},
                    "wall_texture": {"spearman_rho": -0.7640},
                    "sprite": {"spearman_rho": -0.7910},
                    "difficulty": {"spearman_rho": -0.7745},
                },
            },
        }

        phase2_dummy = {
            "Frozen ResNet-18": {
                "k=5": {"geodesic_distance": 0.4812},
                "k=10": {"geodesic_distance": 0.7105},
                "k=20": {"geodesic_distance": 1.0824},
            }
        }

        phase3_dummy = {
            "unadapted_baseline": {
                "mean_tracking_error_px": 162.40,
                "floor_collapse_rate": 1.0,
            },
            "naive_ttt": {
                "mean_tracking_error_px": 184.10,
                "floor_collapse_rate": 1.0,
            },
            "safeguarded_ttss": {
                "mean_tracking_error_px": 48.20,
                "floor_collapse_rate": 0.15,
                "trust_region_acceptance_rate": 0.842,
                "avg_latency_ms": 2.48,
            },
        }

        paths = UnifiedEvaluationEngine.export_headline_results(
            phase1_dummy, phase2_dummy, phase3_dummy, output_dir=self.tmp_dir
        )

        for key, filepath in paths.items():
            self.assertTrue(os.path.exists(filepath), f"File {filepath} was not created!")
            self.assertGreater(os.path.getsize(filepath), 0)


if __name__ == "__main__":
    unittest.main()
