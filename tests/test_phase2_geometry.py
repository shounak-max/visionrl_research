"""
Unit tests for Phase 2 Geodesic Geometry module.
"""

import unittest
import torch
import numpy as np

from geometry import GrassmannMapper, GeodesicDistance, GrassmannPoint


class TestPhase2Geometry(unittest.TestCase):

    def test_svd_subspace_extraction_and_determinism(self):
        act1 = torch.randn(50, 512, generator=torch.manual_seed(42))
        p1_a = GrassmannMapper.to_subspace(act1, subspace_dim=10)
        p1_b = GrassmannMapper.to_subspace(act1, subspace_dim=10)

        self.assertEqual(p1_a.subspace_dim, 10)
        self.assertEqual(p1_a.feature_dim, 512)

        # Verify determinism
        np.testing.assert_array_almost_equal(
            p1_a.to_numpy(), p1_b.to_numpy(), decimal=5
        )

        # Orthonormal basis check Uk^T Uk = I_k
        gram = torch.matmul(p1_a.subspace.T, p1_a.subspace)
        np.testing.assert_array_almost_equal(
            gram.detach().numpy(), np.eye(10), decimal=4
        )

    def test_geodesic_distance_invariants(self):
        """Verifies zero self-distance and metric symmetry invariants."""
        act1 = torch.randn(50, 512, generator=torch.manual_seed(42))
        act2 = torch.randn(50, 512, generator=torch.manual_seed(123))

        p1 = GrassmannMapper.to_subspace(act1, subspace_dim=10)
        p2 = GrassmannMapper.to_subspace(act2, subspace_dim=10)

        # 1. Zero self-distance invariant d(p1, p1) == 0.0
        d_self_geo = GeodesicDistance.compute(p1, p1, metric_type="geodesic")
        d_self_proj = GeodesicDistance.compute(p1, p1, metric_type="projection")

        self.assertAlmostEqual(d_self_geo, 0.0, places=4)
        self.assertAlmostEqual(d_self_proj, 0.0, places=4)

        # 2. Symmetry invariant d(p1, p2) == d(p2, p1)
        d12 = GeodesicDistance.compute(p1, p2, metric_type="geodesic")
        d21 = GeodesicDistance.compute(p2, p1, metric_type="geodesic")

        self.assertAlmostEqual(d12, d21, places=5)
        self.assertGreater(d12, 0.0)

    def test_subspace_dimension_sweep(self):
        """Sweeps subspace dimensions k in {5, 10, 20}."""
        act1 = torch.randn(50, 512, generator=torch.manual_seed(42))
        act2 = torch.randn(50, 512, generator=torch.manual_seed(100))

        for k in [5, 10, 20]:
            pk1 = GrassmannMapper.to_subspace(act1, subspace_dim=k)
            pk2 = GrassmannMapper.to_subspace(act2, subspace_dim=k)

            self.assertEqual(pk1.subspace_dim, k)
            dist = GeodesicDistance.compute(pk1, pk2, metric_type="geodesic")
            self.assertGreater(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
