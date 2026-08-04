"""Tests for exact local cardinality-curve refinement."""

import unittest

import torch

from chemvector.components.atoms.objects.g3_cardinality_curve import (
    G3CardinalityCurveConfig,
    G3CardinalityCurveRefiner,
    g3_cardinality_curve_features,
    g3_cardinality_offset_target,
)


class G3CardinalityCurveTests(unittest.TestCase):
    def test_curve_shape_and_exact_assignment_scores(self) -> None:
        config = G3CardinalityCurveConfig(radius=1, point_dim=13, max_count=3)
        base = torch.tensor((
            (3.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 1.0),
        ))
        expert = 2.0 * base
        logits = torch.tensor((0.0, 0.0, 4.0, 0.0))
        curve, counts, coarse = g3_cardinality_curve_features(
            base, expert, logits, config=config
        )
        self.assertEqual(coarse, 2)
        self.assertEqual(counts.tolist(), [1, 2, 3])
        self.assertEqual(tuple(curve.shape), (3, 13))
        self.assertAlmostEqual(float(curve[0, 3]), 3.0 / 3.0)
        self.assertAlmostEqual(float(curve[1, 3]), 5.0 / 3.0)
        self.assertAlmostEqual(float(curve[1, 9]), 2.0)
        self.assertAlmostEqual(float(curve[1, 10]), 4.0)

    def test_boundary_marks_out_of_capacity_points_invalid(self) -> None:
        config = G3CardinalityCurveConfig(radius=1, point_dim=13, max_count=3)
        scores = torch.eye(3)
        logits = torch.tensor((5.0, 0.0, 0.0, 0.0))
        curve, counts, coarse = g3_cardinality_curve_features(
            scores, scores, logits, config=config
        )
        self.assertEqual(coarse, 0)
        self.assertEqual(counts.tolist(), [-1, 0, 1])
        self.assertEqual(float(curve[0, 2]), 0.0)
        self.assertEqual(float(curve[1, 2]), 1.0)

    def test_target_is_clipped_but_window_membership_is_reported(self) -> None:
        self.assertEqual(g3_cardinality_offset_target(13, 10), (7, True))
        self.assertEqual(g3_cardinality_offset_target(18, 10), (8, False))
        self.assertEqual(g3_cardinality_offset_target(2, 10), (0, False))

    def test_refiner_shape(self) -> None:
        model = G3CardinalityCurveRefiner()
        logits = model(torch.randn(2, 9, 13))
        self.assertEqual(tuple(logits.shape), (2, 9))


if __name__ == "__main__":
    unittest.main()
