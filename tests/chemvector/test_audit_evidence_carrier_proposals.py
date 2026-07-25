from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from chemvector.audit_evidence_carrier_proposals import (
    TargetLine,
    dense_local_expansion,
    detect_lsd_lines,
    line_direction_support_matrix,
    line_distance_matrix,
    one_to_one_matches,
    proposals_pixels_to_q,
    translate_grayscale,
)


class EvidenceCarrierProposalAuditTest(unittest.TestCase):
    def test_line_distance_is_endpoint_order_invariant(self) -> None:
        proposals = np.asarray([[[10.0, 20.0], [40.0, 20.0]]])
        targets = [
            TargetLine(0, "single_line", ((40.0, 20.0), (10.0, 20.0)), False, False)
        ]
        self.assertAlmostEqual(float(line_distance_matrix(proposals, targets)[0, 0]), 0.0)

    def test_dense_expansion_keeps_original_and_moves_endpoints(self) -> None:
        raw = np.asarray([[[10.0, 20.0], [40.0, 20.0]]])
        dense = dense_local_expansion(raw)
        cost = line_distance_matrix(dense, raw)
        self.assertEqual(len(one_to_one_matches(cost, 0.0)), 1)
        self.assertTrue((dense[:, 0, 0] < 10.0).any())
        self.assertTrue((dense[:, 1, 0] > 40.0).any())

    def test_matching_maximizes_valid_edges_before_distance(self) -> None:
        cost = np.asarray(
            (
                (0.0, 0.9, 100.0),
                (0.9, 1.1, 100.0),
            )
        )
        matches = one_to_one_matches(cost, 1.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(set(matches), {0, 1})

    def test_direction_support_allows_incomplete_collinear_segment(self) -> None:
        proposals = np.asarray([[[30.0, 50.5], [55.0, 50.5]]])
        targets = [
            TargetLine(
                0,
                "single_line",
                ((10.0 / 100.0 * 2047.0, 50.0 / 100.0 * 2047.0),
                 (90.0 / 100.0 * 2047.0, 50.0 / 100.0 * 2047.0)),
                True,
                False,
            )
        ]
        support = line_direction_support_matrix(
            proposals,
            targets,
            width=100,
            height=100,
            maximum_angle_degrees=5.0,
            maximum_perpendicular_pixels=1.0,
            minimum_overlap_pixels=8.0,
        )
        self.assertTrue(bool(support[0, 0]))

    def test_lsd_detects_a_simple_bond_in_both_raster_polarities(self) -> None:
        combined = Image.new("L", (96, 96), 255)
        ImageDraw.Draw(combined).line((12, 48, 84, 48), fill=0, width=2)
        bond = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(bond).line((12, 48, 84, 48), fill=255, width=2)
        self.assertGreater(
            len(detect_lsd_lines(combined, oracle_bond_layer=False, minimum_length_pixels=1.0)),
            0,
        )
        self.assertGreater(
            len(detect_lsd_lines(bond, oracle_bond_layer=True, minimum_length_pixels=1.0)),
            0,
        )

    def test_translation_uses_mode_specific_background(self) -> None:
        combined = Image.new("L", (8, 8), 255)
        combined.putpixel((1, 1), 0)
        shifted = np.asarray(
            translate_grayscale(combined, dx=2, dy=3, oracle_bond_layer=False)
        )
        self.assertEqual(int(shifted[4, 3]), 0)
        self.assertEqual(int(shifted[0, 0]), 255)

        bond = Image.new("L", (8, 8), 0)
        bond.putpixel((1, 1), 255)
        shifted_bond = np.asarray(
            translate_grayscale(bond, dx=2, dy=3, oracle_bond_layer=True)
        )
        self.assertEqual(int(shifted_bond[4, 3]), 255)
        self.assertEqual(int(shifted_bond[0, 0]), 0)

    def test_pixel_to_q_scaling(self) -> None:
        proposals = np.asarray([[[0.0, 0.0], [100.0, 200.0]]])
        scaled = proposals_pixels_to_q(proposals, 100, 200)
        np.testing.assert_allclose(scaled[0, 1], (2047.0, 2047.0))


if __name__ == "__main__":
    unittest.main()
