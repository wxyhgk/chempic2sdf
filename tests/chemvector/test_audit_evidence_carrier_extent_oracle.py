from __future__ import annotations

import unittest

import numpy as np
import torch

from chemvector.audit_evidence_carrier_extent_oracle import (
    clip_infinite_line_to_image,
    reconstruct_continuous_target_segments,
    tolerance_key,
    unordered_endpoint_error_q,
)


class EvidenceCarrierExtentOracleTest(unittest.TestCase):
    def test_clips_horizontal_and_negative_slope_lines(self) -> None:
        horizontal = clip_infinite_line_to_image(
            np.asarray((5.0, 4.0)),
            np.asarray((1.0, 0.0)),
            width=11,
            height=9,
        )
        np.testing.assert_allclose(horizontal[0], (0.0, 4.0))
        np.testing.assert_allclose(horizontal[1], (10.0, 4.0))
        diagonal = clip_infinite_line_to_image(
            np.asarray((5.0, 4.0)),
            np.asarray((1.0, -1.0)),
            width=11,
            height=9,
        )
        self.assertIsNotNone(diagonal)
        for point in diagonal:
            self.assertTrue(0.0 <= point[0] <= 10.0)
            self.assertTrue(0.0 <= point[1] <= 8.0)

    def test_reconstructs_offset_and_extent_in_pixel_center_coordinates(self) -> None:
        target = {
            "carrier_segments": torch.tensor([[[0.0, 0.5], [1.0, 0.5]]]),
            "carrier_mask": torch.tensor([True]),
            "geometry_loss_mask": torch.tensor([True]),
            "normal_offset_pixels": torch.tensor([1.0]),
            "extent_left_fraction": torch.tensor([0.2]),
            "extent_right_fraction": torch.tensor([0.8]),
        }
        reconstructed = reconstruct_continuous_target_segments(
            target,
            width=11,
            height=11,
        )
        np.testing.assert_allclose(
            reconstructed[0],
            np.asarray(((2.0, 6.0), (8.0, 6.0))),
            atol=1e-6,
        )

    def test_endpoint_error_is_order_invariant(self) -> None:
        predicted = np.asarray(((2.0, 3.0), (8.0, 3.0)))
        expected = (
            (8.0 / 10.0 * 2047.0, 3.0 / 10.0 * 2047.0),
            (2.0 / 10.0 * 2047.0, 3.0 / 10.0 * 2047.0),
        )
        self.assertAlmostEqual(
            unordered_endpoint_error_q(
                predicted,
                expected,
                width=11,
                height=11,
            ),
            0.0,
            places=5,
        )

    def test_tolerance_keys_do_not_collapse_fractional_values(self) -> None:
        self.assertEqual(tolerance_key(8.1), "8.1q")
        self.assertEqual(tolerance_key(8.9), "8.9q")


if __name__ == "__main__":
    unittest.main()
