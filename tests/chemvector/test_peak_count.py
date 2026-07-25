"""Tests for dense peak count calibration."""

from __future__ import annotations

import unittest

import torch

from chemvector.peak_count import (
    PeakCountCalibrator,
    PeakCountConfig,
    decode_vertices_by_count,
    peak_count_features,
    peak_count_loss,
    predict_peak_counts,
)


class PeakCountTests(unittest.TestCase):
    def test_features_loss_and_decode_have_stable_shapes(self) -> None:
        config = PeakCountConfig(maximum_vertices=8, hidden_dim=16)
        output = {
            "vertex_heatmap_logits": torch.randn(2, 1, 8, 8),
            "vertex_offset_logits": torch.randn(2, 2, 8, 8),
            "vertex_aromatic_logits": torch.randn(2, 1, 8, 8),
        }
        features = peak_count_features(output, config)
        logits = PeakCountCalibrator(config)(features)
        mask = torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
        )
        loss = peak_count_loss(logits, mask)
        decoded = decode_vertices_by_count(
            output,
            torch.tensor([2, 3]),
            maximum_vertices=8,
        )

        self.assertEqual(features.shape, (2, 17))
        self.assertEqual(logits.shape, (2, 9))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([len(items) for items in decoded], [2, 3])

    def test_regression_objective_rounds_and_clamps_counts(self) -> None:
        config = PeakCountConfig(
            maximum_vertices=8,
            hidden_dim=16,
            objective="regression",
        )
        calibrator = PeakCountCalibrator(config)
        output = calibrator(torch.randn(2, 17))
        mask = torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
        )
        loss = peak_count_loss(output, mask)
        counts = predict_peak_counts(torch.tensor([[-2.0], [9.2]]), 8)

        self.assertEqual(output.shape, (2, 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(counts.tolist(), [0, 8])


if __name__ == "__main__":
    unittest.main()
