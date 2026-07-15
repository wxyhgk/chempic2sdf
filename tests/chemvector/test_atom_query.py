"""Tests for the shadow-mode Atom Query detector."""

from __future__ import annotations

import unittest

import torch

from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    atom_query_loss,
    decode_atom_queries,
    snap_queries_to_dense_candidates,
)


class AtomQueryTests(unittest.TestCase):
    def test_forward_loss_and_decode_are_finite(self) -> None:
        config = AtomQueryConfig(
            input_channels=16,
            d_model=32,
            queries=8,
            layers=1,
            heads=4,
            ffn_dim=64,
        )
        detector = AtomQueryDetector(config)
        output = detector(torch.randn(2, 16, 8, 8))
        anchors = torch.tensor(
            [
                [[0.2, 0.3], [0.8, 0.7]],
                [[0.4, 0.5], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, True], [True, False]])
        aromatic = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

        losses = atom_query_loss(output, anchors, mask, aromatic)
        decoded = decode_atom_queries(output, threshold=0.0)

        self.assertEqual(output["centers"].shape, (2, 8, 2))
        self.assertTrue(
            torch.allclose(output["centers"][0], detector.reference_points, atol=1e-6)
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertEqual([len(items) for items in decoded], [8, 8])

    def test_invalid_head_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AtomQueryConfig(d_model=190, heads=6)

    def test_dense_candidates_snap_one_to_one(self) -> None:
        queries = [
            {"x": 0.1, "y": 0.1, "score": 1.0, "aromatic": False},
            {"x": 0.9, "y": 0.9, "score": 1.0, "aromatic": False},
        ]
        candidates = [
            {"x": 0.12, "y": 0.1, "score": 0.8, "aromatic": False},
            {"x": 0.88, "y": 0.9, "score": 0.9, "aromatic": False},
        ]

        snapped = snap_queries_to_dense_candidates(
            queries,
            candidates,
            image_size=100,
            maximum_distance_pixels=5,
        )

        self.assertEqual(snapped[0]["x"], 0.12)
        self.assertEqual(snapped[1]["x"], 0.88)


if __name__ == "__main__":
    unittest.main()
