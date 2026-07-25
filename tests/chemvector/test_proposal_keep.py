"""Tests for one-to-one dense proposal filtering."""

from __future__ import annotations

import unittest

import torch

from chemvector.proposal_keep import (
    ProposalKeepConfig,
    ProposalKeepHead,
    decode_vertices_with_proposal_keep,
    proposal_keep_loss,
    scale_vertex_batch,
)


class ProposalKeepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProposalKeepConfig(
            input_channels=8,
            hidden_dim=4,
            maximum_proposals=8,
            proposal_threshold=0.1,
            matching_radius_pixels=8.0,
        )

    def test_head_emits_one_logit_per_dense_cell(self) -> None:
        head = ProposalKeepHead(self.config)
        features = torch.randn(2, 8, 16, 16)
        dense = {
            "vertex_heatmap_logits": torch.randn(2, 1, 16, 16),
            "vertex_offset_logits": torch.randn(2, 2, 16, 16),
            "vertex_aromatic_logits": torch.randn(2, 1, 16, 16),
        }

        self.assertEqual(head(features, dense).shape, (2, 1, 16, 16))

    def test_one_to_one_loss_marks_only_one_duplicate_positive(self) -> None:
        heatmap = torch.full((1, 1, 16, 16), -20.0)
        heatmap[0, 0, 8, 8] = 10.0
        heatmap[0, 0, 8, 10] = 9.0
        dense = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 16, 16),
            "vertex_aromatic_logits": torch.zeros(1, 1, 16, 16),
        }
        anchors = torch.tensor([[[8.5 / 16, 8.5 / 16]]])
        mask = torch.tensor([[True]])

        metrics = proposal_keep_loss(
            torch.zeros(1, 1, 16, 16),
            dense,
            anchors,
            mask,
            self.config,
            image_size=64,
        )

        self.assertEqual(metrics["proposal_count"], 2.0)
        self.assertEqual(metrics["positive_count"], 1.0)
        self.assertEqual(metrics["proposal_recall"], 1.0)
        self.assertTrue(torch.isfinite(metrics["loss"]))

    def test_decode_uses_keep_score_instead_of_dense_threshold(self) -> None:
        heatmap = torch.full((1, 1, 8, 8), -20.0)
        heatmap[0, 0, 3, 3] = 10.0
        heatmap[0, 0, 3, 5] = 9.0
        dense = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }
        keep = torch.full((1, 1, 8, 8), -10.0)
        keep[0, 0, 3, 5] = 10.0

        decoded = decode_vertices_with_proposal_keep(
            dense,
            keep,
            self.config,
            keep_threshold=0.5,
        )

        self.assertEqual(len(decoded[0]), 1)
        self.assertAlmostEqual(decoded[0][0]["x"], 5.5 / 8)

    def test_scale_augmentation_transforms_pixels_and_anchors_together(self) -> None:
        pixels = torch.ones(1, 1, 8, 8)
        pixels[0, 0, 2:6, 2:6] = -1.0
        anchors = torch.tensor([[[0.25, 0.75]]])

        scaled_pixels, scaled_anchors, scales = scale_vertex_batch(
            pixels,
            anchors,
            minimum_scale=0.5,
            maximum_scale=0.5,
            scales=torch.tensor([0.5]),
        )

        self.assertEqual(scales.tolist(), [0.5])
        self.assertTrue(torch.allclose(scaled_anchors, torch.tensor([[[0.375, 0.625]]])))
        self.assertTrue(torch.equal(scaled_pixels[0, 0, 3:5, 3:5], -torch.ones(2, 2)))
        self.assertEqual(float(scaled_pixels[0, 0, 0, 0]), 1.0)

    def test_cardinality_weight_changes_total_loss(self) -> None:
        heatmap = torch.full((1, 1, 8, 8), -20.0)
        heatmap[0, 0, 3, 3] = 10.0
        heatmap[0, 0, 5, 5] = 9.0
        dense = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }
        anchors = torch.tensor([[[3.5 / 8, 3.5 / 8]]])
        mask = torch.tensor([[True]])
        keep_logits = torch.full((1, 1, 8, 8), 2.0)

        base = proposal_keep_loss(
            keep_logits,
            dense,
            anchors,
            mask,
            self.config,
            image_size=64,
        )
        weighted = proposal_keep_loss(
            keep_logits,
            dense,
            anchors,
            mask,
            self.config,
            image_size=64,
            cardinality_loss_weight=2.0,
        )

        self.assertGreater(float(weighted["loss"]), float(base["loss"]))
        self.assertGreater(float(weighted["cardinality_loss"]), 0.0)

    def test_minimum_dense_fraction_restores_best_rejected_candidates(self) -> None:
        heatmap = torch.full((1, 1, 8, 8), -20.0)
        heatmap[0, 0, 2, 2] = 10.0
        heatmap[0, 0, 4, 4] = 9.0
        heatmap[0, 0, 6, 6] = 8.0
        dense = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }
        keep = torch.full((1, 1, 8, 8), -10.0)
        keep[0, 0, 2, 2] = 10.0
        keep[0, 0, 4, 4] = 0.0
        keep[0, 0, 6, 6] = -1.0

        unbounded = decode_vertices_with_proposal_keep(
            dense,
            keep,
            self.config,
            keep_threshold=0.8,
        )[0]
        bounded = decode_vertices_with_proposal_keep(
            dense,
            keep,
            self.config,
            keep_threshold=0.8,
            minimum_dense_fraction=2 / 3,
        )[0]

        self.assertEqual(len(unbounded), 1)
        self.assertEqual(len(bounded), 2)
        self.assertAlmostEqual(bounded[1]["x"], 4.5 / 8)


if __name__ == "__main__":
    unittest.main()
