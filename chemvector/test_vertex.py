"""Tests for dense visual vertex detection."""

from __future__ import annotations

import unittest

import torch

from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.vertex import build_vertex_heatmap, decode_vertices, vertex_detection_loss


class VertexDetectionTests(unittest.TestCase):
    def test_targets_and_offsets_are_finite(self) -> None:
        anchors = torch.tensor([[[0.25, 0.5], [0.75, 0.5]]])
        mask = torch.tensor([[True, True]])
        aromatic = torch.tensor([[0.0, 1.0]])
        output = {
            "vertex_heatmap_logits": torch.zeros(1, 1, 16, 16),
            "vertex_offset_logits": torch.zeros(1, 2, 16, 16),
            "vertex_aromatic_logits": torch.zeros(1, 1, 16, 16),
        }
        target = build_vertex_heatmap(anchors, mask, 16, 16)
        losses = vertex_detection_loss(output, anchors, mask, aromatic)
        self.assertEqual(int(target.sum()), 2)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_decoder_uses_nms_and_subcell_offsets(self) -> None:
        heatmap = torch.full((1, 1, 8, 8), -10.0)
        heatmap[0, 0, 3, 4] = 10.0
        output = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }
        decoded = decode_vertices(output, threshold=0.3)
        self.assertEqual(len(decoded[0]), 1)
        self.assertAlmostEqual(decoded[0][0]["x"], 4.5 / 8)
        self.assertAlmostEqual(decoded[0][0]["y"], 3.5 / 8)

    def test_model_vertex_head_matches_stride_four_feature_map(self) -> None:
        config = ChemVectorConfig(
            vocab_size=64,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            max_target_length=32,
            vision_dims=(16, 32, 48, 64),
            vision_depths=(1, 1, 1, 1),
            vertex_enabled=True,
            vertex_feature_dim=16,
        )
        model = ChemVectorModel(config)
        features = model.vision_encoder.forward_stroke_resolution(
            torch.randn(2, 3, 128, 128)
        )
        output = model.vertex_head(features)
        self.assertEqual(output["vertex_heatmap_logits"].shape, (2, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
