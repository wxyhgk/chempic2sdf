"""Tests for dense visual vertex detection."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import torch

from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.pretrain_vertex import set_vertex_trainable
from chemvector.vertex import (
    build_vertex_heatmap,
    decode_vertices,
    vertex_detection_loss,
    vertex_teacher_consistency_loss,
)


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

    def test_extreme_bfloat16_heatmap_logits_are_finite(self) -> None:
        anchors = torch.tensor([[[0.25, 0.5]]])
        mask = torch.tensor([[True]])
        aromatic = torch.tensor([[0.0]])
        output = {
            "vertex_heatmap_logits": torch.full(
                (1, 1, 16, 16),
                100.0,
                dtype=torch.bfloat16,
            ),
            "vertex_offset_logits": torch.zeros(1, 2, 16, 16),
            "vertex_aromatic_logits": torch.zeros(1, 1, 16, 16),
        }
        losses = vertex_detection_loss(output, anchors, mask, aromatic)
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

    def test_zero_initialized_adapter_preserves_vertex_outputs(self) -> None:
        base_config = ChemVectorConfig(
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
        base = ChemVectorModel(base_config).eval()
        adapted = ChemVectorModel(
            ChemVectorConfig(
                **{
                    **asdict(base_config),
                    "structure_residual_adapter": True,
                    "structure_adapter_dim": 8,
                }
            )
        ).eval()
        adapted.load_state_dict(base.state_dict(), strict=False)
        features = torch.randn(2, 16, 24, 24)
        expected = base.vertex_head(features)
        actual = adapted.vertex_head(features)
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]))

        trainable = set_vertex_trainable(adapted, adapter_only=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in trainable),
            sum(
                parameter.numel()
                for parameter in adapted.vertex_head.output_adapter.parameters()
            ),
        )

    def test_teacher_consistency_is_zero_only_for_identical_outputs(self) -> None:
        teacher = {
            "vertex_heatmap_logits": torch.randn(2, 1, 8, 8),
            "vertex_offset_logits": torch.randn(2, 2, 8, 8),
            "vertex_aromatic_logits": torch.randn(2, 1, 8, 8),
        }
        identical = vertex_teacher_consistency_loss(teacher, teacher)
        self.assertEqual(float(identical["teacher_loss"]), 0.0)
        changed = {key: value.clone() for key, value in teacher.items()}
        changed["vertex_heatmap_logits"][0, 0, 2, 3] += 2.0
        changed_loss = vertex_teacher_consistency_loss(changed, teacher)
        self.assertGreater(float(changed_loss["teacher_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
