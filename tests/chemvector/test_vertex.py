"""Tests for dense visual vertex detection."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import torch
from PIL import Image

from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.pretrain_vertex import set_vertex_trainable, vertex_metric_key
from chemvector.vertex import (
    build_vertex_heatmap,
    build_vertex_role_heatmap,
    decode_vertices,
    VertexCollator,
    vertex_detection_loss,
    vertex_teacher_consistency_loss,
)


class VertexDetectionTests(unittest.TestCase):
    def test_collator_uses_actual_rendered_visibility_for_roles(self) -> None:
        sample = {
            "key": "hidden-ch",
            "image": Image.new("RGB", (8, 8), "white"),
            "render": {
                "width": 8,
                "height": 8,
                "atoms": [{"atom_id": 0, "image_x": 4.0, "image_y": 4.0}],
            },
            "chemmol": {
                "atoms": [
                    {
                        "element": "C",
                        "aromatic": False,
                        "formal_charge": 0,
                        "implicit_h_count": 1,
                        "no_implicit": True,
                    }
                ]
            },
            "layers": {
                "atoms": [
                    {"atom_id": 0, "visible_text": False, "text": ""}
                ]
            },
        }

        batch = VertexCollator(image_size=8)([sample])

        self.assertEqual(batch["role_targets"].tolist(), [[0]])

    def test_checkpoint_metric_prioritizes_whole_image_exactness(self) -> None:
        lower_exact = {
            "vertex_exact_rate": 0.1,
            "count_exact_rate": 0.9,
            "f1": 0.99,
            "count_mae": 0.1,
        }
        higher_exact = dict(lower_exact, vertex_exact_rate=0.2, f1=0.8)

        self.assertGreater(
            vertex_metric_key(higher_exact),
            vertex_metric_key(lower_exact),
        )

    def test_role_head_only_freezes_existing_vertex_outputs(self) -> None:
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

        set_vertex_trainable(model, role_head_only=True)
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("vertex_head.role_heatmap.") for name in trainable)
        )

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

    def test_role_heatmap_separates_implicit_and_visible_vertices(self) -> None:
        anchors = torch.tensor([[[0.25, 0.5], [0.75, 0.5]]])
        mask = torch.tensor([[True, True]])
        roles = torch.tensor([[0, 1]])

        target = build_vertex_role_heatmap(anchors, mask, roles, 8, 8)

        self.assertEqual(int(target[:, 0].sum()), 1)
        self.assertEqual(int(target[:, 1].sum()), 1)

    def test_gaussian_heatmap_softens_neighboring_cells(self) -> None:
        anchors = torch.tensor([[[0.5, 0.5]]])
        mask = torch.tensor([[True]])

        target = build_vertex_heatmap(
            anchors,
            mask,
            8,
            8,
            radius_cells=2,
        )

        self.assertEqual(float(target[0, 0, 4, 4]), 1.0)
        self.assertGreater(float(target[0, 0, 4, 5]), 0.0)
        self.assertLess(float(target[0, 0, 4, 5]), 1.0)

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
            "vertex_role_heatmap_logits": torch.cat((heatmap, heatmap - 1), dim=1),
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }
        decoded = decode_vertices(output, threshold=0.3)
        self.assertEqual(len(decoded[0]), 1)
        self.assertAlmostEqual(decoded[0][0]["x"], 4.5 / 8)
        self.assertAlmostEqual(decoded[0][0]["y"], 3.5 / 8)

        role_decoded = decode_vertices(output, threshold=0.3, role_aware=True)
        self.assertEqual(role_decoded[0][0]["role"], "implicit_junction")

    def test_decoder_supports_wider_nms(self) -> None:
        heatmap = torch.full((1, 1, 16, 16), -20.0)
        heatmap[0, 0, 5, 5] = 10.0
        heatmap[0, 0, 5, 9] = 9.0
        output = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 16, 16),
            "vertex_aromatic_logits": torch.zeros(1, 1, 16, 16),
        }

        narrow = decode_vertices(output, threshold=0.5, nms_kernel_size=3)[0]
        wide = decode_vertices(output, threshold=0.5, nms_kernel_size=9)[0]

        self.assertEqual(len(narrow), 2)
        self.assertEqual(len(wide), 1)

    def test_decoder_rejects_even_nms_kernel(self) -> None:
        output = {
            "vertex_heatmap_logits": torch.zeros(1, 1, 4, 4),
            "vertex_offset_logits": torch.zeros(1, 2, 4, 4),
            "vertex_aromatic_logits": torch.zeros(1, 1, 4, 4),
        }

        with self.assertRaisesRegex(ValueError, "positive odd"):
            decode_vertices(output, nms_kernel_size=4)

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
