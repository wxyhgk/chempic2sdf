"""Tests for the lightweight G3 AtomCenterField."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    ContinuousStride2AtomCenterField,
    G3AtomCenterField,
    G3AtomCenterFieldHighResolutionResidual,
    G3AtomCenterFieldLocalResidual,
    VirtualStride2AtomCenterField,
    VirtualStride2AtomCenterResidual,
    atom_center_field_loss,
    atom_center_peak_margin_loss,
    build_atom_center_field_targets,
    decode_atom_center_field,
    warmstart_atom_center_field_from_p1a,
)


class G3AtomCenterFieldTests(unittest.TestCase):
    @staticmethod
    def _p1a_head() -> SimpleNamespace:
        return SimpleNamespace(
            presence=nn.Conv2d(256, 3, 1),
            anchor_offset=nn.Conv2d(256, 6, 1),
        )

    def test_forward_is_lightweight_and_detaches_frozen_hidden(self) -> None:
        field = G3AtomCenterField()
        hidden = torch.randn(2, 256, 7, 9, requires_grad=True)

        output = field(hidden)
        output["center_logits"].sum().backward()

        self.assertEqual(output["center_logits"].shape, (2, 1, 7, 9))
        self.assertEqual(output["offset_logits"].shape, (2, 2, 7, 9))
        self.assertEqual(field.parameter_count(), 771)
        self.assertIsNone(hidden.grad)
        self.assertIsNotNone(field.center.weight.grad)
        self.assertFalse(hasattr(field, "count"))

    def test_warmstart_copies_p1a_slot_zero_exactly(self) -> None:
        presence = nn.Conv2d(256, 3, 1)
        anchor_offset = nn.Conv2d(256, 6, 1)
        atom_head = SimpleNamespace(
            presence=presence,
            anchor_offset=anchor_offset,
        )
        with torch.no_grad():
            presence.weight.copy_(torch.arange(presence.weight.numel()).reshape_as(presence.weight))
            presence.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
            anchor_offset.weight.copy_(
                torch.arange(anchor_offset.weight.numel()).reshape_as(anchor_offset.weight)
            )
            anchor_offset.bias.copy_(torch.arange(6, dtype=torch.float32))
        field = G3AtomCenterField()

        warmstart_atom_center_field_from_p1a(field, atom_head)

        torch.testing.assert_close(field.center.weight, presence.weight[0:1])
        torch.testing.assert_close(field.center.bias, presence.bias[0:1])
        torch.testing.assert_close(
            field.offset.weight, anchor_offset.weight.reshape(3, 2, 256, 1, 1)[0]
        )
        torch.testing.assert_close(
            field.offset.bias, anchor_offset.bias.reshape(3, 2)[0]
        )

    def test_default_target_uses_lexicographic_offset_for_collisions(self) -> None:
        anchors = torch.tensor([[[0.11, 0.21], [0.19, 0.29]]])
        mask = torch.tensor([[True, True]])

        target = build_atom_center_field_targets(
            anchors, mask, height=10, width=10
        )

        self.assertEqual(float(target["heatmap"].sum()), 1.0)
        self.assertEqual(int(target["occupancy"][0, 0, 2, 1]), 2)
        torch.testing.assert_close(
            target["offsets"][0, :, 2, 1], torch.tensor([0.1, 0.1])
        )
        self.assertEqual(float(target["sigma_cells"].sum()), 0.0)
        self.assertFalse(target["adaptive_gaussian"])

    def test_adaptive_sigma_uses_order_independent_nearest_neighbor_p75(self) -> None:
        anchors = torch.tensor(
            [[[0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.5, 0.5]]]
        )
        mask = torch.ones(1, 4, dtype=torch.bool)
        permutation = torch.tensor([3, 1, 0, 2])

        original = build_atom_center_field_targets(
            anchors,
            mask,
            height=100,
            width=100,
            adaptive_gaussian=True,
        )
        permuted = build_atom_center_field_targets(
            anchors[:, permutation],
            mask,
            height=100,
            width=100,
            adaptive_gaussian=True,
        )

        self.assertAlmostEqual(float(original["sigma_cells"][0]), 1.5, places=6)
        torch.testing.assert_close(original["sigma_cells"], permuted["sigma_cells"])
        torch.testing.assert_close(original["heatmap"], permuted["heatmap"])
        torch.testing.assert_close(original["offsets"], permuted["offsets"])

    def test_adaptive_sigma_clamps_single_and_widely_separated_atoms(self) -> None:
        single = build_atom_center_field_targets(
            torch.tensor([[[0.5, 0.5]]]),
            torch.tensor([[True]]),
            height=100,
            width=100,
            adaptive_gaussian=True,
        )
        wide = build_atom_center_field_targets(
            torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
            torch.tensor([[True, True]]),
            height=100,
            width=100,
            adaptive_gaussian=True,
        )

        self.assertEqual(float(single["sigma_cells"][0]), 1.0)
        self.assertEqual(float(wide["sigma_cells"][0]), 4.0)

    def test_loss_updates_only_field_with_default_one_hot_target(self) -> None:
        field = G3AtomCenterField()
        hidden = torch.randn(1, 256, 8, 8, requires_grad=True)
        anchors = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
        mask = torch.tensor([[True, True]])

        output = field(hidden)
        losses = atom_center_field_loss(output, anchors, mask)
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(losses["heatmap_loss"]))
        self.assertTrue(torch.isfinite(losses["offset_loss"]))
        self.assertEqual(float(losses["sigma_cells_mean"]), 0.0)
        torch.testing.assert_close(
            losses["loss"], losses["heatmap_loss"] + 8.0 * losses["offset_loss"]
        )
        self.assertIsNone(hidden.grad)
        self.assertIsNotNone(field.center.weight.grad)
        self.assertIsNotNone(field.offset.weight.grad)

    def test_combined_loss_can_enable_local_peak_margin(self) -> None:
        field = G3AtomCenterField()
        hidden = torch.randn(1, 256, 8, 8)
        anchors = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
        mask = torch.tensor([[True, True]])

        losses = atom_center_field_loss(
            field(hidden), anchors, mask, peak_margin_weight=1.0
        )

        self.assertGreater(float(losses["peak_margin_loss"].detach()), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_dense_decode_has_no_threshold_and_stable_cell_ids(self) -> None:
        output = {
            "center_logits": torch.zeros(1, 1, 2, 3),
            "offset_logits": torch.zeros(1, 2, 2, 3),
        }

        decoded = decode_atom_center_field(output)

        self.assertEqual(
            tuple(inspect.signature(decode_atom_center_field).parameters), ("output",)
        )
        self.assertEqual(decoded["scores"].shape, (1, 2, 3))
        self.assertEqual(decoded["scores"].dtype, torch.float32)
        self.assertEqual(decoded["anchors"].shape, (1, 2, 2, 3))
        self.assertEqual(decoded["anchors"].dtype, torch.float32)
        self.assertEqual(decoded["source_ids"].tolist(), [[[0, 1, 2], [3, 4, 5]]])
        torch.testing.assert_close(
            decoded["anchors"][0, :, 0, 0], torch.tensor([1.0 / 6.0, 0.25])
        )
        torch.testing.assert_close(
            decoded["anchors"][0, :, 1, 2], torch.tensor([5.0 / 6.0, 0.75])
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        field = G3AtomCenterField()
        with self.assertRaisesRegex(ValueError, r"\[batch,256,h,w\]"):
            field(torch.zeros(1, 255, 4, 4))
        with self.assertRaisesRegex(ValueError, "anchor mask differs"):
            build_atom_center_field_targets(
                torch.zeros(1, 2, 2),
                torch.ones(1, 1, dtype=torch.bool),
                height=4,
                width=4,
            )
        with self.assertRaisesRegex(ValueError, "normalized"):
            build_atom_center_field_targets(
                torch.tensor([[[1.1, 0.5]]]),
                torch.tensor([[True]]),
                height=4,
                width=4,
            )
        bad_head = SimpleNamespace(
            presence=nn.Conv2d(256, 2, 1),
            anchor_offset=nn.Conv2d(256, 4, 1),
        )
        with self.assertRaisesRegex(ValueError, "three-slot protocol"):
            warmstart_atom_center_field_from_p1a(field, bad_head)

    def test_local_residual_is_exact_slot_zero_at_step_zero(self) -> None:
        atom_head = self._p1a_head()
        field = G3AtomCenterFieldLocalResidual(atom_head)
        hidden = torch.randn(2, 256, 5, 7, requires_grad=True)

        output = field(hidden)

        self.assertEqual(field.trainable_parameter_count(), 8_643)
        self.assertTrue(torch.equal(output["center_logits"], atom_head.presence(hidden)[:, 0:1]))
        self.assertTrue(
            torch.equal(output["offset_logits"], atom_head.anchor_offset(hidden)[:, 0:2])
        )
        self.assertEqual(int(torch.count_nonzero(output["center_residual"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["offset_residual"])), 0)
        self.assertEqual(int(torch.count_nonzero(field.adapter[-1].weight)), 0)
        self.assertEqual(int(torch.count_nonzero(field.adapter[-1].bias)), 0)

    def test_local_residual_copies_frozen_buffers_and_detaches_hidden(self) -> None:
        atom_head = self._p1a_head()
        field = G3AtomCenterFieldLocalResidual(atom_head)
        parameters = dict(field.named_parameters())
        buffers = dict(field.named_buffers())
        original_buffers = {name: value.clone() for name, value in buffers.items()}

        self.assertEqual(
            set(buffers),
            {
                "base_center_weight",
                "base_center_bias",
                "base_offset_weight",
                "base_offset_bias",
            },
        )
        self.assertTrue(all(name.startswith("adapter.") for name in parameters))
        self.assertNotEqual(
            field.base_center_weight.untyped_storage().data_ptr(),
            atom_head.presence.weight.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            field.base_offset_weight.untyped_storage().data_ptr(),
            atom_head.anchor_offset.weight.untyped_storage().data_ptr(),
        )

        hidden = torch.randn(1, 256, 4, 4, requires_grad=True)
        field(hidden)["center_logits"].sum().backward()

        self.assertIsNone(hidden.grad)
        for parameter in field.parameters():
            self.assertTrue(parameter.requires_grad)
        for name, value in field.named_buffers():
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, original_buffers[name])
        self.assertIsNotNone(field.adapter[-1].weight.grad)
        self.assertEqual(
            set(field.state_dict()),
            set(buffers) | set(parameters),
        )

    def test_local_residual_rejects_invalid_p1a_shape(self) -> None:
        bad_head = SimpleNamespace(
            presence=nn.Conv2d(256, 2, 1),
            anchor_offset=nn.Conv2d(256, 6, 1),
        )
        with self.assertRaisesRegex(ValueError, "three-slot protocol"):
            G3AtomCenterFieldLocalResidual(bad_head)

    def test_high_resolution_residual_is_exact_upsampled_base_at_step_zero(self) -> None:
        atom_head = self._p1a_head()
        field = G3AtomCenterFieldHighResolutionResidual(atom_head)
        hidden = torch.randn(1, 256, 192, 192, requires_grad=True)

        output = field(hidden)
        expected_center = torch.nn.functional.interpolate(
            atom_head.presence(hidden)[:, 0:1],
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

        self.assertEqual(field.trainable_parameter_count(), 8_643)
        self.assertEqual(output["center_logits"].shape, (1, 1, 384, 384))
        self.assertEqual(output["offset_logits"].shape, (1, 2, 384, 384))
        self.assertTrue(torch.equal(output["center_logits"], expected_center))
        self.assertEqual(int(torch.count_nonzero(output["center_residual"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["offset_logits"])), 0)
        torch.testing.assert_close(
            output["offsets"], torch.full_like(output["offsets"], 0.5)
        )
        self.assertEqual(int(torch.count_nonzero(field.output_projection.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(field.output_projection.bias)), 0)

    def test_high_resolution_residual_detaches_hidden_and_freezes_base(self) -> None:
        atom_head = self._p1a_head()
        field = G3AtomCenterFieldHighResolutionResidual(atom_head)
        hidden = torch.randn(1, 256, 192, 192, requires_grad=True)
        original_weight = field.base_center_weight.clone()
        original_bias = field.base_center_bias.clone()

        output = field(hidden)
        (output["center_logits"].mean() + output["offset_logits"].mean()).backward()

        self.assertIsNone(hidden.grad)
        self.assertEqual(
            set(dict(field.named_buffers())),
            {"base_center_weight", "base_center_bias"},
        )
        self.assertTrue(
            all(name not in dict(field.named_parameters()) for name in dict(field.named_buffers()))
        )
        self.assertNotEqual(
            field.base_center_weight.untyped_storage().data_ptr(),
            atom_head.presence.weight.untyped_storage().data_ptr(),
        )
        torch.testing.assert_close(field.base_center_weight, original_weight)
        torch.testing.assert_close(field.base_center_bias, original_bias)
        self.assertIsNotNone(field.output_projection.weight.grad)

    def test_high_resolution_residual_requires_192_square_hidden(self) -> None:
        field = G3AtomCenterFieldHighResolutionResidual(self._p1a_head())
        with self.assertRaisesRegex(ValueError, r"\[batch,256,192,192\]"):
            field(torch.zeros(1, 256, 191, 192))

    def test_virtual_stride2_field_has_exact_protocol_and_decode_shape(self) -> None:
        field = VirtualStride2AtomCenterField()
        features = torch.randn(1, 96, 192, 192, requires_grad=True)

        output = field(features)
        decoded = decode_atom_center_field(output)

        self.assertEqual(field.trainable_parameter_count(), 7_628)
        self.assertEqual(output["center_logits"].shape, (1, 1, 384, 384))
        self.assertEqual(output["offset_logits"].shape, (1, 2, 384, 384))
        self.assertEqual(output["offsets"].shape, (1, 2, 384, 384))
        self.assertEqual(decoded["scores"].shape, (1, 384, 384))
        self.assertEqual(decoded["anchors"].shape, (1, 2, 384, 384))
        self.assertEqual(decoded["source_ids"].shape, (1, 384, 384))
        self.assertFalse(hasattr(field, "count"))
        self.assertFalse(hasattr(field, "slots"))

    def test_virtual_stride2_field_initializes_subpixel_channel_biases(self) -> None:
        field = VirtualStride2AtomCenterField()

        torch.testing.assert_close(
            field.subpixel_projection.bias[:4],
            torch.full((4,), -4.6),
        )
        torch.testing.assert_close(
            field.subpixel_projection.bias[4:],
            torch.zeros(8),
        )
        with torch.no_grad():
            for module in (field.input_projection, field.local, field.subpixel_projection):
                module.weight.zero_()
            field.input_projection.bias.zero_()
            field.local.bias.zero_()
        output = field(torch.zeros(1, 96, 192, 192))
        torch.testing.assert_close(
            output["center_logits"],
            torch.full_like(output["center_logits"], -4.6),
        )
        torch.testing.assert_close(
            output["offsets"], torch.full_like(output["offsets"], 0.5)
        )

    def test_virtual_stride2_field_detaches_and_validates_stroke_features(self) -> None:
        field = VirtualStride2AtomCenterField()
        features = torch.randn(1, 96, 192, 192, requires_grad=True)

        (field(features)["center_logits"].mean()).backward()

        self.assertIsNone(features.grad)
        self.assertIsNotNone(field.subpixel_projection.weight.grad)
        with self.assertRaisesRegex(ValueError, r"\[batch,96,192,192\]"):
            field(torch.zeros(1, 96, 192, 191))

    def test_virtual_stride2_residual_is_exact_upsampled_base_at_step_zero(self) -> None:
        atom_head = self._p1a_head()
        field = VirtualStride2AtomCenterResidual(atom_head)
        objects = torch.randn(1, 256, 192, 192, requires_grad=True)
        stroke = torch.randn(1, 96, 192, 192, requires_grad=True)

        output = field(objects, stroke)
        expected = torch.nn.functional.interpolate(
            atom_head.presence(objects)[:, 0:1],
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

        self.assertEqual(field.trainable_parameter_count(), 7_628)
        self.assertTrue(torch.equal(output["center_logits"], expected))
        self.assertEqual(int(torch.count_nonzero(output["center_residual"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["offset_logits"])), 0)
        self.assertEqual(int(torch.count_nonzero(field.subpixel_projection.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(field.subpixel_projection.bias)), 0)

    def test_virtual_stride2_residual_detaches_both_feature_inputs(self) -> None:
        field = VirtualStride2AtomCenterResidual(self._p1a_head())
        objects = torch.randn(1, 256, 192, 192, requires_grad=True)
        stroke = torch.randn(1, 96, 192, 192, requires_grad=True)

        output = field(objects, stroke)
        (output["center_logits"].mean() + output["offset_logits"].mean()).backward()

        self.assertIsNone(objects.grad)
        self.assertIsNone(stroke.grad)
        self.assertIsNotNone(field.subpixel_projection.weight.grad)
        with self.assertRaisesRegex(ValueError, r"\[batch,256,192,192\]"):
            field(torch.zeros(1, 256, 191, 192), torch.zeros(1, 96, 192, 192))

    def test_continuous_stride2_field_has_step_zero_base_and_shared_feature_map(self) -> None:
        atom_head = self._p1a_head()
        field = ContinuousStride2AtomCenterField(atom_head)
        objects = torch.randn(1, 256, 192, 192, requires_grad=True)
        stroke = torch.randn(1, 96, 192, 192, requires_grad=True)

        output = field(objects, stroke)
        expected = torch.nn.functional.interpolate(
            atom_head.presence(objects)[:, 0:1],
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

        self.assertEqual(field.trainable_parameter_count(), 12_835)
        self.assertEqual(output["high_resolution_features"].shape, (1, 32, 384, 384))
        self.assertTrue(torch.equal(output["center_logits"], expected))
        self.assertEqual(int(torch.count_nonzero(output["center_residual"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["offset_logits"])), 0)

    def test_continuous_stride2_field_allows_stroke_but_not_base_gradient(self) -> None:
        field = ContinuousStride2AtomCenterField(self._p1a_head())
        objects = torch.randn(1, 256, 192, 192, requires_grad=True)
        stroke = torch.randn(1, 96, 192, 192, requires_grad=True)

        output = field(objects, stroke)
        (output["center_logits"].mean() + output["offset_logits"].mean()).backward()

        self.assertIsNone(objects.grad)
        self.assertIsNotNone(stroke.grad)
        self.assertIsNotNone(field.output_projection.weight.grad)

    def test_continuous_stride1_field_is_lossless_768_grid_at_step_zero(self) -> None:
        atom_head = self._p1a_head()
        field = ContinuousStride1AtomCenterField(atom_head)
        objects = torch.randn(1, 256, 192, 192, requires_grad=True)
        stroke = torch.randn(1, 96, 192, 192, requires_grad=True)

        output = field(objects, stroke)
        expected = torch.nn.functional.interpolate(
            atom_head.presence(objects)[:, 0:1],
            scale_factor=4.0,
            mode="bilinear",
            align_corners=False,
        )

        self.assertEqual(field.trainable_parameter_count(), 17_379)
        self.assertEqual(output["stride2_features"].shape, (1, 32, 384, 384))
        self.assertEqual(output["stride1_features"].shape, (1, 32, 768, 768))
        self.assertTrue(torch.equal(output["center_logits"], expected))
        self.assertEqual(int(torch.count_nonzero(output["center_residual"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["offset_logits"])), 0)

    def test_continuous_stride1_field_passes_only_residual_stroke_gradient(self) -> None:
        field = ContinuousStride1AtomCenterField(self._p1a_head())
        objects = torch.randn(1, 256, 192, 192, requires_grad=True)
        stroke = torch.randn(1, 96, 192, 192, requires_grad=True)

        output = field(objects, stroke)
        (output["center_logits"].mean() + output["offset_logits"].mean()).backward()

        self.assertIsNone(objects.grad)
        self.assertIsNotNone(stroke.grad)
        self.assertIsNotNone(field.output_projection.weight.grad)

    def test_peak_margin_excludes_adjacent_gt_centers(self) -> None:
        logits = torch.full((1, 1, 3, 4), -4.0)
        logits[0, 0, 1, 1] = 2.0
        logits[0, 0, 1, 2] = 20.0
        logits[0, 0, 0, 1] = 1.0
        positive = torch.zeros_like(logits, dtype=torch.bool)
        positive[0, 0, 1, 1:3] = True

        loss = atom_center_peak_margin_loss(logits, positive)

        expected_first = torch.nn.functional.softplus(torch.tensor(1.0 + 1.0 - 2.0))
        expected_second = torch.nn.functional.softplus(torch.tensor(1.0 + 1.0 - 20.0))
        torch.testing.assert_close(loss, (expected_first + expected_second) / 2.0)

    def test_peak_margin_has_zero_differentiable_loss_without_negatives(self) -> None:
        logits = torch.tensor([[[[2.0]]]], requires_grad=True)
        positive = torch.ones_like(logits, dtype=torch.bool)

        loss = atom_center_peak_margin_loss(logits, positive)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
