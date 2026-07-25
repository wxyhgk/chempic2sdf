"""Tests for spatially grounded dense Bond Objects."""

import unittest

import torch

from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
    _injective_slot_permutations,
    align_dense_bond_targets_to_output,
    build_dense_bond_targets,
    decode_dense_bond_objects,
    dense_bond_object_loss,
)


class DenseBondObjectTests(unittest.TestCase):
    def _targets(self) -> dict[str, torch.Tensor]:
        return {
            "endpoints": torch.tensor([[[[0.2, 0.2], [0.8, 0.8]]]]),
            "bbox": torch.zeros(1, 1, 4),
            "display": torch.tensor([[1]]),
            "chemical": torch.tensor([[1]]),
            "direction": torch.tensor([[0]]),
            "mask": torch.tensor([[True]]),
        }

    def _same_cell_targets(self) -> dict[str, torch.Tensor]:
        return {
            "endpoints": torch.tensor(
                [
                    [
                        [[0.49, 0.50], [0.51, 0.50]],
                        [[0.50, 0.49], [0.50, 0.51]],
                    ]
                ]
            ),
            "bbox": torch.zeros(1, 2, 4),
            "display": torch.tensor([[0, 1]]),
            "chemical": torch.tensor([[1, 2]]),
            "direction": torch.tensor([[0, 0]]),
            "mask": torch.tensor([[True, True]]),
        }

    def test_loss_and_decode_have_stable_shapes(self) -> None:
        model = DenseBondObjectDetector(
            DenseBondObjectConfig(input_channels=8, hidden_channels=16, slots=2)
        )
        output = model(torch.randn(1, 8, 16, 16))
        losses = dense_bond_object_loss(
            output, self._targets(), heatmap_set_weight=1.0
        )
        losses["loss"].backward()
        decoded = decode_dense_bond_objects(output, threshold=0.0, maximum_objects=4)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(losses["heatmap_set_loss"]))
        self.assertEqual(output["endpoints"].shape, (1, 2, 2, 2, 16, 16))
        self.assertEqual(len(decoded[0]), 4)
        self.assertEqual(decoded[0][0]["feature"].shape, (16,))

    def test_same_center_uses_multiple_slots(self) -> None:
        targets = self._targets()
        targets["endpoints"] = torch.tensor(
            [[[[0.2, 0.2], [0.8, 0.8]], [[0.2, 0.8], [0.8, 0.2]]]]
        )
        targets["bbox"] = torch.zeros(1, 2, 4)
        targets["display"] = torch.tensor([[0, 1]])
        targets["chemical"] = torch.tensor([[0, 1]])
        targets["direction"] = torch.tensor([[0, 0]])
        targets["mask"] = torch.tensor([[True, True]])
        dense = build_dense_bond_targets(targets, height=16, width=16, slots=2)
        self.assertEqual(int(dense["positive"].sum()), 2)
        self.assertTrue(bool(dense["positive"][0, 0, 8, 8]))
        self.assertTrue(bool(dense["positive"][0, 1, 8, 8]))

    def test_hungarian_loss_is_invariant_to_target_order_and_endpoint_flip(self) -> None:
        torch.manual_seed(7)
        model = DenseBondObjectDetector(
            DenseBondObjectConfig(input_channels=8, hidden_channels=16, slots=3)
        )
        output = model(torch.randn(1, 8, 16, 16))
        original = self._same_cell_targets()
        equivalent = {
            "endpoints": torch.stack(
                (
                    original["endpoints"][:, 1],
                    original["endpoints"][:, 0].flip(1),
                ),
                dim=1,
            ),
            "bbox": original["bbox"][:, [1, 0]],
            "display": original["display"][:, [1, 0]],
            "chemical": original["chemical"][:, [1, 0]],
            "direction": original["direction"][:, [1, 0]],
            "mask": original["mask"][:, [1, 0]],
        }

        first = dense_bond_object_loss(
            output, original, slot_matching="hungarian"
        )
        second = dense_bond_object_loss(
            output, equivalent, slot_matching="hungarian"
        )

        for name in first:
            self.assertTrue(
                torch.allclose(first[name], second[name], atol=1e-6, rtol=0.0),
                name,
            )

    def test_hungarian_alignment_keeps_unmatched_slots_negative(self) -> None:
        model = DenseBondObjectDetector(
            DenseBondObjectConfig(input_channels=8, hidden_channels=16, slots=3)
        )
        output = model(torch.randn(1, 8, 16, 16))
        targets = self._same_cell_targets()
        dense = build_dense_bond_targets(targets, height=16, width=16, slots=3)

        aligned = align_dense_bond_targets_to_output(
            output, dense, slot_matching="hungarian"
        )

        self.assertEqual(int(aligned["positive"].sum()), int(targets["mask"].sum()))
        cell_positive = aligned["positive"][0, :, 8, 8]
        self.assertEqual(int(cell_positive.sum()), 2)
        unmatched = torch.where(~cell_positive)[0]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(float(aligned["heatmap"][0, unmatched[0], 8, 8]), 0.0)
        self.assertEqual(int(aligned["heatmap"].sum()), 2)

    def test_hungarian_loss_has_finite_gradients(self) -> None:
        model = DenseBondObjectDetector(
            DenseBondObjectConfig(input_channels=8, hidden_channels=16, slots=3)
        )
        output = model(torch.randn(1, 8, 16, 16))
        losses = dense_bond_object_loss(
            output,
            self._same_cell_targets(),
            slot_matching="hungarian",
            cell_count_weight=0.1,
            duplicate_weight=0.05,
        )

        losses["loss"].backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertTrue(any(bool(gradient.abs().sum() > 0) for gradient in gradients))

    def test_injective_slot_permutations_find_known_optimum(self) -> None:
        cost = torch.tensor(
            (
                (5.0, 1.0),
                (1.0, 5.0),
                (2.0, 2.0),
            )
        )
        assignments = _injective_slot_permutations(slots=3, targets=2)
        target_indices = torch.arange(2)
        assignment_costs = cost[assignments, target_indices].sum(-1)
        selected = assignments[assignment_costs.argmin()]

        self.assertEqual(selected.tolist(), [1, 0])
        self.assertEqual(len(set(selected.tolist())), 2)
        self.assertEqual(float(assignment_costs.min()), 2.0)

    def test_slot_overflow_is_rejected(self) -> None:
        targets = self._targets()
        targets["endpoints"] = targets["endpoints"].expand(1, 2, 2, 2).clone()
        targets["bbox"] = torch.zeros(1, 2, 4)
        targets["display"] = torch.zeros(1, 2, dtype=torch.long)
        targets["chemical"] = torch.zeros(1, 2, dtype=torch.long)
        targets["direction"] = torch.zeros(1, 2, dtype=torch.long)
        targets["mask"] = torch.tensor([[True, True]])
        with self.assertRaisesRegex(ValueError, "slot capacity"):
            build_dense_bond_targets(targets, height=16, width=16, slots=1)

    def test_zero_initialized_multiscale_context_preserves_base_output(self) -> None:
        config = DenseBondObjectConfig(
            input_channels=8, hidden_channels=16, slots=2
        )
        base = DenseBondObjectDetector(config).eval()
        multiscale = MultiScaleDenseBondObjectDetector(
            config, stride8_channels=12, stride16_channels=20
        ).eval()
        multiscale.base.load_state_dict(base.state_dict())
        stride4 = torch.randn(1, 8, 16, 16)
        expected = base(stride4)
        actual = multiscale(
            stride4,
            torch.randn(1, 12, 8, 8),
            torch.randn(1, 20, 4, 4),
        )
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_multiscale_adapter_receives_gradients(self) -> None:
        config = DenseBondObjectConfig(
            input_channels=8, hidden_channels=16, slots=2
        )
        model = MultiScaleDenseBondObjectDetector(
            config, stride8_channels=12, stride16_channels=20
        )
        output = model(
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 12, 8, 8),
            torch.randn(1, 20, 4, 4),
        )
        output["heatmap_logits"].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                for parameter in model.adapter_parameters()
            )
        )

    def test_preserved_type_logits_match_stride4_base(self) -> None:
        config = DenseBondObjectConfig(
            input_channels=8, hidden_channels=16, slots=2
        )
        model = MultiScaleDenseBondObjectDetector(
            config,
            stride8_channels=12,
            stride16_channels=20,
            preserve_base_types=True,
        ).eval()
        with torch.no_grad():
            model.stride8_adapter[-1].weight.fill_(0.1)
        stride4 = torch.randn(1, 8, 16, 16)
        base = model.base(stride4)
        output = model(
            stride4,
            torch.randn(1, 12, 8, 8),
            torch.randn(1, 20, 4, 4),
        )
        for key in ("display_logits", "chemical_logits", "direction_logits"):
            self.assertTrue(torch.equal(base[key], output[key]))

    def test_separate_type_base_is_frozen(self) -> None:
        config = DenseBondObjectConfig(
            input_channels=8, hidden_channels=16, slots=2
        )
        source = DenseBondObjectDetector(config)
        model = MultiScaleDenseBondObjectDetector(
            config, stride8_channels=12, stride16_channels=20
        )
        model.install_frozen_type_base(source.state_dict())
        self.assertIsNotNone(model.type_base)
        self.assertFalse(any(p.requires_grad for p in model.type_base.parameters()))


if __name__ == "__main__":
    unittest.main()
