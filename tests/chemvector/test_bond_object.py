"""Tests for semantic Bond Object queries."""

import unittest

import torch

from chemvector.bond_object import (
    BondObjectConfig,
    BondObjectDetector,
    bond_object_loss,
    unordered_endpoint_cost,
)


class BondObjectTests(unittest.TestCase):
    def test_unordered_endpoint_cost_accepts_reversed_bond(self) -> None:
        predicted = torch.tensor([[[0.8, 0.8], [0.2, 0.2]]])
        target = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        self.assertAlmostEqual(float(unordered_endpoint_cost(predicted, target)[0, 0]), 0.0)

    def test_detector_loss_is_finite(self) -> None:
        config = BondObjectConfig(
            input_channels=8,
            d_model=32,
            queries=4,
            layers=1,
            heads=4,
            ffn_dim=64,
            memory_stride=2,
        )
        detector = BondObjectDetector(config)
        output = detector(torch.randn(1, 8, 16, 16))
        targets = {
            "endpoints": torch.tensor([[[[0.2, 0.2], [0.8, 0.8]]]]),
            "bbox": torch.tensor([[[0.5, 0.5, 0.6, 0.6]]]),
            "display": torch.tensor([[1]]),
            "chemical": torch.tensor([[1]]),
            "direction": torch.tensor([[0]]),
            "mask": torch.tensor([[True]]),
        }
        losses = bond_object_loss(output, targets)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(output["endpoints"].shape, (1, 4, 2, 2))

    def test_endpoint_tail_weight_penalizes_worst_object(self) -> None:
        output = {
            "presence_logits": torch.tensor([[8.0, 8.0]], requires_grad=True),
            "endpoints": torch.tensor(
                [[[[0.2, 0.2], [0.8, 0.8]], [[0.1, 0.1], [0.6, 0.6]]]],
                requires_grad=True,
            ),
            "bbox": torch.tensor(
                [[[0.5, 0.5, 0.6, 0.6], [0.35, 0.35, 0.5, 0.5]]],
                requires_grad=True,
            ),
            "display_logits": torch.zeros(1, 2, 4, requires_grad=True),
            "chemical_logits": torch.zeros(1, 2, 4, requires_grad=True),
            "direction_logits": torch.zeros(1, 2, 5, requires_grad=True),
        }
        targets = {
            "endpoints": torch.tensor(
                [[[[0.2, 0.2], [0.8, 0.8]], [[0.1, 0.1], [0.9, 0.9]]]]
            ),
            "bbox": torch.tensor([[[0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.8, 0.8]]]),
            "display": torch.tensor([[0, 0]]),
            "chemical": torch.tensor([[0, 0]]),
            "direction": torch.tensor([[0, 0]]),
            "mask": torch.tensor([[True, True]]),
        }
        baseline = bond_object_loss(output, targets)
        tailed = bond_object_loss(output, targets, endpoint_tail_weight=12.0)
        self.assertGreater(
            float(tailed["loss"].detach()), float(baseline["loss"].detach())
        )
        self.assertGreater(float(tailed["endpoint_tail_loss"].detach()), 0.0)

    def test_negative_endpoint_tail_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bond_object_loss({}, {}, endpoint_tail_weight=-1.0)

    def test_endpoint_barrier_ignores_errors_inside_margin(self) -> None:
        output = {
            "presence_logits": torch.tensor([[8.0]], requires_grad=True),
            "endpoints": torch.tensor(
                [[[[0.2, 0.2], [0.8, 0.8]]]], requires_grad=True
            ),
            "bbox": torch.tensor(
                [[[0.5, 0.5, 0.6, 0.6]]], requires_grad=True
            ),
            "display_logits": torch.zeros(1, 1, 4, requires_grad=True),
            "chemical_logits": torch.zeros(1, 1, 4, requires_grad=True),
            "direction_logits": torch.zeros(1, 1, 5, requires_grad=True),
        }
        targets = {
            "endpoints": torch.tensor([[[[0.201, 0.2], [0.8, 0.8]]]]),
            "bbox": torch.tensor([[[0.5, 0.5, 0.6, 0.6]]]),
            "display": torch.tensor([[0]]),
            "chemical": torch.tensor([[0]]),
            "direction": torch.tensor([[0]]),
            "mask": torch.tensor([[True]]),
        }
        losses = bond_object_loss(
            output,
            targets,
            endpoint_barrier_weight=1.0,
            endpoint_barrier_margin_pixels=4.0,
        )
        self.assertAlmostEqual(float(losses["endpoint_barrier_loss"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
