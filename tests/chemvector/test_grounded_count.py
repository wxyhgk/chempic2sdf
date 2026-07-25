"""Tests for grounded graph cardinality calibration."""

from __future__ import annotations

import unittest

import torch

from chemvector.grounded_count import (
    GroundedCountCalibrator,
    GroundedCountConfig,
    GroundedRoleCountCalibrator,
    grounded_count_loss,
    grounded_role_count_loss,
    grounded_count_node_features,
    predict_grounded_counts,
)


class GroundedCountTests(unittest.TestCase):
    def test_features_model_and_loss_have_stable_shapes(self) -> None:
        config = GroundedCountConfig(maximum_candidates=4, hidden_dim=16)
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        node_logits = torch.randn(2, 4)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        edge_logits = torch.randn(2, len(pairs))

        node_features, global_features = grounded_count_node_features(
            node_logits, mask, edge_logits, pairs
        )
        logits = GroundedCountCalibrator(config)(node_logits, mask, edge_logits, pairs)
        loss = grounded_count_loss(logits, torch.tensor([3, 2]))
        loss.backward()

        self.assertEqual(node_features.shape, (2, 4, 6))
        self.assertEqual(global_features.shape, (2, 6))
        self.assertEqual(logits.shape, (2, 5))
        self.assertTrue(torch.isfinite(loss))

    def test_count_prediction_does_not_modify_node_ranking(self) -> None:
        logits = torch.tensor([[0.0, 5.0, 1.0, -2.0]])
        predicted = predict_grounded_counts(torch.tensor([[0.0, 1.0, 9.0, 2.0, 0.0]]))
        selected = logits.topk(int(predicted[0])).indices.tolist()[0]

        self.assertEqual(predicted.tolist(), [2])
        self.assertEqual(selected, [1, 2])

    def test_role_conditioned_count_features(self) -> None:
        config = GroundedCountConfig(
            maximum_candidates=4,
            hidden_dim=8,
            role_count=2,
        )
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        logits = GroundedCountCalibrator(config)(
            torch.randn(1, 4),
            torch.tensor([[True, True, True, False]]),
            torch.randn(1, len(pairs)),
            pairs,
            torch.tensor([[0, 1, 0, 0]]),
        )

        self.assertEqual(logits.shape, (1, 5))

    def test_role_count_calibrator_predicts_each_role(self) -> None:
        config = GroundedCountConfig(
            maximum_candidates=4,
            hidden_dim=8,
            role_count=2,
        )
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        logits = GroundedRoleCountCalibrator(config)(
            torch.randn(2, 4),
            torch.tensor([[True, True, True, False], [True, True, False, False]]),
            torch.randn(2, len(pairs)),
            pairs,
            torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]]),
        )
        loss = grounded_role_count_loss(logits, torch.tensor([[2, 1], [1, 1]]))
        loss.backward()

        self.assertEqual(logits.shape, (2, 2, 5))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
