"""Tests for grounded candidate node-edge reasoning."""

from __future__ import annotations

import unittest

import torch

from chemvector.grounded_graph import (
    GroundedCandidateGraphRefiner,
    GroundedGraphConfig,
    grounded_graph_loss,
    grounded_graph_targets,
    radius_aware_linear_assignment,
)
from chemvector.geometry import stable_pairwise_distances


class GroundedGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GroundedGraphConfig(
            candidate_dim=16,
            node_dim=16,
            edge_dim=8,
            maximum_candidates=4,
            line_samples=8,
            line_band=3,
            incident_topk=2,
            matching_radius_pixels=8.0,
        )

    def test_targets_transfer_true_adjacency(self) -> None:
        points = torch.tensor([[[0.2, 0.5], [0.8, 0.5], [0.5, 0.5], [0.0, 0.0]]])
        mask = torch.tensor([[True, True, True, False]])
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        anchors = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]])
        anchor_mask = torch.tensor([[True, True]])
        adjacency = torch.zeros(1, 2, 2, dtype=torch.bool)
        adjacency[0, 0, 1] = adjacency[0, 1, 0] = True

        nodes, edges, assignments = grounded_graph_targets(
            points,
            mask,
            pairs,
            anchors,
            anchor_mask,
            adjacency,
            matching_radius_pixels=8.0,
            image_size=64,
        )

        self.assertEqual(nodes.tolist(), [[1.0, 1.0, 0.0, 0.0]])
        self.assertEqual(edges[0, 0].item(), 1.0)
        self.assertEqual(assignments[0, :3].tolist(), [0, 1, -1])

    def test_radius_assignment_prioritizes_match_count_over_distance_sum(self) -> None:
        distances = torch.tensor([[0.0, 5.0], [5.0, 7.0]])

        rows, columns = radius_aware_linear_assignment(distances, 6.0)

        self.assertEqual(rows.tolist(), [0, 1])
        self.assertEqual(columns.tolist(), [1, 0])

    def test_pairwise_distance_is_stable_under_high_matmul_precision(self) -> None:
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            first = torch.tensor([[0.500000, 0.500000], [0.501302, 0.500000]])
            second = torch.tensor([[0.500651, 0.500000]])

            distances = stable_pairwise_distances(first, second) * 768

            self.assertTrue(torch.allclose(distances, torch.tensor([[0.5], [0.5]]), atol=0.01))
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_middle_candidate_is_visible_as_blocker(self) -> None:
        model = GroundedCandidateGraphRefiner(self.config)
        points = torch.tensor([[[0.2, 0.5], [0.8, 0.5], [0.5, 0.5], [0.5, 0.1]]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        features = model._blocker_features(
            points,
            pairs,
            mask,
            torch.full((1, 4), 8.0),
            64,
        )

        pair_01 = torch.where((pairs == torch.tensor([0, 1])).all(dim=1))[0].item()
        pair_03 = torch.where((pairs == torch.tensor([0, 3])).all(dim=1))[0].item()
        self.assertGreater(features[0, pair_01, 0].item(), 0.9)
        self.assertLess(features[0, pair_03, 0].item(), 0.2)

    def test_zero_residual_preserves_selector_logits_and_loss_is_finite(self) -> None:
        model = GroundedCandidateGraphRefiner(self.config)
        hidden = torch.randn(1, 4, 16)
        pixels = torch.ones(1, 3, 32, 32)
        pixels[:, :, 16, 6:27] = -1.0
        points = torch.tensor([[[0.2, 0.5], [0.8, 0.5], [0.5, 0.1], [0.0, 0.0]]])
        mask = torch.tensor([[True, True, True, False]])
        base = torch.tensor([[2.0, 1.0, -1.0, -30.0]])
        output = model(hidden, pixels, points, mask, base)
        anchors = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]])
        anchor_mask = torch.tensor([[True, True]])
        adjacency = torch.zeros(1, 2, 2, dtype=torch.bool)
        adjacency[0, 0, 1] = adjacency[0, 1, 0] = True
        losses = grounded_graph_loss(
            output,
            anchors,
            anchor_mask,
            adjacency,
            matching_radius_pixels=8.0,
            image_size=32,
        )
        losses["loss"].backward()

        self.assertTrue(torch.equal(output["node_logits"][mask], base[mask]))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(output["edge_logits"].shape, (1, 6))
        self.assertEqual(output["blocker_features"].shape, (1, 6, 3))

    def test_bounded_node_residual_cannot_overwrite_selector(self) -> None:
        config = GroundedGraphConfig(
            candidate_dim=16,
            node_dim=16,
            edge_dim=8,
            maximum_candidates=4,
            line_samples=8,
            line_band=3,
            incident_topk=2,
            node_residual_scale=1.5,
        )
        model = GroundedCandidateGraphRefiner(config)
        torch.nn.init.constant_(model.node_residual[-1].bias, 100.0)
        output = model(
            torch.randn(1, 4, 16),
            torch.ones(1, 3, 32, 32),
            torch.rand(1, 4, 2),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4),
        )

        self.assertLessEqual(output["node_residual_logits"].abs().max().item(), 1.5)

    def test_whole_set_loss_targets_weakest_true_and_strongest_false(self) -> None:
        model = GroundedCandidateGraphRefiner(self.config)
        hidden = torch.randn(1, 4, 16)
        pixels = torch.ones(1, 3, 32, 32)
        points = torch.tensor([[[0.2, 0.5], [0.8, 0.5], [0.5, 0.1], [0.0, 0.0]]])
        mask = torch.tensor([[True, True, True, False]])
        output = model(hidden, pixels, points, mask, torch.tensor([[2.0, -1.0, 3.0, -30.0]]))
        anchors = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]])
        anchor_mask = torch.tensor([[True, True]])
        adjacency = torch.zeros(1, 2, 2, dtype=torch.bool)
        adjacency[0, 0, 1] = adjacency[0, 1, 0] = True

        losses = grounded_graph_loss(
            output,
            anchors,
            anchor_mask,
            adjacency,
            matching_radius_pixels=8.0,
            image_size=32,
            whole_set_weight=2.0,
            whole_set_margin=1.0,
        )

        expected = torch.nn.functional.softplus(torch.tensor(1.0 + 3.0 - (-1.0)))
        self.assertTrue(torch.allclose(losses["whole_set_loss"], expected))


if __name__ == "__main__":
    unittest.main()
