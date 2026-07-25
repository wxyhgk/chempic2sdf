"""Tests for role-aware dual vertex candidate preparation."""

from __future__ import annotations

import unittest

import torch

from chemvector.dual_vertex_selector import (
    IMPLICIT_ROLE,
    TEXT_ROLE,
    merge_dual_candidates,
    role_aware_candidate_targets,
    role_aware_candidate_groups,
    role_aware_match_count,
    role_suppressed_topk_indices,
    role_suppressed_threshold_indices,
    role_suppressed_role_threshold_indices,
    grouped_candidate_selection_loss,
    line_interior_candidate_mask,
)


class DualVertexSelectorTests(unittest.TestCase):
    def test_line_interior_mask_excludes_real_and_endpoint_candidates(self) -> None:
        points = torch.tensor([[[0.2, 0.5], [0.5, 0.5], [0.8, 0.5], [0.51, 0.7]]])
        candidate_mask = torch.ones(1, 4, dtype=torch.bool)
        groups = torch.tensor([[0, -1, 1, -1]])
        targets = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]])
        target_mask = torch.ones(1, 2, dtype=torch.bool)
        adjacency = torch.tensor([[[False, True], [True, False]]])
        result = line_interior_candidate_mask(
            points,
            candidate_mask,
            groups,
            targets,
            target_mask,
            adjacency,
            image_size=100,
        )
        self.assertEqual(result.tolist(), [[False, True, False, False]])

    def test_line_interior_ranking_uses_hardest_explicit_false_candidate(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 1.8, -2.0]], requires_grad=True)
        groups = torch.tensor([[0, 1, -1, -1]])
        valid = torch.ones_like(groups, dtype=torch.bool)
        line_mask = torch.tensor([[False, False, True, False]])
        losses = grouped_candidate_selection_loss(
            logits,
            groups,
            valid,
            ranking_weight=0.0,
            background_weight=0.0,
            line_interior_mask=line_mask,
            line_interior_weight=2.0,
            line_interior_margin=1.0,
        )
        self.assertGreater(
            float(losses["line_interior_ranking_loss"].detach()), 0.0
        )
        losses["loss"].backward()
        self.assertGreater(float(logits.grad[0, 2]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)

    def test_merge_preserves_text_capacity_before_implicit_peaks(self) -> None:
        text = [
            {"x": 0.1, "y": 0.1, "score": 0.2},
            {"x": 0.2, "y": 0.2, "score": 0.9},
        ]
        implicit = [
            {"x": 0.3, "y": 0.3, "score": 0.99},
            {"x": 0.4, "y": 0.4, "score": 0.8},
        ]

        merged, roles = merge_dual_candidates(
            text, implicit, maximum_candidates=3
        )

        self.assertEqual(roles, [TEXT_ROLE, TEXT_ROLE, IMPLICIT_ROLE])
        self.assertEqual([item["score"] for item in merged], [0.9, 0.2, 0.99])

    def test_role_aware_matching_rejects_cross_role_credit(self) -> None:
        points = torch.tensor([[[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        roles = torch.tensor([[TEXT_ROLE, IMPLICIT_ROLE, IMPLICIT_ROLE]])
        target_points = torch.tensor([[[0.1, 0.1], [0.5, 0.5]]])
        target_mask = torch.ones((1, 2), dtype=torch.bool)
        # The first point is geometrically exact but has the wrong role.
        target_roles = torch.tensor([[IMPLICIT_ROLE, IMPLICIT_ROLE]])

        targets, assignments = role_aware_candidate_targets(
            points,
            mask,
            roles,
            target_points,
            target_mask,
            target_roles,
            matching_radius_pixels=6.0,
            image_size=768,
        )

        self.assertEqual(targets.tolist(), [[0.0, 1.0, 0.0]])
        self.assertEqual(assignments.tolist(), [[-1, 1, -1]])

    def test_role_aware_match_count_accepts_alternative_nearby_candidate(self) -> None:
        matches = role_aware_match_count(
            torch.tensor([[0.1, 0.1], [0.101, 0.1]]),
            torch.tensor([TEXT_ROLE, IMPLICIT_ROLE]),
            torch.tensor([[0.1, 0.1]]),
            torch.tensor([IMPLICIT_ROLE]),
            matching_radius_pixels=6.0,
            image_size=768,
        )

        self.assertEqual(matches, 1)

    def test_role_suppression_keeps_next_distinct_candidate(self) -> None:
        selected = role_suppressed_topk_indices(
            torch.tensor([5.0, 4.0, 3.0]),
            torch.tensor([[0.1, 0.1], [0.102, 0.1], [0.3, 0.3]]),
            torch.ones(3, dtype=torch.bool),
            torch.tensor([IMPLICIT_ROLE, IMPLICIT_ROLE, IMPLICIT_ROLE]),
            torch.tensor([2, 0]),
            suppression_radius_pixels=6.0,
            image_size=768,
        )

        self.assertEqual(selected.tolist(), [0, 2])

    def test_group_targets_keep_alternative_peaks_equivalent(self) -> None:
        groups = role_aware_candidate_groups(
            torch.tensor([[[0.1, 0.1], [0.102, 0.1], [0.8, 0.8]]]),
            torch.ones((1, 3), dtype=torch.bool),
            torch.tensor([[IMPLICIT_ROLE, IMPLICIT_ROLE, IMPLICIT_ROLE]]),
            torch.tensor([[[0.1, 0.1]]]),
            torch.ones((1, 1), dtype=torch.bool),
            torch.tensor([[IMPLICIT_ROLE]]),
            matching_radius_pixels=6.0,
            image_size=768,
        )

        self.assertEqual(groups.tolist(), [[0, 0, -1]])

    def test_group_loss_rewards_any_representative_and_rejects_background(self) -> None:
        logits = torch.tensor([[3.0, -2.0, 2.0]], requires_grad=True)
        losses = grouped_candidate_selection_loss(
            logits,
            torch.tensor([[0, 0, -1]]),
            torch.ones((1, 3), dtype=torch.bool),
        )
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)

    def test_absolute_boundary_pushes_hard_background_below_ceiling(self) -> None:
        logits = torch.tensor([[0.0, 4.0]], requires_grad=True)
        losses = grouped_candidate_selection_loss(
            logits,
            torch.tensor([[0, -1]]),
            torch.ones((1, 2), dtype=torch.bool),
            ranking_weight=0.0,
            background_weight=0.0,
            absolute_boundary_weight=1.0,
            positive_floor_probability=0.8,
            background_ceiling_probability=0.2,
        )
        losses["loss"].backward()

        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 1]), 0.0)

    def test_threshold_decode_suppresses_duplicate_real_peaks(self) -> None:
        selected = role_suppressed_threshold_indices(
            torch.tensor([5.0, 4.0, -5.0]),
            torch.tensor([[0.1, 0.1], [0.102, 0.1], [0.3, 0.3]]),
            torch.ones(3, dtype=torch.bool),
            torch.tensor([IMPLICIT_ROLE, IMPLICIT_ROLE, IMPLICIT_ROLE]),
            probability_threshold=0.5,
            suppression_radius_pixels=6.0,
            image_size=768,
        )

        self.assertEqual(selected.tolist(), [0])

    def test_role_thresholds_can_lower_only_implicit_boundary(self) -> None:
        selected = role_suppressed_role_threshold_indices(
            torch.tensor([0.2, 0.2]),
            torch.tensor([[0.1, 0.1], [0.8, 0.8]]),
            torch.ones(2, dtype=torch.bool),
            torch.tensor([IMPLICIT_ROLE, TEXT_ROLE]),
            probability_thresholds=torch.tensor([0.5, 0.6]),
            suppression_radius_pixels=6.0,
            image_size=768,
        )

        self.assertEqual(selected.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
