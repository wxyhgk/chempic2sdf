"""Tests for exact G3 latent-positive Atom matching."""

from itertools import permutations
import unittest

import torch

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_best_invalid_assignment,
    g3_candidate_set_covers_gt,
    g3_candidate_set_gt_coverage,
    g3_group_constrained_fixed_cardinality_assignment,
    g3_hall_witnesses,
    g3_latent_positive_assignment,
    g3_unrestricted_fixed_cardinality_assignment,
)


def brute_force_latent_score(scores: torch.Tensor, compatibility: torch.Tensor) -> float:
    gt_count = len(compatibility)
    best = float("-inf")
    for queries in permutations(range(scores.size(0)), gt_count):
        for candidates in permutations(range(scores.size(1)), gt_count):
            if all(bool(compatibility[gt, candidates[gt]]) for gt in range(gt_count)):
                best = max(
                    best,
                    sum(float(scores[queries[gt], candidates[gt]]) for gt in range(gt_count)),
                )
    return best


def brute_force_invalid_score(scores: torch.Tensor, compatibility: torch.Tensor) -> float:
    count = len(compatibility)
    best = float("-inf")
    for queries in permutations(range(scores.size(0)), count):
        for candidates in permutations(range(scores.size(1)), count):
            candidate_tensor = torch.tensor(candidates)
            if not g3_candidate_set_covers_gt(candidate_tensor, compatibility):
                best = max(
                    best,
                    sum(float(scores[queries[offset], candidates[offset]]) for offset in range(count)),
                )
    return best


class G3LatentMatchingTests(unittest.TestCase):
    def test_latent_solver_matches_bruteforce_with_shared_candidate(self) -> None:
        scores = torch.tensor([
            [5.0, 4.0, 0.0, 0.0],
            [4.5, 0.0, 5.0, 0.0],
            [0.0, 3.0, 4.0, 5.0],
        ])
        compatibility = torch.tensor([
            [True, True, False, False],
            [True, False, True, True],
        ])
        result = g3_latent_positive_assignment(scores, compatibility)
        self.assertAlmostEqual(
            result["score"], brute_force_latent_score(scores, compatibility)
        )
        self.assertEqual(result["ambiguous_candidates"], 1)
        self.assertEqual(result["ownership_combinations"], 2)
        self.assertTrue(g3_candidate_set_covers_gt(
            result["candidate_indices"], compatibility
        ))

    def test_latent_score_is_permutation_invariant(self) -> None:
        scores = torch.tensor([
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 4.0],
            [1.0, 3.0, 2.0],
        ])
        compatibility = torch.tensor([
            [True, True, False],
            [False, False, True],
        ])
        baseline = g3_latent_positive_assignment(scores, compatibility)["score"]
        query_order = torch.tensor([2, 0, 1])
        candidate_order = torch.tensor([2, 0, 1])
        gt_order = torch.tensor([1, 0])
        permuted = g3_latent_positive_assignment(
            scores[query_order][:, candidate_order],
            compatibility[gt_order][:, candidate_order],
        )["score"]
        self.assertAlmostEqual(baseline, permuted)

    def test_unrestricted_assignment_enforces_fixed_cardinality(self) -> None:
        result = g3_unrestricted_fixed_cardinality_assignment(
            torch.randn(5, 6), 3
        )
        self.assertEqual(len(result["query_indices"]), 3)
        self.assertEqual(len(result["query_indices"].unique()), 3)
        self.assertEqual(len(result["candidate_indices"].unique()), 3)

    def test_candidate_coverage_rejects_duplicate_gt_neighborhood(self) -> None:
        compatibility = torch.tensor([
            [True, True, False],
            [False, False, True],
        ])
        self.assertFalse(g3_candidate_set_covers_gt(
            torch.tensor([0, 1]), compatibility
        ))
        self.assertTrue(g3_candidate_set_covers_gt(
            torch.tensor([0, 2]), compatibility
        ))

    def test_ownership_cap_is_enforced(self) -> None:
        compatibility = torch.ones(3, 4, dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "exceed cap"):
            g3_latent_positive_assignment(
                torch.randn(4, 4), compatibility, max_ownerships=16
            )

    def test_group_constrained_assignment_matches_bruteforce(self) -> None:
        scores = torch.tensor([
            [5.0, 4.0, 3.0, 0.0],
            [4.0, 5.0, 0.0, 3.0],
            [3.0, 0.0, 5.0, 4.0],
        ])
        group = torch.tensor([True, True, False, False])
        result = g3_group_constrained_fixed_cardinality_assignment(
            scores, 2, group, 1
        )
        brute = float("-inf")
        for queries in permutations(range(3), 2):
            for candidates in permutations(range(4), 2):
                if sum(bool(group[value]) for value in candidates) <= 1:
                    brute = max(brute, sum(
                        float(scores[queries[i], candidates[i]]) for i in range(2)
                    ))
        self.assertAlmostEqual(result["score"], brute)

    def test_best_invalid_matches_bruteforce(self) -> None:
        scores = torch.tensor([
            [5.0, 4.5, 0.0, 1.0],
            [4.0, 0.0, 5.0, 1.0],
            [0.0, 3.0, 4.0, 5.0],
        ])
        compatibility = torch.tensor([
            [True, True, False, False],
            [True, False, True, True],
        ])
        result = g3_best_invalid_assignment(scores, compatibility)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(
            result["score"], brute_force_invalid_score(scores, compatibility)
        )
        self.assertFalse(g3_candidate_set_covers_gt(
            result["candidate_indices"], compatibility
        ))

    def test_hall_witnesses_cover_singleton_and_shared_components(self) -> None:
        compatibility = torch.tensor([
            [True, True, False, False],
            [True, False, True, False],
            [False, False, False, True],
        ])
        witnesses = g3_hall_witnesses(compatibility)
        self.assertEqual(len(witnesses), 4)
        self.assertEqual(max(int(value["group_cap"]) for value in witnesses), 1)

    def test_candidate_set_gt_coverage_counts_maximum_matching(self) -> None:
        compatibility = torch.tensor([
            [True, True, False, False],
            [True, False, False, False],
            [False, False, True, False],
        ])
        self.assertEqual(
            g3_candidate_set_gt_coverage(torch.tensor([0, 1, 3]), compatibility),
            2,
        )
        self.assertEqual(
            g3_candidate_set_gt_coverage(torch.tensor([0, 1, 2]), compatibility),
            3,
        )
        self.assertEqual(
            g3_candidate_set_gt_coverage(torch.empty(0, dtype=torch.long), compatibility),
            0,
        )

    def test_candidate_set_gt_coverage_rejects_duplicate_indices(self) -> None:
        compatibility = torch.eye(2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            g3_candidate_set_gt_coverage(torch.tensor([0, 0]), compatibility)

    def test_best_invalid_rejects_candidate_width_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate widths differ"):
            g3_best_invalid_assignment(
                torch.zeros(2, 3), torch.ones(2, 4, dtype=torch.bool)
            )

    def test_group_constraint_handles_scores_below_old_sentinel(self) -> None:
        scores = torch.tensor([
            [-2.0e9, -3.0e9, -4.0e9],
            [-3.0e9, -2.0e9, -4.0e9],
        ], dtype=torch.float64)
        result = g3_group_constrained_fixed_cardinality_assignment(
            scores,
            2,
            torch.tensor([True, True, False]),
            1,
        )
        self.assertIsNotNone(result)
        self.assertEqual(int(result["selected_group_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
