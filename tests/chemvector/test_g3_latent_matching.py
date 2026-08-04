"""Tests for exact G3 latent-positive Atom matching."""

from itertools import permutations
import unittest

import torch

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_candidate_set_covers_gt,
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


if __name__ == "__main__":
    unittest.main()
