"""Tests for P3-D loss-augmented fixed-cardinality matching."""

import unittest

import torch

from chemvector.training.atoms.train_g3_p3d_structured_matching_replay_train4937 import (
    loss_augmented_fixed_cardinality_assignment,
    structured_matching_loss,
)


class G3P3DStructuredMatchingTests(unittest.TestCase):
    def test_loss_augmented_assignment_has_exact_cardinality_and_uniqueness(self) -> None:
        scores = torch.randn(5, 6)
        target_queries = torch.tensor([0, 3, 4])
        target_candidates = torch.tensor([1, 2, 5])
        queries, candidates, hamming = loss_augmented_fixed_cardinality_assignment(
            scores, 3, target_queries, target_candidates
        )
        self.assertEqual(len(queries), 3)
        self.assertEqual(len(queries.unique()), 3)
        self.assertEqual(len(candidates.unique()), 3)
        self.assertGreaterEqual(hamming, 0.0)
        self.assertLessEqual(hamming, 1.0)

    def test_structured_loss_rewards_target_joint_matching(self) -> None:
        assignment = [{
            "query_indices": torch.tensor([0, 2]),
            "candidate_indices": torch.tensor([0, 1]),
        }]
        existence = torch.tensor([[5.0, 2.0, 4.0]], requires_grad=True)
        good_pointer = torch.tensor([[
            [8.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ]], requires_grad=True)
        bad_pointer = torch.tensor([[
            [0.0, 0.0, 8.0],
            [7.0, 0.0, 0.0],
            [0.0, 0.0, 8.0],
        ]])
        good = structured_matching_loss(existence, good_pointer, assignment)
        bad = structured_matching_loss(existence.detach(), bad_pointer, assignment)
        self.assertLess(
            float(good["structured_matching_loss"].detach()),
            float(bad["structured_matching_loss"].detach()),
        )
        good["structured_matching_loss"].backward()
        self.assertTrue(torch.isfinite(existence.grad).all())
        self.assertTrue(torch.isfinite(good_pointer.grad).all())


if __name__ == "__main__":
    unittest.main()
