from __future__ import annotations

import unittest

import torch

from chemvector.audit_atom_pair_two_move_reachability import (
    bounded_reachable,
    error_ranks,
    residual_kind,
)


class AtomPairTwoMoveReachabilityTest(unittest.TestCase):
    def test_residual_kind_names_bounded_edits(self) -> None:
        self.assertEqual(residual_kind(0, 0), "exact")
        self.assertEqual(residual_kind(2, 1), "two_delete_one_add")
        self.assertEqual(residual_kind(3, 0), "larger")

    def test_error_ranks_follow_current_selection(self) -> None:
        logits = torch.tensor((2.0, 0.1, -0.2, -3.0, 1.0, -0.1))
        selected = torch.tensor((True, True, False, False, True, False))
        expected = torch.tensor((True, False, True, False, True, True))

        delete_ranks, add_ranks = error_ranks(logits, selected, expected)

        self.assertEqual(delete_ranks, [1])
        self.assertEqual(add_ranks, [2, 1])

    def test_bounded_reachability_checks_count_and_rank(self) -> None:
        self.assertTrue(
            bounded_reachable(
                [1, 4],
                [2, 8],
                maximum_deletes=2,
                maximum_adds=2,
                delete_candidates=4,
                add_candidates=8,
            )
        )
        self.assertFalse(
            bounded_reachable(
                [1, 5],
                [2],
                maximum_deletes=2,
                maximum_adds=2,
                delete_candidates=4,
                add_candidates=8,
            )
        )


if __name__ == "__main__":
    unittest.main()
