"""Tests for the P2-C fixed allocation selection rule."""

import unittest

from chemvector.evaluation.audits.atoms.audit_g3_p2c_route128_scale_allocation import (
    ALLOCATIONS,
    EXPANDED_BUDGETS,
    choose_allocation,
)


class P2CRoute128ScaleAllocationTests(unittest.TestCase):
    def test_allocations_keep_total128(self) -> None:
        self.assertEqual(len(ALLOCATIONS), 9)
        self.assertTrue(all(left + right == 128 for left, right in ALLOCATIONS))
        self.assertEqual(
            sorted(set(left + right for left, right in EXPANDED_BUDGETS)),
            [160, 192, 256],
        )

    def test_selection_prefers_exact_then_atoms_then_cheaper_stride1(self) -> None:
        results = [
            {"stride2_k": 64, "stride1_k": 64, "metrics": {"exact_images": 98, "matched_atoms": 3078}},
            {"stride2_k": 80, "stride1_k": 48, "metrics": {"exact_images": 98, "matched_atoms": 3079}},
            {"stride2_k": 96, "stride1_k": 32, "metrics": {"exact_images": 98, "matched_atoms": 3079}},
        ]
        self.assertEqual(choose_allocation(results)["stride1_k"], 32)


if __name__ == "__main__":
    unittest.main()
