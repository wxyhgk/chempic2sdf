"""Tests for the P1-W frozen multiscale peak-rank audit."""

from __future__ import annotations

import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1w_multiscale_peak_rank import (
    choose_protocol,
    evaluate_budget,
    minimum_sufficient_k,
)


class P1WMultiscalePeakRankAuditTests(unittest.TestCase):
    def test_minimum_sufficient_k_obeys_one_to_one_matching(self) -> None:
        expected = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
        predicted = torch.tensor(
            [[0.1, 0.1], [0.101, 0.101], [0.2, 0.2], [0.8, 0.8]]
        )
        self.assertEqual(minimum_sufficient_k(predicted, expected), 3)

    def test_budget_concatenates_scales_without_target_dependent_merging(self) -> None:
        expected = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
        stride2 = {
            "sample": {
                "expected": expected,
                "anchors": torch.tensor([[0.1, 0.1], [0.8, 0.8]]),
            }
        }
        stride1 = {
            "sample": {
                "expected": expected,
                "anchors": torch.tensor([[0.2, 0.2], [0.9, 0.9]]),
            }
        }
        result = evaluate_budget(stride2, stride1, stride2_k=1, stride1_k=1)
        self.assertEqual(result["matched_atoms"], 2)
        self.assertEqual(result["exact_images"], 1)

    def test_protocol_choice_respects_preregistered_budget_order(self) -> None:
        budgets = []
        for domain in ("overfit32", "adjacent32"):
            budgets.extend(
                [
                    {"domain": domain, "stride2_k": 128, "stride1_k": 0, "exact_images": 31},
                    {"domain": domain, "stride2_k": 96, "stride1_k": 32, "exact_images": 32},
                    {"domain": domain, "stride2_k": 64, "stride1_k": 64, "exact_images": 32},
                ]
            )
        selected = choose_protocol(budgets)
        self.assertEqual((selected["stride2_k"], selected["stride1_k"]), (96, 32))


if __name__ == "__main__":
    unittest.main()
