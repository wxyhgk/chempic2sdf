"""Tests for P3-E22 minimum candidate budget selection."""

import unittest

from chemvector.evaluation.audits.atoms.audit_g3_p3e22_candidate_budgets import (
    BUDGETS,
    select_budget,
)
from chemvector.evaluation.audits.atoms import build_g3_p3e21_shared_cache as p3e21


class G3P3E22CandidateBudgetTests(unittest.TestCase):
    def test_selects_smallest_budget_passing_every_split(self) -> None:
        metrics = {
            split: {
                str(budget): {"atom_recall": 0.98 if budget == 128 else 0.995}
                for budget in BUDGETS
            }
            for split in p3e21.MANIFESTS
        }
        selected, checks = select_budget(metrics)
        self.assertEqual(selected, 192)
        self.assertFalse(checks["128"])
        self.assertTrue(checks["192"])

    def test_returns_none_when_any_split_stays_below_gate(self) -> None:
        metrics = {
            split: {
                str(budget): {"atom_recall": 0.995} for budget in BUDGETS
            }
            for split in p3e21.MANIFESTS
        }
        metrics["dev"]["256"]["atom_recall"] = 0.98
        metrics["dev"]["192"]["atom_recall"] = 0.98
        metrics["dev"]["128"]["atom_recall"] = 0.98
        selected, _ = select_budget(metrics)
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
