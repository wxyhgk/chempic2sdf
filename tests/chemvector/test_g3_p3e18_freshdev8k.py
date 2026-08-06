"""Tests for P3-E18 fixed cross-font metrics and gates."""

import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p3e18_freshdev8k import (
    GATES,
    candidate_stratified_metrics,
    compact_decomposition,
    gate,
    stratified_metrics,
)
from chemvector.training.atoms.audit_g3_p3e17_p3e16_count_router import POLICY


def row(key: str, *, count: int, atom_count: int, correct: bool, variant: str) -> dict:
    prediction = {
        "decoded_count": count,
        "correct": correct,
        "matched_atoms": atom_count if correct else atom_count - 1,
        "target_atoms": atom_count,
        "candidate_indices": tuple(range(count)),
    }
    return {
        "sample_key": key,
        "variant": variant,
        "atom_count": atom_count,
        "base": prediction,
        "expert": prediction,
        "features": {POLICY["feature"]: POLICY["threshold"] + 1.0},
    }


class G3P3E18FreshDevTests(unittest.TestCase):
    def test_stratification_preserves_all_rows(self) -> None:
        known = [
            row("a", count=10, atom_count=10, correct=True, variant="coordgen"),
            row("b", count=11, atom_count=10, correct=False, variant="stress"),
        ]
        metrics = stratified_metrics(known, known, POLICY)
        self.assertEqual(metrics["overall"]["images"], 2)
        self.assertEqual(metrics["by_variant"]["coordgen"]["images"], 1)
        self.assertEqual(metrics["by_atom_count"]["1-20"]["images"], 2)

    def test_gate_uses_all_five_preregistered_thresholds(self) -> None:
        metrics = {
            "overall": {
                "images": 10,
                "count": {
                    "count_accuracy": GATES["count_accuracy_min"],
                    "count_mae": GATES["count_mae_max"],
                },
                "known": {"router": {"exact": 3}},
                "predicted": {"router": {"exact": 2}},
            }
        }
        decision, checks = gate(metrics, {"atom_recall": 0.99})
        self.assertEqual(decision, "freshdev_heldout_font_go")
        self.assertTrue(all(checks.values()))
        decision, checks = gate(metrics, {"atom_recall": 0.989})
        self.assertEqual(decision, "freshdev_heldout_font_no_go")
        self.assertFalse(checks["candidate_atom_recall_min99"])

    def test_candidate_metrics_use_representable_subset(self) -> None:
        records = [
            {
                "sample_key": "x_coordgen",
                "atom_mask": torch.tensor([True, False, False]),
                "full_atom_mask": torch.tensor([True, True, False]),
            }
        ]
        metrics = candidate_stratified_metrics(records)
        self.assertEqual(metrics["overall"]["matched_atoms"], 1)
        self.assertEqual(metrics["overall"]["target_atoms"], 2)
        self.assertEqual(metrics["overall"]["exact_images"], 0)

    def test_decomposition_compaction_drops_only_key_lists(self) -> None:
        value = {
            "known_policy": {"exact": 3, "routed_sample_keys": ["a"]},
            "predicted_policy": {"exact": 2, "routed_sample_keys": ["b"]},
            "count_correct": 4,
        }
        compact = compact_decomposition(value)
        self.assertNotIn("routed_sample_keys", compact["known_policy"])
        self.assertEqual(compact["count_correct"], 4)


if __name__ == "__main__":
    unittest.main()
