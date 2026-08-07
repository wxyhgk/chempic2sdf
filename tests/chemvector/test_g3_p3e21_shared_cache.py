"""Tests for the P3-E21 shared route cache."""

import unittest

import torch

from chemvector.evaluation.audits.atoms.build_g3_p3e21_shared_cache import (
    merge_shared_record,
    stratified_reachability,
)


class G3P3E21SharedCacheTests(unittest.TestCase):
    def test_merge_preserves_geometry_and_compresses_frozen_evidence(self) -> None:
        candidate = {
            "sample_key": "a",
            "full_atom_mask": torch.tensor([True, True, False]),
            "candidate_anchors": torch.zeros(128, 2),
        }
        evidence = {
            "sample_key": "a",
            "query": torch.zeros(128, 8),
            "candidate": torch.zeros(128, 41),
            "global": torch.zeros(256),
            "label": 2,
            "baseline_count": 1,
        }
        model_output = {
            "existence_logits": torch.zeros(128),
            "pointer_logits": torch.zeros(128, 128),
        }
        merged = merge_shared_record(
            candidate,
            evidence,
            model_output,
            model_output,
            torch.zeros(129),
        )
        self.assertEqual(merged["label"], 2)
        self.assertEqual(merged["base_pointer_logits"].dtype, torch.float16)
        self.assertEqual(merged["p3e16_count_logits"].shape, (129,))

    def test_merge_rejects_geometry_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "Count differs"):
            merge_shared_record(
                {"sample_key": "a", "full_atom_mask": torch.tensor([True])},
                {
                    "sample_key": "a",
                    "query": torch.zeros(128, 8),
                    "candidate": torch.zeros(128, 41),
                    "global": torch.zeros(256),
                    "label": 2,
                    "baseline_count": 1,
                },
                {"existence_logits": torch.zeros(128), "pointer_logits": torch.zeros(128, 128)},
                {"existence_logits": torch.zeros(128), "pointer_logits": torch.zeros(128, 128)},
                torch.zeros(129),
            )

    def test_reachability_is_stratified_by_count_and_variant(self) -> None:
        records = [
            {
                "sample_key": "x_stress",
                "atom_mask": torch.tensor([True, False]),
                "full_atom_mask": torch.tensor([True, True]),
            }
        ]
        metrics = stratified_reachability(records)
        self.assertEqual(metrics["count:1-20"]["atom_recall"], 0.5)
        self.assertEqual(metrics["variant:stress"]["exact_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
