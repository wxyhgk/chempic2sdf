from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import chemvector.audit_evidence_carrier_fit_candidate_pool_oracle as audit
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


def _source_record() -> dict:
    return {
        "identity": {
            "sample_key": "sample",
            "molecule_id": "molecule",
            "source_structure_id": "source",
            "source_pilot": "fixture",
            "source_split": "train",
            "source_record": "record",
            "render_variant": "fixture",
        },
        "partition": "fit",
        "anchor_segments_pixels": torch.tensor(
            (
                ((0.0, 0.0), (1.0, 0.0)),
                ((0.0, 1.0), (1.0, 1.0)),
                ((2.0, 0.0), (3.0, 0.0)),
                ((2.0, 1.0), (3.0, 1.0)),
            )
        ),
        "anchor_target_line_ids": torch.tensor((0, 0, 1, 1)),
        "anchor_source_indices": torch.arange(4),
        "anchor_group_ids": torch.tensor((0, 0, 1, 1)),
        "anchor_group_types": torch.tensor((2, 2, 2, 2)),
        "pair_indices": torch.tensor(((0, 1), (0, 2), (2, 3))),
        "pair_labels": torch.tensor((1, 0, 1)),
    }


def _fit_record() -> dict:
    source = _source_record()
    pair_tokens = torch.zeros((3, 21), dtype=torch.float32)
    pair_tokens[:, -1] = torch.tensor((1.0, 2.0, 1.0))
    return {
        "identity": dict(source["identity"]),
        "partition": "fit",
        "target_free": {
            "pair_tokens": pair_tokens,
            "source_to_canonical_candidate_indices": torch.arange(3),
        },
        "target_aware": {"pair_labels": torch.tensor((1, 0, 1))},
    }


class FitCandidatePoolOracleTest(unittest.TestCase):
    def test_oracle_scores_are_exact_relative_plus_minus_one(self) -> None:
        scores = audit.oracle_pair_scores(torch.tensor((1, 0, 1)))
        expected = torch.tensor(
            (
                SAMPLE_THRESHOLD_BASELINE_TAU + 1.0,
                SAMPLE_THRESHOLD_BASELINE_TAU - 1.0,
                SAMPLE_THRESHOLD_BASELINE_TAU + 1.0,
            ),
            dtype=torch.float64,
        )
        torch.testing.assert_close(scores, expected, rtol=0.0, atol=0.0)

    def test_oracle_selects_only_disjoint_positive_pairs(self) -> None:
        with (
            patch.object(audit, "EXPECTED_SAMPLE_COUNT", 1),
            patch.object(audit, "EXPECTED_PAIR_COUNT", 3),
            patch.object(audit, "EXPECTED_POSITIVE_COUNT", 2),
            patch.object(audit, "EXPECTED_NEGATIVE_COUNT", 1),
        ):
            result = audit.evaluate_fit_candidate_pool_oracle(
                [_fit_record()], [_source_record()]
            )

        metrics = result["oracle_selected_metrics"]
        self.assertEqual(metrics["selected_cross_target_pairs"], 0)
        self.assertEqual(metrics["selected_same_target_pairs"], 2)
        self.assertEqual(metrics["selected_pair_precision"], 1.0)
        self.assertEqual(result["oracle_failure_sample_count"], 0)

    def test_oracle_rejects_nonbinary_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            audit.oracle_pair_scores(torch.tensor((1, 2)))


if __name__ == "__main__":
    unittest.main()
