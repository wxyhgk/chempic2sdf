from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from chemvector.audit_evidence_carrier_decoded_line_clustering_oracle import (
    evaluate_frozen_partition,
    gate_decision,
    summarize_records,
)
from chemvector.evidence_carrier_decoded_line_clustering import DecodedLineCluster


@dataclass(frozen=True)
class _Line:
    group_index: int
    group_kind: str
    points_q: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class _Group:
    kind: str
    line_indices: tuple[int, ...]


def _cluster(members: tuple[int, ...], representative: int) -> DecodedLineCluster:
    lines = (
        ((10.0, 20.0), (110.0, 20.0)),
        ((10.2, 20.0), (110.2, 20.0)),
        ((10.0, 24.0), (110.0, 24.0)),
    )
    return DecodedLineCluster(
        member_indices=members,
        member_source_ids=tuple(f"id-{index}" for index in members),
        representative_index=representative,
        representative_source_id=f"id-{representative}",
        representative_segment_pixels=lines[representative],
    )


class EvidenceCarrierDecodedLineClusteringOracleAuditTest(unittest.TestCase):
    def test_evaluation_detects_exact_partition_and_preserves_parallel_two(self) -> None:
        decoded = np.asarray(
            [
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.2, 20.0), (110.2, 20.0)),
                ((10.0, 24.0), (110.0, 24.0)),
            ]
        )
        scale = 2047.0 / 767.0
        target_lines = [
            _Line(0, "parallel_2", tuple(map(tuple, decoded[0] * scale))),
            _Line(0, "parallel_2", tuple(map(tuple, decoded[2] * scale))),
        ]
        result = evaluate_frozen_partition(
            clusters=(_cluster((0, 1), 0), _cluster((2,), 2)),
            target_line_ids=np.asarray((0, 0, 1)),
            decoded_segments_pixels=decoded,
            target_lines=target_lines,
            target_groups=[_Group("parallel_2", (0, 1))],
            width=768,
            height=768,
        )
        self.assertEqual(result["pairs"]["f1"], 1.0)
        self.assertTrue(result["positive_partition_eligible"])
        self.assertTrue(result["positive_partition_exact"])
        self.assertEqual(result["cross_target_merge"]["clusters"], 0)
        self.assertEqual(result["q8_recall"]["loss_lines"], 0)
        self.assertEqual(result["parallel_2_distinct_preservation"]["rate"], 1.0)

    def test_evaluation_counts_cross_target_merge(self) -> None:
        decoded = np.asarray(
            [
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.0, 24.0), (110.0, 24.0)),
            ]
        )
        scale = 2047.0 / 767.0
        result = evaluate_frozen_partition(
            clusters=(_cluster((0, 1), 0),),
            target_line_ids=np.asarray((0, 1)),
            decoded_segments_pixels=decoded,
            target_lines=[
                _Line(0, "parallel_2", tuple(map(tuple, decoded[0] * scale))),
                _Line(0, "parallel_2", tuple(map(tuple, decoded[1] * scale))),
            ],
            target_groups=[_Group("parallel_2", (0, 1))],
            width=768,
            height=768,
        )
        self.assertEqual(result["pairs"]["false_positive"], 1)
        self.assertEqual(result["cross_target_merge"]["clusters"], 1)
        self.assertTrue(result["positive_partition_eligible"])
        self.assertFalse(result["positive_partition_exact"])
        self.assertEqual(result["parallel_2_distinct_preservation"]["rate"], 0.0)

    def test_summary_and_gate_are_non_vacuous(self) -> None:
        record = {
            "positive_decoded_lines": 3,
            "predicted_clusters": 2,
            "pairs": {
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
            },
            "cross_target_merge": {
                "clusters": 0,
                "pairs": 0,
                "sample_has_merge": False,
            },
            "positive_partition_eligible": True,
            "positive_partition_exact": True,
            "q8_recall": {
                "expected": 2,
                "before_clustering_matched": 2,
                "after_clustering_matched": 2,
            },
            "parallel_2_distinct_preservation": {
                "total_groups": 1,
                "both_members_supported": 1,
                "preserved": 1,
                "merged": 0,
            },
        }
        summary = summarize_records([record])
        self.assertEqual(summary["pairs"]["f1"], 1.0)
        self.assertEqual(gate_decision(summary)["status"], "GO")

    def test_empty_positive_sample_is_excluded_from_partition_exact(self) -> None:
        result = evaluate_frozen_partition(
            clusters=(),
            target_line_ids=np.zeros((0,), dtype=np.int64),
            decoded_segments_pixels=np.zeros((0, 2, 2), dtype=np.float64),
            target_lines=[],
            target_groups=[],
            width=768,
            height=768,
        )
        self.assertFalse(result["positive_partition_eligible"])
        self.assertFalse(result["positive_partition_exact"])
        record = {
            "positive_decoded_lines": 0,
            "predicted_clusters": 0,
            **result,
        }
        summary = summarize_records([record])
        self.assertEqual(summary["positive_partition"]["eligible_samples"], 0)
        self.assertEqual(
            summary["positive_partition"]["excluded_empty_positive_samples"], 1
        )
        self.assertIsNone(summary["positive_partition"]["exact_rate"])


if __name__ == "__main__":
    unittest.main()
