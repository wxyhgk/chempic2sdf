from __future__ import annotations

import unittest

import numpy as np
import torch

from chemvector.bond_group_set import PARALLEL_2, SINGLE_LINE
from chemvector.evidence_carrier_pair_evaluation import (
    SINGLETON_GAIN,
    aggregate_evidence_carrier_pair_evaluations,
    calibrate_evidence_carrier_pair_threshold,
    evaluate_evidence_carrier_pair_sample,
    score_evidence_carrier_pair_record,
    summarize_evidence_carrier_pair_records,
)
from chemvector.evidence_carrier_conditional_pair_data import (
    ConditionalPairSampleIdentity,
)
from chemvector.evidence_carrier_pair_cache import (
    compact_evidence_carrier_pair_record,
)


def pair_record(
    target_ids: list[int],
    pairs: list[tuple[int, int]],
    *,
    group_ids: list[int] | None = None,
    group_types: list[int] | None = None,
) -> dict:
    anchor_count = len(target_ids)
    group_ids = group_ids or list(target_ids)
    group_types = group_types or [SINGLE_LINE] * anchor_count
    segments = np.asarray(
        [((10.0, 10.0 + index * 2), (90.0, 10.0 + index * 2)) for index in range(anchor_count)],
        dtype=np.float32,
    )
    if not anchor_count:
        segments = np.empty((0, 2, 2), dtype=np.float32)
    pair_array = np.asarray(pairs, dtype=np.int64)
    if not pairs:
        pair_array = np.empty((0, 2), dtype=np.int64)
    return {
        "anchor_segments_pixels": torch.tensor(segments),
        "anchor_target_line_ids": torch.tensor(target_ids, dtype=torch.long),
        "anchor_source_indices": torch.arange(anchor_count),
        "pair_indices": torch.tensor(pair_array, dtype=torch.long),
        "raw_keep_labels": torch.ones(anchor_count, dtype=torch.long),
        "raw_target_line_ids": torch.tensor(target_ids, dtype=torch.long),
        "raw_group_ids": torch.tensor(group_ids, dtype=torch.long),
        "raw_group_types": torch.tensor(group_types, dtype=torch.long),
    }


class EvidenceCarrierPairEvaluationTest(unittest.TestCase):
    def test_compact_cache_record_uses_anchor_group_protocol_without_raw_fields(self) -> None:
        source = pair_record(
            [0, 1],
            [(0, 1)],
            group_ids=[5, 5],
            group_types=[PARALLEL_2, PARALLEL_2],
        )
        target = {
            **source,
            "anchor_segments": source["anchor_segments_pixels"] / torch.tensor(
                (99.0, 99.0)
            ),
            "pair_features": torch.zeros((1, 8), dtype=torch.float32),
            "pair_labels": torch.tensor([0], dtype=torch.long),
            "pair_target_line_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "pair_overlap_centerline": torch.zeros((1, 2, 2)),
            "pair_overlap_centerline_pixels": torch.zeros((1, 2, 2)),
        }
        identity = ConditionalPairSampleIdentity(
            sample_key="cache-sample",
            molecule_id="mol-cache",
            source_structure_id="pilot|train|cache-source",
            source_pilot="pilot",
            source_split="train",
            source_record="cache-source",
            render_variant="original",
        )
        record = compact_evidence_carrier_pair_record(
            identity,
            target,
            partition="calibration",
            canvas_size=(100, 100),
        )

        result = score_evidence_carrier_pair_record(
            record,
            [4.0],
            0.0,
            verify_permutation=True,
        )

        self.assertEqual(result["group_protocol_source"], "anchor_cache")
        self.assertEqual(result["selected_cross_target_pairs"], 1)
        self.assertEqual(result["parallel_2_group_count"], 1)
        self.assertEqual(result["parallel_2_distinct_preservation"], 0.0)

    def test_empty_sample_is_optimal_without_vacuous_exact_metrics(self) -> None:
        result = evaluate_evidence_carrier_pair_sample(
            pair_record([], []),
            [],
        )

        self.assertEqual(result["selected_pair_count"], 0)
        self.assertEqual(result["selected_singleton_count"], 0)
        self.assertIsNone(result["selected_pair_precision"])
        self.assertIsNone(result["multiplicity_2_recall"])
        self.assertIsNone(result["parallel_2_distinct_preservation"])
        self.assertIsNone(result["ordinary_support_loss_pp"])
        self.assertFalse(result["pairable_sample_eligible"])
        self.assertIsNone(result["pairable_sample_exact"])
        self.assertTrue(result["matching_optimal"])
        self.assertTrue(result["permutation_deterministic"])

    def test_conflicting_matching_uses_global_gain_and_reports_cross_pairs(self) -> None:
        record = pair_record(
            [0, 0, 1, 1],
            [(0, 1), (0, 2), (1, 3)],
            group_ids=[10, 10, 11, 11],
        )

        result = evaluate_evidence_carrier_pair_sample(
            record,
            [3.0, 2.0, 2.0],
        )

        self.assertEqual(result["selected_anchor_pairs"], [[0, 2], [1, 3]])
        self.assertEqual(result["selected_same_target_pairs"], 0)
        self.assertEqual(result["selected_cross_target_pairs"], 2)
        self.assertEqual(result["selected_pair_precision"], 0.0)
        self.assertEqual(result["multiplicity_2_recall"], 0.0)
        self.assertEqual(result["ordinary_support_loss_pp"], 100.0)
        self.assertFalse(result["pairable_sample_eligible"])
        self.assertIsNone(result["pairable_sample_exact"])
        self.assertTrue(result["permutation_deterministic"])

    def test_parallel_two_member_merge_is_a_distinct_preservation_failure(self) -> None:
        record = pair_record(
            [0, 1],
            [(0, 1)],
            group_ids=[5, 5],
            group_types=[PARALLEL_2, PARALLEL_2],
        )

        result = evaluate_evidence_carrier_pair_sample(record, [4.0])

        self.assertEqual(result["selected_cross_target_pairs"], 1)
        self.assertEqual(result["parallel_2_group_count"], 1)
        self.assertEqual(result["parallel_2_violated_group_count"], 1)
        self.assertEqual(result["parallel_2_distinct_preservation"], 0.0)
        self.assertEqual(result["ordinary_support_loss_pp"], 100.0)

    def test_candidate_permutation_cannot_change_selected_partition(self) -> None:
        record = pair_record(
            [0, 0, 1, 1],
            [(0, 1), (0, 2), (1, 3), (2, 3)],
            group_ids=[10, 10, 11, 11],
        )
        baseline = evaluate_evidence_carrier_pair_sample(
            record,
            [1.0, 1.0, 1.0, 1.0],
        )
        permutation = [3, 1, 0, 2]
        permuted_record = {
            **record,
            "pair_indices": record["pair_indices"][permutation],
        }
        permuted = evaluate_evidence_carrier_pair_sample(
            permuted_record,
            torch.tensor([1.0] * 4)[permutation],
        )

        self.assertEqual(
            baseline["selected_anchor_pairs"], permuted["selected_anchor_pairs"]
        )
        self.assertTrue(baseline["permutation_deterministic"])
        self.assertTrue(permuted["permutation_deterministic"])

    def test_aggregate_is_count_weighted_and_reports_latency(self) -> None:
        same = evaluate_evidence_carrier_pair_sample(
            pair_record([0, 0], [(0, 1)]),
            [2.0],
        )
        none = evaluate_evidence_carrier_pair_sample(
            pair_record([1], []),
            [],
        )

        aggregate = aggregate_evidence_carrier_pair_evaluations([same, none])

        self.assertEqual(aggregate["sample_count"], 2)
        self.assertEqual(aggregate["selected_pair_precision"], 1.0)
        self.assertEqual(aggregate["multiplicity_2_recall"], 1.0)
        self.assertEqual(aggregate["pairable_sample_exact"], 1.0)
        self.assertEqual(aggregate["ordinary_support_loss_pp"], 0.0)
        self.assertTrue(aggregate["permutation_deterministic"])
        self.assertGreaterEqual(aggregate["matching_latency_ms"]["max"], 0.0)

    def test_train_cal_threshold_obeys_hard_constraints_before_recall(self) -> None:
        ordinary = pair_record(
            [0, 0, 1],
            [(0, 1), (1, 2)],
            group_ids=[10, 10, 11],
        )
        parallel_two = pair_record(
            [2, 3],
            [(0, 1)],
            group_ids=[20, 20],
            group_types=[PARALLEL_2, PARALLEL_2],
        )

        calibrated = calibrate_evidence_carrier_pair_threshold(
            [ordinary, parallel_two],
            [[0.9, 0.8], [0.7]],
        )

        expected = 0.8 / 2.0 + 0.9 / 2.0
        self.assertEqual(calibrated["threshold_source_score"], expected)
        self.assertEqual(calibrated["threshold"], expected)
        self.assertEqual(
            calibrated["threshold_rule"],
            "midpoint(adjacent_selected_scores)",
        )
        self.assertFalse(calibrated["threshold_strictly_above_source_score"])
        self.assertFalse(calibrated["threshold_strictly_below_source_score"])
        self.assertEqual(calibrated["singleton_gain"], SINGLETON_GAIN)
        selected = calibrated["selected_metrics"]
        self.assertEqual(selected["selected_cross_target_pairs"], 0)
        self.assertEqual(selected["parallel_2_violated_group_count"], 0)
        self.assertEqual(selected["multiplicity_2_recall"], 1.0)
        self.assertEqual(selected["pairable_sample_exact"], 1.0)
        self.assertTrue(selected["permutation_deterministic"])

    def test_threshold_cap_and_stable_trainer_aliases(self) -> None:
        record = pair_record([0, 0], [(0, 1)])
        scored = score_evidence_carrier_pair_record(
            record,
            [0.9],
            0.8,
            verify_permutation=True,
        )
        summary = summarize_evidence_carrier_pair_records([scored])
        calibrated = calibrate_evidence_carrier_pair_threshold(
            [record, pair_record([1, 1], [(0, 1)])],
            [[0.9], [0.7]],
            max_threshold_candidates=5,
        )

        self.assertEqual(scored["selected_same_target_pairs"], 1)
        self.assertEqual(summary["multiplicity_2_recall"], 1.0)
        self.assertEqual(calibrated["candidate_threshold_count"], 5)
        self.assertEqual(calibrated["total_unique_calibration_score_count"], 2)
        self.assertEqual(calibrated["max_threshold_candidates"], 5)
        self.assertTrue(
            calibrated["threshold_candidate_protocol"][
                "all_mandatory_scores_retained"
            ]
        )

    def test_threshold_cap_always_retains_max_negative_score(self) -> None:
        first = pair_record(
            [0, 0, 1],
            [(0, 1), (0, 2), (1, 2)],
            group_ids=[10, 10, 11],
        )
        second = pair_record(
            [2, 2, 3],
            [(0, 1), (0, 2), (1, 2)],
            group_ids=[20, 20, 21],
        )

        calibrated = calibrate_evidence_carrier_pair_threshold(
            [first, second],
            [[0.9, 0.4, 0.1], [0.8, 0.3, 0.2]],
            max_threshold_candidates=7,
        )

        protocol = calibrated["threshold_candidate_protocol"]
        self.assertEqual(protocol["max_negative_score"], 0.4)
        self.assertIn(0.4, protocol["mandatory_source_scores"])
        self.assertTrue(protocol["all_mandatory_scores_retained"])
        self.assertLessEqual(calibrated["candidate_threshold_count"], 7)

    def test_interval_candidates_recover_safe_higher_cardinality_matching(self) -> None:
        record = pair_record(
            [0, 0, 1, 1, 2, 2],
            [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)],
        )

        calibrated = calibrate_evidence_carrier_pair_threshold(
            [record],
            [[0.41, 0.41, 0.41, 0.60, 0.60]],
        )

        self.assertEqual(calibrated["threshold"], 0.0)
        self.assertEqual(calibrated["threshold_rule"], "fixed_zero")
        selected = calibrated["selected_metrics"]
        self.assertEqual(selected["selected_cross_target_pairs"], 0)
        self.assertEqual(selected["multiplicity_2_recall"], 1.0)
        self.assertEqual(selected["pairable_sample_exact"], 1.0)
        protocol = calibrated["threshold_candidate_protocol"]
        self.assertTrue(protocol["bounded_candidate_search"])
        self.assertFalse(protocol["mathematically_complete_parametric_matching_search"])
        self.assertGreaterEqual(protocol["selected_interval_midpoint_count"], 1)

    def test_threshold_cap_rejects_missing_mandatory_interval_candidates(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "mandatory safe threshold candidate count 5",
        ):
            calibrate_evidence_carrier_pair_threshold(
                [pair_record([0, 0], [(0, 1)]), pair_record([1, 1], [(0, 1)])],
                [[0.9], [0.7]],
                max_threshold_candidates=4,
            )

    def test_single_positive_pair_uses_mandatory_lower_bound_threshold(self) -> None:
        record = pair_record([0, 0], [(0, 1)])

        calibrated = calibrate_evidence_carrier_pair_threshold(
            [record],
            [[0.9]],
        )

        expected = float(np.nextafter(0.9, -np.inf))
        self.assertEqual(calibrated["threshold"], expected)
        self.assertEqual(calibrated["threshold_source_score"], 0.9)
        self.assertEqual(calibrated["threshold_rule"], "nextafter(global_min,-inf)")
        self.assertTrue(calibrated["threshold_strictly_below_source_score"])
        self.assertFalse(calibrated["threshold_strictly_above_source_score"])
        self.assertEqual(calibrated["selected_metrics"]["multiplicity_2_recall"], 1.0)
        lower_rows = [
            row
            for row in calibrated["threshold_trace"]
            if row["rule"] == "nextafter(global_min,-inf)"
        ]
        self.assertEqual(len(lower_rows), 1)
        self.assertEqual(lower_rows[0]["threshold"], expected)


if __name__ == "__main__":
    unittest.main()
