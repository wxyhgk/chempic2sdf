from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import chemvector.audit_evidence_carrier_sample_threshold_reachability as reachability
import chemvector.build_evidence_carrier_sample_threshold_interval_manifest as manifest


def _record(
    target_ids: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
) -> dict:
    segments = [
        ((0.0, float(index * 3)), (10.0, float(index * 3)))
        for index in range(len(target_ids))
    ]
    pair_indices = torch.tensor(pairs, dtype=torch.long)
    if not pairs:
        pair_indices = torch.empty((0, 2), dtype=torch.long)
    targets = torch.tensor(target_ids, dtype=torch.long)
    labels = (
        targets[pair_indices[:, 0]] == targets[pair_indices[:, 1]]
    ).to(torch.long)
    return {
        "identity": {
            "sample_key": "train-fixture",
            "molecule_id": "fixture:1",
            "source_structure_id": "fixture|train|1",
            "source_pilot": "fixture",
            "source_split": "train",
            "source_record": "1",
            "render_variant": "original",
        },
        "anchor_segments_pixels": torch.tensor(segments, dtype=torch.float32),
        "anchor_target_line_ids": targets,
        "anchor_source_indices": torch.arange(len(target_ids), dtype=torch.long),
        "anchor_group_ids": targets.clone(),
        "anchor_group_types": torch.ones(len(target_ids), dtype=torch.long),
        "pair_indices": pair_indices,
        "pair_labels": labels,
    }


class IntervalManifestTest(unittest.TestCase):
    def test_interval_normalization_and_five_percent_inset(self) -> None:
        finite = manifest.robust_loss_interval(1.0, 3.0)
        self.assertEqual(finite["normalization_scale"], 2.0)
        self.assertEqual(finite["loss_lower_threshold"], 1.1)
        self.assertEqual(finite["loss_upper_threshold"], 2.9)

        lower_unbounded = manifest.robust_loss_interval(None, 1.0)
        self.assertEqual(lower_unbounded["lower_threshold"], -math.inf)
        self.assertEqual(lower_unbounded["normalization_scale"], 1.0)
        self.assertEqual(lower_unbounded["loss_lower_threshold"], -math.inf)
        self.assertEqual(lower_unbounded["loss_upper_threshold"], 0.95)

        upper_unbounded = manifest.robust_loss_interval(3.0, None)
        expected_scale = max(1.0, abs(3.0 - manifest.TAU0))
        self.assertEqual(upper_unbounded["normalization_scale"], expected_scale)
        self.assertEqual(
            upper_unbounded["loss_lower_threshold"],
            3.0 + 0.05 * expected_scale,
        )
        self.assertEqual(upper_unbounded["loss_upper_threshold"], math.inf)

        fully_unbounded = manifest.robust_loss_interval(None, None)
        self.assertEqual(fully_unbounded["normalization_scale"], 1.0)
        self.assertEqual(fully_unbounded["loss_lower_threshold"], -math.inf)
        self.assertEqual(fully_unbounded["loss_upper_threshold"], math.inf)

    def test_target_free_pair_tokens_and_global_count_contract(self) -> None:
        record = _record((0, 1, 2, 3), ((2, 3), (0, 1), (1, 2)))
        scores = torch.tensor(
            (manifest.TAU0, manifest.TAU0 + 0.01, manifest.TAU0 + 0.02),
            dtype=torch.float32,
        )
        descriptors = torch.arange(60, dtype=torch.float32).reshape(3, 20)

        result = manifest.build_sample_features(record, scores, descriptors)
        target_free = result["target_free"]

        self.assertEqual(target_free["pair_tokens"].shape, (3, 21))
        torch.testing.assert_close(
            target_free["pair_tokens"][:, :20], descriptors[[1, 2, 0]]
        )
        torch.testing.assert_close(
            target_free["pair_tokens"][:, 20], scores[[1, 2, 0]]
        )
        self.assertEqual(
            target_free["pair_indices"].tolist(),
            [[0, 1], [1, 2], [2, 3]],
        )
        self.assertEqual(
            target_free["canonical_to_source_candidate_indices"].tolist(),
            [1, 2, 0],
        )
        self.assertEqual(
            target_free["source_to_canonical_candidate_indices"].tolist(),
            [2, 0, 1],
        )
        self.assertEqual(
            target_free["raw_global_counts"].tolist(), [4, 3, 2, 3]
        )
        torch.testing.assert_close(
            target_free["global_count_features"],
            torch.log1p(torch.tensor((4, 3, 2, 3), dtype=torch.float32)),
        )
        self.assertNotIn("anchor_target_line_ids", result)
        self.assertEqual(
            manifest.PAIR_TOKEN_FEATURE_NAMES[-1], "raw_pair_gain"
        )
        self.assertEqual(
            manifest.GLOBAL_COUNT_FEATURE_NAMES,
            (
                "log1p_anchor_count",
                "log1p_candidate_pair_count",
                "log1p_gain_above_tau0_count",
                "log1p_gain_above_tau0_incident_unique_anchor_count",
            ),
        )

    def test_candidate_unreachable_is_explicit_and_excluded_from_loss(self) -> None:
        record = _record((0, 0), ())

        result = manifest.build_sample_oracle_label(record, torch.empty(0))

        self.assertFalse(result["candidate_reachable"])
        self.assertEqual(result["candidate_unreachable_target_count"], 1)
        self.assertEqual(result["candidate_unreachable_anchor_pairs"], [[0, 1]])
        self.assertFalse(result["loss_eligible"])
        self.assertEqual(result["exclusion_reason"], "candidate_unreachable")
        self.assertEqual(result["oracle_equivalent_robust_intervals"], [])
        self.assertFalse(result["early_certificate_used"])
        self.assertTrue(result["parametric_hull_fully_enumerated"])

    def test_interval_must_remain_non_degenerate_in_fp32(self) -> None:
        with self.assertRaisesRegex(
            manifest.IntervalManifestError,
            "FP32",
        ):
            manifest.robust_loss_interval(1.00000001, 1.00000002)

    def test_full_hull_emits_real_boundary_without_early_shortcut(self) -> None:
        record = _record((0, 0), ((0, 1),))
        scores = torch.tensor((1.0,), dtype=torch.float32)

        with mock.patch.object(
            reachability,
            "perfect_max_cardinality_open_certificate",
            side_effect=AssertionError("early shortcut must not be called"),
        ):
            result = manifest.build_sample_oracle_label(record, scores)

        self.assertTrue(result["loss_eligible"])
        self.assertIsNone(result["exclusion_reason"])
        self.assertFalse(result["early_certificate_used"])
        self.assertTrue(result["parametric_hull_fully_enumerated"])
        self.assertEqual(len(result["oracle_equivalent_robust_intervals"]), 1)
        interval = result["oracle_equivalent_robust_intervals"][0]
        self.assertEqual(interval["lower_threshold"], -math.inf)
        self.assertEqual(interval["upper_threshold"], 1.0)
        self.assertEqual(interval["normalization_scale"], 1.0)
        self.assertEqual(interval["loss_upper_threshold"], 0.95)
        self.assertEqual(interval["selected_candidate_indices"], [0])
        self.assertTrue(interval["production_certified_open_interval"])
        self.assertTrue(interval["permutation_deterministic"])

    def test_candidate_reachable_rank_failure_is_scalar_unreachable(self) -> None:
        record = _record(
            (0, 0, 1, 1),
            ((0, 1), (2, 3), (0, 2), (1, 3)),
        )
        scores = torch.tensor((2.0, 2.0, 3.0, 3.0), dtype=torch.float32)

        result = manifest.build_sample_oracle_label(record, scores)

        self.assertTrue(result["candidate_reachable"])
        self.assertEqual(result["candidate_unreachable_target_count"], 0)
        self.assertFalse(result["scalar_reachable_target_consistent"])
        self.assertFalse(result["loss_eligible"])
        self.assertEqual(result["exclusion_reason"], "scalar_unreachable")
        self.assertEqual(result["oracle_equivalent_robust_intervals"], [])

    def test_sample_without_m2_target_can_remain_loss_eligible(self) -> None:
        record = _record((0, 1), ())

        result = manifest.build_sample_oracle_label(record, torch.empty(0))

        self.assertEqual(result["multiplicity_2_target_count"], 0)
        self.assertTrue(result["candidate_reachable"])
        self.assertTrue(result["scalar_reachable_target_consistent"])
        self.assertTrue(result["loss_eligible"])
        self.assertEqual(len(result["oracle_equivalent_robust_intervals"]), 1)
        interval = result["oracle_equivalent_robust_intervals"][0]
        self.assertEqual(interval["lower_threshold"], -math.inf)
        self.assertEqual(interval["upper_threshold"], math.inf)

    def test_all_structurally_equivalent_robust_intervals_are_retained(self) -> None:
        record = _record((0, 0, 0, 0), ((0, 1), (2, 3)))
        scores = torch.tensor((2.0, 0.5), dtype=torch.float32)

        result = manifest.build_sample_oracle_label(record, scores)

        intervals = result["oracle_equivalent_robust_intervals"]
        self.assertTrue(result["loss_eligible"])
        self.assertEqual(len(intervals), 2)
        self.assertEqual(
            [(item["lower_threshold"], item["upper_threshold"]) for item in intervals],
            [(-math.inf, 0.5), (0.5, 2.0)],
        )

    def test_worker_payload_uses_compact_numpy_arrays_without_torch_storage(self) -> None:
        record = _record((0, 0, 1), ((0, 1), (1, 2)))
        scores = torch.tensor((1.0, -1.0), dtype=torch.float32)

        compact, compact_scores = manifest._oracle_process_payload(
            record,
            scores,
        )

        self.assertEqual(set(compact), set(manifest._ORACLE_RECORD_FIELDS))
        self.assertTrue(
            all(
                isinstance(value, manifest.np.ndarray)
                for value in compact.values()
            )
        )
        self.assertIsInstance(compact_scores, manifest.np.ndarray)
        self.assertTrue(all(value.flags.c_contiguous for value in compact.values()))
        self.assertTrue(compact_scores.flags.c_contiguous)
        original = manifest.build_sample_oracle_label(record, scores)
        transported = manifest.build_sample_oracle_label(compact, compact_scores)
        self.assertEqual(
            transported["oracle_equivalent_robust_intervals"],
            original["oracle_equivalent_robust_intervals"],
        )

    def test_implementation_commitment_binds_p0_pair_training_dependency(self) -> None:
        commitment = manifest._implementation_sha256()
        dependency = Path(manifest.p0_pair_training.__file__).resolve()

        self.assertEqual(
            commitment["p0_pair_training"],
            manifest._file_sha256(dependency),
        )
        predictor_dependency = Path(
            manifest.threshold_predictor.__file__
        ).resolve()
        self.assertEqual(
            commitment["sample_threshold_predictor"],
            manifest._file_sha256(predictor_dependency),
        )

    def test_cli_is_train_only_and_defaults_to_both_train_partitions(self) -> None:
        parser = manifest.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertEqual(
            destinations,
            {
                "help",
                "base_cache",
                "augmentation_cache",
                "checkpoint",
                "output",
                "summary_output",
                "partitions",
                "limit_per_partition",
                "batch_size",
                "device",
                "workers",
            },
        )
        defaults = parser.parse_args([])
        self.assertEqual(defaults.partitions, ["fit", "calibration"])
        self.assertEqual(defaults.device, "cpu")
        self.assertEqual(defaults.workers, 4)

    def test_build_lock_is_exclusive_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "manifest.pt"
            summary = root / "audit.json"
            args = manifest.build_parser().parse_args(
                [
                    "--output",
                    str(output),
                    "--summary-output",
                    str(summary),
                    "--limit-per-partition",
                    "1",
                ]
            )
            lock_path, descriptor = manifest._acquire_build_lock(
                output, summary, args
            )
            try:
                with self.assertRaises(FileExistsError):
                    manifest._acquire_build_lock(output, summary, args)
                running = lock_path.read_text(encoding="utf-8")
                self.assertIn('"status": "running"', running)
            finally:
                manifest._discard_build_lock(lock_path, descriptor)
            self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
