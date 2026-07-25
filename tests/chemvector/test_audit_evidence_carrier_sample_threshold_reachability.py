from __future__ import annotations

import itertools
import unittest
from unittest import mock

import numpy as np
import torch

import chemvector.audit_evidence_carrier_sample_threshold_reachability as audit


def _record(
    target_ids: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
) -> dict:
    segments = []
    for index in range(len(target_ids)):
        y = float(index * 3)
        segments.append(((0.0, y), (10.0, y)))
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    if not pairs:
        pair_tensor = torch.empty((0, 2), dtype=torch.long)
    target_tensor = torch.tensor(target_ids, dtype=torch.long)
    labels = (
        target_tensor[pair_tensor[:, 0]] == target_tensor[pair_tensor[:, 1]]
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
        "anchor_target_line_ids": target_tensor,
        "anchor_source_indices": torch.arange(len(target_ids), dtype=torch.long),
        "anchor_group_ids": target_tensor.clone(),
        "anchor_group_types": torch.ones(len(target_ids), dtype=torch.long),
        "pair_indices": pair_tensor,
        "pair_labels": labels,
    }


def _state_set(sample: dict) -> set[tuple[tuple[int, int], ...]]:
    return {
        tuple(tuple(pair) for pair in state["selected_anchor_pairs"])
        for state in sample["reachable_states"]
    }


class SampleThresholdReachabilityAuditTest(unittest.TestCase):
    def test_milp_retries_without_thread_override_after_scheduler_conflict(self) -> None:
        failed = mock.Mock(status=4, x=None)
        solved = mock.Mock(status=0, x=np.asarray([1.0]))
        with mock.patch.object(audit, "milp", side_effect=(failed, solved)) as solver:
            result = audit._solve_milp(
                c=np.asarray([-1.0]),
                integrality=np.asarray([1], dtype=np.uint8),
                bounds=audit.Bounds(0.0, 1.0),
                constraints=audit.LinearConstraint(
                    np.asarray([[1.0]]),
                    np.asarray([-np.inf]),
                    np.asarray([1.0]),
                ),
            )

        self.assertIs(result, solved)
        self.assertEqual(solver.call_count, 2)
        self.assertEqual(
            solver.call_args_list[0].kwargs["options"],
            {"mip_rel_gap": 0.0, "threads": 1},
        )
        self.assertEqual(
            solver.call_args_list[1].kwargs["options"],
            {"mip_rel_gap": 0.0},
        )

    def test_parametric_hull_matches_exhaustive_small_graph_objective(self) -> None:
        record = _record(
            (0, 1, 2, 3, 4, 5),
            ((0, 1), (2, 3), (4, 5), (0, 2), (1, 4), (3, 5)),
        )
        scores = torch.tensor((4.0, 1.0, -2.0, 3.5, 2.25, 2.0))

        hull = audit.discover_parametric_upper_hull(record, scores)
        pairs = record["pair_indices"].tolist()
        raw_scores = scores.tolist()

        def brute(threshold: float) -> float:
            best = 0.0
            for mask in itertools.product((0, 1), repeat=len(pairs)):
                selected = [index for index, value in enumerate(mask) if value]
                anchors = [anchor for index in selected for anchor in pairs[index]]
                if len(set(anchors)) != len(anchors):
                    continue
                value = sum(raw_scores[index] - threshold for index in selected)
                best = max(best, value)
            return best

        for threshold in np.linspace(-8.0, 8.0, num=65):
            discovered = max(state.line_value(float(threshold)) for state in hull)
            self.assertAlmostEqual(discovered, brute(float(threshold)), places=9)
        self.assertEqual(
            hull[0].cardinality,
            audit.maximum_candidate_matching_cardinality(record, scores),
        )

    def test_three_lines_meeting_at_one_point_have_no_robust_middle_state(self) -> None:
        record = _record(
            (0, 1, 2, 3, 4, 5),
            ((0, 1), (2, 3), (4, 5)),
        )
        scores = torch.ones(3)

        hull = audit.discover_parametric_upper_hull(record, scores)
        replay = audit.replay_parametric_open_intervals(record, scores, hull)

        self.assertEqual([state.cardinality for state in hull], [3, 0])
        self.assertEqual(
            {state["cardinality"] for state in replay["states"]},
            {0, 3},
        )
        self.assertNotIn(1, {state["cardinality"] for state in replay["states"]})
        self.assertNotIn(2, {state["cardinality"] for state in replay["states"]})

    def test_open_interval_requires_three_consistent_production_witnesses(self) -> None:
        record = _record((0, 0), ((0, 1),))
        scores = torch.tensor((1.0,))

        hull = audit.discover_parametric_upper_hull(record, scores)
        replay = audit.replay_parametric_open_intervals(record, scores, hull)

        self.assertTrue(replay["interval_certifications"])
        for interval in replay["interval_certifications"]:
            self.assertEqual(len(interval["production_states"]), 3)
            self.assertTrue(interval["production_state_consistent"])
            self.assertTrue(interval["production_certified_open_interval"])

    def test_empty_candidate_sample_has_one_robust_empty_state(self) -> None:
        record = _record((0, 1), ())

        sample = audit.audit_sample_threshold_reachability(
            record,
            torch.empty((0,), dtype=torch.float32),
        )

        self.assertEqual(len(sample["fixed_cardinality_frontier"]), 1)
        self.assertEqual(sample["fixed_cardinality_frontier"][0]["cardinality"], 0)
        self.assertEqual(_state_set(sample), {()})
        self.assertEqual(sample["robust_open_interval_state_count"], 1)
        self.assertEqual(sample["oracle_evaluation"]["selected_pair_count"], 0)

    def test_negative_score_boundary_is_open_on_the_selected_side(self) -> None:
        record = _record((0, 0), ((0, 1),))
        score = -2.0

        sample = audit.audit_sample_threshold_reachability(
            record,
            torch.tensor((score,), dtype=torch.float32),
        )

        certification = next(
            item
            for item in sample["boundary_certifications"]
            if item["threshold"] == score
        )
        self.assertEqual(certification["left_nextafter_state"], [[0, 1]])
        self.assertEqual(certification["exact_state"], [])
        self.assertEqual(certification["right_nextafter_state"], [])
        self.assertTrue(sample["scalar_reachable_pairable_exact"])

    def test_cross_cardinality_boundary_changes_matching(self) -> None:
        record = _record(
            (0, 0, 1, 1),
            ((0, 1), (2, 3), (0, 2)),
        )
        scores = torch.tensor((3.0, 3.0, 5.0))

        sample = audit.audit_sample_threshold_reachability(record, scores)

        states = _state_set(sample)
        self.assertIn(((0, 1), (2, 3)), states)
        self.assertIn(((0, 2),), states)
        self.assertIn((), states)
        self.assertFalse(sample["same_cardinality_rank_failure"])
        self.assertTrue(sample["scalar_reachable_pairable_exact"])
        self.assertTrue(sample["oracle_evaluation"]["pairable_sample_exact"])
        certification = next(
            item
            for item in sample["boundary_certifications"]
            if item["threshold"] == 1.0
        )
        # The production solver's 64-ulp objective tolerance may keep the
        # same state at both immediate floating-point neighbours.  Those
        # probes are diagnostic only; robust midpoint witnesses above prove
        # that the two open regions are both reachable.
        self.assertIsNotNone(certification["left_nextafter_state"])
        self.assertIsNotNone(certification["right_nextafter_state"])
        self.assertEqual(
            certification["left_nextafter_state"],
            certification["right_nextafter_state"],
        )
        robust_states = {
            tuple(tuple(pair) for pair in state["selected_anchor_pairs"])
            for state in sample["reachable_states"]
            if state["production_certified_open_interval"]
        }
        self.assertIn(((0, 1), (2, 3)), robust_states)
        self.assertIn(((0, 2),), robust_states)
        self.assertTrue(
            certification["exact_matches_left_nextafter"]
            or certification["exact_matches_right_nextafter"]
        )

    def test_same_cardinality_rank_failure_is_not_scalar_reachable(self) -> None:
        record = _record(
            (0, 0, 1, 1),
            ((0, 1), (2, 3), (0, 2), (1, 3)),
        )
        scores = torch.tensor((2.0, 2.0, 3.0, 3.0))

        sample = audit.audit_sample_threshold_reachability(record, scores)

        self.assertTrue(sample["same_cardinality_rank_failure"])
        self.assertFalse(sample["scalar_reachable_pairable_exact"])
        fixed_two = next(
            item
            for item in sample["fixed_cardinality_frontier"]
            if item["cardinality"] == 2
        )
        self.assertEqual(
            {tuple(pair) for pair in fixed_two["selected_anchor_pairs"]},
            {(0, 2), (1, 3)},
        )

    def test_same_cardinality_raw_score_tie_is_canonical_and_order_invariant(self) -> None:
        first = _record((0, 1, 2), ((0, 2), (0, 1)))
        second = _record((0, 1, 2), ((0, 1), (0, 2)))
        scores = torch.tensor((1.0, 1.0))

        left = audit.solve_fixed_cardinality_matching(first, scores, 1)
        right = audit.solve_fixed_cardinality_matching(
            second,
            torch.flip(scores, dims=(0,)),
            1,
        )

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertEqual(left.selected_anchor_pairs, right.selected_anchor_pairs)
        self.assertIn("optimal", left.solver_status)

    def test_64_ulp_tolerance_prefers_canonical_earlier_slightly_lower_edge(self) -> None:
        record = _record((0, 1, 2), ((0, 1), (0, 2)))
        scores = np.asarray(
            (np.nextafter(1.0, 0.0), 1.0),
            dtype=np.float64,
        )

        fixed = audit.solve_fixed_cardinality_matching(record, scores, 1)
        production = audit.evaluate_evidence_carrier_pair_sample(
            record,
            scores,
            threshold=0.0,
            verify_permutation=True,
        )

        self.assertIsNotNone(fixed)
        self.assertEqual(fixed.selected_anchor_pairs, ((0, 1),))
        self.assertEqual(production["selected_anchor_pairs"], [[0, 1]])
        self.assertTrue(production["permutation_deterministic"])

    def test_positive_scaling_can_change_production_tie_state(self) -> None:
        record = _record((0, 1, 2), ((0, 1), (0, 2)))
        scores = np.asarray((1e-15, 2e-15), dtype=np.float64)

        base = audit.evaluate_evidence_carrier_pair_sample(
            record,
            scores,
            threshold=0.0,
            verify_permutation=True,
        )
        scaled = audit.evaluate_evidence_carrier_pair_sample(
            record,
            scores * 1000.0,
            threshold=0.0,
            verify_permutation=True,
        )

        self.assertEqual(base["selected_anchor_pairs"], [[0, 1]])
        self.assertEqual(scaled["selected_anchor_pairs"], [[0, 2]])

    def test_perfect_state_uses_conservative_open_interval_early_certificate(self) -> None:
        record = _record((0, 0), ((0, 1),))
        scores = torch.tensor((1.0,))

        early = audit.audit_sample_threshold_reachability(
            record,
            scores,
            detail_policy="none",
        )
        full = audit.audit_sample_threshold_reachability(
            record,
            scores,
            detail_policy="all",
        )

        self.assertTrue(early["early_certificate"])
        self.assertTrue(early["early_certificate_not_training_interval"])
        self.assertFalse(early["parametric_hull_fully_enumerated"])
        self.assertFalse(full["early_certificate"])
        self.assertFalse(full["early_certificate_not_training_interval"])
        self.assertEqual(
            early["oracle_evaluation"]["selected_anchor_pairs"],
            full["oracle_evaluation"]["selected_anchor_pairs"],
        )
        interval = early["oracle_interval"]
        lower = interval["lower_threshold"]
        upper = interval["upper_threshold"]
        thresholds = interval["thresholds"]
        self.assertEqual(len(thresholds), 3)
        self.assertTrue(all(lower < threshold < upper for threshold in thresholds))
        self.assertEqual(
            thresholds,
            [lower + (upper - lower) * fraction for fraction in (0.25, 0.5, 0.75)],
        )
        self.assertIn(early["oracle_threshold"], thresholds)
        self.assertTrue(interval["early_certificate_not_training_interval"])

    def test_failure_detail_indices_are_capped_after_formal_samples(self) -> None:
        record = _record((0, 0), ((0, 1),))
        success = audit.audit_sample_threshold_reachability(
            record,
            torch.tensor((1.0,)),
            detail_policy="none",
        )
        first_failure = dict(success)
        first_failure["same_cardinality_rank_failure"] = True
        second_failure = dict(first_failure)

        indices, truncated = audit._detailed_frontier_indices(
            (success, first_failure, second_failure),
            detail_policy="failures",
            maximum=1,
        )

        self.assertEqual(indices, (1,))
        self.assertEqual(truncated, 1)
        self.assertEqual(
            audit._detailed_frontier_indices(
                (success, first_failure, second_failure),
                detail_policy="failures",
                maximum=0,
            ),
            ((), 2),
        )
        self.assertEqual(
            audit._detailed_frontier_indices(
                (success, first_failure, second_failure),
                detail_policy="none",
                maximum=10,
            ),
            ((), 0),
        )

    def test_auto_device_is_cpu_for_safe_parallel_audit(self) -> None:
        self.assertEqual(audit._device("auto"), torch.device("cpu"))

    def test_matching_latency_is_diagnostic_for_scalar_capability_gate(self) -> None:
        base = {
            "status": "NO-GO",
            "checks": {
                "structural_check": True,
                "matching_p95_below_20ms": False,
            },
            "thresholds": {
                "structural_threshold": 1.0,
                "maximum_matching_p95_ms": 20.0,
            },
        }

        result = audit._scalar_oracle_capability_gate(base)

        self.assertEqual(result["status"], "GO")
        self.assertEqual(result["checks"], {"structural_check": True})
        self.assertEqual(
            result["diagnostic_checks"],
            {"matching_p95_below_20ms": False},
        )
        self.assertEqual(result["base_pair_scorer_gate_status"], "NO-GO")

    def test_w1_w2_degenerate_boundary_is_reported_but_open_states_drive_oracle(self) -> None:
        record = _record(
            (0, 0, 1, 1),
            ((0, 1), (2, 3), (0, 2)),
        )
        # W1=3 and W2=4 meet at threshold 1.  Production behavior at exactly
        # one is reported independently from the two adjacent open intervals.
        scores = torch.tensor((2.0, 2.0, 3.0))

        sample = audit.audit_sample_threshold_reachability(record, scores)

        certification = next(
            item
            for item in sample["boundary_certifications"]
            if item["threshold"] == 1.0
        )
        self.assertIsNotNone(certification["left_nextafter_state"])
        self.assertIsNotNone(certification["right_nextafter_state"])
        oracle_state = tuple(
            tuple(pair) for pair in sample["oracle_evaluation"]["selected_anchor_pairs"]
        )
        robust_states = {
            tuple(tuple(pair) for pair in state["selected_anchor_pairs"])
            for state in sample["reachable_states"]
            if state["production_certified_open_interval"]
        }
        self.assertIn(oracle_state, robust_states)

    def test_probe_protocol_contains_exact_nextafter_and_open_midpoint(self) -> None:
        record = _record((0, 0), ((0, 1),))
        scores = torch.tensor((1.0,))
        frontier = audit.fixed_cardinality_frontier(record, scores)
        envelope = audit.cardinality_envelope(frontier, scores)

        probes = audit.threshold_probe_values(scores, envelope)
        by_value = {item["threshold"]: item for item in probes}

        self.assertIn(1.0, by_value)
        self.assertIn("boundary", by_value[1.0]["kinds"])
        self.assertIn(float(np.nextafter(1.0, -np.inf)), by_value)
        self.assertIn(float(np.nextafter(1.0, np.inf)), by_value)
        self.assertTrue(any(item["robust_open_interval"] for item in probes))

    def test_forbidden_paths_are_rejected(self) -> None:
        for value in (
            "/tmp/fresh_dev2/cache.pt",
            "/tmp/validation/cache.pt",
            "/tmp/sealed/cache.pt",
            "/tmp/internal-test/cache.pt",
            "/tmp/MR-TADF/cache.pt",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit._reject_forbidden_path(value, label="fixture")

    def test_cli_has_no_fresh_or_validation_input(self) -> None:
        parser = audit.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertEqual(
            destinations,
            {
                "help",
                "base_cache",
                "augmentation_cache",
                "checkpoint",
                "output",
                "partitions",
                "limit_per_partition",
                "batch_size",
                "device",
                "workers",
                "detail_policy",
                "max_detailed_frontiers",
            },
        )
        defaults = parser.parse_args([])
        self.assertEqual(defaults.partitions, ["calibration"])
        self.assertEqual(defaults.device, "cpu")
        self.assertEqual(defaults.workers, 4)

    def test_implementation_hashes_bind_all_production_components(self) -> None:
        hashes = audit._implementation_sha256()

        self.assertEqual(
            set(hashes),
            {"audit", "matching", "evaluation", "p0_pair_scorer"},
        )
        self.assertTrue(
            all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in hashes.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
