from __future__ import annotations

import itertools
import unittest

from chemvector.evidence_carrier_pair_matching import (
    EvidenceCarrierPairCandidate,
    EvidenceCarrierPairMatchingValidationError,
    canonical_pair_geometry_key,
    solve_evidence_carrier_pair_matching,
)


def candidate(
    first: int,
    second: int,
    gain: float,
    *,
    geometry: tuple[float, ...] | None = None,
) -> EvidenceCarrierPairCandidate:
    return EvidenceCarrierPairCandidate(
        anchor_ids=(first, second),
        relative_gain=gain,
        canonical_geometry_key=geometry,
    )


def selected_ids(result) -> tuple[tuple[int, int], ...]:
    return tuple(pair.anchor_ids for pair in result.selected_pairs)


class EvidenceCarrierPairMatchingTest(unittest.TestCase):
    def test_maximum_total_gain_beats_greedy_conflicting_pair(self) -> None:
        result = solve_evidence_carrier_pair_matching(
            (0, 1, 2, 3),
            (
                candidate(0, 1, 7.0),
                candidate(0, 2, 4.0),
                candidate(1, 3, 4.0),
            ),
        )

        self.assertTrue(result.feasible)
        self.assertTrue(result.optimal)
        self.assertEqual(result.solver_status, "optimal")
        self.assertEqual(selected_ids(result), ((0, 2), (1, 3)))
        self.assertEqual(result.selected_singletons, ())
        self.assertAlmostEqual(result.total_gain, 8.0)
        self.assertGreater(result.tie_break_solve_count, 0)

    def test_unmatched_active_anchor_is_returned_as_singleton(self) -> None:
        result = solve_evidence_carrier_pair_matching(
            (9, 3, 7),
            (candidate(3, 7, 2.5),),
        )

        self.assertEqual(selected_ids(result), ((3, 7),))
        self.assertEqual(result.selected_singletons, (9,))
        self.assertEqual(result.selected_singleton_anchor_ids, (9,))
        self.assertAlmostEqual(result.total_gain, 2.5)

    def test_zero_and_negative_gains_leave_all_anchors_singleton(self) -> None:
        for gains in ((0.0,), (-1.0,), (0.0, -2.0)):
            pairs = tuple(
                candidate(index, index + 1, gain)
                for index, gain in enumerate(gains)
            )
            active = tuple(range(len(gains) + 1))
            result = solve_evidence_carrier_pair_matching(active, pairs)

            self.assertEqual(result.selected_pairs, ())
            self.assertEqual(result.selected_singletons, active)
            self.assertEqual(result.total_gain, 0.0)
            self.assertEqual(result.solver_status, "optimal")
            self.assertEqual(result.positive_candidate_count, 0)
            self.assertEqual(result.tie_break_solve_count, 0)

    def test_empty_anchor_set_is_an_optimal_empty_partition(self) -> None:
        result = solve_evidence_carrier_pair_matching((), ())

        self.assertEqual(result.selected_pairs, ())
        self.assertEqual(result.selected_singletons, ())
        self.assertTrue(result.optimal)
        self.assertEqual(result.total_gain, 0.0)

    def test_candidate_and_active_input_permutations_cannot_change_result(self) -> None:
        pairs = (
            candidate(0, 1, 3.0),
            candidate(2, 0, 3.0),
            candidate(1, 3, 3.0),
            candidate(3, 2, 3.0),
        )
        expected = None
        for active in ((0, 1, 2, 3), (3, 1, 0, 2)):
            for ordering in itertools.permutations(pairs):
                result = solve_evidence_carrier_pair_matching(active, ordering)
                partition = (selected_ids(result), result.selected_singletons)
                if expected is None:
                    expected = partition
                self.assertEqual(partition, expected)

        self.assertEqual(expected, (((0, 1), (2, 3)), ()))

    def test_equal_gain_tie_prefers_canonical_anchor_ids_without_geometry(self) -> None:
        result = solve_evidence_carrier_pair_matching(
            (1, 2, 3),
            (
                candidate(1, 3, 5.0),
                candidate(1, 2, 5.0),
            ),
        )

        self.assertEqual(selected_ids(result), ((1, 2),))
        self.assertEqual(result.selected_singletons, (3,))

    def test_geometry_key_owns_equal_gain_tie_before_anchor_ids(self) -> None:
        result = solve_evidence_carrier_pair_matching(
            (0, 1, 2),
            (
                candidate(0, 1, 4.0, geometry=(9.0,)),
                candidate(0, 2, 4.0, geometry=(2.0,)),
            ),
        )

        self.assertEqual(selected_ids(result), ((0, 2),))
        self.assertEqual(result.selected_singletons, (1,))

    def test_lexicographic_tie_is_unique_across_equal_partition_sums(self) -> None:
        result = solve_evidence_carrier_pair_matching(
            (0, 1, 2, 3),
            (
                candidate(0, 2, 1.0),
                candidate(1, 3, 1.0),
                candidate(0, 1, 1.0),
                candidate(2, 3, 1.0),
            ),
        )

        self.assertEqual(selected_ids(result), ((0, 1), (2, 3)))
        self.assertEqual(result.total_gain, 2.0)

    def test_small_graph_gain_matches_exhaustive_reference(self) -> None:
        pairs = (
            candidate(0, 1, 2.5),
            candidate(0, 2, 4.0),
            candidate(1, 2, 1.0),
            candidate(1, 3, 4.0),
            candidate(2, 4, 5.5),
            candidate(3, 4, 2.0),
        )
        expected_gain = 0.0
        for selected_count in range(len(pairs) + 1):
            for selection in itertools.combinations(pairs, selected_count):
                members = [
                    anchor_id
                    for pair in selection
                    for anchor_id in pair.anchor_ids
                ]
                if len(members) != len(set(members)):
                    continue
                expected_gain = max(
                    expected_gain,
                    sum(pair.relative_gain for pair in selection),
                )

        result = solve_evidence_carrier_pair_matching(range(5), pairs)

        self.assertAlmostEqual(result.total_gain, expected_gain)
        self.assertEqual(result.total_gain, 9.5)

    def test_canonical_pair_geometry_ignores_segment_and_endpoint_order(self) -> None:
        first = ((10.0, 20.0), (30.0, 40.0))
        second = ((11.0, 21.0), (31.0, 41.0))

        expected = canonical_pair_geometry_key(first, second)
        self.assertEqual(
            canonical_pair_geometry_key(first[::-1], second[::-1]),
            expected,
        )
        self.assertEqual(
            canonical_pair_geometry_key(second, first),
            expected,
        )

    def test_validation_rejects_malformed_matching_contracts(self) -> None:
        with self.assertRaisesRegex(
            EvidenceCarrierPairMatchingValidationError,
            "duplicates",
        ):
            solve_evidence_carrier_pair_matching((0, 0), ())

        with self.assertRaisesRegex(
            EvidenceCarrierPairMatchingValidationError,
            "inactive anchor",
        ):
            solve_evidence_carrier_pair_matching(
                (0, 1),
                (candidate(0, 2, 1.0),),
            )

        with self.assertRaisesRegex(
            EvidenceCarrierPairMatchingValidationError,
            "Duplicate pair",
        ):
            solve_evidence_carrier_pair_matching(
                (0, 1),
                (candidate(0, 1, 1.0), candidate(1, 0, 2.0)),
            )

        with self.assertRaisesRegex(
            EvidenceCarrierPairMatchingValidationError,
            "distinct",
        ):
            candidate(0, 0, 1.0)

        with self.assertRaisesRegex(
            EvidenceCarrierPairMatchingValidationError,
            "finite",
        ):
            candidate(0, 1, float("nan"))

        with self.assertRaisesRegex(
            EvidenceCarrierPairMatchingValidationError,
            "every candidate or none",
        ):
            solve_evidence_carrier_pair_matching(
                (0, 1, 2),
                (
                    candidate(0, 1, 1.0, geometry=(1.0,)),
                    candidate(0, 2, 1.0),
                ),
            )

        with self.assertRaisesRegex(
            EvidenceCarrierPairMatchingValidationError,
            "non-negative",
        ):
            solve_evidence_carrier_pair_matching((), (), score_tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
