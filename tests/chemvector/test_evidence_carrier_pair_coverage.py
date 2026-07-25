from __future__ import annotations

import unittest

import numpy as np

from chemvector.evidence_carrier_pair_coverage import (
    EvidenceCarrierPairCoverageConfig,
    apply_synergistic_pair_coverage_veto,
)


def line(x0: float, x1: float, y: float) -> list[list[float]]:
    return [[x0, y], [x1, y]]


class EvidenceCarrierPairCoverageTest(unittest.TestCase):
    def test_angle_dot_tolerance_is_explicit_and_nonnegative(self) -> None:
        config = EvidenceCarrierPairCoverageConfig()
        self.assertEqual(config.angle_dot_tolerance, 1e-7)
        with self.assertRaisesRegex(ValueError, "angle_dot_tolerance"):
            EvidenceCarrierPairCoverageConfig(angle_dot_tolerance=-1e-7)

    def test_joint_stronger_pairs_veto_a_bridge_pair(self) -> None:
        pairs = np.asarray(((0, 1), (2, 3), (4, 5)), dtype=np.int64)
        decoded = np.asarray(
            (
                line(0, 5, 0),
                line(0, 5, 1),
                line(5, 10, 0),
                line(5, 10, 1),
                line(0, 10, 0),
                line(0, 10, 1),
            ),
            dtype=np.float64,
        )

        result = apply_synergistic_pair_coverage_veto(
            (0, 1, 2),
            np.asarray((5.0, 4.0, 3.0)),
            pairs,
            decoded,
            config=EvidenceCarrierPairCoverageConfig(tube_radius_pixels=0.1),
        )

        self.assertEqual(result["vetoed_candidate_indices"], [2])
        self.assertEqual(result["kept_candidate_indices"], [0, 1])
        self.assertEqual(result["veto_diagnostics"][0]["contributor_count"], 2)
        self.assertAlmostEqual(
            result["veto_diagnostics"][0]["union_common_coverage"], 1.0
        )

    def test_one_stronger_pair_cannot_trigger_synergy_veto(self) -> None:
        pairs = np.asarray(((0, 1), (2, 3)), dtype=np.int64)
        decoded = np.asarray(
            (
                line(0, 10, 0),
                line(0, 10, 1),
                line(0, 10, 0),
                line(0, 10, 1),
            ),
            dtype=np.float64,
        )
        result = apply_synergistic_pair_coverage_veto(
            (0, 1), np.asarray((5.0, 3.0)), pairs, decoded
        )
        self.assertEqual(result["vetoed_candidate_indices"], [])

    def test_a_single_pair_that_explains_both_lines_blocks_veto(self) -> None:
        pairs = np.asarray(((0, 1), (2, 3), (4, 5)), dtype=np.int64)
        decoded = np.asarray(
            (
                line(0, 10, 0),
                line(0, 10, 1),
                line(0, 5, 0),
                line(0, 5, 1),
                line(0, 10, 0),
                line(0, 10, 1),
            ),
            dtype=np.float64,
        )
        result = apply_synergistic_pair_coverage_veto(
            (0, 1, 2), np.asarray((5.0, 4.0, 3.0)), pairs, decoded
        )
        self.assertEqual(result["vetoed_candidate_indices"], [])

    def test_lower_scoring_pairs_do_not_explain_a_stronger_pair(self) -> None:
        pairs = np.asarray(((0, 1), (2, 3), (4, 5)), dtype=np.int64)
        decoded = np.asarray(
            (
                line(0, 10, 0),
                line(0, 10, 1),
                line(0, 5, 0),
                line(0, 5, 1),
                line(5, 10, 0),
                line(5, 10, 1),
            ),
            dtype=np.float64,
        )
        result = apply_synergistic_pair_coverage_veto(
            (0, 1, 2), np.asarray((5.0, 4.0, 3.0)), pairs, decoded
        )
        self.assertNotIn(0, result["vetoed_candidate_indices"])

    def test_discrete_sample_union_is_not_pruned_by_continuous_gaps(self) -> None:
        segments = []
        for index in range(9):
            start = 70 * index
            segments.extend(
                (line(start, start + 60, 0), line(start, start + 60, 10))
            )
        segments.extend((line(630, 640, 0), line(630, 640, 10)))
        segments.extend((line(0, 640, 0), line(0, 640, 10)))
        pairs = np.arange(22, dtype=np.int64).reshape((-1, 2))

        result = apply_synergistic_pair_coverage_veto(
            tuple(range(11)),
            np.arange(11, 0, -1, dtype=np.float64),
            pairs,
            np.asarray(segments, dtype=np.float64),
        )

        self.assertEqual(result["vetoed_candidate_indices"], [10])
        self.assertAlmostEqual(
            result["veto_diagnostics"][0]["union_common_coverage"], 1.0
        )

    def test_contributor_fraction_uses_closed_ten_percent_boundary(self) -> None:
        pairs = np.arange(6, dtype=np.int64).reshape((-1, 2))
        scores = np.asarray((3.0, 2.0, 1.0))
        exact_boundary = np.asarray(
            (
                line(0, 570, 0),
                line(0, 570, 10),
                line(580, 640, 0),
                line(590, 640, 10),
                line(0, 640, 0),
                line(0, 640, 10),
            ),
            dtype=np.float64,
        )
        below_boundary = exact_boundary.copy()
        below_boundary[2] = np.asarray(line(590, 640, 0))

        exact = apply_synergistic_pair_coverage_veto(
            (0, 1, 2), scores, pairs, exact_boundary
        )
        below = apply_synergistic_pair_coverage_veto(
            (0, 1, 2), scores, pairs, below_boundary
        )

        self.assertEqual(exact["vetoed_candidate_indices"], [2])
        self.assertEqual(exact["veto_diagnostics"][0]["contributor_count"], 2)
        self.assertEqual(below["vetoed_candidate_indices"], [])

    def test_candidate_permutation_preserves_anchor_pair_decisions(self) -> None:
        pairs = np.asarray(((0, 1), (2, 3), (4, 5)), dtype=np.int64)
        scores = np.asarray((5.0, 4.0, 3.0))
        decoded = np.asarray(
            (
                line(0, 5, 0),
                line(0, 5, 1),
                line(5, 10, 0),
                line(5, 10, 1),
                line(0, 10, 0),
                line(0, 10, 1),
            ),
            dtype=np.float64,
        )
        direct = apply_synergistic_pair_coverage_veto(
            (0, 1, 2), scores, pairs, decoded
        )
        order = np.asarray((2, 0, 1))
        permuted = apply_synergistic_pair_coverage_veto(
            (0, 1, 2), scores[order], pairs[order], decoded
        )

        direct_veto = {
            tuple(pairs[index]) for index in direct["vetoed_candidate_indices"]
        }
        permuted_veto = {
            tuple(pairs[order][index])
            for index in permuted["vetoed_candidate_indices"]
        }
        self.assertEqual(direct_veto, permuted_veto)

    def test_selected_pairs_must_be_anchor_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchor-disjoint"):
            apply_synergistic_pair_coverage_veto(
                (0, 1),
                np.asarray((2.0, 1.0)),
                np.asarray(((0, 1), (1, 2))),
                np.asarray((line(0, 1, 0), line(0, 1, 1), line(0, 1, 2))),
            )


if __name__ == "__main__":
    unittest.main()
