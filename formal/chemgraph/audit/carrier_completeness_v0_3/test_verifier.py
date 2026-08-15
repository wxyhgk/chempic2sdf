from __future__ import annotations

import json
from pathlib import Path
import unittest

from formal.chemgraph.audit.carrier_candidate_set_v0_3._common import (
    canonical_json_bytes,
)
from formal.chemgraph.audit.carrier_candidate_set_v0_3.validate_candidate_set import (
    validate_candidate_set_bytes,
)

from .verify_boundary_completeness import (
    compare_boundary_matchings,
    enumerate_boundary_matchings,
    root_projection,
    verify_candidate_bytes,
)


ROOT = Path(__file__).resolve().parent
COUNTEREXAMPLES = ROOT / "counterexamples"
STRUCTURAL_FIXTURES = ROOT.parent / "carrier_candidate_set_v0_3" / "fixtures"
INPUT_RAW = (COUNTEREXAMPLES / "input.json").read_bytes()
FULL_RAW = (COUNTEREXAMPLES / "full_collision.json").read_bytes()
OMITTED_RAW = (COUNTEREXAMPLES / "omitted_alternative.json").read_bytes()
PAIRED_ONLY_RAW = (COUNTEREXAMPLES / "paired_only_matching.json").read_bytes()


def _boundaries(count: int) -> list[dict]:
    return [{"id": index} for index in range(count)]


def _pairs(*members: tuple[int, int]) -> list[dict]:
    return [
        {"boundary_ids": list(pair), "id": index}
        for index, pair in enumerate(members)
    ]


class MatchingEnumeratorTests(unittest.TestCase):
    def test_no_compatible_pairs_has_only_all_unmatched(self) -> None:
        result = enumerate_boundary_matchings(_boundaries(2), [])
        self.assertFalse(result.overflow)
        self.assertEqual(list(result.records), [{
            "compatible_pair_ids": [],
            "matching_index": 0,
            "unmatched_boundary_ids": [0, 1],
        }])

    def test_one_pair_has_paired_and_all_unmatched(self) -> None:
        result = enumerate_boundary_matchings(_boundaries(2), _pairs((0, 1)))
        self.assertFalse(result.overflow)
        self.assertEqual(len(result.records), 2)
        bodies = {
            (tuple(item["compatible_pair_ids"]), tuple(item["unmatched_boundary_ids"]))
            for item in result.records
        }
        self.assertEqual(bodies, {((), (0, 1)), ((0,), ())})

    def test_triangle_has_empty_plus_three_single_pairs(self) -> None:
        result = enumerate_boundary_matchings(
            _boundaries(3), _pairs((0, 1), (0, 2), (1, 2)),
        )
        self.assertFalse(result.overflow)
        self.assertEqual(len(result.records), 4)

    def test_two_disjoint_pairs_have_four_matchings(self) -> None:
        result = enumerate_boundary_matchings(
            _boundaries(4), _pairs((0, 1), (2, 3)),
        )
        self.assertFalse(result.overflow)
        self.assertEqual(len(result.records), 4)
        self.assertIn(
            ((0, 1), ()),
            {
                (tuple(item["compatible_pair_ids"]), tuple(item["unmatched_boundary_ids"]))
                for item in result.records
            },
        )

    def test_capacity_overflow_is_closed(self) -> None:
        result = enumerate_boundary_matchings(
            _boundaries(16), _pairs(*((2 * index, 2 * index + 1) for index in range(8))),
        )
        self.assertTrue(result.overflow)
        self.assertEqual(result.records, ())

    def test_comparison_detects_missing_matching(self) -> None:
        expected = list(enumerate_boundary_matchings(_boundaries(2), _pairs((0, 1))).records)
        comparison = compare_boundary_matchings(expected, expected[1:])
        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["missing_matching_count"], 1)
        self.assertEqual(comparison["extra_matching_count"], 0)

    def test_comparison_distinguishes_noncanonical_order_from_missing_set(self) -> None:
        expected = list(enumerate_boundary_matchings(_boundaries(2), _pairs((0, 1))).records)
        comparison = compare_boundary_matchings(expected, list(reversed(expected)))
        self.assertFalse(comparison["exact"])
        self.assertTrue(comparison["canonical_order_mismatch"])
        self.assertEqual(comparison["missing_matching_count"], 0)
        self.assertEqual(comparison["extra_matching_count"], 0)


class ExecutableNoGoTests(unittest.TestCase):
    def test_full_and_omitted_documents_are_both_structurally_valid(self) -> None:
        self.assertEqual(
            validate_candidate_set_bytes(FULL_RAW, INPUT_RAW)["status"],
            "STRUCTURALLY_VALID_NOT_COMPLETE",
        )
        self.assertEqual(
            validate_candidate_set_bytes(OMITTED_RAW, INPUT_RAW)["status"],
            "STRUCTURALLY_VALID_NOT_COMPLETE",
        )

    def test_full_and_omitted_documents_have_identical_root_projection(self) -> None:
        full = json.loads(FULL_RAW)
        omitted = json.loads(OMITTED_RAW)
        self.assertEqual(root_projection(full), root_projection(omitted))
        self.assertNotEqual(full["interpretations"], omitted["interpretations"])
        self.assertEqual(len(full["interpretations"]), 2)
        self.assertEqual(len(omitted["interpretations"]), 1)

    def test_root_only_probe_refuses_interpretation_completeness_for_both(self) -> None:
        for raw in (FULL_RAW, OMITTED_RAW):
            report = verify_candidate_bytes(raw, INPUT_RAW)
            self.assertEqual(
                report["status"],
                "BOUNDARY_MATCHINGS_COMPLETE_INTERPRETATIONS_UNPROVABLE",
            )
            self.assertFalse(report["interpretation_completeness_certified"])
            self.assertEqual(len(report["missing_root_domains"]), 8)

    def test_paired_only_structural_document_omits_all_unmatched_matching(self) -> None:
        structural = validate_candidate_set_bytes(PAIRED_ONLY_RAW, INPUT_RAW)
        self.assertEqual(structural["status"], "STRUCTURALLY_VALID_NOT_COMPLETE")
        report = verify_candidate_bytes(PAIRED_ONLY_RAW, INPUT_RAW)
        self.assertEqual(report["status"], "BOUNDARY_MATCHINGS_INCOMPLETE")
        self.assertEqual(report["missing_matching_count"], 1)
        self.assertEqual(report["extra_matching_count"], 0)

    def test_closed_reject_cannot_be_replayed_without_root_universes(self) -> None:
        reject_raw = (STRUCTURAL_FIXTURES / "valid_reject.json").read_bytes()
        report = verify_candidate_bytes(reject_raw, INPUT_RAW)
        self.assertEqual(report["status"], "ROOT_UNAVAILABLE_FROM_REJECT")

    def test_invalid_candidate_stays_invalid(self) -> None:
        value = json.loads(FULL_RAW)
        value["training_authorized"] = True
        report = verify_candidate_bytes(canonical_json_bytes(value), INPUT_RAW)
        self.assertEqual(report["status"], "INVALID")


if __name__ == "__main__":
    unittest.main()
