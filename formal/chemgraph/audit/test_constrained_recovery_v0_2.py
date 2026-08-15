from __future__ import annotations

from dataclasses import replace
import unittest

from formal.chemgraph.audit.constrained_recovery_v0_2 import (
    BondBundle,
    BondLane,
    CalibratedInterval,
    CalibrationEvidence,
    CandidateHypothesis,
    CarrierEvidence,
    CoverageCertificate,
    SegmentGeometry,
    candidate_set_sha256,
    certify_graph,
    graph_has_complete_provenance,
    graph_record,
)


SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64


def segment(primitive_id: str, begin: tuple[int, int], end: tuple[int, int]) -> SegmentGeometry:
    return SegmentGeometry(f"segment:{primitive_id}", begin, end)


def carrier(
    primitive_id: str,
    bond_id: str,
    geometry: SegmentGeometry,
    *,
    score: int = 100,
    minimum: int = 50,
    pixels: tuple[str, ...] | None = None,
) -> CarrierEvidence:
    return CarrierEvidence(
        carrier_id=f"carrier:{primitive_id}",
        primitive_id=primitive_id,
        support_pixels=pixels if pixels is not None else (f"pixel:{primitive_id}",),
        support_segments=(geometry,),
        support_score=score,
        minimum_support_score=minimum,
        owner_kind="bond_lane",
        owner_id=bond_id,
    )


def lane(
    primitive_id: str,
    bond_id: str,
    begin_node: int,
    end_node: int,
    begin: tuple[int, int],
    end: tuple[int, int],
    **carrier_options,
) -> BondLane:
    geometry = segment(primitive_id, begin, end)
    return BondLane(
        primitive_id=primitive_id,
        geometry=geometry,
        support=carrier(primitive_id, bond_id, geometry, **carrier_options),
        bond_id=bond_id,
        begin_node=begin_node,
        end_node=end_node,
    )


def single_bundle(
    primitive_id: str,
    bond_id: str,
    begin_node: int,
    end_node: int,
    begin: tuple[int, int],
    end: tuple[int, int],
) -> BondBundle:
    return BondBundle(
        bond_id=bond_id,
        order=1,
        lanes=(lane(primitive_id, bond_id, begin_node, end_node, begin, end),),
    )


def path_graph() -> dict:
    return graph_record(
        (0, 1, 2),
        (
            single_bundle("p-ab", "b-ab", 0, 1, (0, 0), (10, 0)),
            single_bundle("p-bc", "b-bc", 1, 2, (10, 0), (20, 0)),
        ),
    )


def single_graph() -> dict:
    return graph_record(
        (0, 1),
        (single_bundle("p-ab", "b-ab", 0, 1, (0, 0), (10, 0)),),
    )


def hypotheses() -> list[CandidateHypothesis]:
    return [
        CandidateHypothesis("candidate-single", single_graph()),
        CandidateHypothesis("candidate-path", path_graph()),
    ]


def coverage(candidates: list[CandidateHypothesis]) -> CoverageCertificate:
    identifiers = tuple(sorted(candidate.candidate_id for candidate in candidates))
    return CoverageCertificate(
        evidence_id="finite-redteam-evidence-v0.2",
        evidence_sha256=SHA_A,
        enumerator_id="finite-enumerator-v0.2",
        coverage_audit_sha256=SHA_B,
        candidate_ids=identifiers,
        supported_hypothesis_ids=identifiers,
        candidate_set_sha256=candidate_set_sha256(candidates),
    )


def calibration(epsilon: int = 3) -> CalibrationEvidence:
    return CalibrationEvidence(
        calibration_id="finite-calibration-v0.2",
        calibration_sha256=SHA_C,
        sample_count=128,
        epsilon=epsilon,
        maximum_observed_error=epsilon,
    )


class CarrierAndLaneTests(unittest.TestCase):
    def test_zero_support_path_shortcut_cannot_become_edge(self) -> None:
        geometry = segment("p-ac", (0, 0), (20, 0))
        unsupported = carrier("p-ac", "b-ac", geometry, pixels=())
        self.assertIn("empty_pixel_support", unsupported.admissibility_errors())
        with self.assertRaisesRegex(ValueError, "admissible carrier support"):
            BondLane("p-ac", geometry, unsupported, "b-ac", 0, 2)
        self.assertEqual({(edge["begin"], edge["end"]) for edge in path_graph()["edges"]},
                         {(0, 1), (1, 2)})

    def test_low_support_fan_chord_cannot_become_edge(self) -> None:
        geometry = segment("p-outer", (0, 10), (20, 10))
        weak = carrier("p-outer", "b-outer", geometry, score=49, minimum=50)
        self.assertIn("support_below_threshold", weak.admissibility_errors())
        with self.assertRaisesRegex(ValueError, "admissible carrier support"):
            BondLane("p-outer", geometry, weak, "b-outer", 1, 3)

    def test_one_stroke_is_incomplete_for_double_or_triple_bond(self) -> None:
        only_lane = lane("p-1", "b-multi", 0, 1, (0, 0), (10, 0))
        for order in (2, 3):
            with self.subTest(order=order):
                with self.assertRaisesRegex(ValueError, "exact physical lane count"):
                    BondBundle("b-multi", order, (only_lane,))

    def test_double_and_triple_require_exact_distinct_lanes(self) -> None:
        for order in (2, 3):
            lanes = tuple(
                lane(f"p-{index}", "b-multi", 0, 1, (0, index), (10, index))
                for index in range(order)
            )
            bundle = BondBundle("b-multi", order, lanes)
            self.assertEqual(len(bundle.lanes), order)
            self.assertTrue(all(item.endpoint(False) == 0 for item in lanes))
            self.assertTrue(all(item.endpoint(True) == 1 for item in lanes))

    def test_lane_endpoint_depends_on_that_lane(self) -> None:
        first = lane("p-1", "b-1", 0, 1, (0, 0), (10, 0))
        second = lane("p-2", "b-2", 2, 3, (0, 10), (10, 10))
        self.assertEqual((first.endpoint(False), first.endpoint(True)), (0, 1))
        self.assertEqual((second.endpoint(False), second.endpoint(True)), (2, 3))

    def test_lane_support_must_be_mutually_exclusive(self) -> None:
        left = lane("p-1", "b-double", 0, 1, (0, 0), (10, 0), pixels=("shared",))
        right = lane("p-2", "b-double", 0, 1, (0, 1), (10, 1), pixels=("shared",))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            BondBundle("b-double", 2, (left, right))

    def test_graph_forbids_reusing_one_primitive_for_two_edges(self) -> None:
        first = single_bundle("p-shared", "b-1", 0, 1, (0, 0), (10, 0))
        second = single_bundle("p-shared", "b-2", 1, 2, (10, 0), (20, 0))
        with self.assertRaisesRegex(ValueError, "primitive ownership"):
            graph_record((0, 1, 2), (first, second))


class ProvenanceAndAuthorityTests(unittest.TestCase):
    def test_graph_edge_retains_complete_lane_and_carrier_provenance(self) -> None:
        graph = path_graph()
        self.assertTrue(graph_has_complete_provenance(graph))
        for edge in graph["edges"]:
            self.assertEqual(len(edge["primitive_ids"]), edge["order"])
            self.assertEqual(len(edge["carrier_ids"]), edge["order"])
            self.assertEqual(len(edge["lane_provenance"]), edge["order"])

    def test_old_edge_without_provenance_cannot_be_graph_output(self) -> None:
        old_graph = {
            "nodes": [{"id": 0, "role": "implicit"}, {"id": 1, "role": "implicit"}],
            "edges": [{"begin": 0, "end": 1, "order": 1}],
            "role_hypotheses": [],
        }
        candidates = [CandidateHypothesis("old", old_graph)]
        result = certify_graph(candidates, coverage=coverage(candidates))
        self.assertEqual(result, {
            "decision": "Ambiguous",
            "reason": "incomplete_edge_provenance",
        })

    def test_singleton_without_coverage_certificate_is_not_graph(self) -> None:
        candidates = [CandidateHypothesis("single", single_graph())]
        result = certify_graph(candidates, coverage=None)
        self.assertEqual(result, {
            "decision": "Ambiguous",
            "reason": "coverage_not_certified",
        })

    def test_singleton_with_bound_coverage_certificate_is_graph(self) -> None:
        candidates = [CandidateHypothesis("single", single_graph())]
        result = certify_graph(candidates, coverage=coverage(candidates))
        self.assertEqual(result["decision"], "Graph")
        self.assertEqual(result["authority"], "coverage_singleton")

    def test_tampered_candidate_coverage_is_not_graph(self) -> None:
        candidates = [CandidateHypothesis("single", single_graph())]
        invalid = replace(coverage(candidates), candidate_set_sha256="0" * 64)
        result = certify_graph(candidates, coverage=invalid)
        self.assertEqual(result["reason"], "coverage_not_certified")

    def test_arbitrary_intervals_without_calibration_are_not_graph(self) -> None:
        candidates = hypotheses()
        arbitrary = [
            CalibratedInterval("candidate-single", 0, 0, 3, "", ""),
            CalibratedInterval("candidate-path", 7, 4, 10, "", ""),
        ]
        result = certify_graph(
            candidates, coverage=coverage(candidates), calibration=None, intervals=arbitrary,
        )
        self.assertEqual(result, {
            "decision": "Ambiguous",
            "reason": "calibration_not_certified",
        })

    def test_forged_interval_not_derived_from_epsilon_is_not_graph(self) -> None:
        candidates = hypotheses()
        calibrated = calibration()
        intervals = [
            CalibratedInterval.from_score("candidate-single", 0, calibrated),
            CalibratedInterval.from_score("candidate-path", 7, calibrated),
        ]
        intervals[0] = replace(intervals[0], upper=2)
        result = certify_graph(
            candidates,
            coverage=coverage(candidates),
            calibration=calibrated,
            intervals=intervals,
        )
        self.assertEqual(result["reason"], "interval_binding_invalid")

    def test_calibrated_strict_margin_can_authorize_graph(self) -> None:
        candidates = hypotheses()
        calibrated = calibration(epsilon=3)
        intervals = [
            CalibratedInterval.from_score("candidate-single", 0, calibrated),
            CalibratedInterval.from_score("candidate-path", 7, calibrated),
        ]
        result = certify_graph(
            candidates,
            coverage=coverage(candidates),
            calibration=calibrated,
            intervals=intervals,
        )
        self.assertEqual(result["decision"], "Graph")
        self.assertEqual(result["authority"], "coverage_and_calibrated_interval")
        self.assertTrue(graph_has_complete_provenance(result["graph"]))

    def test_two_epsilon_boundary_remains_ambiguous(self) -> None:
        candidates = hypotheses()
        calibrated = calibration(epsilon=3)
        intervals = [
            CalibratedInterval.from_score("candidate-single", 0, calibrated),
            CalibratedInterval.from_score("candidate-path", 6, calibrated),
        ]
        result = certify_graph(
            candidates,
            coverage=coverage(candidates),
            calibration=calibrated,
            intervals=intervals,
        )
        self.assertEqual(result["reason"], "score_intervals_overlap")


if __name__ == "__main__":
    unittest.main()
