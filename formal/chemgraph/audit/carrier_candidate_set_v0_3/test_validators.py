from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from formal.chemgraph.audit.blind_carrier_v0_2._common import canonical_json_bytes

from .validate_candidate_set import (
    _angle_millidegrees,
    _stage_chain_sha256,
    validate_candidate_set_bytes,
)
from .validate_input import validate_input_bytes


FIXTURES = Path(__file__).resolve().parent / "fixtures"
INPUT_RAW = (FIXTURES / "valid_input.json").read_bytes()
SINGLE_RAW = (FIXTURES / "valid_single_bond.json").read_bytes()
GLYPH_RAW = (FIXTURES / "valid_glyph_only.json").read_bytes()
COLLISION_RAW = (FIXTURES / "valid_glyph_bond_collision.json").read_bytes()
REJECT_RAW = (FIXTURES / "valid_reject.json").read_bytes()
CROSSING_INPUT_RAW = (FIXTURES / "valid_crossing_input.json").read_bytes()
CROSSING_RAW = (FIXTURES / "valid_crossing_junction_collision.json").read_bytes()


def _codes(report: dict) -> set[str]:
    return {item["code"] for item in report.get("errors", [])}


def _mutate(raw: bytes, change) -> bytes:
    value = json.loads(raw)
    change(value)
    return canonical_json_bytes(value)


class PositiveFixtureTests(unittest.TestCase):
    def test_input_is_canonical(self) -> None:
        self.assertEqual(validate_input_bytes(INPUT_RAW)["status"], "STRUCTURALLY_VALID_NOT_SCORED")

    def test_single_bond_is_structurally_valid_but_not_complete(self) -> None:
        report = validate_candidate_set_bytes(SINGLE_RAW, INPUT_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_COMPLETE")
        self.assertEqual(report["graph_class_count"], 1)

    def test_glyph_only_spatial_empty_graph_is_representable(self) -> None:
        report = validate_candidate_set_bytes(GLYPH_RAW, INPUT_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_COMPLETE")
        self.assertEqual(report["graph_class_count"], 1)

    def test_glyph_vs_bond_collision_retains_two_graph_classes(self) -> None:
        report = validate_candidate_set_bytes(COLLISION_RAW, INPUT_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_COMPLETE")
        self.assertEqual(report["graph_class_count"], 2)
        self.assertEqual(report["interpretation_count"], 2)

    def test_crossing_vs_junction_retains_two_graph_classes(self) -> None:
        report = validate_candidate_set_bytes(CROSSING_RAW, CROSSING_INPUT_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_COMPLETE")
        self.assertEqual(report["graph_class_count"], 2)
        self.assertEqual(report["interpretation_count"], 2)

    def test_closed_reject_is_valid_without_graph(self) -> None:
        report = validate_candidate_set_bytes(REJECT_RAW, INPUT_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_COMPLETE")
        self.assertEqual(report["graph_class_count"], 0)


class CanonicalAndBindingMutationTests(unittest.TestCase):
    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = REJECT_RAW.replace(b'"reason":"unsupported_primitive",', b'"reason":"unsupported_primitive","reason":"unsupported_geometry",')
        self.assertIn("duplicate_key", _codes(validate_candidate_set_bytes(raw, INPUT_RAW)))

    def test_noncanonical_json_is_rejected(self) -> None:
        raw = json.dumps(json.loads(REJECT_RAW), indent=2).encode("ascii")
        self.assertIn("noncanonical_json", _codes(validate_candidate_set_bytes(raw, INPUT_RAW)))

    def test_invalid_png_is_rejected_before_candidate(self) -> None:
        bad = _mutate(INPUT_RAW, lambda value: value["image"].update(bytes_base64="iVBORw0KGgo="))
        self.assertEqual(validate_input_bytes(bad)["status"], "INVALID")

    def test_wrong_png_digest_is_rejected(self) -> None:
        forged = _mutate(SINGLE_RAW, lambda value: value["image"].update(png_sha256="0" * 64))
        self.assertIn("image_digest_mismatch", _codes(validate_candidate_set_bytes(forged, INPUT_RAW)))

    def test_forbidden_winner_field_is_rejected(self) -> None:
        forged = _mutate(SINGLE_RAW, lambda value: value.update(winner=0))
        self.assertIn("forbidden_semantic_field", _codes(validate_candidate_set_bytes(forged, INPUT_RAW)))

    def test_reject_is_closed(self) -> None:
        forged = _mutate(REJECT_RAW, lambda value: value.update(detail="hidden failure"))
        self.assertIn("jsonschema", _codes(validate_candidate_set_bytes(forged, INPUT_RAW)))


class RasterAndGeometryMutationTests(unittest.TestCase):
    def test_overlapping_rle_is_rejected(self) -> None:
        def change(value):
            value["image"]["final_foreground_runs"] = [[384, 100, 300], [384, 200, 301]]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("rle_overlap", _codes(report))

    def test_forged_rle_hash_is_rejected(self) -> None:
        forged = _mutate(SINGLE_RAW, lambda value: value["image"].update(final_foreground_rle_sha256="0" * 64))
        self.assertIn("mask_digest_mismatch", _codes(validate_candidate_set_bytes(forged, INPUT_RAW)))

    def test_boundary_bond_support_must_be_its_own_foreground(self) -> None:
        def change(value):
            for boundary in (value["boundary_universe"], value["interpretations"][0]["boundaries"]):
                boundary[0]["final_foreground_support_runs"] = [[384, 100, 200]]
                boundary[0]["final_foreground_support_pixel_count"] = 200

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("boundary_bond_support_not_foreground", _codes(report))

    def test_boundary_cannot_disappear_from_fragment_fiber(self) -> None:
        def change(value):
            value["interpretations"][0]["stroke_fragments"][0]["boundary_member_ids"] = [1]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("boundary_without_fragment", _codes(report))

    def test_unmatched_fragment_cannot_change_segment(self) -> None:
        def change(value):
            value["interpretations"][0]["stroke_fragments"][0]["segment_q4"] = [[1608, 6152], [7992, 6152]]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("unmatched_fragment_segment_mismatch", _codes(report))

    def test_bond_lane_cannot_be_relabelled_unknown(self) -> None:
        def change(value):
            value["interpretations"][0]["stroke_fragments"][0]["primitive_kind"] = "unknown"

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertTrue({"owner_primitive_mismatch", "unsupported_bond_primitive"} <= _codes(report))


class GraphRelationMutationTests(unittest.TestCase):
    def test_wrong_atom_pair_provenance_is_rejected_without_crash(self) -> None:
        def change(value):
            value["interpretations"][0]["bonds"][0]["atom_pair"] = [0, 2]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("atom_pair_invalid", _codes(report))

    def test_wrong_direct_pair_id_is_rejected(self) -> None:
        def change(value):
            value["interpretations"][0]["bonds"][0]["atom_pair_evidence"]["pair_id"] = 1

        self.assertIn("unknown_direct_pair", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_missing_direct_pair_universe_member_is_rejected(self) -> None:
        forged = _mutate(SINGLE_RAW, lambda value: value.update(direct_pair_universe=[]))
        self.assertIn("direct_pair_universe_incomplete", _codes(validate_candidate_set_bytes(forged, INPUT_RAW)))

    def test_inactive_atom_proposal_cannot_be_instantiated(self) -> None:
        def change(value):
            proposal = value["atom_proposal_universe"][0]
            proposal.update(heatmap_score_q24=0, set_decision="inactive", set_logit_q20=-300000)

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("inactive_atom_proposal_used", _codes(report))

    def test_endpoint_relation_must_use_frozen_angle_gate(self) -> None:
        def change(value):
            value["interpretations"][0]["endpoint_atom_relations"][0]["geometry_evidence"]["angle_millidegrees"] = 12001

        self.assertIn("endpoint_attachment_predicate_failed", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_squared_distance_gate_rejects_flooring_counterexample(self) -> None:
        def change(value):
            anchor = [1352, 6153]
            proposal = value["atom_proposal_universe"][0]
            proposal["anchor_q4"] = anchor
            proposal["source_cell_q4"] = anchor
            atom = value["interpretations"][0]["atoms"][0]
            atom["anchor_q4"] = anchor
            atom["source_cell_q4"] = anchor
            evidence = value["interpretations"][0]["endpoint_atom_relations"][0]["geometry_evidence"]
            evidence.update(
                absolute_cross_q8=6400,
                angle_millidegrees=_angle_millidegrees((1608, 6152), (8008, 6152), tuple(anchor)),
                atom_anchor_q4=anchor,
                distance_q4=256,
                outward_dot_q8=1638400,
            )

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("endpoint_outside_attachment_radius", _codes(report))
        self.assertIn("endpoint_attachment_predicate_failed", _codes(report))

    def test_inward_endpoint_attachment_is_rejected_by_dot_sign(self) -> None:
        def change(value):
            anchor = [1700, 6152]
            proposal = value["atom_proposal_universe"][0]
            proposal["anchor_q4"] = anchor
            proposal["source_cell_q4"] = anchor
            atom = value["interpretations"][0]["atoms"][0]
            atom["anchor_q4"] = anchor
            atom["source_cell_q4"] = anchor
            evidence = value["interpretations"][0]["endpoint_atom_relations"][0]["geometry_evidence"]
            evidence.update(
                absolute_cross_q8=0,
                angle_millidegrees=0,
                atom_anchor_q4=anchor,
                distance_q4=92,
                outward_dot_q8=-588800,
            )

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("endpoint_attachment_predicate_failed", _codes(report))

    def test_port_and_relation_must_be_bijective(self) -> None:
        def change(value):
            value["interpretations"][0]["ports"][0]["fragment_endpoint_refs"][0]["endpoint_index"] = 1

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertTrue({"port_relation_missing", "endpoint_relation_not_mirrored"} <= _codes(report))

    def test_fragment_has_exactly_one_owner(self) -> None:
        def change(value):
            value["interpretations"][0]["ownership"].append({"fragment_id": 0, "owner_id": 0, "owner_kind": "residual"})

        self.assertIn("ownership_not_exact", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_bond_order_equals_lane_count(self) -> None:
        def change(value):
            value["interpretations"][0]["bonds"][0]["order"] = 2

        self.assertIn("order_lane_count_mismatch", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_direct_decision_is_recomputed_from_three_bands(self) -> None:
        def change(value):
            value["direct_pair_universe"][0]["presence_score_q24"] = 10000000

        self.assertIn("direct_pair_decision_mismatch", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_duplicate_spatial_atom_anchor_is_rejected(self) -> None:
        def change(value):
            value["interpretations"][0]["atoms"][1]["anchor_q4"] = [1608, 6152]

        self.assertIn("duplicate_atom_anchor", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_bond_lane_owner_must_name_actual_lane(self) -> None:
        def change(value):
            value["interpretations"][0]["ownership"][0]["owner_id"] = 1

        self.assertIn("bond_lane_ownership_fiber_mismatch", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))


class EnumerationAndTraceMutationTests(unittest.TestCase):
    def test_incompatible_pair_observations_are_rejected(self) -> None:
        def change(value):
            value["compatible_boundary_pairs"] = [{
                "boundary_ids": [0, 1],
                "canonical_centerline_q4": [[1608, 6152], [8008, 6152]],
                "id": 0,
                "interior_darkness_q24": 0,
                "interior_ink_occupancy_ppm": 0,
                "observed_angle_mdeg": 90000,
                "overlap_q4": 0,
                "separation_q4": 0,
                "shorter_overlap_ppm": 0,
            }]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("compatible_pair_predicate_failed", _codes(report))

    def test_threshold_passing_pair_metrics_are_recomputed_from_boundaries(self) -> None:
        def change(value):
            value["compatible_boundary_pairs"] = [{
                "boundary_ids": [0, 1],
                "canonical_centerline_q4": [[1608, 6152], [8008, 6152]],
                "id": 0,
                "interior_darkness_q24": 5000000,
                "interior_ink_occupancy_ppm": 900000,
                "observed_angle_mdeg": 0,
                "overlap_q4": 3200,
                "separation_q4": 16,
                "shorter_overlap_ppm": 800000,
            }]

        report = validate_candidate_set_bytes(_mutate(CROSSING_RAW, change), CROSSING_INPUT_RAW)
        self.assertIn("compatible_pair_geometry_mismatch", _codes(report))

    def test_enumeration_counts_bind_root_universes(self) -> None:
        forged = _mutate(SINGLE_RAW, lambda value: value["enumeration"].update(compatible_boundary_pair_count=1))
        self.assertIn("enumeration_count_mismatch", _codes(validate_candidate_set_bytes(forged, INPUT_RAW)))

    def test_duplicate_stage_order_is_rejected(self) -> None:
        def change(value):
            value["stage_trace"]["stages"][1]["index"] = 0
            value["stage_trace"]["stages"][1]["stage"] = "preprocess"

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertTrue({"stage_indices_invalid", "stage_order_invalid"} <= _codes(report))

    def test_stage_component_is_in_chain_and_bound_to_provenance(self) -> None:
        def change(value):
            value["stage_trace"]["stages"][0]["component_sha256"] = "0" * 64
            value["stage_trace"]["ordered_chain_sha256"] = _stage_chain_sha256(value["stage_trace"]["stages"])

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("stage_component_mismatch", _codes(report))
        self.assertNotIn("stage_chain_digest_mismatch", _codes(report))

    def test_forged_stage_output_is_rejected(self) -> None:
        def change(value):
            value["stage_trace"]["stages"][2]["output_sha256"] = "0" * 64
            value["stage_trace"]["stages"][3]["input_sha256"] = "0" * 64
            value["stage_trace"]["ordered_chain_sha256"] = _stage_chain_sha256(value["stage_trace"]["stages"])

        self.assertIn("stage_output_mismatch", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_false_singleton_collision_is_rejected(self) -> None:
        def change(value):
            value["ambiguity_taxonomy"]["entries"] = []
            value["ambiguity_taxonomy"]["entry_count"] = 0

        self.assertIn("ambiguity_taxonomy_missing", _codes(validate_candidate_set_bytes(_mutate(COLLISION_RAW, change), INPUT_RAW)))

    def test_junction_region_cannot_be_relabelled_crossing(self) -> None:
        def change(value):
            junction = next(
                item for item in value["interpretations"]
                if item["intersection_regions"][0]["kind"] == "junction"
            )
            junction["intersection_regions"][0]["kind"] = "crossing"

        report = validate_candidate_set_bytes(_mutate(CROSSING_RAW, change), CROSSING_INPUT_RAW)
        self.assertIn("intersection_kind_port_mismatch", _codes(report))

    def test_junction_region_must_list_center_port(self) -> None:
        def change(value):
            junction = next(
                item for item in value["interpretations"]
                if item["intersection_regions"][0]["kind"] == "junction"
            )
            junction["intersection_regions"][0]["port_member_ids"] = []

        report = validate_candidate_set_bytes(_mutate(CROSSING_RAW, change), CROSSING_INPUT_RAW)
        self.assertIn("intersection_port_union_mismatch", _codes(report))

    def test_taxonomy_code_kind_mapping_is_frozen(self) -> None:
        def change(value):
            value["ambiguity_taxonomy"]["entries"][0]["taxonomy_code"] = "A03"

        self.assertIn("ambiguity_code_kind_mismatch", _codes(validate_candidate_set_bytes(_mutate(COLLISION_RAW, change), INPUT_RAW)))

    def test_graph_classes_are_recomputed(self) -> None:
        def change(value):
            value["graph_equivalence"]["graph_classes"][0]["canonical_spatial_graph_sha256"] = "0" * 64

        self.assertIn("graph_classes_mismatch", _codes(validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)))

    def test_far_white_gap_support_is_rejected(self) -> None:
        def change(value):
            interpretation = value["interpretations"][0]
            interpretation["white_gap_certificates"] = [{
                "bond_probability_min_q24": 16000000,
                "gap_begin_q4": [1608, 6152],
                "gap_end_q4": [1708, 6152],
                "gap_kind": "raster_break",
                "gap_length_q4": 100,
                "id": 0,
                "input_png_sha256": value["image"]["png_sha256"],
                "left_endpoint_index": 0,
                "left_fragment_id": 0,
                "right_endpoint_index": 1,
                "right_fragment_id": 0,
                "rule_id": "white_gap_v0_3",
                "span_q4": [[1608, 6152], [1708, 6152]],
                "support_pixel_count": 1,
                "support_runs": [[0, 0, 1]],
                "text_probability_max_q24": 0,
                "white_occupancy_ppm": 1000000,
            }]
            interpretation["lanes"][0]["white_gap_certificate_ids"] = [0]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("white_gap_support_far_from_span", _codes(report))

    def test_unknown_white_gap_reference_returns_invalid_not_exception(self) -> None:
        def change(value):
            value["interpretations"][0]["lanes"][0]["white_gap_certificate_ids"] = [511]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("lane_gap_reference_invalid", _codes(report))

    def test_forged_lane_separation_geometry_is_rejected(self) -> None:
        def change(value):
            interpretation = value["interpretations"][0]
            interpretation["lane_separation_certificates"] = [{
                "adjacent_lane_index": 0,
                "bond_id": 0,
                "id": 0,
                "input_png_sha256": value["image"]["png_sha256"],
                "left_lane_id": 0,
                "observed_maximum_gap_q4": 16,
                "observed_minimum_gap_q4": 16,
                "predicate_passed": True,
                "right_lane_id": 0,
                "rule_id": "lane_separation_v0_3",
                "span_q4": [[1608, 6152], [1708, 6152]],
                "support_pixel_count": 1,
                "support_runs": [[0, 0, 1]],
                "white_occupancy_ppm": 1000000,
            }]

        report = validate_candidate_set_bytes(_mutate(SINGLE_RAW, change), INPUT_RAW)
        self.assertIn("lane_separation_support_far_from_span", _codes(report))
        self.assertIn("lane_separation_geometry_mismatch", _codes(report))


if __name__ == "__main__":
    unittest.main()
