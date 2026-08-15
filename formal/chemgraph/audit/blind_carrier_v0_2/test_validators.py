from __future__ import annotations

import base64
import json
from pathlib import Path
import unittest

from formal.chemgraph.audit.blind_carrier_v0_1 import test_validators as v01_tests
from formal.chemgraph.audit.blind_carrier_v0_2._common import (
    canonical_grayscale_png,
    canonical_json_bytes,
)
from formal.chemgraph.audit.blind_carrier_v0_2.validate_inbox import validate_inbox_bytes
from formal.chemgraph.audit.blind_carrier_v0_2.validate_predictions import validate_prediction_bytes


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
INBOX_RAW = (FIXTURES / "valid_inbox.json").read_bytes()
PREDICTION_RAW = (FIXTURES / "valid_prediction.json").read_bytes()
DOUBLE_INBOX_RAW = (FIXTURES / "valid_double_inbox.json").read_bytes()
DOUBLE_PREDICTION_RAW = (FIXTURES / "valid_double_prediction.json").read_bytes()
TRIPLE_INBOX_RAW = (FIXTURES / "valid_triple_inbox.json").read_bytes()
TRIPLE_PREDICTION_RAW = (FIXTURES / "valid_triple_prediction.json").read_bytes()


def error_codes(report: dict) -> set[str]:
    return {error["code"] for error in report.get("errors", [])}


def mutate(raw: bytes, change) -> bytes:
    value = json.loads(raw)
    change(value)
    return canonical_json_bytes(value)


def inbox_for_pixels(pixels: set[tuple[int, int]]) -> bytes:
    decoded = bytearray([255]) * (768 * 768)
    for x, y in pixels:
        decoded[y * 768 + x] = 0
    value = json.loads(INBOX_RAW)
    value["image"]["bytes_base64"] = base64.b64encode(
        canonical_grayscale_png(bytes(decoded)),
    ).decode("ascii")
    return canonical_json_bytes(value)


class V02RegressionMixin:
    def setUp(self) -> None:
        super_method = getattr(super(), "setUp", None)
        if super_method is not None:
            super_method()
        names = (
            "ROOT", "FIXTURES", "INBOX_RAW", "PREDICTION_RAW",
            "DOUBLE_INBOX_RAW", "DOUBLE_PREDICTION_RAW",
            "validate_inbox_bytes", "validate_prediction_bytes",
        )
        self._saved_v01 = {name: getattr(v01_tests, name) for name in names}
        replacements = {
            "ROOT": ROOT,
            "FIXTURES": FIXTURES,
            "INBOX_RAW": INBOX_RAW,
            "PREDICTION_RAW": PREDICTION_RAW,
            "DOUBLE_INBOX_RAW": DOUBLE_INBOX_RAW,
            "DOUBLE_PREDICTION_RAW": DOUBLE_PREDICTION_RAW,
            "validate_inbox_bytes": validate_inbox_bytes,
            "validate_prediction_bytes": validate_prediction_bytes,
        }
        for name, value in replacements.items():
            setattr(v01_tests, name, value)

    def tearDown(self) -> None:
        for name, value in self._saved_v01.items():
            setattr(v01_tests, name, value)
        super_method = getattr(super(), "tearDown", None)
        if super_method is not None:
            super_method()


class InboxValidationTests(V02RegressionMixin, v01_tests.InboxValidationTests):
    pass


class PredictionFormatTests(V02RegressionMixin, v01_tests.PredictionFormatTests):
    pass


class RasterSupportTests(V02RegressionMixin, v01_tests.RasterSupportTests):
    def test_proper_crossing_may_share_one_pixel_when_both_segments_have_exclusive_support(self) -> None:
        pixels = {(column, 384) for column in range(100, 501)}
        pixels |= {(300, row) for row in range(300, 469)}
        inbox = inbox_for_pixels(pixels)
        value = json.loads(PREDICTION_RAW)
        value["segments"] = [
            {"id": 0, "polyline_q4": [[1600, 6144], [8000, 6144]],
             "support_pixel_count": 401, "support_runs": [[384, 100, 401]], "width_q4": 16},
            {"id": 1, "polyline_q4": [[4800, 4800], [4800, 7488]],
             "support_pixel_count": 169, "support_runs": [[row, 300, 1] for row in range(300, 469)], "width_q4": 16},
        ]
        value["terminals"] = [
            {"center_q4": [1600, 6144], "endpoint_alignment": [{"endpoint_index": 0, "segment_id": 0}], "id": 0, "support_segment_ids": [0]},
            {"center_q4": [8000, 6144], "endpoint_alignment": [{"endpoint_index": 1, "segment_id": 0}], "id": 1, "support_segment_ids": [0]},
            {"center_q4": [4800, 4800], "endpoint_alignment": [{"endpoint_index": 0, "segment_id": 1}], "id": 2, "support_segment_ids": [1]},
            {"center_q4": [4800, 7488], "endpoint_alignment": [{"endpoint_index": 1, "segment_id": 1}], "id": 3, "support_segment_ids": [1]},
        ]
        value["lane_hypotheses"] = [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0, "segment_ids": [0]},
            {"begin_terminal_id": 2, "end_terminal_id": 3, "id": 1, "segment_ids": [1]},
        ]
        value["bond_hypotheses"] = [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0,
             "lane_endpoint_alignment": [{"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 0}], "lane_ids": [0], "order": 1},
            {"begin_terminal_id": 2, "end_terminal_id": 3, "id": 1,
             "lane_endpoint_alignment": [{"begin_terminal_id": 2, "end_terminal_id": 3, "lane_id": 1}], "lane_ids": [1], "order": 1},
        ]
        value["interpretations"] = [{"id": 0, "ownership": [
            {"owner_id": 0, "owner_kind": "bond_lane", "segment_id": 0},
            {"owner_id": 1, "owner_kind": "bond_lane", "segment_id": 1},
        ], "selected_bond_ids": [0, 1]}]
        report = validate_prediction_bytes(canonical_json_bytes(value), inbox)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")

    def test_geometry_support_binding_rejects_origin_hallucination_and_accepts_radius_boundary(self) -> None:
        origin = mutate(PREDICTION_RAW, lambda value: value["segments"][0].update(
            polyline_q4=[[0, 0], [16, 0]],
        ))
        self.assertTrue(
            {"support_far_from_geometry", "polyline_vertex_unsupported"}
            & error_codes(validate_prediction_bytes(origin, INBOX_RAW)),
        )

        def boundary(value):
            for segment in value["segments"]:
                segment["polyline_q4"] = [
                    [segment["polyline_q4"][0][0] + 8, 6128],
                    [segment["polyline_q4"][-1][0] + 8, 6128],
                ]
            value["terminals"][0]["center_q4"] = [1608, 6128]
            value["terminals"][1]["center_q4"] = [4808, 6128]
            value["terminals"][2]["center_q4"] = [8008, 6128]

        report = validate_prediction_bytes(mutate(PREDICTION_RAW, boundary), INBOX_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")

    def test_terminal_must_be_near_referenced_polyline_endpoint(self) -> None:
        forged = mutate(PREDICTION_RAW, lambda value: value["terminals"][0].update(
            center_q4=[1601, 6144],
        ))
        self.assertIn("terminal_center_mismatch", error_codes(
            validate_prediction_bytes(forged, INBOX_RAW),
        ))


class TopologyAndOwnershipTests(V02RegressionMixin, v01_tests.TopologyAndOwnershipTests):
    pass


class CliAndIsolationTests(V02RegressionMixin, v01_tests.CliAndIsolationTests):
    pass


class TerminalRegionV02Tests(unittest.TestCase):
    def test_chain_shared_terminal_aligns_previous_end_and_next_begin(self) -> None:
        report = validate_prediction_bytes(PREDICTION_RAW, INBOX_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")
        terminal = json.loads(PREDICTION_RAW)["terminals"][1]
        self.assertEqual(terminal["endpoint_alignment"], [
            {"endpoint_index": 1, "segment_id": 0},
            {"endpoint_index": 0, "segment_id": 1},
        ])

    def test_fan_terminal_may_contain_multiple_selected_bond_endpoints(self) -> None:
        horizontal = {(column, 384) for column in range(100, 501)}
        vertical = {(300, row) for row in range(200, 385)}
        inbox = inbox_for_pixels(horizontal | vertical)
        segments = [
            {"id": 0, "polyline_q4": [[4808, 6152], [1608, 6152]],
             "support_pixel_count": 201, "support_runs": [[384, 100, 201]], "width_q4": 16},
            {"id": 1, "polyline_q4": [[4808, 6152], [4808, 3208]],
             "support_pixel_count": 185, "support_runs": [[row, 300, 1] for row in range(200, 385)], "width_q4": 16},
            {"id": 2, "polyline_q4": [[4808, 6152], [8008, 6152]],
             "support_pixel_count": 201, "support_runs": [[384, 300, 201]], "width_q4": 16},
        ]
        terminals = [
            {"center_q4": [4808, 6152], "endpoint_alignment": [
                {"endpoint_index": 0, "segment_id": 0},
                {"endpoint_index": 0, "segment_id": 1},
                {"endpoint_index": 0, "segment_id": 2},
            ], "id": 0, "support_segment_ids": [0, 1, 2]},
            {"center_q4": [1608, 6152], "endpoint_alignment": [{"endpoint_index": 1, "segment_id": 0}], "id": 1, "support_segment_ids": [0]},
            {"center_q4": [4808, 3208], "endpoint_alignment": [{"endpoint_index": 1, "segment_id": 1}], "id": 2, "support_segment_ids": [1]},
            {"center_q4": [8008, 6152], "endpoint_alignment": [{"endpoint_index": 1, "segment_id": 2}], "id": 3, "support_segment_ids": [2]},
        ]
        lanes = [
            {"begin_terminal_id": 0, "end_terminal_id": index + 1, "id": index, "segment_ids": [index]}
            for index in range(3)
        ]
        bonds = [
            {"begin_terminal_id": 0, "end_terminal_id": index + 1, "id": index,
             "lane_endpoint_alignment": [{"begin_terminal_id": 0, "end_terminal_id": index + 1, "lane_id": index}],
             "lane_ids": [index], "order": 1}
            for index in range(3)
        ]
        value = {
            "bond_hypotheses": bonds,
            "image": {"coordinate_format": "Q4-pixel", "height": 768, "width": 768},
            "interpretations": [{"id": 0, "ownership": [
                {"owner_id": index, "owner_kind": "bond_lane", "segment_id": index}
                for index in range(3)
            ], "selected_bond_ids": [0, 1, 2]}],
            "lane_hypotheses": lanes,
            "schema_version": "chemgraph-blind-carrier-prediction-v0.2",
            "segments": segments,
            "status": "ok",
            "terminals": terminals,
            "training_authorized": False,
        }
        report = validate_prediction_bytes(canonical_json_bytes(value), inbox)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")

    def test_standard_parallel_triple_with_tangent_disk_path_is_valid(self) -> None:
        report = validate_prediction_bytes(TRIPLE_PREDICTION_RAW, TRIPLE_INBOX_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")
        self.assertEqual(report["lane_count"], 3)
        self.assertEqual(report["bond_count"], 1)

    def test_tangent_disk_boundary_accepts_and_plus_one_q4_rejects(self) -> None:
        valid = validate_prediction_bytes(TRIPLE_PREDICTION_RAW, TRIPLE_INBOX_RAW)
        self.assertEqual(valid["status"], "STRUCTURALLY_VALID_NOT_SCORED")

        value = json.loads(TRIPLE_PREDICTION_RAW)
        value["segments"][0]["polyline_q4"] = [[1608, 6103], [8008, 6103]]
        value["terminals"][0]["center_q4"] = [1608, 6152]
        value["terminals"][1]["center_q4"] = [8008, 6152]
        report = validate_prediction_bytes(canonical_json_bytes(value), TRIPLE_INBOX_RAW)
        codes = error_codes(report)
        self.assertIn("terminal_region_disconnected", codes)
        self.assertIn("bond_terminal_region_disconnected", codes)

    def test_terminal_center_uses_frozen_half_up_rounding(self) -> None:
        value = json.loads(DOUBLE_PREDICTION_RAW)
        value["segments"][1]["polyline_q4"] = [[1608, 6169], [8008, 6169]]
        value["terminals"][0]["center_q4"] = [1608, 6153]
        value["terminals"][1]["center_q4"] = [8008, 6153]
        report = validate_prediction_bytes(canonical_json_bytes(value), DOUBLE_INBOX_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")

    def test_endpoint_index_center_and_support_union_forgery_reject(self) -> None:
        wrong_index = mutate(PREDICTION_RAW, lambda value: value["terminals"][0][
            "endpoint_alignment"
        ][0].update(endpoint_index=1))
        wrong_center = mutate(PREDICTION_RAW, lambda value: value["terminals"][1].update(
            center_q4=[4801, 6144],
        ))
        wrong_union = mutate(PREDICTION_RAW, lambda value: value["terminals"][1].update(
            support_segment_ids=[0],
        ))
        self.assertIn("lane_endpoint_indices_not_distinct", error_codes(
            validate_prediction_bytes(wrong_index, INBOX_RAW),
        ))
        self.assertIn("terminal_center_mismatch", error_codes(
            validate_prediction_bytes(wrong_center, INBOX_RAW),
        ))
        self.assertIn("terminal_endpoint_union_mismatch", error_codes(
            validate_prediction_bytes(wrong_union, INBOX_RAW),
        ))

    def test_selected_endpoint_cannot_belong_to_two_terminals(self) -> None:
        def duplicate(value):
            value["terminals"][2]["endpoint_alignment"].insert(
                0, {"endpoint_index": 1, "segment_id": 0},
            )
            value["terminals"][2]["support_segment_ids"] = [0, 1]
            value["terminals"][2]["center_q4"] = [6400, 6144]

        report = validate_prediction_bytes(mutate(PREDICTION_RAW, duplicate), INBOX_RAW)
        self.assertIn("selected_endpoint_reused", error_codes(report))

    def test_unselected_lane_cannot_bridge_selected_bundle_terminal(self) -> None:
        value = json.loads(TRIPLE_PREDICTION_RAW)
        value["bond_hypotheses"] = [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0,
             "lane_endpoint_alignment": [
                 {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 0},
                 {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 2},
             ], "lane_ids": [0, 2], "order": 2},
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 1,
             "lane_endpoint_alignment": [
                 {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 1},
             ], "lane_ids": [1], "order": 1},
        ]
        value["interpretations"][0]["selected_bond_ids"] = [0]
        value["interpretations"][0]["ownership"][1] = {
            "owner_id": 0, "owner_kind": "residual", "segment_id": 1,
        }
        report = validate_prediction_bytes(canonical_json_bytes(value), TRIPLE_INBOX_RAW)
        codes = error_codes(report)
        self.assertIn("bond_terminal_region_disconnected", codes)
        self.assertIn("selected_terminal_region_disconnected", codes)

    def test_v02_rejects_multi_segment_lane_and_equal_terminal_centers(self) -> None:
        multi = mutate(PREDICTION_RAW, lambda value: value["lane_hypotheses"][0].update(
            segment_ids=[0, 1],
        ))
        equal = mutate(PREDICTION_RAW, lambda value: value["terminals"][2].update(
            center_q4=[4800, 6144],
        ))
        self.assertIn("lane_not_single_segment", error_codes(
            validate_prediction_bytes(multi, INBOX_RAW),
        ))
        self.assertIn("lane_terminal_centers_equal", error_codes(
            validate_prediction_bytes(equal, INBOX_RAW),
        ))


if __name__ == "__main__":
    unittest.main()
