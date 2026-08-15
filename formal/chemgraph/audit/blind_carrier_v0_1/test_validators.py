from __future__ import annotations

import ast
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from PIL import Image

from formal.chemgraph.audit.blind_carrier_v0_1._common import (
    canonical_grayscale_png,
    canonical_json_bytes,
)
from formal.chemgraph.audit.blind_carrier_v0_1.validate_inbox import validate_inbox_bytes
from formal.chemgraph.audit.blind_carrier_v0_1.validate_predictions import validate_prediction_bytes


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
INBOX_RAW = (FIXTURES / "valid_inbox.json").read_bytes()
PREDICTION_RAW = (FIXTURES / "valid_prediction.json").read_bytes()
DOUBLE_INBOX_RAW = (FIXTURES / "valid_double_inbox.json").read_bytes()
DOUBLE_PREDICTION_RAW = (FIXTURES / "valid_double_prediction.json").read_bytes()


def mutate_prediction(change) -> bytes:
    value = json.loads(PREDICTION_RAW)
    change(value)
    return canonical_json_bytes(value)


def mutate_inbox_png(png: bytes) -> bytes:
    value = json.loads(INBOX_RAW)
    value["image"]["bytes_base64"] = base64.b64encode(png).decode("ascii")
    return canonical_json_bytes(value)


def error_codes(report: dict) -> set[str]:
    return {error["code"] for error in report.get("errors", [])}


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def grayscale_png_with_encoding(decoded: bytes, *, level: int, split_idat: bool) -> bytes:
    scanlines = b"".join(
        b"\x00" + decoded[row * 768:(row + 1) * 768]
        for row in range(768)
    )
    ihdr = struct.pack(">IIBBBBB", 768, 768, 8, 0, 0, 0, 0)
    compressed = zlib.compress(scanlines, level=level)
    if split_idat:
        middle = len(compressed) // 2
        idat = png_chunk(b"IDAT", compressed[:middle]) + png_chunk(b"IDAT", compressed[middle:])
    else:
        idat = png_chunk(b"IDAT", compressed)
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr)
            + idat + png_chunk(b"IEND", b""))


class InboxValidationTests(unittest.TestCase):
    def test_valid_synthetic_inbox_decodes_complete_png(self) -> None:
        report = validate_inbox_bytes(INBOX_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")
        self.assertEqual(report["foreground_pixel_count"], 401)
        self.assertIs(report["training_authorized"], False)

    def test_duplicate_key_nonfinite_noncanonical_and_extra_field_reject(self) -> None:
        duplicate = INBOX_RAW.replace(
            b'"training_authorized":false}',
            b'"training_authorized":false,"training_authorized":false}',
        )
        nonfinite = b'{"x":NaN}\n'
        noncanonical = INBOX_RAW.rstrip()
        value = json.loads(INBOX_RAW)
        value["sample_id"] = "forbidden"
        cases = (
            (duplicate, "duplicate_key"),
            (nonfinite, "invalid_json"),
            (noncanonical, "noncanonical_json"),
            (canonical_json_bytes(value), "jsonschema"),
        )
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, error_codes(validate_inbox_bytes(raw)))

    def test_png_signature_dimensions_crc_and_decode_are_checked(self) -> None:
        original = (FIXTURES / "synthetic.png").read_bytes()
        bad_signature = b"BAD!" + original[4:]
        bad_crc = bytearray(original)
        bad_crc[-5] ^= 1
        buffer = BytesIO()
        Image.new("L", (64, 64), 255).save(buffer, format="PNG")
        cases = (
            (bad_signature, "invalid_png_signature"),
            (bytes(bad_crc), "png_crc_mismatch"),
            (buffer.getvalue(), "png_dimensions"),
        )
        for png, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, error_codes(validate_inbox_bytes(mutate_inbox_png(png))))

    def test_png_metadata_alpha_rgb_and_empty_foreground_reject(self) -> None:
        original = (FIXTURES / "synthetic.png").read_bytes()
        iend_offset = original.rfind(b"\x00\x00\x00\x00IEND")
        with_text = original[:iend_offset] + png_chunk(b"tEXt", b"key\x00value") + original[iend_offset:]

        alpha_buffer = BytesIO()
        alpha_image = Image.new("RGBA", (768, 768), (255, 255, 255, 255))
        alpha_image.putpixel((0, 0), (255, 0, 0, 0))
        alpha_image.save(alpha_buffer, format="PNG")

        chroma_buffer = BytesIO()
        chroma_image = Image.new("RGB", (768, 768), (255, 255, 255))
        chroma_image.putpixel((100, 384), (0, 1, 0))
        chroma_image.save(chroma_buffer, format="PNG")

        empty = canonical_grayscale_png(bytes([255]) * (768 * 768))
        cases = (
            (with_text, "forbidden_png_chunk"),
            (alpha_buffer.getvalue(), "unsupported_png_profile"),
            (chroma_buffer.getvalue(), "unsupported_png_profile"),
            (empty, "empty_foreground"),
        )
        for png, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, error_codes(validate_inbox_bytes(mutate_inbox_png(png))))

    def test_rgb_profile_is_rejected_even_when_every_pixel_is_grayscale(self) -> None:
        image = Image.new("RGB", (768, 768), (255, 255, 255))
        for column in range(100, 501):
            image.putpixel((column, 384), (0, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        report = validate_inbox_bytes(mutate_inbox_png(buffer.getvalue()))
        self.assertIn("unsupported_png_profile", error_codes(report))

    def test_same_grayscale_pixels_require_frozen_compression_and_single_idat(self) -> None:
        with Image.open(BytesIO((FIXTURES / "synthetic.png").read_bytes())) as image:
            decoded = image.tobytes()
        compression_variant = grayscale_png_with_encoding(
            decoded, level=1, split_idat=False,
        )
        split_variant = grayscale_png_with_encoding(
            decoded, level=9, split_idat=True,
        )
        for png in (compression_variant, split_variant):
            with self.subTest(split_idat=png is split_variant):
                self.assertIn("noncanonical_png_bytes", error_codes(
                    validate_inbox_bytes(mutate_inbox_png(png)),
                ))


class PredictionFormatTests(unittest.TestCase):
    def test_valid_ok_and_reject_predictions_are_not_scored(self) -> None:
        report = validate_prediction_bytes(PREDICTION_RAW, INBOX_RAW)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")
        self.assertEqual(report["prediction_status"], "ok")
        reject_raw = (FIXTURES / "valid_reject_prediction.json").read_bytes()
        reject = validate_prediction_bytes(reject_raw, INBOX_RAW)
        self.assertEqual(reject["status"], "STRUCTURALLY_VALID_NOT_SCORED")
        self.assertEqual(reject["prediction_status"], "reject")
        self.assertIs(reject["training_authorized"], False)

    def test_prediction_duplicate_nonfinite_noncanonical_and_extra_reject(self) -> None:
        duplicate = PREDICTION_RAW.replace(
            b'"training_authorized":false}',
            b'"training_authorized":false,"training_authorized":false}',
        )
        value = json.loads(PREDICTION_RAW)
        value["score"] = float("inf")
        nonfinite = json.dumps(value).replace("Infinity", "NaN").encode() + b"\n"
        extra = json.loads(PREDICTION_RAW)
        extra["atoms"] = []
        cases = (
            (duplicate, "duplicate_key"),
            (nonfinite, "invalid_json"),
            (PREDICTION_RAW.rstrip(), "noncanonical_json"),
            (canonical_json_bytes(extra), "forbidden_chemical_field"),
        )
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, error_codes(validate_prediction_bytes(raw, INBOX_RAW)))

    def test_ids_and_q4_bounds_are_enforced(self) -> None:
        noncontiguous = mutate_prediction(lambda value: value["segments"][1].update(id=2))
        out_of_bounds = mutate_prediction(
            lambda value: value["segments"][0]["polyline_q4"][0].__setitem__(0, 12289),
        )
        self.assertIn("noncontiguous_ids", error_codes(validate_prediction_bytes(noncontiguous, INBOX_RAW)))
        self.assertIn("jsonschema", error_codes(validate_prediction_bytes(out_of_bounds, INBOX_RAW)))


class RasterSupportTests(unittest.TestCase):
    def test_rle_must_be_sorted_nonoverlapping_and_counted(self) -> None:
        unsorted = mutate_prediction(
            lambda value: value["segments"][0].update(
                support_runs=[[384, 200, 1], [384, 100, 1]], support_pixel_count=2,
            ),
        )
        overlap = mutate_prediction(
            lambda value: value["segments"][0].update(
                support_runs=[[384, 100, 151], [384, 200, 100]], support_pixel_count=251,
            ),
        )
        mismatch = mutate_prediction(
            lambda value: value["segments"][0].update(support_pixel_count=200),
        )
        self.assertIn("unsorted_support_runs", error_codes(validate_prediction_bytes(unsorted, INBOX_RAW)))
        self.assertIn("overlapping_support_runs", error_codes(validate_prediction_bytes(overlap, INBOX_RAW)))
        self.assertIn("support_count_mismatch", error_codes(validate_prediction_bytes(mismatch, INBOX_RAW)))

    def test_segment_support_must_reference_decoded_foreground(self) -> None:
        background = mutate_prediction(
            lambda value: value["segments"][0].update(
                support_runs=[[0, 0, 1]], support_pixel_count=1,
            ),
        )
        self.assertIn("support_not_foreground", error_codes(
            validate_prediction_bytes(background, INBOX_RAW),
        ))

    def test_segment_support_may_overlap_but_must_not_duplicate_or_lose_exclusive_pixels(self) -> None:
        valid = validate_prediction_bytes(PREDICTION_RAW, INBOX_RAW)
        self.assertEqual(valid["status"], "STRUCTURALLY_VALID_NOT_SCORED")

        duplicate = mutate_prediction(
            lambda value: value["segments"][1].update(
                support_runs=[[384, 100, 201]], support_pixel_count=201,
            ),
        )
        no_exclusive = mutate_prediction(
            lambda value: value["segments"][1].update(
                support_runs=[[384, 100, 401]], support_pixel_count=401,
            ),
        )
        duplicate_codes = error_codes(validate_prediction_bytes(duplicate, INBOX_RAW))
        self.assertIn("duplicate_segment_support", duplicate_codes)
        self.assertIn("segment_has_no_exclusive_support", duplicate_codes)
        self.assertIn("segment_has_no_exclusive_support", error_codes(
            validate_prediction_bytes(no_exclusive, INBOX_RAW),
        ))

    def test_proper_crossing_may_share_one_pixel_when_both_segments_have_exclusive_support(self) -> None:
        image = Image.new("L", (768, 768), 255)
        for column in range(100, 501):
            image.putpixel((column, 384), 0)
        for row in range(300, 469):
            image.putpixel((300, row), 0)
        inbox = mutate_inbox_png(canonical_grayscale_png(image.tobytes()))

        value = json.loads(PREDICTION_RAW)
        value["segments"] = [
            {
                "id": 0,
                "polyline_q4": [[1600, 6144], [8000, 6144]],
                "support_pixel_count": 401,
                "support_runs": [[384, 100, 401]],
                "width_q4": 16,
            },
            {
                "id": 1,
                "polyline_q4": [[4800, 4800], [4800, 7488]],
                "support_pixel_count": 169,
                "support_runs": [[row, 300, 1] for row in range(300, 469)],
                "width_q4": 16,
            },
        ]
        value["terminals"] = [
            {"center_q4": [1600, 6144], "id": 0, "support_segment_ids": [0]},
            {"center_q4": [8000, 6144], "id": 1, "support_segment_ids": [0]},
            {"center_q4": [4800, 4800], "id": 2, "support_segment_ids": [1]},
            {"center_q4": [4800, 7488], "id": 3, "support_segment_ids": [1]},
        ]
        value["lane_hypotheses"] = [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0, "segment_ids": [0]},
            {"begin_terminal_id": 2, "end_terminal_id": 3, "id": 1, "segment_ids": [1]},
        ]
        value["bond_hypotheses"] = [
            {
                "begin_terminal_id": 0,
                "end_terminal_id": 1,
                "id": 0,
                "lane_endpoint_alignment": [{
                    "begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 0,
                }],
                "lane_ids": [0],
                "order": 1,
            },
            {
                "begin_terminal_id": 2,
                "end_terminal_id": 3,
                "id": 1,
                "lane_endpoint_alignment": [{
                    "begin_terminal_id": 2, "end_terminal_id": 3, "lane_id": 1,
                }],
                "lane_ids": [1],
                "order": 1,
            },
        ]
        value["interpretations"] = [{
            "id": 0,
            "ownership": [
                {"owner_id": 0, "owner_kind": "bond_lane", "segment_id": 0},
                {"owner_id": 1, "owner_kind": "bond_lane", "segment_id": 1},
            ],
            "selected_bond_ids": [0, 1],
        }]
        report = validate_prediction_bytes(canonical_json_bytes(value), inbox)
        self.assertEqual(report["status"], "STRUCTURALLY_VALID_NOT_SCORED")

    def test_geometry_support_binding_rejects_origin_hallucination_and_accepts_radius_boundary(self) -> None:
        origin = mutate_prediction(
            lambda value: value["segments"][0].update(polyline_q4=[[0, 0], [16, 0]]),
        )
        origin_codes = error_codes(validate_prediction_bytes(origin, INBOX_RAW))
        self.assertTrue(
            {"support_far_from_geometry", "polyline_vertex_unsupported"} & origin_codes,
        )

        boundary = mutate_prediction(
            lambda value: [segment.update(
                polyline_q4=[[segment["polyline_q4"][0][0] + 8, 6128],
                             [segment["polyline_q4"][-1][0] + 8, 6128]],
            ) for segment in value["segments"]],
        )
        boundary_report = validate_prediction_bytes(boundary, INBOX_RAW)
        self.assertEqual(boundary_report["status"], "STRUCTURALLY_VALID_NOT_SCORED")

    def test_terminal_must_be_near_referenced_polyline_endpoint(self) -> None:
        far_terminal = mutate_prediction(
            lambda value: value["terminals"][0].update(center_q4=[1600, 0]),
        )
        self.assertIn("terminal_not_near_segment_endpoint", error_codes(
            validate_prediction_bytes(far_terminal, INBOX_RAW),
        ))

    def test_same_or_near_centerline_cannot_be_split_into_fake_double_bond(self) -> None:
        def fake_double(value, offset_q4=0):
            even_runs = [[384, column, 1] for column in range(100, 301, 2)]
            odd_runs = [[384, column, 1] for column in range(101, 300, 2)]
            value["segments"][0].update(
                support_runs=even_runs,
                support_pixel_count=len(even_runs),
                polyline_q4=[[1600, 6144], [4800, 6144]],
            )
            value["segments"][1].update(
                support_runs=odd_runs,
                support_pixel_count=len(odd_runs),
                polyline_q4=[[1600, 6144 + offset_q4], [4800, 6144 + offset_q4]],
            )
            value["terminals"][0].update(support_segment_ids=[0, 1])
            value["terminals"][1].update(support_segment_ids=[0, 1])
            value["terminals"][2].update(center_q4=[4800, 6144], support_segment_ids=[1])
            value["lane_hypotheses"][1].update(begin_terminal_id=0, end_terminal_id=1)
            value["bond_hypotheses"] = [{
                "begin_terminal_id": 0,
                "end_terminal_id": 1,
                "id": 0,
                "lane_endpoint_alignment": [
                    {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 0},
                    {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 1},
                ],
                "lane_ids": [0, 1],
                "order": 2,
            }]
            value["interpretations"][0]["selected_bond_ids"] = [0]

        for offset_q4 in (0, 1):
            forged = mutate_prediction(lambda value: fake_double(value, offset_q4))
            with self.subTest(offset_q4=offset_q4):
                self.assertIn("bond_lane_geometry_overlaid", error_codes(
                    validate_prediction_bytes(forged, INBOX_RAW),
                ))

    def test_parallel_double_lane_boundary_is_valid_and_lane_support_must_be_disjoint(self) -> None:
        valid = validate_prediction_bytes(DOUBLE_PREDICTION_RAW, DOUBLE_INBOX_RAW)
        self.assertEqual(valid["status"], "STRUCTURALLY_VALID_NOT_SCORED")

        value = json.loads(DOUBLE_PREDICTION_RAW)
        value["segments"][1].update(
            support_pixel_count=402,
            support_runs=[[383, 300, 1], [385, 100, 401]],
        )
        overlapping = canonical_json_bytes(value)
        self.assertIn("bond_lane_support_overlap", error_codes(
            validate_prediction_bytes(overlapping, DOUBLE_INBOX_RAW),
        ))


class TopologyAndOwnershipTests(unittest.TestCase):
    def test_terminal_and_lane_references_must_exist_and_be_sorted(self) -> None:
        terminal_unknown = mutate_prediction(
            lambda value: value["terminals"][0].update(support_segment_ids=[2]),
        )
        terminal_unsorted = mutate_prediction(
            lambda value: value["terminals"][1].update(support_segment_ids=[1, 0]),
        )
        lane_unknown = mutate_prediction(
            lambda value: value["lane_hypotheses"][0].update(segment_ids=[2]),
        )
        lane_unsorted = mutate_prediction(
            lambda value: value["lane_hypotheses"][0].update(segment_ids=[1, 0]),
        )
        cases = (
            (terminal_unknown, "unknown_terminal_segment"),
            (terminal_unsorted, "terminal_refs_not_sorted_unique"),
            (lane_unknown, "unknown_lane_segment"),
            (lane_unsorted, "lane_segments_not_sorted_unique"),
        )
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, error_codes(validate_prediction_bytes(raw, INBOX_RAW)))

    def test_bond_lane_count_union_and_shared_endpoints_are_exact(self) -> None:
        wrong_count = mutate_prediction(
            lambda value: value["bond_hypotheses"][0].update(order=2),
        )
        wrong_union = mutate_prediction(
            lambda value: value["bond_hypotheses"][0]["lane_endpoint_alignment"][0].update(lane_id=1),
        )
        wrong_endpoint = mutate_prediction(
            lambda value: value["bond_hypotheses"][0]["lane_endpoint_alignment"][0].update(end_terminal_id=2),
        )
        cases = (
            (wrong_count, "bond_lane_count_mismatch"),
            (wrong_union, "bond_lane_union_mismatch"),
            (wrong_endpoint, "bond_lane_endpoint_mismatch"),
        )
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, error_codes(validate_prediction_bytes(raw, INBOX_RAW)))

    def test_interpretation_owns_every_segment_exactly_once(self) -> None:
        missing = mutate_prediction(
            lambda value: value["interpretations"][0].update(
                ownership=value["interpretations"][0]["ownership"][:1],
            ),
        )
        duplicate = mutate_prediction(
            lambda value: value["interpretations"][0]["ownership"].append(
                copy.deepcopy(value["interpretations"][0]["ownership"][0]),
            ),
        )
        for raw in (missing, duplicate):
            with self.subTest():
                self.assertIn("ownership_not_exact", error_codes(
                    validate_prediction_bytes(raw, INBOX_RAW),
                ))

    def test_selected_bonds_cannot_reuse_lane_and_must_match_ownership(self) -> None:
        def reuse(value):
            bond = value["bond_hypotheses"][1]
            bond["lane_ids"] = [0]
            bond["lane_endpoint_alignment"] = [{
                "begin_terminal_id": 1, "end_terminal_id": 2, "lane_id": 0,
            }]

        reused = mutate_prediction(reuse)
        wrong_owner = mutate_prediction(
            lambda value: value["interpretations"][0]["ownership"][1].update(owner_id=0),
        )
        self.assertIn("selected_lane_reused", error_codes(validate_prediction_bytes(reused, INBOX_RAW)))
        self.assertIn("selected_bond_ownership_mismatch", error_codes(
            validate_prediction_bytes(wrong_owner, INBOX_RAW),
        ))


class CliAndIsolationTests(unittest.TestCase):
    def test_all_cli_errors_are_canonical_structured_exit_two(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "validate_inbox.py"),
            "-",
        ]
        result = subprocess.run(command, input=INBOX_RAW.rstrip(), capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        report = json.loads(result.stderr)
        self.assertEqual(result.stderr, canonical_json_bytes(report))
        self.assertEqual(report["status"], "INVALID")
        self.assertIs(report["training_authorized"], False)

    def test_prediction_and_argument_cli_errors_are_structured_exit_two(self) -> None:
        missing_arguments = subprocess.run(
            [sys.executable, str(ROOT / "validate_predictions.py")],
            capture_output=True,
            check=False,
        )
        self.assertEqual(missing_arguments.returncode, 2)
        missing_report = json.loads(missing_arguments.stderr)
        self.assertEqual(missing_arguments.stderr, canonical_json_bytes(missing_report))
        self.assertIn("invalid_arguments", error_codes(missing_report))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prediction_path = directory / "prediction.json"
            inbox_path = directory / "inbox.json"
            prediction_path.write_bytes(PREDICTION_RAW.rstrip())
            inbox_path.write_bytes(INBOX_RAW)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "validate_predictions.py"),
                    str(prediction_path),
                    "--inbox",
                    str(inbox_path),
                ],
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")
        report = json.loads(result.stderr)
        self.assertEqual(result.stderr, canonical_json_bytes(report))
        self.assertIn("noncanonical_json", error_codes(report))

    def test_validator_sources_do_not_select_data_read_gt_or_run_models(self) -> None:
        sources = [ROOT / "validate_inbox.py", ROOT / "validate_predictions.py", ROOT / "_common.py"]
        forbidden_imports = {"chemvector", "rdkit", "torch", "subprocess"}
        for source_path in sources:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0])
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {"glob", "rglob", "walk"})
            self.assertFalse(imports & forbidden_imports)
            source = source_path.read_text(encoding="utf-8")
            self.assertNotIn("/sealed/", source)
            self.assertNotIn("MR-TADF", source)


if __name__ == "__main__":
    unittest.main()
