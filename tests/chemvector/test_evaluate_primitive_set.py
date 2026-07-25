import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from Data.sdf.visual_svg_sequence_v0_1 import (
    ParsedVisualSVGSequenceV01,
    serialize_parsed_visual_svg_v0_1,
)
from chemvector.evaluate_primitive_set import (
    build_decode_groups,
    decoded_rows_to_parsed,
    evaluate_decoded_sample,
    normalize_decode_modes,
    normalize_thresholds,
    summarize_prediction_rows,
    validate_shards,
)
from chemvector.primitive_set import LINE, QUAD, TEXT_BOX, TRIANGLE


def decoded_row(
    primitive_type,
    geometry,
    *,
    stroke=0.5,
    query_index=0,
    score=0.9,
):
    return {
        "query_index": query_index,
        "score": score,
        "type": primitive_type,
        "geometry": torch.tensor(geometry, dtype=torch.float32),
        "stroke": stroke,
    }


class PrimitiveSetEvaluationTest(unittest.TestCase):
    def test_decoded_rows_quantize_all_types_and_make_text_box_positive(self):
        rows = [
            decoded_row(LINE, [0.0, 0.0, 1.0, 1.0, 0, 0, 0, 0], stroke=0.0),
            decoded_row(
                TRIANGLE,
                [0.1, 0.1, 0.8, 0.1, 0.2, 0.7, 0, 0],
                stroke=1.0,
                query_index=1,
            ),
            decoded_row(
                QUAD,
                [0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8],
                query_index=2,
            ),
            decoded_row(
                TEXT_BOX,
                [1.0, 1.0, 0.0, 0.0, 0, 0, 0, 0],
                query_index=3,
            ),
        ]

        result = decoded_rows_to_parsed(rows)

        self.assertTrue(result.valid)
        self.assertEqual(result.decoded_counts, {"line": 1, "triangle": 1, "quad": 1, "text_box": 1})
        self.assertEqual(result.renderable_counts, result.decoded_counts)
        self.assertEqual(result.parsed.lines[0].stroke_width, 1)
        self.assertEqual(result.parsed.triangles[0].stroke_width, 63)
        box = result.parsed.text_boxes[0]
        self.assertEqual((box.x0, box.y0, box.x1, box.y1), (2046, 2046, 2047, 2047))
        self.assertIsNotNone(result.sequence)

    def test_invalid_geometry_is_recorded_and_not_counted_as_renderable(self):
        rows = [
            decoded_row(LINE, [0.1, 0.1, 0.8, 0.8, 0, 0, 0, 0]),
            decoded_row(LINE, [0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0], query_index=1),
            decoded_row(
                TRIANGLE,
                [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0, 0],
                query_index=2,
            ),
            decoded_row(
                QUAD,
                [0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.1],
                query_index=3,
            ),
        ]

        result = decoded_rows_to_parsed(rows)

        self.assertEqual(len(result.invalid_objects), 3)
        self.assertEqual({row["query_index"] for row in result.invalid_objects}, {1, 2, 3})
        self.assertEqual(result.decoded_counts["line"], 2)
        self.assertEqual(result.renderable_counts["line"], 1)
        self.assertEqual(result.renderable_counts["triangle"], 0)
        self.assertEqual(result.renderable_counts["quad"], 0)
        self.assertIsNotNone(result.sequence)
        self.assertFalse(result.valid)

    def test_missing_structural_primitive_reports_serialization_error(self):
        result = decoded_rows_to_parsed(
            [decoded_row(TEXT_BOX, [0.5, 0.5, 0.2, 0.1, 0, 0, 0, 0])]
        )

        self.assertIsNone(result.sequence)
        self.assertIn("at least one bond primitive", result.serialization_error)
        self.assertEqual(result.renderable_counts["text_box"], 1)

    def test_invalid_prediction_cannot_be_primitive_exact(self):
        expected_result = decoded_rows_to_parsed(
            [decoded_row(LINE, [0.1, 0.1, 0.8, 0.8, 0, 0, 0, 0])]
        )
        expected_sequence = serialize_parsed_visual_svg_v0_1(expected_result.parsed)
        rows = [
            decoded_row(LINE, [0.1, 0.1, 0.8, 0.8, 0, 0, 0, 0]),
            decoded_row(TRIANGLE, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0, 0]),
        ]

        evaluated = evaluate_decoded_sample(
            sample_key="sample",
            threshold=0.5,
            decoded_rows=rows,
            expected_sequence=expected_sequence,
            bond_layer=Image.new("L", (32, 32), 0),
            line_tolerance=0.0,
            text_box_iou=0.5,
        )

        self.assertFalse(evaluated["primitive_exact"])
        self.assertEqual(evaluated["decoded_counts"]["triangle"], 1)
        self.assertEqual(evaluated["renderable_counts"]["triangle"], 0)
        self.assertEqual(evaluated["invalid_polygon_count"], 1)
        self.assertFalse(evaluated["conditional_valid"])

    def test_summary_uses_decoded_invalid_objects_in_precision_and_counts(self):
        row = {
            "primitive_exact": False,
            "decoded_counts": {"line": 2, "triangle": 1, "quad": 0, "text_box": 0},
            "expected_counts": {"line": 1, "triangle": 0, "quad": 0, "text_box": 0},
            "true_positives": {"line": 1, "triangle": 0, "quad": 0, "text_box": 0},
            "bond_iou": 0.5,
            "invalid_objects": [{"type": "triangle"}],
            "invalid_polygon_count": 1,
            "serialization_error": None,
            "render_error": None,
            "conditional_valid": False,
        }

        summary = summarize_prediction_rows([row])

        self.assertEqual(summary["classes"]["line"]["precision"], 0.5)
        self.assertEqual(summary["classes"]["line"]["count_mae"], 1.0)
        self.assertEqual(summary["classes"]["triangle"]["predicted"], 1)
        self.assertEqual(summary["invalid"]["polygon_objects"], 1)
        self.assertEqual(summary["conditional_on_valid"]["samples"], 0)

    def test_threshold_sweep_and_shard_guard(self):
        self.assertEqual(normalize_decode_modes(None), ("threshold",))
        self.assertEqual(
            normalize_decode_modes(["threshold", "count_head", "threshold"]),
            ("threshold", "count_head"),
        )
        self.assertEqual(
            build_decode_groups(("threshold", "count_head"), (0.3, 0.5)),
            (("threshold", 0.3), ("threshold", 0.5), ("count_head", None)),
        )
        with self.assertRaises(ValueError):
            normalize_decode_modes(["unknown"])
        self.assertEqual(normalize_thresholds(None), (0.5,))
        self.assertEqual(normalize_thresholds([0.7, 0.3, 0.7]), (0.3, 0.7))
        with self.assertRaises(ValueError):
            normalize_thresholds([1.1])
        with tempfile.TemporaryDirectory() as directory:
            validation = Path(directory) / "validation-000000.tar"
            validation.touch()
            internal = Path(directory) / "internal_test-000000.tar"
            internal.touch()
            self.assertEqual(validate_shards([str(validation)]), (str(validation.resolve()),))
            with self.assertRaises(ValueError):
                validate_shards([str(internal)])
            with self.assertRaises(ValueError):
                validate_shards([str(Path(directory) / "*.tar")])

if __name__ == "__main__":
    unittest.main()
