import unittest

from chemvector.audit_primitive_set_residuals import (
    build_residual_records,
    bond_length_signature,
    classify_bond_group,
    compare_coarse_outputs,
    coarse_error_bin,
    length_bin,
    orientation_bin,
    summarize_buckets,
    summarize_records,
    text_distance_bin,
)


def sequence(*records):
    return "\n".join(
        ["<visual_svg v=0.1 units=q2047>", *records, "<visual_svg_end>", ""]
    )


class PrimitiveSetResidualAuditTest(unittest.TestCase):
    def test_bucket_boundaries_and_bond_roles(self):
        self.assertEqual(length_bin(15.9), "00:<16q")
        self.assertEqual(length_bin(16.0), "01:16-32q")
        self.assertEqual(orientation_bin(0.0), "00:horizontal_0-15deg")
        self.assertEqual(orientation_bin(90.0), "03:vertical_75-90deg")
        self.assertEqual(text_distance_bin(0.0), "00:touching_0-4q")
        self.assertEqual(coarse_error_bin(129.0), "03:128-256q")
        self.assertEqual(
            bond_length_signature("hashed_wedge", 20.0),
            "hashed_wedge|short_<32q",
        )
        self.assertEqual(
            classify_bond_group({"direction": "BEGINDASH", "display_type": "SINGLE"}),
            "hashed_wedge",
        )
        self.assertEqual(
            classify_bond_group({"direction": "NONE", "display_type": "DOUBLE"}),
            "double",
        )

    def test_residual_records_keep_query_coarse_and_sidecar_role(self):
        row = {
            "sample_key": "sample",
            "expected_sequence": sequence(
                "<line 0 0 100 0 2>",
                "<line 0 50 100 50 2>",
                "<text_box 100 0 120 20>",
            ),
            "renderable_objects": [
                {
                    "kind": "line",
                    "query_index": 7,
                    "score": 0.9,
                    "quantized_points": [[1, 1], [101, 1]],
                    "stroke_width_q": 2,
                },
                {
                    "kind": "line",
                    "query_index": 9,
                    "score": 0.8,
                    "quantized_points": [[0, 70], [100, 70]],
                    "stroke_width_q": 2,
                },
            ],
        }
        primitives = {
            "primitives": [
                {
                    "canonical_index": 0,
                    "primitive_id": 10,
                    "kind": "line",
                    "quantized_points": [[0, 0], [100, 0]],
                    "stroke_width_q": 2,
                },
                {
                    "canonical_index": 1,
                    "primitive_id": 11,
                    "kind": "line",
                    "quantized_points": [[0, 50], [100, 50]],
                    "stroke_width_q": 2,
                },
                {
                    "canonical_index": 2,
                    "primitive_id": 12,
                    "kind": "text_box",
                    "quantized_points": [[100, 0], [120, 20]],
                },
            ],
            "bond_groups": [
                {
                    "bond_id": 3,
                    "display_type": "DOUBLE",
                    "chemical_type": "DOUBLE",
                    "direction": "NONE",
                    "primitive_ids": [10, 11],
                }
            ],
        }
        coarse = {
            7: [0.0, 0.0, 100 / 2047.0, 0.0, 0, 0, 0, 0],
            9: [0.0, 60 / 2047.0, 100 / 2047.0, 60 / 2047.0, 0, 0, 0, 0],
        }

        records, counts = build_residual_records(row, primitives, coarse)

        self.assertEqual(counts["expected_lines"], 2)
        self.assertEqual(records[0]["bond_category"], "double")
        self.assertEqual(records[0]["group_line_count"], 2)
        self.assertEqual(records[0]["query_index"], 7)
        self.assertAlmostEqual(records[0]["final_error_q"], 2**0.5)
        self.assertAlmostEqual(records[0]["coarse_error_q"], 0.0)
        self.assertEqual(records[0]["text_distance_bin"], "00:touching_0-4q")
        self.assertEqual(records[0]["endpoint_failure_mode"], "00:both_within_8q")
        self.assertEqual(records[1]["final_error_q"], 20.0)
        self.assertEqual(records[1]["endpoint_failure_mode"], "02:both_outside_8q")

    def test_summaries_report_failure_share(self):
        records = [
            {"final_error_q": 2.0, "coarse_error_q": 20.0, "refinement_gain_q": 18.0, "kind": "a"},
            {"final_error_q": 10.0, "coarse_error_q": 30.0, "refinement_gain_q": 20.0, "kind": "a"},
            {"final_error_q": 20.0, "coarse_error_q": 40.0, "refinement_gain_q": 20.0, "kind": "b"},
        ]

        overall = summarize_records(records)
        buckets = summarize_buckets(records, "kind")

        self.assertAlmostEqual(overall["recall_8q"], 1 / 3)
        self.assertEqual(overall["failure_8q"], 2)
        self.assertAlmostEqual(buckets["a"]["failure_share"], 0.5)
        self.assertAlmostEqual(buckets["b"]["failure_share"], 0.5)

    def test_coarse_output_comparison_is_query_aligned(self):
        left = {"sample": {0: [0.0, 0.5], 1: [1.0, 0.25]}}
        right = {"sample": {0: [0.0, 0.5], 1: [1.0, 0.5]}}

        result = compare_coarse_outputs(left, right)

        self.assertEqual(result["exact_query_rate"], 0.5)
        self.assertAlmostEqual(result["query_max_abs_q"]["max"], 2047.0 * 0.25)


if __name__ == "__main__":
    unittest.main()
