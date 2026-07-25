import unittest

import torch

from Data.sdf.visual_svg_sequence_v0_1 import (
    LineRecordV01,
    ParsedVisualSVGSequenceV01,
)
from chemvector.audit_primitive_set_geometry import (
    build_parser,
    compare_geometry_outputs,
    distribution,
    line_geometry_error_records,
    summarize_line_geometry,
)


def parsed_lines(*lines):
    return ParsedVisualSVGSequenceV01(tuple(lines), (), (), ())


class PrimitiveSetGeometryAuditTest(unittest.TestCase):
    def test_line_geometry_is_endpoint_order_invariant(self):
        expected = parsed_lines(LineRecordV01(0, 0, 10, 0, 2))
        predicted = parsed_lines(LineRecordV01(10, 0, 0, 0, 2))

        rows = line_geometry_error_records(predicted, expected)

        self.assertEqual(len(rows), 1)
        for value in rows[0].values():
            self.assertAlmostEqual(value, 0.0)

    def test_line_geometry_reports_center_endpoint_length_and_angle(self):
        expected = parsed_lines(LineRecordV01(0, 0, 10, 0, 2))
        predicted = parsed_lines(LineRecordV01(5, 0, 5, 20, 2))

        row = line_geometry_error_records(predicted, expected)[0]

        self.assertAlmostEqual(row["center_error_q"], 10.0)
        self.assertAlmostEqual(row["max_endpoint_error_q"], 20.6155281281)
        self.assertAlmostEqual(row["length_error_q"], 10.0)
        self.assertAlmostEqual(row["angle_error_degrees"], 90.0)

    def test_line_summary_has_fixed_recall_buckets_and_quantiles(self):
        errors = [
            {
                "center_error_q": 1.0,
                "max_endpoint_error_q": 2.0,
                "length_error_q": 3.0,
                "angle_error_degrees": 4.0,
            },
            {
                "center_error_q": 9.0,
                "max_endpoint_error_q": 10.0,
                "length_error_q": 11.0,
                "angle_error_degrees": 12.0,
            },
        ]

        summary = summarize_line_geometry(
            expected_lines=4,
            tolerance_hits={
                1.0: 0,
                2.0: 1,
                4.0: 1,
                8.0: 1,
                16.0: 2,
                32.0: 2,
                64.0: 3,
                128.0: 4,
            },
            errors=errors,
        )

        self.assertEqual(
            summary["recall"],
            {
                "1q": 0.0,
                "2q": 0.25,
                "4q": 0.25,
                "8q": 0.25,
                "16q": 0.5,
                "32q": 0.5,
                "64q": 0.75,
                "128q": 1.0,
            },
        )
        self.assertEqual(summary["matched_pairs"], 2)
        self.assertEqual(summary["unmatched_expected_lines"], 2)
        self.assertEqual(summary["center_error_q"]["p50"], 5.0)
        self.assertAlmostEqual(summary["center_error_q"]["p90"], 8.2)

    def test_real_white_comparison_reports_geometry_and_decisions(self):
        real = {
            "geometry": torch.zeros(1, 2, 8),
            "text_geometry": torch.zeros(1, 2, 8),
            "type_logits": torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
            "count_logits": torch.tensor([[0.0, 2.0, 0.0]]),
        }
        white = {
            "geometry": torch.ones(1, 2, 8) / 2047.0,
            "text_geometry": torch.ones(1, 2, 8) * 2.0 / 2047.0,
            "type_logits": real["type_logits"].clone(),
            "count_logits": torch.tensor([[2.0, 0.0, 0.0]]),
        }

        result = compare_geometry_outputs(real, white)

        self.assertAlmostEqual(result["geometry_coordinate_abs_q"]["p50"], 1.0)
        self.assertAlmostEqual(result["text_geometry_coordinate_abs_q"]["p90"], 2.0)
        self.assertEqual(result["type_argmax_same_rate"], 1.0)
        self.assertEqual(result["count_argmax_same_rate"], 0.0)

    def test_empty_distribution_and_cli_contract(self):
        self.assertEqual(distribution([])["count"], 0)
        args = build_parser().parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--shard",
                "validation.tar",
                "--output",
                "audit.json",
                "--compare-white",
            ]
        )
        self.assertTrue(args.compare_white)


if __name__ == "__main__":
    unittest.main()
