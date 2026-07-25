import unittest

import torch

from chemvector.audit_primitive_set_output_heads import (
    class_count_oracle_indices,
    distribution,
    global_topk_indices,
    summarize_text_boxes,
    text_box_error_record,
    text_box_iou_from_geometry,
)


class PrimitiveSetOutputHeadAuditTest(unittest.TestCase):
    def test_global_topk_reproduces_object_minus_background_ranking(self):
        probabilities = torch.tensor(
            [
                [0.10, 0.70, 0.10, 0.05, 0.05],
                [0.45, 0.50, 0.02, 0.02, 0.01],
                [0.01, 0.10, 0.10, 0.10, 0.69],
            ]
        )

        selected = global_topk_indices(probabilities, 2)

        self.assertEqual(selected.tolist(), [2, 0])

    def test_class_count_oracle_is_unique_and_obeys_requested_counts(self):
        probabilities = torch.tensor(
            [
                [0.01, 0.90, 0.01, 0.01, 0.07],
                [0.01, 0.80, 0.01, 0.01, 0.17],
                [0.01, 0.10, 0.01, 0.01, 0.87],
                [0.01, 0.20, 0.01, 0.01, 0.77],
            ]
        )

        selected = class_count_oracle_indices(probabilities, {1: 2, 4: 1})

        self.assertEqual(len({query for query, _ in selected}), 3)
        self.assertEqual(sum(type_id == 1 for _, type_id in selected), 2)
        self.assertEqual(sum(type_id == 4 for _, type_id in selected), 1)
        self.assertEqual({query for query, _ in selected}, {0, 1, 2})

    def test_text_box_iou_and_reference_error_are_interpretable(self):
        expected = torch.tensor([0.50, 0.50, 0.20, 0.10])
        predicted = torch.tensor([0.55, 0.50, 0.20, 0.10])
        reference = torch.tensor([0.40, 0.50])

        iou = text_box_iou_from_geometry(predicted, expected)
        record = text_box_error_record(predicted, expected, reference)
        summary = summarize_text_boxes([record])

        self.assertAlmostEqual(iou, 0.6, places=5)
        self.assertAlmostEqual(record["center_error_q"], 0.05 * 2047, places=3)
        self.assertAlmostEqual(
            record["target_center_to_reference_q"], 0.10 * 2047, places=3
        )
        self.assertEqual(summary["iou_recall"]["at_0.5"], 1.0)

    def test_empty_distribution_is_explicit(self):
        self.assertEqual(distribution([])["count"], 0)


if __name__ == "__main__":
    unittest.main()
