"""Tests for automatic predicted-vertex error classification."""

import unittest

from chemvector.audit_predicted_vertices import summarize_records


class PredictedVertexAuditTests(unittest.TestCase):
    def test_under_count_selects_global_count_hypothesis(self) -> None:
        records = [
            {
                "predicted_vertices": 2,
                "expected_vertices": 3,
                "unmatched_vertices": [],
                "missing_vertices": [
                    {"visible_text": False, "candidate_available": False}
                ],
            },
            {
                "predicted_vertices": 3,
                "expected_vertices": 3,
                "unmatched_vertices": [
                    {
                        "near_multiple_bond": True,
                        "nearest_bond_display_type": "DOUBLE",
                        "nearest_bond_distance_pixels": 2.5,
                    }
                ],
                "missing_vertices": [],
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["count"]["exact"], 0.5)
        self.assertEqual(summary["count"]["under_rate"], 0.5)
        self.assertEqual(
            summary["recommended_hypothesis"], "global_count_and_set_selection"
        )
        self.assertEqual(
            summary["unmatched_vertices"]["nearest_multiple_bond_types"],
            {"DOUBLE": 1},
        )
        self.assertEqual(summary["missing_vertices"]["without_candidate"], 1)
        self.assertEqual(summary["missing_vertices"]["with_candidate"], 0)
        self.assertEqual(
            summary["unmatched_vertices"]["within_bond_3px_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
