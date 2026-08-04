"""Tests for P3-E8 Count bottleneck summaries."""

import unittest

from chemvector.training.atoms.audit_g3_p3e8_predicted_count_router import (
    count_summary,
)


class G3P3E8PredictedCountTests(unittest.TestCase):
    def test_count_summary_reports_signed_errors(self) -> None:
        rows = [
            {"atom_count": 3, "base": {"decoded_count": 3}},
            {"atom_count": 4, "base": {"decoded_count": 2}},
            {"atom_count": 2, "base": {"decoded_count": 3}},
        ]
        summary = count_summary(rows)
        self.assertEqual(summary["count_exact"], 1)
        self.assertEqual(summary["count_mae"], 1.0)
        self.assertEqual(summary["count_error_min"], -2)
        self.assertEqual(summary["count_error_max"], 1)


if __name__ == "__main__":
    unittest.main()
