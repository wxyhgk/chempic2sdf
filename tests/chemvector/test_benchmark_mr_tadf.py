"""Tests for MR-TADF benchmark aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from chemvector.benchmark_mr_tadf import (
    ensure_benchmark_config,
    failure_category,
    summarize_records,
)


class MrTadfBenchmarkMetricTests(unittest.TestCase):
    def test_summary_reports_overall_and_difficulty_rates(self) -> None:
        records = [
            {
                "difficulty": "standard_bond_length",
                "sdf_valid": True,
                "formula_match": True,
                "atom_count_match": True,
                "bond_count_match": True,
                "nonstereo_structure_match": True,
                "chemical_match": True,
                "seconds": 0.2,
            },
            {
                "difficulty": "extreme_long_bond",
                "sdf_valid": False,
                "formula_match": False,
                "atom_count_match": False,
                "bond_count_match": False,
                "nonstereo_structure_match": False,
                "chemical_match": False,
                "failure_category": "disconnected",
                "seconds": 0.4,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["chemical_match_rate"], 0.5)
        self.assertAlmostEqual(summary["average_seconds"], 0.3)
        self.assertEqual(summary["failure_categories"], {"disconnected": 1})
        self.assertEqual(
            summary["difficulty_rates"]["extreme_long_bond"]["sdf_valid_rate"],
            0.0,
        )

    def test_failure_categories_are_stable(self) -> None:
        self.assertEqual(
            failure_category(ValueError("Ink graph is not connected")),
            "disconnected",
        )
        self.assertEqual(failure_category(ValueError("Explicit valence")), "valence")
        self.assertEqual(failure_category(ValueError("Can't kekulize")), "kekulize")

    def test_benchmark_config_requires_an_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark_config.json"
            ensure_benchmark_config(path, {"checkpoint_sha256": "abc"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"checkpoint_sha256": "abc"},
            )
            ensure_benchmark_config(path, {"checkpoint_sha256": "abc"})
            with self.assertRaises(ValueError):
                ensure_benchmark_config(
                    path,
                    {"checkpoint_sha256": "abc", "new_parameter": 1},
                )


if __name__ == "__main__":
    unittest.main()
