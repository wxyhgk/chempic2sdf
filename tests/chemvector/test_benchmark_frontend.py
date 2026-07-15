"""Tests for benchmark frontend status classification."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from frontend.server import BenchmarkRepository, sample_status


class BenchmarkFrontendTests(unittest.TestCase):
    def test_status_uses_exact_then_validity_then_count_agreement(self) -> None:
        self.assertEqual(sample_status({"chemical_match": True}), "exact")
        self.assertEqual(
            sample_status({"chemical_match": False, "sdf_valid": False}),
            "invalid",
        )
        self.assertEqual(
            sample_status(
                {
                    "chemical_match": False,
                    "sdf_valid": True,
                    "atom_count_match": True,
                    "bond_count_match": True,
                }
            ),
            "similar",
        )
        self.assertEqual(
            sample_status(
                {
                    "chemical_match": False,
                    "sdf_valid": True,
                    "atom_count_match": False,
                    "bond_count_match": True,
                }
            ),
            "mismatch",
        )

    def test_repository_skips_non_benchmark_evaluation_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, config in (
                ("benchmark", {"benchmark_manifest": "manifest.csv"}),
                ("diagnostic", {"dataset_root": "pilot"}),
            ):
                run = root / name
                run.mkdir()
                (run / "summary.json").write_text("{}", encoding="utf-8")
                (run / "records.jsonl").write_text("", encoding="utf-8")
                (run / "benchmark_config.json").write_text(
                    json.dumps(config),
                    encoding="utf-8",
                )

            runs = BenchmarkRepository(root).run_directories()

            self.assertEqual([path.name for path in runs], ["benchmark"])


if __name__ == "__main__":
    unittest.main()
