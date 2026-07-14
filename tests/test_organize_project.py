"""Tests for generated-artifact organization."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from tools.organize_project import organize_outputs, write_manifest


class OrganizeProjectTests(unittest.TestCase):
    def test_artifacts_are_classified_and_moved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            outputs.mkdir()
            checkpoint = outputs / "model-run"
            checkpoint.mkdir()
            (checkpoint / "last.pt").write_bytes(b"checkpoint")
            (outputs / "validation-100").mkdir()
            (outputs / "sample-gd906").mkdir()
            (outputs / "preview").mkdir()
            (outputs / "legacy.sdf").write_text("test", encoding="utf-8")

            rows = organize_outputs(outputs, apply=True)

            categories = {row["name"]: row["category"] for row in rows}
            self.assertEqual(categories["model-run"], "checkpoints")
            self.assertEqual(categories["validation-100"], "evaluations")
            self.assertEqual(categories["sample-gd906"], "predictions")
            self.assertEqual(categories["preview"], "diagnostics")
            self.assertEqual(categories["legacy.sdf"], "diagnostics")
            self.assertTrue((outputs / "checkpoints/model-run/last.pt").is_file())
            self.assertTrue(
                (outputs / "diagnostics/loose-files/legacy.sdf").is_file()
            )

    def test_manifest_is_relative_and_preserved_on_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            run = outputs / "first-run"
            run.mkdir(parents=True)
            (run / "last.pt").write_bytes(b"checkpoint")
            first_rows = organize_outputs(outputs, apply=True)
            write_manifest(outputs, first_rows)

            second_rows = organize_outputs(outputs, apply=True)
            write_manifest(outputs, second_rows)

            with (outputs / "MANIFEST.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["old_path"], "outputs/first-run")
            self.assertEqual(
                manifest[0]["new_path"],
                "outputs/checkpoints/first-run",
            )


if __name__ == "__main__":
    unittest.main()
