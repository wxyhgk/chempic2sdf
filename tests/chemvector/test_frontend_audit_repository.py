"""Tests for persisted causal audit annotations."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from frontend.server import AuditRepository


class AuditRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        assets = self.root / "assets" / "M1"
        assets.mkdir(parents=True)
        (assets / "input.png").write_bytes(b"png")
        (self.root / "audit.json").write_text(
            json.dumps(
                {
                    "summary": {"samples": 1},
                    "records": [
                        {
                            "molecule_id": "M1",
                            "artifacts": {"input": "assets/M1/input.png"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (self.root / "annotations.json").write_text(
            json.dumps({"selection_sha256": "fixed", "annotations": {}}),
            encoding="utf-8",
        )
        self.repository = AuditRepository(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_save_annotation_round_trip(self) -> None:
        saved = self.repository.save_annotation(
            "M1",
            {
                "primary_cause": "ocr_transcription",
                "confidence": "high",
                "visual_tags": ["large_font", "large_font"],
                "note": "N was read as B",
            },
        )

        payload = self.repository.payload()
        self.assertEqual(saved["visual_tags"], ["large_font"])
        self.assertEqual(
            payload["annotations"]["M1"]["primary_cause"],
            "ocr_transcription",
        )

    def test_rejects_unknown_cause_without_writing(self) -> None:
        with self.assertRaisesRegex(ValueError, "未知"):
            self.repository.save_annotation(
                "M1",
                {
                    "primary_cause": "not-a-cause",
                    "confidence": "high",
                    "visual_tags": [],
                    "note": "",
                },
            )

        self.assertEqual(self.repository.payload()["annotations"], {})

    def test_artifact_must_belong_to_selected_record(self) -> None:
        self.assertEqual(
            self.repository.artifact_path("M1", "input"),
            (self.root / "assets/M1/input.png").resolve(),
        )
        with self.assertRaises(FileNotFoundError):
            self.repository.artifact_path("M1", "prediction")


if __name__ == "__main__":
    unittest.main()
