"""Tests for OCR teacher crop preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from chemvector.ocr_teacher import load_anchors, prepare_anchor_crop


class OCRTeacherTests(unittest.TestCase):
    def test_anchor_crop_has_stable_shape(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        crop = prepare_anchor_crop(image, 50, 40)
        self.assertEqual(crop.shape, (96, 192, 3))

    def test_anchor_json_accepts_wrapped_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.json"
            path.write_text(
                json.dumps({"anchors": [{"id": "n1", "x": 12, "y": 34}]}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_anchors(path),
                [{"id": "n1", "x": 12.0, "y": 34.0}],
            )


if __name__ == "__main__":
    unittest.main()
