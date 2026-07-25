from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


exporter = load_module(
    "export_real_predicted_bboxes",
    "Data/ocr/open_text_v0_1/export_real_predicted_bboxes.py",
)


class OpenTextPredictionExportTests(unittest.TestCase):
    def test_predicted_bbox_uses_bbox_center_and_clips(self) -> None:
        box = exporter.predicted_bbox(
            {
                "x": 0.5,
                "y": 0.5,
                "bbox_x": 0.01,
                "bbox_y": 0.99,
                "half_width": 0.02,
                "half_height": 0.03,
            }
        )
        self.assertEqual(box, [0.0, 737.28, 23.04, 768.0])

    def test_match_predictions_is_one_to_one_and_radius_limited(self) -> None:
        objects = [
            {"x": 10 / 768, "y": 10 / 768},
            {"x": 21 / 768, "y": 20 / 768},
            {"x": 500 / 768, "y": 500 / 768},
        ]
        atoms = [
            {
                "atom_id": 1,
                "visible_text": True,
                "bbox": [1, 1, 2, 2],
                "center": [10, 10],
            },
            {
                "atom_id": 2,
                "visible_text": True,
                "bbox": [1, 1, 2, 2],
                "center": [20, 20],
            },
            {
                "atom_id": 3,
                "visible_text": False,
                "bbox": None,
                "center": [500, 500],
            },
        ]
        matches = exporter.match_predictions(objects, atoms, radius_pixels=8.0)
        self.assertEqual(
            [
                (atom["atom_id"], round(distance, 3))
                for atom, _, distance in matches
            ],
            [(1, 0.0), (2, 1.0)],
        )

    def test_select_balanced_requires_exact_quota(self) -> None:
        records = [
            {
                "split": "validation",
                "text": label,
                "source_sample_key": f"{index:09d}",
                "source_atom_id": 0,
                "matching_distance_pixels": float(index),
                "score": 1.0,
            }
            for index, label in enumerate(("N", "O", "N"))
        ]
        selected = exporter.select_balanced(
            records, split="validation", count=3
        )
        self.assertEqual(len(selected), 3)
        with self.assertRaisesRegex(RuntimeError, "only 3"):
            exporter.select_balanced(records, split="validation", count=4)


if __name__ == "__main__":
    unittest.main()
