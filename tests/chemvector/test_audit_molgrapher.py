"""Tests for MolGrapher keypoint audit helpers."""

from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from chemvector.audit_molgrapher import analyze_sample, select_clean_records


def sample_with_line(key: str = "sample") -> dict:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.line((24, 64, 104, 64), fill="black", width=3)
    atoms = [
        {
            "id": index,
            "element": "C",
            "formal_charge": 0,
            "isotope": 0,
            "radical_electrons": 0,
            "explicit_h_count": 0,
            "implicit_h_count": 3,
            "no_implicit": False,
        }
        for index in range(2)
    ]
    return {
        "key": key,
        "image": image,
        "chemmol": {"molecule_id": key, "atoms": atoms},
        "render": {
            "width": 128,
            "height": 128,
            "atoms": [
                {"atom_id": 0, "image_x": 24.0, "image_y": 64.0},
                {"atom_id": 1, "image_x": 104.0, "image_y": 64.0},
            ],
            "bonds": [
                {
                    "begin_image_x": 24.0,
                    "begin_image_y": 64.0,
                    "end_image_x": 104.0,
                    "end_image_y": 64.0,
                    "display_type": "SINGLE",
                }
            ],
        },
    }


class AuditMolGrapherTests(unittest.TestCase):
    def test_clean_line_has_aligned_anchors(self) -> None:
        record, unexplained = analyze_sample(sample_with_line(), 220)

        self.assertEqual(record["flags"], [])
        self.assertEqual(record["atoms"], 2)
        self.assertEqual(record["maximum_anchor_dark_distance"], 0.0)
        self.assertFalse(unexplained.any())

    def test_selection_round_robins_clean_buckets(self) -> None:
        records = []
        for index, atoms in enumerate((8, 12, 30, 9, 13, 31)):
            records.append(
                {
                    "key": f"sample-{index}",
                    "atoms": atoms,
                    "visible_text_atoms": index % 2,
                    "multiple_bonds": index % 3 == 0,
                    "unexplained_ink_fraction": 0.0,
                    "maximum_anchor_dark_distance": 0.0,
                    "flags": [],
                    "warnings": [],
                }
            )

        selected = select_clean_records(records, 4)

        self.assertEqual(len(selected), 4)
        self.assertEqual(len({item["key"] for item in selected}), 4)


if __name__ == "__main__":
    unittest.main()
