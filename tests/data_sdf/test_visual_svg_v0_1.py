import json
import unittest

from Data.sdf.visual_primitives_v0_1 import extract_visual_primitives_v0_1
from Data.sdf.visual_svg_sequence_v0_1 import (
    LineRecordV01,
    ParsedVisualSVGSequenceV01,
    QuadRecordV01,
    TextBoxRecordV01,
    TriangleRecordV01,
    VisualSVGSequenceError,
    parse_visual_svg_v0_1,
    serialize_parsed_visual_svg_v0_1,
)
from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


class VisualSVGSequenceV01Test(unittest.TestCase):
    def test_sequence_and_tokenizer_roundtrip(self):
        parsed = ParsedVisualSVGSequenceV01(
            (LineRecordV01(20, 10, 5, 4, 15),),
            (TriangleRecordV01(1, 1, 4, 1, 2, 5, 6),),
            (QuadRecordV01(10, 10, 20, 10, 20, 20, 10, 20, 6),),
            (TextBoxRecordV01(30, 30, 40, 45),),
        )
        sequence = serialize_parsed_visual_svg_v0_1(parsed)
        tokenizer = VisualSVGTokenizerV01()
        encoded = tokenizer.encode(sequence)
        self.assertEqual(tokenizer.decode(encoded), sequence)
        self.assertEqual(parse_visual_svg_v0_1(sequence).primitive_count, 4)

    def test_extracts_physical_lines_and_polygons(self):
        svg = b"""<svg xmlns='http://www.w3.org/2000/svg' width='768' height='768'>
<path class='bond-0 atom-0 atom-1' d='M 10,10 L 100,10' style='fill:none;stroke:#000;stroke-width:2.0px'/>
<path class='bond-0 atom-0 atom-1' d='M 10,20 L 100,20' style='fill:none;stroke:#000;stroke-width:2.0px'/>
<path class='bond-1 atom-1 atom-2' d='M 100,10 L 150,30 L 150,50 Z' style='fill:#000;stroke:#000;stroke-width:0.6px'/>
<path class='bond-2 atom-2 atom-3' d='M 150,50 L 180,40 L 190,60 L 160,70 Z' style='fill:#000;stroke:#000;stroke-width:0.6px'/>
<path class='atom-3' d='M 1,1 L 2,2'/>
</svg>"""
        layers = {
            "sample_key": "sample",
            "molecule_id": "molecule",
            "width": 768,
            "height": 768,
            "atoms": [
                {"atom_id": 0, "visible_text": False, "bbox": None},
                {"atom_id": 1, "visible_text": False, "bbox": None},
                {"atom_id": 2, "visible_text": False, "bbox": None},
                {"atom_id": 3, "visible_text": True, "bbox": [190, 50, 210, 70], "text": "N"},
            ],
            "bonds": [
                {"bond_id": 0, "begin_atom": 0, "end_atom": 1, "chemical_type": "DOUBLE", "display_type": "DOUBLE", "direction": "NONE"},
                {"bond_id": 1, "begin_atom": 1, "end_atom": 2, "chemical_type": "SINGLE", "display_type": "SINGLE", "direction": "BEGINWEDGE"},
                {"bond_id": 2, "begin_atom": 2, "end_atom": 3, "chemical_type": "SINGLE", "display_type": "SINGLE", "direction": "BEGINWEDGE"},
            ],
        }
        payload = extract_visual_primitives_v0_1(svg, layers)
        counts = {}
        for primitive in payload["primitives"]:
            counts[primitive["kind"]] = counts.get(primitive["kind"], 0) + 1
        self.assertEqual(counts, {"line": 2, "triangle": 1, "quad": 1, "text_box": 1})
        self.assertEqual(len(payload["bond_groups"][0]["primitive_ids"]), 2)

    def test_quad_cycle_has_one_canonical_form(self):
        points = ((10, 10), (30, 10), (25, 30), (12, 25))
        sequences = set()
        for values in (
            points,
            points[1:] + points[:1],
            tuple(reversed(points)),
            tuple(reversed(points[2:] + points[:2])),
        ):
            record = QuadRecordV01(
                *(coordinate for point in values for coordinate in point),
                6,
            )
            sequences.add(
                serialize_parsed_visual_svg_v0_1(
                    ParsedVisualSVGSequenceV01((), (), (record,), ())
                )
            )
        self.assertEqual(len(sequences), 1)

    def test_rejects_self_intersecting_quad(self):
        with self.assertRaises(VisualSVGSequenceError):
            serialize_parsed_visual_svg_v0_1(
                ParsedVisualSVGSequenceV01(
                    (),
                    (),
                    (QuadRecordV01(10, 10, 30, 30, 10, 30, 30, 10, 6),),
                    (),
                )
            )


if __name__ == "__main__":
    unittest.main()
