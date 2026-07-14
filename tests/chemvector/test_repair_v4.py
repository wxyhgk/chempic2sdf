"""Tests for image-grounded ChemSVG graph repair."""

from __future__ import annotations

import unittest

from PIL import Image, ImageDraw
from rdkit import Chem

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    VertexRecordV4,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
    serialize_parsed_chemsvg_v4,
)
from chemvector.repair_v4 import repair_chemsvg_v4_from_image


class InkGraphRepairTests(unittest.TestCase):
    @staticmethod
    def _vertex(x: int, y: int, element: str = "C", aromatic: bool = False):
        return VertexRecordV4(
            x=x,
            y=y,
            element=element,
            formal_charge=0,
            isotope=0,
            aromatic=aromatic,
            explicit_h_count=0,
            no_implicit=False,
            radical_electrons=0,
            canonical_chiral_tag="CHI_UNSPECIFIED",
        )

    @staticmethod
    def _image(vertices, edges):
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        for begin, end in edges:
            left, right = vertices[begin], vertices[end]
            draw.line(
                (
                    left.x / 1023 * 255,
                    left.y / 1023 * 255,
                    right.x / 1023 * 255,
                    right.y / 1023 * 255,
                ),
                fill="black",
                width=2,
            )
        return image

    def test_long_visible_stroke_has_no_distance_cutoff(self) -> None:
        vertices = (self._vertex(100, 512), self._vertex(900, 512))
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, ())
        )
        result = repair_chemsvg_v4_from_image(
            sequence,
            self._image(vertices, ((0, 1),)),
        )
        self.assertEqual(result.strokes, 1)
        self.assertEqual(Chem.MolToSmiles(mol_from_chemsvg_v4(result.sequence)), "CC")

    def test_intermediate_vertex_blocks_collinear_shortcut(self) -> None:
        vertices = tuple(self._vertex(x, 512) for x in (100, 500, 900))
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, ())
        )
        result = repair_chemsvg_v4_from_image(
            sequence,
            self._image(vertices, ((0, 1), (1, 2))),
        )
        self.assertEqual(result.strokes, 2)
        self.assertEqual(Chem.MolToSmiles(mol_from_chemsvg_v4(result.sequence)), "CCC")

    def test_close_duplicate_visible_label_is_merged(self) -> None:
        vertices = (
            self._vertex(100, 512),
            self._vertex(500, 512, "F"),
            self._vertex(510, 512, "F"),
        )
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, ())
        )
        result = repair_chemsvg_v4_from_image(
            sequence,
            self._image(vertices, ((0, 1),)),
        )
        self.assertEqual(result.vertices, 2)
        self.assertEqual(result.removed_coordinates, ((510, 512),))
        self.assertEqual(Chem.MolToSmiles(mol_from_chemsvg_v4(result.sequence)), "CF")

    def test_aromatic_votes_propagate_around_visible_ring(self) -> None:
        vertices = tuple(
            self._vertex(x, y, aromatic=index < 4)
            for index, (x, y) in enumerate(
                ((300, 200), (700, 200), (900, 512), (700, 824), (300, 824), (100, 512))
            )
        )
        edges = tuple((index, (index + 1) % 6) for index in range(6))
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, ())
        )
        result = repair_chemsvg_v4_from_image(sequence, self._image(vertices, edges))
        parsed = parse_chemsvg_v4(result.sequence)
        self.assertEqual(result.aromatic_strokes, 6)
        self.assertTrue(all(vertex.aromatic for vertex in parsed.vertices))
        self.assertEqual(Chem.MolToSmiles(mol_from_chemsvg_v4(result.sequence)), "c1ccccc1")


if __name__ == "__main__":
    unittest.main()
