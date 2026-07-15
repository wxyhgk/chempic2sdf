"""Tests for image-grounded ChemSVG graph repair."""

from __future__ import annotations

import unittest

from PIL import Image, ImageDraw
from rdkit import Chem

from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    StrokeRecordV4,
    VertexRecordV4,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
    serialize_parsed_chemsvg_v4,
)
from chemvector.repair_v4 import InkGraphRepairConfig, repair_chemsvg_v4_from_image


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

    def test_model_candidate_mode_does_not_invent_unscored_edge(self) -> None:
        vertices = (self._vertex(100, 512), self._vertex(900, 512))
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, ())
        )

        with self.assertRaisesRegex(ValueError, "not connected"):
            repair_chemsvg_v4_from_image(
                sequence,
                self._image(vertices, ((0, 1),)),
                config=InkGraphRepairConfig(candidate_pair_mode="model"),
            )

    def test_model_connect_adds_only_the_bridge_needed_for_connectivity(self) -> None:
        vertices = tuple(self._vertex(x, 512) for x in (100, 500, 900))
        source_stroke = StrokeRecordV4(
            begin_x=vertices[0].x,
            begin_y=vertices[0].y,
            end_x=vertices[1].x,
            end_y=vertices[1].y,
            display_type="SINGLE",
            bond_type="SINGLE",
            direction="NONE",
            stereo="STEREONONE",
        )
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, (source_stroke,))
        )

        result = repair_chemsvg_v4_from_image(
            sequence,
            self._image(vertices, ((0, 1), (1, 2))),
            config=InkGraphRepairConfig(candidate_pair_mode="model_connect"),
        )

        self.assertEqual(result.strokes, 2)
        self.assertEqual(Chem.MolToSmiles(mol_from_chemsvg_v4(result.sequence)), "CCC")

    def test_model_probability_can_restore_faint_bridge_and_bond_type(self) -> None:
        vertices = tuple(self._vertex(x, 512) for x in (100, 500, 900))
        source_stroke = StrokeRecordV4(
            begin_x=vertices[0].x,
            begin_y=vertices[0].y,
            end_x=vertices[1].x,
            end_y=vertices[1].y,
            display_type="SINGLE",
            bond_type="SINGLE",
            direction="NONE",
            stereo="STEREONONE",
        )
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, (source_stroke,))
        )
        bridge_coordinates = tuple(
            sorted(((vertices[1].x, vertices[1].y), (vertices[2].x, vertices[2].y)))
        )

        result = repair_chemsvg_v4_from_image(
            sequence,
            self._image(vertices, ((0, 1),)),
            pair_predictions={bridge_coordinates: (0.3, "DOUBLE")},
            config=InkGraphRepairConfig(candidate_pair_mode="model_connect"),
        )
        parsed = parse_chemsvg_v4(result.sequence)

        self.assertEqual(result.model_bridge_strokes, 1)
        self.assertEqual({stroke.bond_type for stroke in parsed.strokes}, {"SINGLE", "DOUBLE"})

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

    def test_invalid_chemistry_can_be_retained_for_diagnostics(self) -> None:
        vertices = (
            self._vertex(100, 512, "Cl"),
            self._vertex(400, 300),
            self._vertex(400, 500),
            self._vertex(400, 700),
        )
        edges = ((0, 1), (0, 2), (0, 3))
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, ())
        )

        result = repair_chemsvg_v4_from_image(
            sequence,
            self._image(vertices, edges),
            validate_chemistry=False,
        )

        self.assertEqual(result.strokes, 3)
        with self.assertRaises(ValueError):
            mol_from_chemsvg_v4(result.sequence)

    def test_text_box_trims_a_large_visible_label_gap(self) -> None:
        vertices = (
            self._vertex(200, 512, "N"),
            self._vertex(820, 512),
        )
        sequence = serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, ())
        )
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        draw.line((90, 128, 205, 128), fill="black", width=2)

        with self.assertRaisesRegex(ValueError, "not connected"):
            repair_chemsvg_v4_from_image(sequence, image)

        result = repair_chemsvg_v4_from_image(
            sequence,
            image,
            endpoint_boxes={(200, 512): (40.0, 110.0, 90.0, 145.0)},
        )

        self.assertEqual(result.strokes, 1)


if __name__ == "__main__":
    unittest.main()
