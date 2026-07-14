#!/usr/bin/env python3
"""Unit tests for coordinate-grounded ChemSVG v0.4."""

from __future__ import annotations

import unittest

from rdkit import Chem

from .chemmol_sequence import ChemMolSequenceError
from .chemsvg_sequence_v4 import (
    ChemSVGCoordinateCollisionError,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
    serialize_chemsvg_v4,
    validate_chemsvg_v4_roundtrip,
)
from .test_chemmol_sequence import make_annotations


class ChemSVGSequenceV4Tests(unittest.TestCase):
    def test_representative_chemistry_roundtrips(self) -> None:
        for smiles in (
            "CCO",
            "c1cc[nH]c1",
            "F/C=C/F",
            "N[C@@H](C)C(=O)O",
            "C[N+](C)(C)C",
            "[CH3]",
        ):
            with self.subTest(smiles=smiles):
                chemmol, render = make_annotations(smiles)
                sequence = serialize_chemsvg_v4(chemmol, render)
                parsed = parse_chemsvg_v4(sequence)
                self.assertEqual(len(parsed.vertices), len(chemmol["atoms"]))
                self.assertEqual(len(parsed.strokes), len(chemmol["bonds"]))
                validate_chemsvg_v4_roundtrip(
                    sequence,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )

    def test_strokes_connect_coordinates_instead_of_atom_ids(self) -> None:
        chemmol, render = make_annotations("CCO")
        sequence = serialize_chemsvg_v4(chemmol, render)
        self.assertNotIn("<b ", sequence)
        parsed = parse_chemsvg_v4(sequence)
        coordinates = {(vertex.x, vertex.y) for vertex in parsed.vertices}
        for stroke in parsed.strokes:
            self.assertIn(stroke.begin, coordinates)
            self.assertIn(stroke.end, coordinates)
        mol = mol_from_chemsvg_v4(sequence)
        self.assertEqual(Chem.MolToSmiles(mol), "CCO")

    def test_duplicate_visual_coordinates_are_rejected(self) -> None:
        chemmol, render = make_annotations("CC")
        render["atoms"][1]["quantized_x"] = render["atoms"][0]["quantized_x"]
        render["atoms"][1]["quantized_y"] = render["atoms"][0]["quantized_y"]
        with self.assertRaises(ChemSVGCoordinateCollisionError):
            serialize_chemsvg_v4(chemmol, render)

    def test_unknown_stroke_endpoint_is_rejected(self) -> None:
        chemmol, render = make_annotations("CC")
        lines = serialize_chemsvg_v4(chemmol, render).splitlines()
        stroke_index = next(index for index, line in enumerate(lines) if line.startswith("<s "))
        fields = lines[stroke_index][1:-1].split()
        fields[1] = "0"
        fields[2] = "0"
        lines[stroke_index] = "<" + " ".join(fields) + ">"
        with self.assertRaises(ChemMolSequenceError):
            parse_chemsvg_v4("\n".join(lines) + "\n")

    def test_unspecified_double_bond_stereo_needs_no_references(self) -> None:
        chemmol, render = make_annotations("FC=CF")
        double_bond = next(bond for bond in chemmol["bonds"] if bond["type"] == "DOUBLE")
        double_bond["stereo"] = "STEREOANY"
        double_bond["stereo_atoms"] = []
        sequence = serialize_chemsvg_v4(chemmol, render)
        stroke = next(
            stroke for stroke in parse_chemsvg_v4(sequence).strokes if stroke.stereo == "STEREOANY"
        )
        self.assertIsNone(stroke.stereo_atom_0)

    def test_stereo_reference_must_be_connected_to_bond_endpoint(self) -> None:
        chemmol, render = make_annotations("CC/C=C/CC")
        lines = serialize_chemsvg_v4(chemmol, render).splitlines()
        parsed = parse_chemsvg_v4("\n".join(lines) + "\n")
        stereo_stroke = next(
            stroke for stroke in parsed.strokes if stroke.stereo_atom_0 is not None
        )
        coordinates = {(vertex.x, vertex.y) for vertex in parsed.vertices}
        adjacency = {coordinate: set() for coordinate in coordinates}
        for stroke in parsed.strokes:
            adjacency[stroke.begin].add(stroke.end)
            adjacency[stroke.end].add(stroke.begin)
        invalid_reference = next(
            coordinate
            for coordinate in adjacency
            if coordinate not in adjacency[stereo_stroke.begin]
            and coordinate != stereo_stroke.begin
        )
        stroke_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith(
                f"<s {stereo_stroke.begin_x} {stereo_stroke.begin_y} "
                f"{stereo_stroke.end_x} {stereo_stroke.end_y} "
            )
        )
        fields = lines[stroke_index][1:-1].split()
        fields[9] = str(invalid_reference[0])
        fields[10] = str(invalid_reference[1])
        lines[stroke_index] = "<" + " ".join(fields) + ">"
        with self.assertRaisesRegex(
            ChemMolSequenceError,
            "not connected to the begin vertex",
        ):
            parse_chemsvg_v4("\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
