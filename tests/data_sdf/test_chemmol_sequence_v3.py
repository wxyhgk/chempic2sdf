#!/usr/bin/env python3
"""Unit tests for spatial ChemMolSeq v0.3."""

from __future__ import annotations

import unittest

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemmol_sequence_v3 import (
    END_TOKEN,
    START_TOKEN,
    mol_from_chemmolseq_v3,
    parse_chemmolseq_v3,
    serialize_chemmolseq_v3,
    validate_chemmolseq_v3_roundtrip,
)
from tests.data_sdf.test_chemmol_sequence import make_annotations


class ChemMolSequenceV3Tests(unittest.TestCase):
    CASES = (
        "c1cc[nH]c1",
        "C[N+](C)(C)C",
        "N[C@@H](C)C(=O)O",
        "F/C=C/F",
        "[CH3]",
        "NCC(=O)N[C@H]1CC[C@H](CC(=O)O)CC1",
    )

    def test_representative_chemistry_roundtrips(self) -> None:
        for smiles in self.CASES:
            with self.subTest(smiles=smiles):
                chemmol, render = make_annotations(smiles)
                sequence = serialize_chemmolseq_v3(chemmol, render)
                self.assertTrue(sequence.startswith(START_TOKEN + "\n"))
                self.assertTrue(sequence.endswith(END_TOKEN + "\n"))
                validate_chemmolseq_v3_roundtrip(
                    sequence,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )

    def test_atom_order_follows_visible_y_then_x(self) -> None:
        chemmol, render = make_annotations("CNO")
        elements = {atom["id"]: atom["element"] for atom in chemmol["atoms"]}
        render["atoms"][0]["quantized_y"] = 800
        render["atoms"][1]["quantized_y"] = 100
        render["atoms"][2]["quantized_y"] = 500
        sequence = serialize_chemmolseq_v3(chemmol, render)
        parsed = parse_chemmolseq_v3(sequence)
        self.assertEqual(
            [atom.element for atom in parsed.atoms],
            [elements[1], elements[2], elements[0]],
        )

    def test_sequence_omits_render_only_bond_fields(self) -> None:
        chemmol, render = make_annotations("F/C=C/F")
        sequence = serialize_chemmolseq_v3(chemmol, render)
        self.assertNotIn("display_type", sequence)
        self.assertNotIn("direction", sequence)
        for line in sequence.splitlines():
            if line.startswith("<a "):
                self.assertEqual(len(line[1:-1].split()), 9)
            if line.startswith("<b "):
                fields = line[1:-1].split()
                self.assertEqual(len(fields), 7)
                self.assertNotEqual(int(fields[1]), int(fields[2]))

    def test_reconstructed_molecule_has_generated_2d_coordinates(self) -> None:
        sequence = serialize_chemmolseq_v3(*make_annotations("CCO"))
        mol = mol_from_chemmolseq_v3(sequence, apply_render_coordinates=True)
        self.assertEqual(mol.GetNumConformers(), 1)
        self.assertFalse(mol.GetConformer().Is3D())

    def test_parser_rejects_duplicate_bonds(self) -> None:
        sequence = serialize_chemmolseq_v3(*make_annotations("CCC"))
        lines = sequence.splitlines()
        bond_lines = [line for line in lines if line.startswith("<b ")]
        insert_at = lines.index(bond_lines[-1]) + 1
        lines.insert(insert_at, bond_lines[0])
        with self.assertRaises(ChemMolSequenceError):
            parse_chemmolseq_v3("\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
