#!/usr/bin/env python3
"""Unit tests for ChemMolSeq v0.2."""

from __future__ import annotations

import unittest

from rdkit import Chem
from rdkit.Chem import rdDepictor

from Data.sdf.chemmol_sequence import (
    ChemMolSequenceError,
    END_TOKEN,
    START_TOKEN,
    mol_from_chemmolseq,
    parse_chemmolseq,
    serialize_chemmolseq,
    validate_chemmolseq_roundtrip,
)


def enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def make_annotations(smiles: str) -> tuple[dict, dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"Cannot parse test SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
    conformer = mol.GetConformer()
    points = [conformer.GetAtomPosition(index) for index in range(mol.GetNumAtoms())]
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    atoms = []
    render_atoms = []
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        point = points[atom_id]
        quantized_x = round(100 + 800 * (point.x - min_x) / span_x)
        quantized_y = round(100 + 800 * (max_y - point.y) / span_y)
        atoms.append(
            {
                "id": atom_id,
                "source_atom_index": atom_id,
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "aromatic": atom.GetIsAromatic(),
                "chiral_tag": enum_name(atom.GetChiralTag()),
                "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
                "explicit_h_count": atom.GetNumExplicitHs(),
                "implicit_h_count": atom.GetNumImplicitHs(),
                "no_implicit": atom.GetNoImplicit(),
                "radical_electrons": atom.GetNumRadicalElectrons(),
            }
        )
        render_atoms.append(
            {
                "atom_id": atom_id,
                "molecule_x": point.x,
                "molecule_y": point.y,
                "image_x": float(quantized_x),
                "image_y": float(quantized_y),
                "quantized_x": quantized_x,
                "quantized_y": quantized_y,
            }
        )

    bonds = []
    render_bonds = []
    for bond in mol.GetBonds():
        bond_id = bond.GetIdx()
        bond_type = enum_name(bond.GetBondType())
        bonds.append(
            {
                "id": bond_id,
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "type": bond_type,
                "aromatic": bond.GetIsAromatic(),
                "conjugated": bond.GetIsConjugated(),
                "stereo": enum_name(bond.GetStereo()),
                "stereo_atoms": list(bond.GetStereoAtoms()),
                "ez": (
                    "E"
                    if bond.GetStereo() == Chem.BondStereo.STEREOE
                    else "Z"
                    if bond.GetStereo() == Chem.BondStereo.STEREOZ
                    else None
                ),
            }
        )
        render_bonds.append(
            {
                "bond_id": bond_id,
                "begin_image_x": render_atoms[bond.GetBeginAtomIdx()]["image_x"],
                "begin_image_y": render_atoms[bond.GetBeginAtomIdx()]["image_y"],
                "end_image_x": render_atoms[bond.GetEndAtomIdx()]["image_x"],
                "end_image_y": render_atoms[bond.GetEndAtomIdx()]["image_y"],
                "display_type": bond_type,
                "direction": enum_name(bond.GetBondDir()),
            }
        )

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    molecule_id = f"test:{Chem.MolToInchiKey(mol)}"
    chemmol = {
        "schema_version": "chemmol-0.1",
        "molecule_id": molecule_id,
        "structure_ref": "test.sdf",
        "canonical_isomeric_smiles": canonical_smiles,
        "inchikey": Chem.MolToInchiKey(mol),
        "atoms": atoms,
        "bonds": bonds,
    }
    render = {
        "schema_version": "render-0.1",
        "molecule_id": molecule_id,
        "variant": "original",
        "width": 1024,
        "height": 1024,
        "atoms": render_atoms,
        "bonds": render_bonds,
        "render_parameters": {},
    }
    return chemmol, render


class ChemMolSequenceTests(unittest.TestCase):
    CASES = {
        "aromatic_explicit_h": "c1cc[nH]c1",
        "charged": "C[N+](C)(C)C",
        "tetrahedral": "N[C@@H](C)C(=O)O",
        "double_bond_stereo": "F/C=C/F",
        "radical": "COc1ccc(/C=C\\c2cc(OC)c(OC)c(OC)c2)cc1NC(=O)C1CC(C)(C)N([O])C(C)(C)C1",
    }

    def test_roundtrip_representative_chemistry(self) -> None:
        for name, smiles in self.CASES.items():
            with self.subTest(name=name):
                chemmol, render = make_annotations(smiles)
                sequence = serialize_chemmolseq(chemmol, render)
                self.assertTrue(sequence.startswith(START_TOKEN + "\n"))
                self.assertTrue(sequence.endswith(END_TOKEN + "\n"))
                parsed = parse_chemmolseq(sequence)
                self.assertEqual(len(parsed.atoms), len(chemmol["atoms"]))
                self.assertEqual(len(parsed.bonds), len(chemmol["bonds"]))
                validate_chemmolseq_roundtrip(
                    sequence,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )

    def test_stereo_atoms_and_radicals_are_serialized(self) -> None:
        stereo, stereo_render = make_annotations("F/C=C/F")
        stereo_sequence = serialize_chemmolseq(stereo, stereo_render)
        stereo_bond = next(bond for bond in stereo["bonds"] if bond["stereo_atoms"])
        self.assertIn(
            f" {stereo_bond['stereo_atoms'][0]} {stereo_bond['stereo_atoms'][1]} ",
            stereo_sequence,
        )

        radical, radical_render = make_annotations("[CH3]")
        radical_sequence = serialize_chemmolseq(radical, radical_render)
        parsed = parse_chemmolseq(radical_sequence)
        self.assertEqual(parsed.atoms[0].radical_electrons, 1)

    def test_quantized_coordinates_create_a_2d_conformer(self) -> None:
        chemmol, render = make_annotations("CCO")
        sequence = serialize_chemmolseq(chemmol, render)
        mol = mol_from_chemmolseq(sequence)
        self.assertEqual(mol.GetNumConformers(), 1)
        self.assertFalse(mol.GetConformer().Is3D())

    def test_display_directions_are_opt_in(self) -> None:
        chemmol, render = make_annotations("N[C@@H](C)C(=O)O")
        render["bonds"][0]["direction"] = "BEGINWEDGE"
        sequence = serialize_chemmolseq(chemmol, render)
        chemical_mol = mol_from_chemmolseq(sequence)
        display_mol = mol_from_chemmolseq(sequence, apply_display_directions=True)
        self.assertTrue(
            all(bond.GetBondDir() == Chem.BondDir.NONE for bond in chemical_mol.GetBonds())
        )
        self.assertEqual(display_mol.GetBondWithIdx(0).GetBondDir(), Chem.BondDir.BEGINWEDGE)

    def test_parser_rejects_missing_stereo_reference(self) -> None:
        chemmol, render = make_annotations("F/C=C/F")
        sequence = serialize_chemmolseq(chemmol, render)
        lines = sequence.splitlines()
        bond_index = next(index for index, line in enumerate(lines) if " STEREOE " in line)
        fields = lines[bond_index][1:-1].split()
        fields[7] = "-1"
        lines[bond_index] = "<" + " ".join(fields) + ">"
        with self.assertRaises(ChemMolSequenceError):
            parse_chemmolseq("\n".join(lines) + "\n")

    def test_serializer_rejects_mismatched_molecule_ids(self) -> None:
        chemmol, render = make_annotations("CC")
        render["molecule_id"] = "test:different"
        with self.assertRaises(ChemMolSequenceError):
            serialize_chemmolseq(chemmol, render)


if __name__ == "__main__":
    unittest.main()
