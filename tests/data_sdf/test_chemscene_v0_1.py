#!/usr/bin/env python3
"""Tests for the ChemScene v0.1 object protocol."""

from __future__ import annotations

import copy
import unittest

from rdkit import Chem

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemscene_v0_1 import (
    build_chemscene_v0_1,
    chemsvg_from_chemscene_v0_1,
    mol_from_chemscene_v0_1,
    parse_chemscene_v0_1,
    sdf_from_chemscene_v0_1,
    validate_chemscene_v0_1_roundtrip,
)
from tests.data_sdf.test_chemmol_sequence import make_annotations


def make_layers(chemmol: dict, render: dict) -> dict:
    atoms = []
    for atom, visual in zip(chemmol["atoms"], render["atoms"]):
        visible = atom["element"] != "C"
        x = visual["image_x"]
        y = visual["image_y"]
        atoms.append(
            {
                "atom_id": atom["id"],
                "element": atom["element"],
                "formal_charge": atom["formal_charge"],
                "aromatic": atom["aromatic"],
                "implicit_h_count": atom["implicit_h_count"],
                "center": [x, y],
                "visible_text": visible,
                "text": atom["element"] if visible else "",
                "bbox": [x - 5, y - 5, x + 5, y + 5] if visible else None,
            }
        )
    chemical_bonds = {item["id"]: item for item in chemmol["bonds"]}
    bonds = []
    for visual in render["bonds"]:
        chemical = chemical_bonds[visual["bond_id"]]
        x0 = visual["begin_image_x"]
        y0 = visual["begin_image_y"]
        x1 = visual["end_image_x"]
        y1 = visual["end_image_y"]
        bonds.append(
            {
                "bond_id": visual["bond_id"],
                "begin_atom": chemical["begin_atom"],
                "end_atom": chemical["end_atom"],
                "begin": [x0, y0],
                "end": [x1, y1],
                "chemical_type": chemical["type"],
                "display_type": visual["display_type"],
                "direction": visual["direction"],
                "bbox": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
            }
        )
    return {
        "schema_version": "chemvision-layers-0.1",
        "molecule_id": chemmol["molecule_id"],
        "width": render["width"],
        "height": render["height"],
        "atoms": atoms,
        "bonds": bonds,
    }


class ChemSceneV01Tests(unittest.TestCase):
    def test_representative_chemistry_roundtrips_through_sdf(self) -> None:
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
                scene = build_chemscene_v0_1(
                    chemmol, render, make_layers(chemmol, render)
                )
                parsed = parse_chemscene_v0_1(scene)
                self.assertEqual(len(parsed.junctions), len(chemmol["atoms"]))
                self.assertEqual(len(parsed.bond_objects), len(chemmol["bonds"]))
                validate_chemscene_v0_1_roundtrip(
                    scene,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )

    def test_double_bond_is_one_object_not_parallel_line_vertices(self) -> None:
        chemmol, render = make_annotations("C=C")
        scene = build_chemscene_v0_1(chemmol, render, make_layers(chemmol, render))
        self.assertEqual(len(scene["junctions"]), 2)
        self.assertEqual(len(scene["bond_objects"]), 1)
        self.assertEqual(scene["bond_objects"][0]["display_type"], "DOUBLE")
        self.assertEqual(scene["bond_objects"][0]["chemical_type"], "DOUBLE")

    def test_long_visible_bond_is_compiled_without_distance_threshold(self) -> None:
        chemmol, render = make_annotations("CC")
        render["atoms"][0].update(
            image_x=1.0, image_y=512.0, quantized_x=1, quantized_y=512
        )
        render["atoms"][1].update(
            image_x=1022.0, image_y=512.0, quantized_x=1022, quantized_y=512
        )
        render["bonds"][0].update(
            begin_image_x=1.0,
            begin_image_y=512.0,
            end_image_x=1022.0,
            end_image_y=512.0,
        )
        scene = build_chemscene_v0_1(chemmol, render, make_layers(chemmol, render))
        bond = parse_chemscene_v0_1(scene).bond_objects[0]
        self.assertEqual(abs(bond.end_x - bond.begin_x), 1021)
        self.assertEqual(Chem.MolToSmiles(mol_from_chemscene_v0_1(scene)), "CC")

    def test_open_text_is_separate_from_geometry(self) -> None:
        chemmol, render = make_annotations("CO")
        layers = make_layers(chemmol, render)
        oxygen = next(item for item in layers["atoms"] if item["element"] == "O")
        oxygen["text"] = "Ph-R1"
        scene = build_chemscene_v0_1(chemmol, render, layers)
        self.assertEqual(scene["text_objects"][0]["text"], "Ph-R1")
        self.assertEqual(Chem.MolToSmiles(mol_from_chemscene_v0_1(scene)), "CO")

    def test_dummy_text_atom_roundtrips_as_sdf_alias(self) -> None:
        chemmol, render = make_annotations("CO")
        scene = build_chemscene_v0_1(
            chemmol, render, make_layers(chemmol, render)
        )
        text_junction = next(
            item for item in scene["junctions"] if item["role"] == "TEXT"
        )
        text_junction["atom"].update(
            element="*", formal_charge=0, isotope=0, aromatic=False,
            explicit_h_count=0, no_implicit=True, radical_electrons=0,
        )
        scene["text_objects"][0]["text"] = "Ph"
        sdf = sdf_from_chemscene_v0_1(scene)
        self.assertIn("\nPh\n", sdf)
        restored = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=False)
        dummy = next(atom for atom in restored.GetAtoms() if atom.GetAtomicNum() == 0)
        self.assertEqual(dummy.GetProp("molFileAlias"), "Ph")

    def test_unknown_bond_endpoint_is_rejected(self) -> None:
        chemmol, render = make_annotations("CC")
        scene = build_chemscene_v0_1(chemmol, render, make_layers(chemmol, render))
        invalid = copy.deepcopy(scene)
        invalid["bond_objects"][0]["endpoints"][1] = [0, 0]
        with self.assertRaisesRegex(ChemMolSequenceError, "distinct Junctions"):
            parse_chemscene_v0_1(invalid)

    def test_compilers_emit_chemsvg_and_single_record_sdf(self) -> None:
        chemmol, render = make_annotations("CCO")
        scene = build_chemscene_v0_1(chemmol, render, make_layers(chemmol, render))
        chemsvg = chemsvg_from_chemscene_v0_1(scene)
        sdf = sdf_from_chemscene_v0_1(scene)
        self.assertIn("<s ", chemsvg)
        self.assertNotIn("<b ", chemsvg)
        self.assertTrue(sdf.endswith("\n$$$$\n"))


if __name__ == "__main__":
    unittest.main()
