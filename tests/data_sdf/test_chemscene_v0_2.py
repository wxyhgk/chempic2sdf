#!/usr/bin/env python3

import copy
import unittest

from rdkit import Chem

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemscene_v0_2 import (
    build_chemscene_v0_2,
    chemscene_v0_1_from_v0_2,
    mol_from_chemscene_v0_2,
    parse_chemscene_v0_2,
    sdf_from_chemscene_v0_2,
    validate_chemscene_v0_2_roundtrip,
)
from tests.data_sdf.test_chemmol_sequence import make_annotations
from tests.data_sdf.test_chemscene_v0_1 import make_layers


class ChemSceneV02Tests(unittest.TestCase):
    def build(self, smiles: str) -> tuple[dict, dict]:
        chemmol, render = make_annotations(smiles)
        return chemmol, build_chemscene_v0_2(
            chemmol, render, make_layers(chemmol, render)
        )

    def test_representative_chemistry_roundtrips(self) -> None:
        for smiles in ("CCO", "c1cc[nH]c1", "F/C=C/F", "N[C@@H](C)C(=O)O"):
            with self.subTest(smiles=smiles):
                chemmol, scene = self.build(smiles)
                validate_chemscene_v0_2_roundtrip(
                    scene,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )

    def test_text_and_anchor_are_owned_by_one_atom_object(self) -> None:
        _, scene = self.build("CO")
        self.assertNotIn("junctions", scene)
        self.assertNotIn("text_objects", scene)
        oxygen = next(
            atom for atom in scene["atom_objects"] if atom["chemical"]["element"] == "O"
        )
        self.assertEqual(oxygen["role"], "TEXT")
        self.assertEqual(oxygen["text"]["value"], "O")
        self.assertTrue(oxygen["incident_ports"])

    def test_text_port_is_clipped_but_implicit_port_uses_anchor(self) -> None:
        _, scene = self.build("CO")
        carbon = next(
            atom for atom in scene["atom_objects"] if atom["chemical"]["element"] == "C"
        )
        oxygen = next(
            atom for atom in scene["atom_objects"] if atom["chemical"]["element"] == "O"
        )
        self.assertEqual(carbon["incident_ports"][0]["source"], "ANCHOR")
        self.assertEqual(carbon["incident_ports"][0]["coordinate"], carbon["anchor"])
        self.assertEqual(oxygen["incident_ports"][0]["source"], "TEXT_BBOX_RAY")
        self.assertNotEqual(oxygen["incident_ports"][0]["coordinate"], oxygen["anchor"])

    def test_port_ownership_mismatch_is_rejected(self) -> None:
        _, scene = self.build("CCO")
        invalid = copy.deepcopy(scene)
        invalid["bond_objects"][0]["ports"][0][0] += 1
        with self.assertRaisesRegex(ChemMolSequenceError, "ports and BondObjects disagree"):
            parse_chemscene_v0_2(invalid)

    def test_anchor_port_cannot_drift_from_atom_anchor(self) -> None:
        _, scene = self.build("CC")
        invalid = copy.deepcopy(scene)
        invalid["atom_objects"][0]["incident_ports"][0]["coordinate"][0] += 1
        with self.assertRaisesRegex(ChemMolSequenceError, "ANCHOR port"):
            parse_chemscene_v0_2(invalid)

    def test_text_bbox_port_requires_text_atom(self) -> None:
        _, scene = self.build("CC")
        invalid = copy.deepcopy(scene)
        invalid["atom_objects"][0]["incident_ports"][0]["source"] = "TEXT_BBOX_RAY"
        with self.assertRaisesRegex(ChemMolSequenceError, "requires a TEXT AtomObject"):
            parse_chemscene_v0_2(invalid)

    def test_duplicate_bond_objects_are_rejected(self) -> None:
        _, scene = self.build("CC")
        invalid = copy.deepcopy(scene)
        duplicate = copy.deepcopy(invalid["bond_objects"][0])
        duplicate["id"] = 1
        invalid["bond_objects"].append(duplicate)
        for endpoint, atom_id in enumerate(duplicate["atom_refs"]):
            source_port = invalid["atom_objects"][atom_id]["incident_ports"][0]
            port = copy.deepcopy(source_port)
            port["bond_id"] = 1
            port["endpoint"] = endpoint
            invalid["atom_objects"][atom_id]["incident_ports"].append(port)
        with self.assertRaisesRegex(ChemMolSequenceError, "duplicate BondObjects"):
            parse_chemscene_v0_2(invalid)

    def test_v01_projection_and_sdf_compilation_are_lossless(self) -> None:
        _, scene = self.build("C[N+](C)(C)C")
        projected = chemscene_v0_1_from_v0_2(scene)
        self.assertEqual(len(projected["junctions"]), len(scene["atom_objects"]))
        self.assertTrue(sdf_from_chemscene_v0_2(scene).endswith("\n$$$$\n"))
        self.assertEqual(Chem.MolToSmiles(mol_from_chemscene_v0_2(scene)), "C[N+](C)(C)C")


if __name__ == "__main__":
    unittest.main()
