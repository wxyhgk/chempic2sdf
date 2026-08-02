"""Tests for the canonical ChemTraverse v0.1 protocol."""

from __future__ import annotations

import copy
import unittest

from rdkit import Chem

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemscene_v0_1 import mol_from_chemscene_v0_1
from Data.sdf.chemtraverse_v0_1 import (
    mol_from_chemtraverse_v0_1,
    parse_chemtraverse_v0_1,
    serialize_chemtraverse_v0_1,
    validate_chemtraverse_v0_1_roundtrip,
)


def atom(x: int, y: int, element: str = "C", role: str = "IMPLICIT") -> dict:
    return {
        "coordinate": [x, y],
        "role": role,
        "atom": {
            "element": element,
            "formal_charge": 0,
            "isotope": 0,
            "aromatic": False,
            "explicit_h_count": 0,
            "no_implicit": False,
            "radical_electrons": 0,
            "canonical_chiral_tag": "CHI_UNSPECIFIED",
        },
    }


def bond(left: list[int], right: list[int], chemical: str = "SINGLE") -> dict:
    return {
        "endpoints": [left, right],
        "display_type": chemical,
        "chemical_type": chemical,
        "direction": "NONE",
        "stereo": "STEREONONE",
        "bbox": None,
    }


def ring_scene() -> dict:
    points = [[100, 100], [200, 100], [250, 180], [200, 260], [100, 260], [50, 180]]
    return {
        "schema_version": "chemscene-0.1",
        "molecule_id": "ring-1",
        "units": "q1023",
        "canvas": {"width": 768, "height": 768},
        "junctions": [atom(*point) for point in points],
        "text_objects": [],
        "bond_objects": [
            bond(points[index], points[(index + 1) % len(points)])
            for index in range(len(points))
        ],
    }


def chiral_scene() -> dict:
    center = [500, 500]
    fluorine = [400, 300]
    chlorine = [700, 400]
    bromine = [450, 750]
    scene = {
        "schema_version": "chemscene-0.1",
        "molecule_id": "chiral-1",
        "units": "q1023",
        "canvas": {"width": 768, "height": 768},
        "junctions": [
            atom(*chlorine, "Cl", "TEXT"),
            atom(*center),
            atom(*bromine, "Br", "TEXT"),
            atom(*fluorine, "F", "TEXT"),
        ],
        "text_objects": [
            {"anchor": point, "center": point, "bbox": None, "text": label}
            for point, label in (
                (fluorine, "F"),
                (chlorine, "Cl"),
                (bromine, "Br"),
            )
        ],
        "bond_objects": [
            bond(center, fluorine),
            bond(center, chlorine),
            bond(center, bromine),
        ],
    }
    scene["junctions"][1]["atom"]["canonical_chiral_tag"] = "CHI_TETRAHEDRAL_CW"
    return scene


class ChemTraverseV01Tests(unittest.TestCase):
    def test_cycle_roundtrip_is_canonical(self) -> None:
        scene = ring_scene()
        sequence = validate_chemtraverse_v0_1_roundtrip(scene)

        self.assertIn("<link ", sequence)
        self.assertEqual(sequence, serialize_chemtraverse_v0_1(parse_chemtraverse_v0_1(sequence)))

    def test_sequence_is_invariant_to_scene_list_order(self) -> None:
        scene = ring_scene()
        reordered = copy.deepcopy(scene)
        reordered["junctions"].reverse()
        reordered["bond_objects"].reverse()

        self.assertEqual(
            serialize_chemtraverse_v0_1(scene),
            serialize_chemtraverse_v0_1(reordered),
        )

    def test_open_utf8_text_and_dummy_alias_roundtrip(self) -> None:
        scene = ring_scene()
        scene["molecule_id"] = "alias-molecule"
        scene["junctions"][0] = atom(100, 100, "*", "TEXT")
        scene["text_objects"] = [
            {
                "anchor": [100, 100],
                "center": [108, 96],
                "bbox": [88, 80, 132, 112],
                "text": "Ph-A1",
            }
        ]

        sequence = validate_chemtraverse_v0_1_roundtrip(scene)
        restored = mol_from_chemtraverse_v0_1(sequence)

        self.assertNotIn("Ph-A1", sequence)
        self.assertIn("utf8:", sequence)
        self.assertEqual(restored.GetAtomWithIdx(0).GetAtomicNum(), 0)
        self.assertTrue(restored.GetAtomWithIdx(0).HasProp("molFileAlias"))

    def test_disconnected_components_roundtrip(self) -> None:
        scene = ring_scene()
        scene["junctions"].append(atom(800, 800, "Cl"))

        sequence = validate_chemtraverse_v0_1_roundtrip(scene)

        self.assertEqual(sequence.count("<component>"), 2)

    def test_molecular_graph_is_preserved(self) -> None:
        scene = ring_scene()
        sequence = validate_chemtraverse_v0_1_roundtrip(scene)
        expected = mol_from_chemscene_v0_1(scene)
        actual = mol_from_chemtraverse_v0_1(sequence)

        self.assertEqual(
            Chem.MolToSmiles(expected, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(actual, canonical=True, isomericSmiles=True),
        )

    def test_tetrahedral_stereo_survives_traversal_atom_reordering(self) -> None:
        scene = chiral_scene()
        expected = mol_from_chemscene_v0_1(scene)
        expected_key = Chem.MolToInchiKey(expected)

        sequence = validate_chemtraverse_v0_1_roundtrip(scene)
        actual = mol_from_chemtraverse_v0_1(sequence)

        self.assertNotIn("CHI_TETRAHEDRAL", sequence)
        self.assertRegex(sequence, r"<atom [^>]+ (?:TCW|TCCW)>\n")
        self.assertEqual(Chem.MolToInchiKey(actual), expected_key)

    def test_forward_link_is_rejected(self) -> None:
        sequence = serialize_chemtraverse_v0_1(ring_scene())
        lines = sequence.splitlines()
        link_index = next(index for index, line in enumerate(lines) if line.startswith("<link "))
        values = lines[link_index].split()
        values[1] = "999"
        lines[link_index] = " ".join(values)

        with self.assertRaisesRegex(ChemMolSequenceError, "earlier atom"):
            parse_chemtraverse_v0_1("\n".join(lines) + "\n")

    def test_invalid_valence_graph_remains_sequence_canonical(self) -> None:
        scene = ring_scene()
        scene["junctions"][0]["atom"]["element"] = "F"

        sequence = serialize_chemtraverse_v0_1(scene)
        restored = parse_chemtraverse_v0_1(sequence)

        self.assertEqual(serialize_chemtraverse_v0_1(restored), sequence)


if __name__ == "__main__":
    unittest.main()
