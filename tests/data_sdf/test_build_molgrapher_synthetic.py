import unittest

from rdkit import Chem

from Data.sdf.build_molgrapher_synthetic import (
    build_chemmol,
    build_render,
    selection_bucket,
)
from Data.sdf.chemsvg_sequence_v4 import validate_chemsvg_v4_roundtrip, serialize_chemsvg_v4


class MolGrapherSyntheticBuildTests(unittest.TestCase):
    def test_standard_molecule_builds_roundtrippable_chemsvg(self) -> None:
        mol = Chem.MolFromSmiles("CC=O")
        molecule_id = "molgrapher:test"
        chemmol = build_chemmol(mol, molecule_id, "test.sdf")
        render = build_render(
            mol,
            molecule_id,
            [(100.0, 300.0), (300.0, 300.0), (500.0, 300.0)],
            (1024, 1024),
        )
        sequence = serialize_chemsvg_v4(chemmol, render)

        validate_chemsvg_v4_roundtrip(
            sequence,
            chemmol["canonical_isomeric_smiles"],
            chemmol["inchikey"],
        )

    def test_selection_bucket_records_visible_and_double_bond_features(self) -> None:
        mol = Chem.MolFromSmiles("FC=O")
        self.assertEqual(selection_bucket(mol), "small:double:text_low")


if __name__ == "__main__":
    unittest.main()
