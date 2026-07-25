import unittest

from rdkit import Chem

from chemvector.benchmark_object_decomposition import compare_molecular_graphs


class BenchmarkObjectDecompositionTest(unittest.TestCase):
    def compare(self, expected: str, predicted: str) -> dict:
        expected_mol = Chem.MolFromSmiles(expected)
        predicted_mol = Chem.MolFromSmiles(predicted)
        self.assertIsNotNone(expected_mol)
        self.assertIsNotNone(predicted_mol)
        return compare_molecular_graphs(expected_mol, predicted_mol)

    def test_atom_error_does_not_hide_correct_skeleton_and_bonds(self) -> None:
        result = self.compare("CCO", "CCN")
        self.assertTrue(result["skeleton_exact"])
        self.assertFalse(result["atom_typed_skeleton_exact"])
        self.assertTrue(result["bond_typed_skeleton_exact"])
        self.assertFalse(result["labeled_graph_exact"])
        self.assertEqual(result["atom_type_correct_max"], 2)
        self.assertEqual(result["atom_type_total"], 3)
        self.assertEqual(result["atom_type_confusions"], {"O->N": 1})

    def test_bond_type_error_does_not_hide_correct_skeleton_and_atoms(self) -> None:
        result = self.compare("CCO", "C=CO")
        self.assertTrue(result["skeleton_exact"])
        self.assertTrue(result["atom_typed_skeleton_exact"])
        self.assertFalse(result["bond_typed_skeleton_exact"])
        self.assertFalse(result["labeled_graph_exact"])
        self.assertEqual(result["bond_type_correct_max"], 1)
        self.assertEqual(result["bond_type_total"], 2)
        self.assertEqual(result["bond_type_confusions"], {"SINGLE->DOUBLE": 1})

    def test_same_counts_do_not_imply_correct_skeleton(self) -> None:
        result = self.compare("CCCC", "CC(C)C")
        self.assertTrue(result["atom_count_match"])
        self.assertTrue(result["bond_count_match"])
        self.assertFalse(result["skeleton_exact"])


if __name__ == "__main__":
    unittest.main()
