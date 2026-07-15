"""Tests for public OCSR SDF graph comparison and aggregation."""

from __future__ import annotations

import unittest

from rdkit import Chem

from chemvector.benchmark_public_ocsr import sdf_graph_match, summarize_records


class PublicOcsrGraphMatchTests(unittest.TestCase):
    def test_atom_order_does_not_affect_graph_match(self) -> None:
        expected = Chem.MolFromSmiles("CC(=O)N")
        predicted = Chem.RenumberAtoms(expected, [3, 1, 0, 2])

        self.assertTrue(sdf_graph_match(expected, predicted, use_chirality=True))

    def test_charge_and_isotope_are_part_of_sdf_graph(self) -> None:
        neutral = Chem.MolFromSmiles("N")
        charged = Chem.MolFromSmiles("[NH4+]")
        carbon = Chem.MolFromSmiles("C")
        carbon_13 = Chem.MolFromSmiles("[13CH4]")

        self.assertFalse(sdf_graph_match(neutral, charged, use_chirality=True))
        self.assertFalse(sdf_graph_match(carbon, carbon_13, use_chirality=True))

    def test_nonstereo_and_full_match_are_reported_separately(self) -> None:
        expected = Chem.MolFromSmiles("F/C=C/F")
        predicted = Chem.MolFromSmiles("F/C=C\\F")

        self.assertTrue(sdf_graph_match(expected, predicted, use_chirality=False))
        self.assertFalse(sdf_graph_match(expected, predicted, use_chirality=True))

    def test_summary_uses_sdf_graph_match_as_primary_rate(self) -> None:
        rows = [
            {
                "sdf_valid": True,
                "formula_match": True,
                "atom_count_match": True,
                "bond_count_match": True,
                "nonstereo_structure_match": True,
                "chemical_match": True,
                "fingerprint_similarity": 1.0,
                "seconds": 0.1,
            },
            {
                "sdf_valid": False,
                "formula_match": False,
                "atom_count_match": False,
                "bond_count_match": False,
                "nonstereo_structure_match": False,
                "chemical_match": False,
                "failure_category": "valence",
                "seconds": 0.3,
            },
        ]

        summary = summarize_records(rows)

        self.assertEqual(summary["sdf_valid_count"], 1)
        self.assertEqual(summary["chemical_match_count"], 1)
        self.assertEqual(summary["chemical_match_rate"], 0.5)
        self.assertEqual(summary["average_fingerprint_similarity_valid_only"], 1.0)
        self.assertEqual(summary["failure_categories"], {"valence": 1})
        self.assertAlmostEqual(summary["average_seconds"], 0.2)


if __name__ == "__main__":
    unittest.main()
