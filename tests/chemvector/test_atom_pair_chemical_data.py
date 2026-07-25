from __future__ import annotations

import unittest

from chemvector.atom_pair_chemical_data import (
    molecule_partition,
    sample_molecule_id,
)


class AtomPairChemicalDataTest(unittest.TestCase):
    def test_partition_is_deterministic_and_has_two_values(self) -> None:
        values = {molecule_partition(f"chembl:{index}") for index in range(100)}
        self.assertEqual(values, {"train_fit", "train_dev"})
        self.assertEqual(
            molecule_partition("chembl:CHEMBL1"),
            molecule_partition("chembl:CHEMBL1"),
        )

    def test_molecule_id_uses_chemmol_then_meta(self) -> None:
        self.assertEqual(
            sample_molecule_id(
                {
                    "chemmol": {"molecule_id": "chembl:1"},
                    "meta": {"molecule_id": "chembl:2"},
                }
            ),
            "chembl:1",
        )
        self.assertEqual(
            sample_molecule_id({"chemmol": {}, "meta": {"molecule_id": "x:1"}}),
            "x:1",
        )


if __name__ == "__main__":
    unittest.main()
