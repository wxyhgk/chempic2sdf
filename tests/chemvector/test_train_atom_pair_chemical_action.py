from __future__ import annotations

import unittest

from chemvector.train_atom_pair_chemical_action import selection_key


class TrainAtomPairChemicalActionTest(unittest.TestCase):
    def test_selection_prioritizes_minimum_domain_chemical_and_breaks(self) -> None:
        baseline = {
            "minimum_domain_chemical_exact_rate": 0.50,
            "pooled_chemical_exact_rate": 0.60,
            "minimum_domain_presence_exact_rate": 0.70,
            "pooled_presence_exact_rate": 0.75,
            "breaks": 2,
        }
        safer = {**baseline, "breaks": 1}
        stronger_domain = {
            **baseline,
            "minimum_domain_chemical_exact_rate": 0.51,
            "pooled_chemical_exact_rate": 0.59,
        }

        self.assertGreater(selection_key(safer), selection_key(baseline))
        self.assertGreater(selection_key(stronger_domain), selection_key(safer))


if __name__ == "__main__":
    unittest.main()
