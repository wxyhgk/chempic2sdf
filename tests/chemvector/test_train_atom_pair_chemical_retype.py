from __future__ import annotations

import unittest

from chemvector.train_atom_pair_chemical_retype import selection_key


class TrainAtomPairChemicalRetypeTest(unittest.TestCase):
    def test_selection_prefers_minimum_domain_then_fewer_breaks(self) -> None:
        base = {
            "minimum_domain_chemical_exact_rate": 0.5,
            "pooled_chemical_exact_rate": 0.6,
            "breaks": 2,
            "rescues": 4,
        }
        safe = {**base, "breaks": 1}
        stronger = {
            **base,
            "minimum_domain_chemical_exact_rate": 0.51,
            "pooled_chemical_exact_rate": 0.59,
        }
        self.assertGreater(selection_key(safe), selection_key(base))
        self.assertGreater(selection_key(stronger), selection_key(safe))

    def test_selection_prefers_fewer_breaks_before_more_rescues(self) -> None:
        safer = {
            "minimum_domain_chemical_exact_rate": 0.5,
            "pooled_chemical_exact_rate": 0.6,
            "breaks": 1,
            "rescues": 2,
        }
        riskier = {**safer, "breaks": 2, "rescues": 10}
        self.assertGreater(selection_key(safer), selection_key(riskier))


if __name__ == "__main__":
    unittest.main()
