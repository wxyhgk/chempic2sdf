import unittest

import torch

from chemvector.bond_equivalence import collapse_incidence_pair_bonds


class BondEquivalenceTests(unittest.TestCase):
    def test_keeps_highest_logit_for_each_unordered_incidence_pair(self) -> None:
        logits = torch.tensor([0.2, 1.5, 0.7, 2.0])
        mask = torch.tensor([True, True, True, False])
        incidence = torch.full((4, 2, 4), -5.0)
        incidence[0, 0, 0] = incidence[0, 1, 1] = 5.0
        incidence[1, 0, 1] = incidence[1, 1, 0] = 5.0
        incidence[2, 0, 1] = incidence[2, 1, 2] = 5.0

        output_logits, output_mask, diagnostics = collapse_incidence_pair_bonds(
            logits, mask, incidence
        )

        self.assertEqual(output_mask.tolist(), [False, True, True, False])
        self.assertEqual(float(output_logits[0]), -10.0)
        self.assertEqual(diagnostics["groups"], 2)
        self.assertEqual(diagnostics["suppressed_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
