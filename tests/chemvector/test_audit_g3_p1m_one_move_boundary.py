import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1m_one_move_boundary import (
    boundary_rescue,
    selection_exact,
)


class G3P1MOneMoveBoundaryTests(unittest.TestCase):
    def test_boundary_swap_recovers_exact_selection(self) -> None:
        compatible = torch.tensor(
            [[True, False], [False, False], [False, True]]
        )
        order = torch.tensor([0, 1, 2])

        self.assertFalse(selection_exact(compatible, order[:2]))
        self.assertTrue(boundary_rescue(compatible, order, 2, 1))

    def test_boundary_excludes_deeper_add_candidate(self) -> None:
        compatible = torch.tensor(
            [[True, False], [False, False], [False, False], [False, True]]
        )
        order = torch.arange(4)

        self.assertFalse(boundary_rescue(compatible, order, 2, 1))
        self.assertTrue(boundary_rescue(compatible, order, 2, 2))


if __name__ == "__main__":
    unittest.main()
