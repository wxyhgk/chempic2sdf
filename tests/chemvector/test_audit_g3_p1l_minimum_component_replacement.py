import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1l_minimum_component_replacement import (
    minimum_replacements,
    sample_statistics,
)


class G3P1LMinimumComponentReplacementTests(unittest.TestCase):
    def test_one_unselected_component_repairs_assignment(self) -> None:
        compatible = torch.tensor(
            [[True, False], [False, False], [False, True]]
        )
        selected = torch.tensor([True, True, False])

        self.assertEqual(minimum_replacements(compatible, selected), 1)
        result = sample_statistics(compatible, selected)
        self.assertEqual(result["selected_matched"], 1)
        self.assertEqual(result["minimum_replacements"], 1)

    def test_oracle_unreachable_returns_none(self) -> None:
        compatible = torch.tensor([[True, False], [False, False]])
        selected = torch.tensor([True, True])

        self.assertIsNone(minimum_replacements(compatible, selected))

    def test_matching_handles_shared_components_globally(self) -> None:
        compatible = torch.tensor(
            [[True, True], [True, False], [False, True]]
        )
        selected = torch.tensor([True, True, False])

        self.assertEqual(minimum_replacements(compatible, selected), 0)
        self.assertTrue(sample_statistics(compatible, selected)["selected_exact"])


if __name__ == "__main__":
    unittest.main()
