import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1h_assignment_stability import (
    _assignment,
    _finalize_turnover,
)


class G3P1HAssignmentStabilityTests(unittest.TestCase):
    def test_assignment_maps_target_to_candidate_identity(self) -> None:
        positive = torch.tensor([[False, True, True]])
        matched = torch.tensor([[-1, 7, 3]])
        ids = torch.tensor([[10, 20, 30]])

        result = _assignment(
            {"positive_mask": positive, "matched_target_indices": matched}, ids, 0
        )

        self.assertEqual(result, {7: 20, 3: 30})

    def test_turnover_rates_use_atom_and_image_denominators(self) -> None:
        result = _finalize_turnover(
            {"images": 10, "atoms": 100, "changed_images": 4, "changed_atoms": 7}
        )

        self.assertEqual(result["image_turnover_rate"], 0.4)
        self.assertEqual(result["atom_turnover_rate"], 0.07)


if __name__ == "__main__":
    unittest.main()
