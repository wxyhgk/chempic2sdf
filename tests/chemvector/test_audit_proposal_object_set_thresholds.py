import unittest

import torch

from chemvector.audit_proposal_object_set_thresholds import sweep_thresholds


class AuditProposalObjectSetThresholdsTests(unittest.TestCase):
    def test_sweep_requires_coverage_and_incidence_for_full_exact(self) -> None:
        records = [
            {
                "atom_logits": torch.tensor([2.0, -2.0]),
                "atom_targets": torch.tensor([True, False]),
                "atom_covered": True,
                "bond_logits": torch.tensor([1.0, -1.0]),
                "bond_targets": torch.tensor([True, False]),
                "bond_covered": True,
                "incidence_exact": True,
            },
            {
                "atom_logits": torch.tensor([2.0]),
                "atom_targets": torch.tensor([True]),
                "atom_covered": False,
                "bond_logits": torch.tensor([2.0]),
                "bond_targets": torch.tensor([True]),
                "bond_covered": True,
                "incidence_exact": True,
            },
        ]

        result = sweep_thresholds(records, [-1.5, 0.0, 1.5])

        self.assertEqual(result["atom_best"]["deployed_exact"], 1)
        self.assertEqual(result["bond_best"]["deployed_exact"], 2)
        self.assertEqual(result["joint_best"]["full_object_exact"], 1)


if __name__ == "__main__":
    unittest.main()
