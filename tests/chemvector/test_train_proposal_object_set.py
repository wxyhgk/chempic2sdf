import unittest

import torch

from chemvector.train_proposal_object_set import _sample_set_metrics


class TrainProposalObjectSetTests(unittest.TestCase):
    def test_full_exact_requires_coverage_selection_and_incidence(self) -> None:
        output = {
            "atom_presence_logits": torch.tensor([[5.0, 5.0, -5.0]]),
            "bond_presence_logits": torch.tensor([[5.0, -5.0]]),
            "incidence_logits": torch.tensor(
                [[[[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0]],
                  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
            ),
        }
        proposals = {
            "atom_mask": torch.tensor([[True, True, True]]),
            "bond_mask": torch.tensor([[True, True]]),
            "atom_presence_targets": torch.tensor([[True, True, False]]),
            "bond_presence_targets": torch.tensor([[True, False]]),
            "atom_target_matched": torch.tensor([[True, True, False]]),
            "bond_target_matched": torch.tensor([[True, False]]),
            "incidence_target_eligible": torch.tensor([[True, False]]),
            "bond_supervision_mask": torch.tensor([[True, False]]),
            "target_atom_refs": torch.tensor([[[0, 1], [0, 0]]]),
        }
        targets = {
            "atom_mask": torch.tensor([[True, True, False]]),
            "bond_mask": torch.tensor([[True, False]]),
        }

        metrics = _sample_set_metrics(output, proposals, targets, 0)

        self.assertTrue(metrics["atom_deployed_exact"])
        self.assertTrue(metrics["bond_deployed_exact"])
        self.assertTrue(metrics["incidence_exact"])
        self.assertTrue(metrics["full_object_exact"])

        proposals["atom_target_matched"][0, 1] = False
        missing = _sample_set_metrics(output, proposals, targets, 0)
        self.assertFalse(missing["atom_deployed_exact"])
        self.assertFalse(missing["full_object_exact"])


if __name__ == "__main__":
    unittest.main()
