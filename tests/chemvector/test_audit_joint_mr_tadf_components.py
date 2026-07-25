import unittest

import torch

from chemvector.audit_joint_mr_tadf_components import (
    decision_metrics,
    targets_from_annotations,
)


class JointMrTadfComponentAuditTests(unittest.TestCase):
    def test_targets_from_annotations_preserves_atom_refs(self) -> None:
        targets = targets_from_annotations(
            [
                {
                    "atoms": [
                        {"atom_id": 0, "image_x": 76.8, "image_y": 153.6},
                        {"atom_id": 1, "image_x": 384.0, "image_y": 460.8},
                    ],
                    "bonds": [{"begin_atom": 0, "end_atom": 1}],
                }
            ],
            torch.device("cpu"),
        )

        self.assertTrue(torch.allclose(targets["atom_anchors"][0, 0], torch.tensor([0.1, 0.2])))
        self.assertEqual(targets["bond_atom_refs"][0, 0].tolist(), [0, 1])

    def test_decision_metrics_requires_no_false_objects(self) -> None:
        proposals = {
            "atom_mask": torch.tensor([[True, True, True]]),
            "bond_mask": torch.tensor([[True, True]]),
            "atom_presence_targets": torch.tensor([[True, True, False]]),
            "bond_presence_targets": torch.tensor([[True, False]]),
            "bond_supervision_mask": torch.tensor([[True, False]]),
            "target_atom_refs": torch.tensor([[[0, 1], [0, 0]]]),
        }
        metrics = decision_metrics(
            atom_logits=torch.tensor([1.0, 1.0, -1.0]),
            bond_logits=torch.tensor([1.0, -1.0]),
            incidence_logits=torch.tensor(
                [[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
            ),
            proposals=proposals,
            batch_index=0,
            atom_threshold=0.0,
            bond_threshold=0.0,
            target_atoms=2,
            target_bonds=1,
        )

        self.assertTrue(metrics["object_incidence_exact"])


if __name__ == "__main__":
    unittest.main()
