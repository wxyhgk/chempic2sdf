import unittest

import torch

from chemvector.train_joint_graph_potentials import _graph_features, _ground_truth_graph


class TrainJointGraphPotentialsTests(unittest.TestCase):
    def test_ground_truth_requires_topk_representable_complete_graph(self) -> None:
        record = {
            "atom_matched_count": 2,
            "atom_target_count": 2,
            "bond_matched_count": 1,
            "bond_target_count": 1,
            "incidence_eligible_count": 1,
            "atom_mask": torch.tensor([True, True]),
            "bond_mask": torch.tensor([True]),
            "atom_presence_targets": torch.tensor([True, True]),
            "bond_presence_targets": torch.tensor([True]),
            "bond_supervision_mask": torch.tensor([True]),
            "target_atom_refs": torch.tensor([[0, 1]]),
        }
        truth = _ground_truth_graph(
            record, torch.tensor([[[5.0, 0.0], [0.0, 5.0]]]), endpoint_topk=1
        )

        self.assertEqual(truth, ({0, 1}, {0: (0, 1)}))

    def test_graph_features_reward_presence_and_incidence(self) -> None:
        features = _graph_features(
            atoms={0, 1},
            bonds={0: (0, 1)},
            atom_presence_logits=torch.tensor([1.0, 2.0]),
            bond_presence_logits=torch.tensor([3.0]),
            incidence_logits=torch.tensor([[[5.0, 0.0], [0.0, 5.0]]]),
            atom_mask=torch.tensor([True, True]),
            atom_threshold=0.0,
            bond_threshold=0.0,
        )

        self.assertEqual(features[:4].tolist(), [3.0, 2.0, 3.0, 1.0])
        self.assertAlmostEqual(float(features[4]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
