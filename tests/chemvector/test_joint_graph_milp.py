import unittest

import torch

from chemvector.joint_graph_milp import JointGraphMilpConfig, solve_joint_graph_milp


class JointGraphMilpTests(unittest.TestCase):
    def test_selects_consistent_atom_bond_graph(self) -> None:
        atoms, bonds, diagnostics = solve_joint_graph_milp(
            atom_presence_logits=torch.tensor([2.0, 2.0, -2.0]),
            atom_mask=torch.tensor([True, True, True]),
            bond_presence_logits=torch.tensor([2.0]),
            bond_mask=torch.tensor([True]),
            incidence_logits=torch.tensor([[[5.0, 0.0, -5.0], [0.0, 5.0, -5.0]]]),
            atom_threshold=0.0,
            bond_threshold=0.0,
        )

        self.assertEqual(atoms, [0, 1])
        self.assertEqual(bonds, {0: (0, 1)})
        self.assertTrue(diagnostics["solver_success"])

    def test_duplicate_atom_pair_keeps_higher_scoring_bond(self) -> None:
        atoms, bonds, _ = solve_joint_graph_milp(
            atom_presence_logits=torch.tensor([2.0, 2.0]),
            atom_mask=torch.tensor([True, True]),
            bond_presence_logits=torch.tensor([2.0, 1.0]),
            bond_mask=torch.tensor([True, True]),
            incidence_logits=torch.tensor(
                [
                    [[5.0, 0.0], [0.0, 5.0]],
                    [[5.0, 0.0], [0.0, 5.0]],
                ]
            ),
            atom_threshold=0.0,
            bond_threshold=0.0,
        )

        self.assertEqual(atoms, [0, 1])
        self.assertEqual(bonds, {0: (0, 1)})

    def test_topk_endpoint_alternative_avoids_self_bond(self) -> None:
        atoms, bonds, _ = solve_joint_graph_milp(
            atom_presence_logits=torch.tensor([2.0, 2.0]),
            atom_mask=torch.tensor([True, True]),
            bond_presence_logits=torch.tensor([2.0]),
            bond_mask=torch.tensor([True]),
            incidence_logits=torch.tensor([[[5.0, 4.0], [5.0, 4.0]]]),
            atom_threshold=0.0,
            bond_threshold=0.0,
            config=JointGraphMilpConfig(endpoint_topk=2, incidence_weight=0.1),
        )

        self.assertEqual(atoms, [0, 1])
        self.assertEqual(set(bonds[0]), {0, 1})


if __name__ == "__main__":
    unittest.main()
