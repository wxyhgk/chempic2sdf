import unittest

import torch

from chemvector.atom_bond_scene import (
    AtomBondSceneConfig,
    AtomBondScenePredictor,
    atom_bond_scene_loss,
    unordered_port_cost,
)
from chemvector.model import ChemVectorConfig


class AtomBondSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene_config = AtomBondSceneConfig(
            atom_queries=4,
            bond_queries=3,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            incidence_dim=16,
        )
        self.vision_config = ChemVectorConfig(
            d_model=32,
            decoder_heads=4,
            vision_dims=(8, 16, 32, 64),
            vision_depths=(1, 1, 1, 1),
            dropout=0.0,
        )

    def test_forward_exposes_joint_incidence_logits(self) -> None:
        model = AtomBondScenePredictor(self.vision_config, self.scene_config)
        output = model(torch.zeros(1, 3, 64, 64))

        self.assertEqual(tuple(output["atom_anchors"].shape), (1, 4, 2))
        self.assertEqual(tuple(output["bond_ports"].shape), (1, 3, 2, 2))
        self.assertEqual(tuple(output["incidence_logits"].shape), (1, 3, 2, 4))

    def test_joint_loss_backpropagates_through_atom_bond_and_incidence(self) -> None:
        model = AtomBondScenePredictor(self.vision_config, self.scene_config)
        output = model(torch.zeros(1, 3, 64, 64))
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.25, 0.5], [0.75, 0.5], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "atom_roles": torch.tensor([[0, 1, 0, 0]]),
            "atom_text_center": torch.tensor(
                [[[0.0, 0.0], [0.75, 0.5], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "atom_text_bbox": torch.tensor(
                [[[0.0, 0.0, 0.0, 0.0], [0.75, 0.5, 0.08, 0.08], [0, 0, 0, 0], [0, 0, 0, 0]]]
            ),
            "atom_text_mask": torch.tensor([[False, True, False, False]]),
            "atom_mask": torch.tensor([[True, True, False, False]]),
            "bond_ports": torch.tensor(
                [[[[0.25, 0.5], [0.71, 0.5]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]]
            ),
            "bond_display": torch.tensor([[0, 0, 0]]),
            "bond_chemical": torch.tensor([[0, 0, 0]]),
            "bond_direction": torch.tensor([[0, 0, 0]]),
            "bond_atom_refs": torch.tensor([[[0, 1], [0, 0], [0, 0]]]),
            "bond_mask": torch.tensor([[True, False, False]]),
        }

        losses = atom_bond_scene_loss(output, targets)
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(model.incidence_query.weight.grad)
        self.assertGreater(float(model.incidence_query.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.atom_anchor[-1].weight.grad)
        self.assertIsNotNone(model.bond_center_vector[-1].weight.grad)

    def test_unordered_port_cost_is_endpoint_swap_invariant(self) -> None:
        predicted = torch.tensor([[[0.1, 0.2], [0.8, 0.7]]])
        expected = torch.tensor([[[0.8, 0.7], [0.1, 0.2]]])

        cost = unordered_port_cost(predicted, expected)

        self.assertAlmostEqual(float(cost[0, 0]), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
