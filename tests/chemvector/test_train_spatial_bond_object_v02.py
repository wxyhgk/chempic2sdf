import unittest

import torch

from chemvector.train_spatial_bond_object_v02 import bond_targets


class SpatialBondObjectV02TrainingTests(unittest.TestCase):
    def test_visible_ports_are_the_dense_endpoint_targets(self) -> None:
        ports = torch.tensor([[[[0.2, 0.3], [0.7, 0.8]]]])
        targets = {
            "bond_ports": ports,
            "bond_display": torch.tensor([[1]]),
            "bond_chemical": torch.tensor([[2]]),
            "bond_direction": torch.tensor([[0]]),
            "bond_mask": torch.tensor([[True]]),
        }

        dense = bond_targets(targets)

        torch.testing.assert_close(dense["endpoints"], ports)
        self.assertIs(dense["mask"], targets["bond_mask"])
        self.assertEqual(tuple(dense["bbox"].shape), (1, 1, 4))

    def test_atom_anchor_geometry_uses_bond_atom_references(self) -> None:
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.1, 0.2], [0.7, 0.8], [0.4, 0.5]]]
            ),
            "bond_atom_refs": torch.tensor([[[2, 0], [0, 1]]]),
            "bond_ports": torch.zeros(1, 2, 2, 2),
            "bond_display": torch.zeros(1, 2, dtype=torch.long),
            "bond_chemical": torch.zeros(1, 2, dtype=torch.long),
            "bond_direction": torch.zeros(1, 2, dtype=torch.long),
            "bond_mask": torch.tensor([[True, True]]),
        }

        dense = bond_targets(targets, "atom_anchors")

        torch.testing.assert_close(
            dense["endpoints"],
            torch.tensor([[[[0.4, 0.5], [0.1, 0.2]], [[0.1, 0.2], [0.7, 0.8]]]]),
        )


if __name__ == "__main__":
    unittest.main()
