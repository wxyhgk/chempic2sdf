import unittest

import torch

from chemvector.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
    decode_spatial_atom_objects,
    spatial_atom_object_loss,
)


class SpatialAtomObjectTests(unittest.TestCase):
    def test_joint_spatial_object_loss_and_decode(self) -> None:
        config = SpatialAtomObjectConfig(
            input_channels=8,
            hidden_channels=16,
            maximum_atoms=4,
        )
        model = SpatialAtomObjectDetector(config)
        output = model(torch.zeros(1, 8, 16, 16))
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.25, 0.5], [0.75, 0.5], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "atom_roles": torch.tensor([[0, 1, 0, 0]]),
            "atom_text_center": torch.tensor(
                [[[0.0, 0.0], [0.75, 0.5], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "atom_text_bbox": torch.tensor(
                [[[0, 0, 0, 0], [0.75, 0.5, 0.08, 0.08], [0, 0, 0, 0], [0, 0, 0, 0]]]
            ),
            "atom_text_mask": torch.tensor([[False, True, False, False]]),
            "atom_mask": torch.tensor([[True, True, False, False]]),
        }

        losses = spatial_atom_object_loss(output, targets)
        losses["loss"].backward()
        decoded = decode_spatial_atom_objects(output, torch.tensor([2]))

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(len(decoded[0]), 2)
        self.assertIsNotNone(model.anchor_offset.weight.grad)
        self.assertIsNotNone(model.role.weight.grad)
        self.assertIsNotNone(model.text_extent.weight.grad)

    def test_multiple_objects_in_one_cell_are_rejected(self) -> None:
        config = SpatialAtomObjectConfig(
            input_channels=8,
            hidden_channels=16,
            maximum_atoms=4,
        )
        output = SpatialAtomObjectDetector(config)(torch.zeros(1, 8, 8, 8))
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.251, 0.5], [0.252, 0.5], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "atom_roles": torch.zeros(1, 4, dtype=torch.long),
            "atom_text_center": torch.zeros(1, 4, 2),
            "atom_text_bbox": torch.zeros(1, 4, 4),
            "atom_text_mask": torch.zeros(1, 4, dtype=torch.bool),
            "atom_mask": torch.tensor([[True, True, False, False]]),
        }

        with self.assertRaisesRegex(ValueError, "one spatial cell"):
            spatial_atom_object_loss(output, targets)


if __name__ == "__main__":
    unittest.main()
