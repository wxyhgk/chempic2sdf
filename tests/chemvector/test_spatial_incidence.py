import unittest

import torch

from chemvector.spatial_incidence import (
    SpatialIncidenceConfig,
    SpatialIncidenceHead,
    spatial_incidence_loss,
)
from chemvector.train_spatial_incidence import extract_gt_associated_proposals


class SpatialIncidenceTests(unittest.TestCase):
    def test_forward_masks_invalid_atoms_and_bonds(self) -> None:
        config = SpatialIncidenceConfig(
            atom_feature_dim=4,
            bond_feature_dim=5,
            bond_type_dim=8,
            hidden_dim=16,
            pair_hidden_dim=8,
        )
        head = SpatialIncidenceHead(config)
        output = head(
            atom_features=torch.randn(1, 3, 4),
            atom_geometry=torch.rand(1, 3, 8),
            atom_mask=torch.tensor([[True, True, False]]),
            bond_features=torch.randn(1, 2, 5),
            bond_endpoints=torch.rand(1, 2, 2, 2),
            bond_types=torch.rand(1, 2, 8),
            bond_mask=torch.tensor([[True, False]]),
        )

        logits = output["incidence_logits"]
        self.assertEqual(tuple(logits.shape), (1, 2, 2, 3))
        self.assertTrue(torch.isfinite(logits[0, 0, :, :2]).all())
        self.assertTrue((logits[0, 0, :, 2] == -1e4).all())
        self.assertTrue((logits[0, 1] == -1e4).all())

    def test_loss_backpropagates_through_content_and_geometry(self) -> None:
        config = SpatialIncidenceConfig(
            atom_feature_dim=4,
            bond_feature_dim=5,
            bond_type_dim=8,
            hidden_dim=16,
            pair_hidden_dim=8,
        )
        head = SpatialIncidenceHead(config)
        output = head(
            atom_features=torch.randn(2, 3, 4),
            atom_geometry=torch.rand(2, 3, 8),
            atom_mask=torch.tensor([[True, True, False], [True, True, True]]),
            bond_features=torch.randn(2, 2, 5),
            bond_endpoints=torch.rand(2, 2, 2, 2),
            bond_types=torch.rand(2, 2, 8),
            bond_mask=torch.tensor([[True, False], [True, True]]),
        )
        losses = spatial_incidence_loss(
            output,
            torch.tensor([[[0, 1], [0, 0]], [[1, 2], [2, 0]]]),
            torch.tensor([[True, False], [True, True]]),
        )

        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(head.atom_projection[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.endpoint_projection[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.pair_geometry[0].weight.grad.abs().sum()), 0.0)

    def test_gt_association_flips_atom_refs_with_reversed_endpoints(self) -> None:
        atom_output = {
            "object_features": torch.randn(1, 4, 2, 2),
            "heatmap_logits": torch.zeros(1, 1, 2, 2),
            "anchors": torch.tensor(
                [[[[0.25, 0.75], [0.25, 0.75]], [[0.25, 0.25], [0.75, 0.75]]]]
            ),
            "role_logits": torch.zeros(1, 2, 2, 2),
            "text_centers": torch.zeros(1, 2, 2, 2),
            "text_extents": torch.zeros(1, 2, 2, 2),
        }
        endpoints = torch.zeros(1, 1, 2, 2, 2, 2)
        endpoints[0, 0, :, :, 1, 1] = torch.tensor(
            [[0.80, 0.50], [0.20, 0.50]]
        )
        bond_output = {
            "object_features": torch.randn(1, 5, 2, 2),
            "heatmap_logits": torch.zeros(1, 1, 2, 2),
            "endpoints": endpoints,
            "display_logits": torch.zeros(1, 1, 4, 2, 2),
            "chemical_logits": torch.zeros(1, 1, 4, 2, 2),
        }
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.20, 0.50], [0.80, 0.50], [0.0, 0.0]]]
            ),
            "atom_mask": torch.tensor([[True, True, False]]),
            "bond_atom_refs": torch.tensor([[[0, 1], [0, 0]]]),
            "bond_mask": torch.tensor([[True, False]]),
        }

        proposals = extract_gt_associated_proposals(
            atom_output, bond_output, targets
        )

        self.assertTrue(
            torch.equal(proposals["target_atom_refs"][0, 0], torch.tensor([1, 0]))
        )
        self.assertTrue(
            torch.allclose(
                proposals["bond_endpoints"][0, 0],
                torch.tensor([[0.80, 0.50], [0.20, 0.50]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
