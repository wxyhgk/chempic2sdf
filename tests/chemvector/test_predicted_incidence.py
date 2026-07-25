import unittest

import torch

from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
    fuse_predicted_atom_proposals,
)


class PredictedIncidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PredictedProposalConfig(
            maximum_atoms=3,
            maximum_bonds=2,
            atom_threshold=0.1,
            bond_threshold=0.1,
            atom_match_radius_px=8.0,
            bond_match_radius_px=8.0,
            image_size=100.0,
        )

    def test_decoders_use_model_peaks_without_target_counts(self) -> None:
        atom_output = {
            "heatmap_logits": torch.tensor([[[[-8.0, 8.0], [7.0, -8.0]]]]),
            "object_features": torch.randn(1, 4, 2, 2),
            "anchors": torch.rand(1, 2, 2, 2),
            "role_logits": torch.randn(1, 2, 2, 2),
            "text_centers": torch.rand(1, 2, 2, 2),
            "text_extents": torch.rand(1, 2, 2, 2),
        }
        bond_output = {
            "heatmap_logits": torch.tensor(
                [[[[8.0, -8.0], [-8.0, -8.0]], [[-8.0, -8.0], [-8.0, 7.0]]]]
            ),
            "object_features": torch.randn(1, 5, 2, 2),
            "endpoints": torch.rand(1, 2, 2, 2, 2, 2),
            "display_logits": torch.randn(1, 2, 4, 2, 2),
            "chemical_logits": torch.randn(1, 2, 4, 2, 2),
        }

        atoms = decode_predicted_atom_proposals(atom_output, self.config)
        bonds = decode_predicted_bond_proposals(bond_output, self.config)

        self.assertEqual(tuple(atoms["atom_features"].shape), (1, 3, 4))
        self.assertEqual(int(atoms["atom_mask"].sum()), 1)
        self.assertEqual(tuple(bonds["bond_endpoints"].shape), (1, 2, 2, 2))
        self.assertEqual(int(bonds["bond_mask"].sum()), 2)

    def test_association_labels_only_covered_predicted_objects(self) -> None:
        atom_proposals = {
            "atom_features": torch.zeros(1, 3, 4),
            "atom_geometry": torch.tensor(
                [[[0.20, 0.50, 0, 0, 0, 0, 0, 0],
                  [0.80, 0.50, 0, 0, 0, 0, 0, 0],
                  [0.50, 0.20, 0, 0, 0, 0, 0, 0]]]
            ),
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_scores": torch.ones(1, 3),
            "atom_indices": torch.arange(3)[None],
        }
        bond_proposals = {
            "bond_features": torch.zeros(1, 2, 5),
            "bond_endpoints": torch.tensor(
                [[[[0.80, 0.50], [0.20, 0.50]],
                  [[0.10, 0.10], [0.10, 0.90]]]]
            ),
            "bond_types": torch.zeros(1, 2, 8),
            "bond_mask": torch.tensor([[True, True]]),
            "bond_scores": torch.ones(1, 2),
            "bond_indices": torch.arange(2)[None],
        }
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.20, 0.50], [0.80, 0.50], [0.0, 0.0]]]
            ),
            "atom_mask": torch.tensor([[True, True, False]]),
            "bond_atom_refs": torch.tensor([[[0, 1], [0, 0]]]),
            "bond_mask": torch.tensor([[True, False]]),
        }

        associated = associate_predicted_proposals(
            atom_proposals, bond_proposals, targets, self.config
        )

        self.assertTrue(associated["atom_target_matched"][0, :2].all())
        self.assertTrue(associated["bond_target_matched"][0, 0])
        self.assertTrue(associated["incidence_target_eligible"][0, 0])
        self.assertTrue(associated["bond_supervision_mask"][0, 0])
        self.assertTrue(associated["atom_presence_targets"][0, :2].all())
        self.assertFalse(associated["atom_presence_targets"][0, 2])
        self.assertTrue(
            torch.equal(
                associated["atom_target_indices"][0], torch.tensor([0, 1, -1])
            )
        )
        self.assertTrue(associated["bond_presence_targets"][0, 0])
        self.assertFalse(associated["bond_presence_targets"][0, 1])
        self.assertTrue(
            torch.equal(associated["target_atom_refs"][0, 0], torch.tensor([1, 0]))
        )

    def test_fusion_keeps_primary_candidates_and_appends_novel_rescue_geometry(self) -> None:
        anchors = torch.zeros(1, 2, 1, 5)
        anchors[0, 0, 0] = torch.linspace(0.1, 0.9, 5)
        anchors[0, 1, 0] = 0.5
        primary = {
            "heatmap_logits": torch.tensor([[[[8.0, -8.0, -8.0, -8.0, 7.0]]]]),
            "object_features": torch.arange(20, dtype=torch.float32).reshape(1, 4, 1, 5),
            "anchors": anchors,
            "role_logits": torch.zeros(1, 2, 1, 5),
            "text_centers": anchors.clone(),
            "text_extents": torch.zeros(1, 2, 1, 5),
        }
        rescue = {
            **primary,
            "heatmap_logits": torch.tensor([[[[-8.0, -8.0, 9.0, -8.0, -8.0]]]]),
            "object_features": torch.full((1, 4, 1, 5), 99.0),
        }
        config = PredictedProposalConfig(
            maximum_atoms=3,
            maximum_bonds=1,
            atom_threshold=0.1,
            image_size=100.0,
        )

        fused = fuse_predicted_atom_proposals(
            primary,
            rescue,
            config,
            primary_capacity=2,
            rescue_capacity=1,
            dedup_radius_px=5.0,
        )

        self.assertEqual(int(fused["atom_mask"].sum()), 3)
        self.assertTrue(torch.equal(fused["atom_sources"], torch.tensor([[0, 0, 1]])))
        self.assertTrue(torch.equal(fused["atom_indices"], torch.tensor([[0, 4, 2]])))
        self.assertTrue(
            torch.equal(
                fused["atom_features"][0, 2], primary["object_features"][0, :, 0, 2]
            )
        )
        self.assertTrue(torch.allclose(fused["atom_geometry"][0, 2, :2], anchors[0, :, 0, 2]))


if __name__ == "__main__":
    unittest.main()
