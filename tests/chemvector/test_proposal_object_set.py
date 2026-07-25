import unittest

import torch

from chemvector.proposal_object_set import (
    ProposalObjectSetConfig,
    ProposalObjectSetHead,
    _graph_exact_surrogate_loss,
    _topology_closure_loss,
    proposal_object_set_loss,
)
from chemvector.spatial_incidence import SpatialIncidenceConfig


class ProposalObjectSetTests(unittest.TestCase):
    def test_zero_initialized_stable_bond_adapter_receives_gradients(self) -> None:
        config = ProposalObjectSetConfig(
            incidence=SpatialIncidenceConfig(
                atom_feature_dim=4,
                bond_feature_dim=5,
                bond_type_dim=8,
                hidden_dim=16,
                pair_hidden_dim=8,
            ),
            hidden_dim=16,
            stable_bond_hidden_dim=12,
        )
        head = ProposalObjectSetHead(config)
        output = head(
            torch.randn(1, 4, 4),
            torch.rand(1, 4, 8),
            torch.rand(1, 4),
            torch.tensor([[True, True, True, False]]),
            torch.randn(1, 3, 5),
            torch.rand(1, 3, 2, 2),
            torch.rand(1, 3, 8),
            torch.rand(1, 3),
            torch.tensor([[True, True, False]]),
        )
        self.assertTrue(
            torch.equal(
                output["bond_presence_logits"][0, :2],
                output["base_bond_presence_logits"][0, :2],
            )
        )
        output["bond_presence_logits"].sum().backward()
        self.assertGreater(
            float(head.stable_bond_presence[-1].weight.grad.abs().sum()), 0.0
        )

    def test_forward_and_joint_loss_have_stable_shapes_and_gradients(self) -> None:
        config = ProposalObjectSetConfig(
            incidence=SpatialIncidenceConfig(
                atom_feature_dim=4,
                bond_feature_dim=5,
                bond_type_dim=8,
                hidden_dim=16,
                pair_hidden_dim=8,
            ),
            hidden_dim=16,
        )
        head = ProposalObjectSetHead(config)
        atom_mask = torch.tensor([[True, True, True, False]])
        bond_mask = torch.tensor([[True, True, False]])
        output = head(
            torch.randn(1, 4, 4),
            torch.rand(1, 4, 8),
            torch.rand(1, 4),
            atom_mask,
            torch.randn(1, 3, 5),
            torch.rand(1, 3, 2, 2),
            torch.rand(1, 3, 8),
            torch.rand(1, 3),
            bond_mask,
        )
        losses = proposal_object_set_loss(
            output,
            torch.tensor([[True, True, False, False]]),
            torch.tensor([[True, False, False]]),
            atom_mask,
            bond_mask,
            torch.tensor([[[0, 1], [0, 0], [0, 0]]]),
            torch.tensor([[True, False, False]]),
            topology_closure_weight=0.2,
        )
        losses["loss"].backward()

        self.assertEqual(tuple(output["atom_presence_logits"].shape), (1, 4))
        self.assertEqual(tuple(output["bond_presence_logits"].shape), (1, 3))
        self.assertTrue((output["atom_presence_logits"][0, 3] == -1e4))
        self.assertTrue((output["bond_presence_logits"][0, 2] == -1e4))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(
            float(head.atom_pre_presence[0].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(head.bond_pre_presence[0].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(head.incidence.atom_projection[0].weight.grad.abs().sum()), 0.0
        )

    def test_topology_closure_penalizes_bond_without_present_endpoint_atoms(self) -> None:
        incidence = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True
        )
        common = {
            "bond_presence_logits": torch.tensor([[5.0]], requires_grad=True),
            "incidence_probabilities": incidence,
        }
        aligned = _topology_closure_loss(
            {**common, "atom_presence_logits": torch.tensor([[5.0, 5.0]])},
            torch.tensor([[True, True]]),
            torch.tensor([[True]]),
        )
        missing = _topology_closure_loss(
            {**common, "atom_presence_logits": torch.tensor([[-5.0, -5.0]])},
            torch.tensor([[True, True]]),
            torch.tensor([[True]]),
        )
        missing.backward()

        self.assertLess(float(aligned.detach()), float(missing.detach()))
        self.assertIsNone(incidence.grad)

    def test_graph_exact_surrogate_tracks_worst_presence_and_incidence_error(self) -> None:
        targets = torch.tensor([[True, False]])
        bond_targets = torch.tensor([[True]])
        atom_mask = torch.tensor([[True, True]])
        bond_mask = torch.tensor([[True]])
        refs = torch.tensor([[[0, 1]]])
        supervised = torch.tensor([[True]])
        good = {
            "atom_presence_logits": torch.tensor([[5.0, -5.0]]),
            "bond_presence_logits": torch.tensor([[5.0]]),
            "incidence_logits": torch.tensor([[[[5.0, -5.0], [-5.0, 5.0]]]]),
        }
        bad = {
            **good,
            "atom_presence_logits": torch.tensor([[-5.0, 5.0]]),
        }

        good_loss = _graph_exact_surrogate_loss(
            good,
            targets,
            bond_targets,
            atom_mask,
            bond_mask,
            refs,
            supervised,
            atom_threshold=0.0,
            bond_threshold=-0.5,
        )
        bad_loss = _graph_exact_surrogate_loss(
            bad,
            targets,
            bond_targets,
            atom_mask,
            bond_mask,
            refs,
            supervised,
            atom_threshold=0.0,
            bond_threshold=-0.5,
        )

        self.assertLess(float(good_loss), float(bad_loss))


if __name__ == "__main__":
    unittest.main()
