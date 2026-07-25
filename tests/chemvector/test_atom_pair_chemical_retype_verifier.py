from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import torch

from chemvector.atom_pair_chemical_retype import (
    AtomPairChemicalRetypeConfig,
    AtomPairChemicalRetypeHead,
    AtomPairRetypeAction,
)
from chemvector.atom_pair_chemical_retype_verifier import (
    AtomPairChemicalRetypeVerifier,
    AtomPairChemicalRetypeVerifierConfig,
    candidate_exact_labels,
    chemical_retype_verifier_loss,
    choose_verified_retype_action,
    initialize_verifier_from_retype,
    load_atom_pair_chemical_retype_verifier,
)


class AtomPairChemicalRetypeVerifierTest(unittest.TestCase):
    def test_labels_identify_only_the_exact_candidate_action(self) -> None:
        actions = [AtomPairRetypeAction(0, 1), AtomPairRetypeAction(1, 2)]
        labels = candidate_exact_labels(
            actions,
            torch.tensor((0, 0)),
            torch.tensor((1, 0)),
        )
        self.assertEqual(labels.tolist(), [1.0, 0.0])

    def test_loss_ranks_positive_above_hardest_negative(self) -> None:
        scores = torch.tensor((0.0, 0.5, -0.2), requires_grad=True)
        values = chemical_retype_verifier_loss(
            scores, torch.tensor((1.0, 0.0, 0.0)), ranking_margin=1.0
        )
        values["loss"].backward()
        self.assertGreater(float(values["ranking_loss"].detach()), 0.0)
        self.assertLess(float(scores.grad[0]), 0.0)
        self.assertGreater(float(scores.grad[1]), 0.0)

    def test_all_negative_sample_pushes_down_current_maximum(self) -> None:
        scores = torch.tensor((-0.5, 0.3), requires_grad=True)
        values = chemical_retype_verifier_loss(
            scores, torch.zeros(2), ranking_margin=1.0
        )
        values["loss"].backward()
        self.assertEqual(float(scores.grad[0]), 0.0)
        self.assertGreater(float(scores.grad[1]), 0.0)

    def test_zero_initialized_verifier_keeps_at_zero_margin(self) -> None:
        model = AtomPairChemicalRetypeVerifier(
            AtomPairChemicalRetypeVerifierConfig(hidden_dim=16, maximum_groups=8)
        )
        edge = torch.randn((2, 26))
        chemical = torch.randn((2, 4))
        selected = torch.tensor((True, True))
        actions = [AtomPairRetypeAction(0, 1), AtomPairRetypeAction(1, 2)]
        scores = model.score_sample(
            edge,
            chemical,
            selected,
            torch.tensor(0.0),
            torch.tensor((0.0, 0.0)),
            actions,
        )
        self.assertIsNone(
            choose_verified_retype_action(scores, actions, minimum_margin=0.0)
        )

    def test_warm_start_copies_edge_and_type_encoders(self) -> None:
        retype = AtomPairChemicalRetypeHead(
            AtomPairChemicalRetypeConfig(hidden_dim=16)
        )
        verifier = AtomPairChemicalRetypeVerifier(
            AtomPairChemicalRetypeVerifierConfig(hidden_dim=16)
        )
        initialize_verifier_from_retype(verifier, retype)
        self.assertTrue(
            torch.equal(
                verifier.edge_encoder[0].weight,
                retype.edge_encoder[0].weight,
            )
        )
        self.assertTrue(
            torch.equal(
                verifier.type_embedding.weight,
                retype.type_embedding.weight,
            )
        )

    def test_checkpoint_loader_round_trips_verifier(self) -> None:
        model = AtomPairChemicalRetypeVerifier(
            AtomPairChemicalRetypeVerifierConfig(hidden_dim=16)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verifier.pt"
            torch.save(
                {
                    "checkpoint_schema_version": (
                        "atom-pair-chemical-retype-verifier-0.1"
                    ),
                    "config": {
                        "input_dim": 26,
                        "hidden_dim": 16,
                        "type_embedding_dim": 16,
                        "maximum_groups": 256,
                        "ranking_margin": 1.0,
                    },
                    "head": model.state_dict(),
                },
                path,
            )
            loaded, _ = load_atom_pair_chemical_retype_verifier(
                path, torch.device("cpu")
            )
        self.assertTrue(
            torch.equal(
                loaded.verifier[-1].weight,
                model.verifier[-1].weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
