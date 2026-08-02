from pathlib import Path
import unittest

import torch

from chemvector.g3_hierarchical_graph import (
    G3AtomPairBondIdentityPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.train_g3_p0k_relation_alignment import (
    DEFAULT_OUTPUT,
    build_parser,
    build_predicted_pair_targets,
    p0k_training_step,
    validate_args,
    validate_pair_optimizer,
)


class G3P0KRelationAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = G3HierarchicalGraphConfig(
            input_channels=8,
            hidden_channels=16,
            d_model=16,
            atom_slots_per_cell=2,
            maximum_atoms=4,
            bond_slots_per_cell=2,
            maximum_bonds=3,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            memory_stride=1,
        )

    def test_predicted_pair_targets_follow_detached_atom_mapping(self) -> None:
        atoms = {
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_anchors": torch.tensor(
                [[[0.10, 0.50], [0.50, 0.50], [0.90, 0.50]]]
            ),
        }
        targets = {
            "atom_mask": atoms["atom_mask"].clone(),
            "atom_anchors": atoms["atom_anchors"].clone(),
            "bond_mask": torch.tensor([[True, True, False]]),
            "bond_atom_refs": torch.tensor([[[0, 1], [1, 2], [0, 0]]]),
            "bond_types": torch.tensor([[1, 2, 0]]),
            "bond_display": torch.tensor([[1, 2, 0]]),
        }
        pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])

        presence, chemical, coverage = build_predicted_pair_targets(
            atoms, targets, pairs, torch.tensor([0, 0, 0])
        )

        self.assertEqual(presence.tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(chemical.tolist(), [1, 0, 2])
        self.assertEqual(coverage["mapped_atoms"], 3)
        self.assertEqual(coverage["covered_bonds"], 2)

    def test_optimizer_rejects_non_pair_parameters(self) -> None:
        model = G3AtomPairBondIdentityPredictor(self.config)
        valid = torch.optim.AdamW(model.pair_bond.parameters(), lr=1e-3)
        validate_pair_optimizer(model, valid)

        invalid = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "only P0-G pair-head"):
            validate_pair_optimizer(model, invalid)

    def test_cpu_training_step_updates_only_pair_head(self) -> None:
        model = G3AtomPairBondIdentityPredictor(self.config)
        optimizer = torch.optim.AdamW(model.pair_bond.parameters(), lr=1e-3)
        atoms = {
            "atom_mask": torch.tensor([[True, True, True, False]]),
            "atom_anchors": torch.tensor(
                [[[0.10, 0.50], [0.50, 0.50], [0.90, 0.50], [0.0, 0.0]]]
            ),
            "atom_scores": torch.tensor([[0.9, 0.9, 0.9, 0.0]]),
            "atom_features": torch.randn(1, 4, 16),
        }
        targets = {
            "atom_mask": atoms["atom_mask"].clone(),
            "atom_anchors": atoms["atom_anchors"].clone(),
            "bond_mask": torch.tensor([[True, True, False]]),
            "bond_atom_refs": torch.tensor([[[0, 1], [1, 2], [0, 0]]]),
            "bond_types": torch.tensor([[1, 2, 0]]),
            "bond_display": torch.tensor([[1, 2, 0]]),
        }
        before = model.pair_bond.presence.weight.detach().clone()

        losses = p0k_training_step(
            model,
            optimizer,
            torch.randn(1, 8, 8, 8),
            atoms,
            targets,
            precision="fp32",
            max_grad_norm=1.0,
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertFalse(torch.equal(before, model.pair_bond.presence.weight))

    def test_cli_requires_fixed_output_and_manifest(self) -> None:
        parser = build_parser()
        valid = parser.parse_args([])
        validate_args(valid)
        self.assertEqual(Path(valid.output_dir), DEFAULT_OUTPUT)

        invalid = parser.parse_args(["--output-dir", "outputs/other"])
        with self.assertRaisesRegex(ValueError, "fixed output"):
            validate_args(invalid)


if __name__ == "__main__":
    unittest.main()
