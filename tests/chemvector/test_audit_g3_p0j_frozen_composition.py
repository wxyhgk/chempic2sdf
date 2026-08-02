import unittest

import torch

from chemvector.audit_g3_p0j_frozen_composition import (
    composition_batch_statistics,
    finalize_mode_totals,
    gather_slot_features,
    maximum_radius_atom_mapping,
    p0j_gate,
)


class G3P0JFrozenCompositionTests(unittest.TestCase):
    def test_radius_mapping_is_one_to_one_and_cardinality_first(self) -> None:
        mapping = maximum_radius_atom_mapping(
            torch.tensor([[0.10, 0.10], [0.11, 0.10]]),
            torch.tensor([0, 1]),
            torch.tensor([[0.10, 0.10], [0.12, 0.10]]),
            torch.tensor([0, 1]),
            radius_pixels=1.1,
            image_size=100.0,
        )

        self.assertEqual(mapping, {0: 0, 1: 1})

    def test_slot_replay_gathers_only_predicted_identities(self) -> None:
        dense = {
            "slot_features": torch.arange(24.0).reshape(1, 2, 3, 2, 2)
        }
        identities = torch.tensor([[[1, 0, 1], [0, 1, 0], [-1, -1, -1]]])
        mask = torch.tensor([[True, True, False]])

        gathered = gather_slot_features(dense, identities, mask)

        torch.testing.assert_close(gathered[0, 0], dense["slot_features"][0, 1, :, 0, 1])
        torch.testing.assert_close(gathered[0, 1], dense["slot_features"][0, 0, :, 1, 0])
        self.assertEqual(gathered[0, 2].tolist(), [0.0, 0.0, 0.0])

    def test_perfect_composed_pair_graph_is_exact(self) -> None:
        atoms = {
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_anchors": torch.tensor(
                [[[0.10, 0.50], [0.50, 0.50], [0.90, 0.50]]]
            ),
        }
        targets = {
            "atom_mask": atoms["atom_mask"].clone(),
            "atom_anchors": atoms["atom_anchors"].clone(),
        }
        pair_output = {
            "pair_indices": torch.tensor([[0, 1], [0, 2], [1, 2]]),
            "sample_indices": torch.tensor([0, 0, 0]),
            "presence_logits": torch.tensor([10.0, -10.0, 9.0]),
            "chemical_logits": torch.tensor(
                [[-5.0, 8.0, -5.0, -5.0], [8.0, -5.0, -5.0, -5.0], [-5.0, -5.0, 8.0, -5.0]]
            ),
            "bond_count_logits": torch.tensor([[-5.0, -5.0, 8.0, -5.0]]),
        }
        normalized = {
            "mask": torch.tensor([[True, True]]),
            "atom_refs": torch.tensor([[[0, 1], [1, 2]]]),
            "chemical": torch.tensor([[1, 2]]),
        }

        result = finalize_mode_totals(
            composition_batch_statistics(pair_output, atoms, targets, normalized)
        )

        self.assertEqual(result["atom_set_exact_images"], 1)
        self.assertEqual(result["representable_bonds"], 2)
        self.assertEqual(result["positive_type_accuracy"], 1.0)
        for mode in ("gt_bond_count", "deploy_bond_count"):
            self.assertEqual(result[mode]["skeleton_exact_images"], 1)
            self.assertEqual(result[mode]["typed_graph_exact_images"], 1)

    def test_gate_distinguishes_native_interface_from_replay(self) -> None:
        oracle_mode = {
            "structural_graph_exact_images": 32,
        }
        oracle = {"gt_count": oracle_mode, "deploy_count": oracle_mode}
        branch = {
            "images": 32,
            "atom_matched": 1424,
            "atom_targets": 1424,
            "atom_set_exact_images": 32,
            "representable_bonds": 1508,
            "target_bonds": 1508,
            "pair_universe": 37380,
            "pair_identity_sha256": "a" * 64,
            "gt_bond_count": {
                "skeleton_exact_images": 32,
                "typed_graph_exact_images": 32,
            },
            "deploy_bond_count": {
                "count_exact_images": 32,
                "skeleton_exact_images": 32,
                "typed_graph_exact_images": 32,
            },
        }
        native = {
            **branch,
            "gt_bond_count": {
                "skeleton_exact_images": 0,
                "typed_graph_exact_images": 0,
            },
            "deploy_bond_count": {
                "count_exact_images": 0,
                "skeleton_exact_images": 0,
                "typed_graph_exact_images": 0,
            },
        }

        gate = p0j_gate(oracle, native, branch)

        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["native_feature_direct_composition"], "failed")
        self.assertEqual(gate["interpretation"], "atom_feature_interface_mismatch")


if __name__ == "__main__":
    unittest.main()
