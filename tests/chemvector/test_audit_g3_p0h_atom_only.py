import json
import unittest

import torch

from chemvector.audit_g3_p0h_atom_only import (
    audit_checkpoint,
    conditional_localization_summary,
    flatten_atom_candidates,
    hungarian_radius_match_count,
    identity_retention,
    model_from_checkpoint,
    ranked_set_retention,
    selector_identity_counts,
    supervision_positive_snapshot,
    validate_p0h_checkpoint,
)
from chemvector.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256
from chemvector.train_g3_hierarchical_graph import DEFAULT_VISION_CHECKPOINT_SHA256


class G3P0HAtomOnlyAuditTests(unittest.TestCase):
    def test_checkpoint_audit_aggregates_selector_identity_metrics(self) -> None:
        dense = {
            "presence_logits": torch.tensor([[[[10.0, 9.0]]]]),
            "anchors": torch.tensor(
                [[[[[0.20, 0.80]], [[0.50, 0.50]]]]]
            ),
            "count_logits": torch.tensor([[0.0, 0.0, 10.0]]),
        }

        class FixedAtomOnly(torch.nn.Module):
            def __init__(self, output):
                super().__init__()
                self.output = output

            def atom_head(self, features):
                return self.output

        report, _ = audit_checkpoint(
            FixedAtomOnly(dense),
            torch.zeros(1, 1, 1, 2),
            {
                "atom_anchors": torch.tensor([[[0.20, 0.50], [0.80, 0.50]]]),
                "atom_mask": torch.tensor([[True, True]]),
            },
            sample_keys=["train-only"],
            batch_size=1,
            precision="fp32",
        )

        metrics = report["metrics"]
        self.assertEqual(metrics["selector_identity_intersection"], 2)
        self.assertEqual(metrics["selector_identity_selected_only"], 0)
        self.assertEqual(metrics["selector_identity_positive_only"], 0)
        self.assertEqual(metrics["selector_identity_micro_precision"], 1.0)
        self.assertEqual(metrics["selector_identity_micro_recall"], 1.0)
        self.assertEqual(metrics["selector_identity_whole_set_exact_images"], 1)
        conditional = report["selected_positive_conditional_localization"]
        self.assertEqual(conditional["selected_positive_atoms"], 2)
        self.assertEqual(conditional["within_6px_accuracy"], 1.0)

    def test_selector_identity_reports_full_confusion_and_conditional_location(
        self,
    ) -> None:
        selected = [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
        positive = [(0, 1, 1), (1, 2, 2), (0, 4, 4)]

        counts = selector_identity_counts(selected, positive)
        location = conditional_localization_summary([2.0, 7.0])

        self.assertEqual(
            counts,
            {
                "selected": 3,
                "positive": 3,
                "intersection": 2,
                "selected_only": 1,
                "positive_only": 1,
                "whole_set_exact": 0,
            },
        )
        self.assertEqual(location["selected_positive_atoms"], 2)
        self.assertEqual(location["within_6px"], 1)
        self.assertEqual(location["within_6px_accuracy"], 0.5)
        self.assertAlmostEqual(location["median_pixels"], 4.5)
        self.assertAlmostEqual(location["p90_pixels"], 6.5)

    def test_real_atom_only_state_dict_strict_loads_through_audit_path(self) -> None:
        config = G3HierarchicalGraphConfig(
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
        source = G3AtomOnlyPredictor(config)
        payload = {
            "model_type": source.model_type,
            "model": source.state_dict(),
        }

        restored = model_from_checkpoint(payload, config, torch.device("cpu"))

        self.assertIsInstance(restored, G3AtomOnlyPredictor)
        self.assertFalse(restored.training)
        self.assertEqual(set(restored.state_dict()), set(source.state_dict()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)

    def test_radius_matching_is_one_to_one_and_cardinality_first(self) -> None:
        predicted = torch.tensor([[0.10, 0.10], [0.11, 0.10]])
        expected = torch.tensor([[0.10, 0.10], [0.12, 0.10]])

        matched = hungarian_radius_match_count(
            predicted, expected, radius_pixels=1.1, image_size=100.0
        )

        self.assertEqual(matched, 2)
        self.assertIs(type(matched), int)
        json.dumps({"matched": matched})

    def test_flattened_candidate_identity_matches_slot_major_layout(self) -> None:
        dense = {
            "presence_logits": torch.arange(8.0).reshape(1, 2, 2, 2),
            "anchors": torch.zeros(1, 2, 2, 2, 2),
        }

        logits, anchors, identities = flatten_atom_candidates(dense)

        self.assertEqual(logits.tolist(), [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        self.assertEqual(tuple(anchors.shape), (1, 8, 2))
        self.assertEqual(
            identities.tolist(),
            [
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
            ],
        )

    def test_supervision_snapshot_is_ordered_by_canonical_target(self) -> None:
        dense = {
            "presence_logits": torch.zeros(1, 2, 1, 2),
            "anchors": torch.tensor(
                [[[[[0.20, 0.70]], [[0.50, 0.50]]], [[[0.80, 0.80]], [[0.50, 0.50]]]]]
            ).reshape(1, 2, 2, 1, 2),
        }
        targets = {
            "atom_anchors": torch.tensor([[[0.20, 0.50], [0.80, 0.50]]]),
            "atom_mask": torch.tensor([[True, True]]),
        }

        identities, errors = supervision_positive_snapshot(
            dense, targets, image_size=100.0
        )

        self.assertEqual(len(identities[0]), 2)
        self.assertEqual(set(identities[0]), {(0, 0, 0), (1, 0, 1)})
        self.assertEqual(errors, [[0.0, 0.0]])

    def test_identity_and_ranking_churn_are_reported_separately(self) -> None:
        previous = [[(0, 1, 1), (1, 2, 2)]]
        current = [[(0, 1, 1), (2, 2, 2)]]

        identity = identity_retention(previous, current)
        ranking = ranked_set_retention(previous, [list(reversed(previous[0]))])

        self.assertEqual(identity["retained"], 1)
        self.assertEqual(identity["changed_images"], 1)
        self.assertEqual(ranking["set_retention"], 1.0)
        self.assertEqual(ranking["rank_position_retention"], 0.0)
        self.assertEqual(ranking["changed_images"], 1)

    def test_checkpoint_validation_binds_atom_only_scope_and_manifest(self) -> None:
        payload = {
            "global_step": 400,
            "structural_scope": {
                "atom_only_capacity": True,
                "bond_refs_and_types": False,
                "count_teacher_forcing": False,
            },
            "training_args": {"atom_only_capacity": True},
            "sample_manifest": {"sha256": OVERFIT32_MANIFEST_SHA256},
            "source_vision_checkpoint_sha256": DEFAULT_VISION_CHECKPOINT_SHA256,
        }
        validate_p0h_checkpoint(payload, step=400)

        invalid = {**payload, "structural_scope": {"atom_only_capacity": True}}
        with self.assertRaisesRegex(ValueError, "Atom-only"):
            validate_p0h_checkpoint(invalid, step=400)

        invalid = {**payload, "sample_manifest": {"sha256": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "overfit32"):
            validate_p0h_checkpoint(invalid, step=400)


if __name__ == "__main__":
    unittest.main()
