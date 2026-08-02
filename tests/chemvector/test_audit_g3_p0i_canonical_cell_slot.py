import unittest

import torch

from chemvector.audit_g3_p0i_canonical_cell_slot import (
    audit_checkpoint,
    canonical_cell_slot_identities,
    cutoff_diagnostic,
    cutoff_summary,
    p0i_audit_gate,
    selected_cell_statistics,
    validate_p0i_checkpoint,
)
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256
from chemvector.train_g3_hierarchical_graph import DEFAULT_VISION_CHECKPOINT_SHA256


class G3P0ICanonicalCellSlotAuditTests(unittest.TestCase):
    def test_fixed_final_audit_gate_requires_every_protocol_layer(self) -> None:
        gate = p0i_audit_gate(
            {
                "target_atoms": 2,
                "occupied_cells": 2,
                "collision_cells": 0,
                "extra_cell_occupancy": 0,
                "maximum_cell_occupancy": 1,
                "overflow_cells": 0,
                "overflow_atoms": 0,
            },
            {
                "metrics": {
                    "deploy_count_exact_images": 1,
                    "all_dense_set_exact_images": 1,
                    "top128_set_exact_images": 1,
                    "selector_identity_micro_precision": 1.0,
                    "selector_identity_micro_recall": 1.0,
                    "selector_identity_whole_set_exact_images": 1,
                    "gt_count_topk_set_exact_images": 1,
                },
                "fixed_positive_localization": {"within_6px": 2},
            },
            {"changed": 0},
            {"status": "passed"},
            expected_images=1,
            expected_atoms=2,
        )

        self.assertEqual(gate["status"], "passed")
        self.assertTrue(all(gate["checks"].values()))

    def test_checkpoint_audit_uses_fixed_identity_and_reports_cutoff(self) -> None:
        dense = {
            "presence_logits": torch.tensor([[[[10.0, 9.0, -3.0]]]]),
            "anchors": torch.tensor(
                [[[[[0.10, 0.50, 0.90]], [[0.50, 0.50, 0.50]]]]]
            ),
            "count_logits": torch.tensor([[0.0, 0.0, 10.0]]),
        }

        class FixedAtomOnly(torch.nn.Module):
            def __init__(self, output):
                super().__init__()
                self.output = output

            def atom_head(self, features):
                return self.output

        report, snapshot = audit_checkpoint(
            FixedAtomOnly(dense),
            torch.zeros(1, 1, 1, 3),
            {
                "atom_anchors": torch.tensor([[[0.10, 0.50], [0.50, 0.50]]]),
                "atom_mask": torch.tensor([[True, True]]),
            },
            sample_keys=["train-only"],
            fixed_identities=[[(0, 0, 0), (0, 0, 1)]],
            batch_size=1,
            precision="fp32",
        )

        self.assertEqual(snapshot, [[(0, 0, 0), (0, 0, 1)]])
        self.assertEqual(report["metrics"]["selector_identity_intersection"], 2)
        self.assertEqual(report["metrics"]["gt_count_topk_matched_atoms"], 2)
        self.assertEqual(report["fixed_positive_localization"]["within_6px"], 2)
        self.assertEqual(report["cutoff_margin"]["minimum_margin"], 12.0)

    def test_canonical_identity_uses_anchor_order_not_source_order(self) -> None:
        anchors = torch.tensor([[[0.24, 0.20], [0.21, 0.22], [0.80, 0.80]]])
        mask = torch.tensor([[True, True, True]])

        identities, stats = canonical_cell_slot_identities(
            anchors, mask, slots=3, height=2, width=2
        )

        self.assertEqual(identities, [[(1, 0, 0), (0, 0, 0), (0, 1, 1)]])
        self.assertEqual(stats["target_atoms"], 3)
        self.assertEqual(stats["collision_cells"], 1)
        self.assertEqual(stats["maximum_cell_occupancy"], 2)
        self.assertEqual(stats["overflow_cells"], 0)

    def test_canonical_identity_rejects_duplicate_coordinates(self) -> None:
        anchors = torch.tensor([[[0.20, 0.20], [0.20, 0.20]]])

        with self.assertRaisesRegex(ValueError, "duplicate GT anchors"):
            canonical_cell_slot_identities(
                anchors,
                torch.tensor([[True, True]]),
                slots=3,
                height=2,
                width=2,
            )

    def test_selected_cell_statistics_distinguish_legal_collision_and_excess(self) -> None:
        fixed = [(0, 0, 0), (1, 0, 0), (0, 1, 1)]
        selected = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]

        stats = selected_cell_statistics(selected, fixed)

        self.assertEqual(stats["selected_multi_candidate_cells"], 1)
        self.assertEqual(stats["selected_duplicate_candidates"], 2)
        self.assertEqual(stats["missing_target_cells"], 1)
        self.assertEqual(stats["missing_target_cell_atoms"], 1)
        self.assertEqual(stats["excess_occupancy_candidates"], 1)
        self.assertEqual(stats["missing_occupancy_atoms"], 1)

    def test_cutoff_reports_margin_and_tie_crossing_selection(self) -> None:
        diagnostic = cutoff_diagnostic(torch.tensor([5.0, 4.0, 4.0, 1.0]), 2)
        summary = cutoff_summary([diagnostic, cutoff_diagnostic(torch.tensor([3.0, 2.0]), 1)])

        self.assertEqual(diagnostic["cutoff_margin"], 0.0)
        self.assertEqual(diagnostic["equal_to_selected_cutoff"], 2)
        self.assertTrue(diagnostic["tie_crosses_cutoff"])
        self.assertEqual(summary["tie_crosses_cutoff_images"], 1)
        self.assertEqual(summary["zero_margin_images"], 1)

    def test_checkpoint_protocol_requires_both_scope_and_cli_flags(self) -> None:
        payload = {
            "global_step": 400,
            "structural_scope": {
                "atom_only_capacity": True,
                "canonical_cell_slot_targets": True,
                "bond_refs_and_types": False,
                "count_teacher_forcing": False,
            },
            "training_args": {
                "atom_only_capacity": True,
                "canonical_cell_slot_targets": True,
            },
            "sample_manifest": {"sha256": OVERFIT32_MANIFEST_SHA256},
            "source_vision_checkpoint_sha256": DEFAULT_VISION_CHECKPOINT_SHA256,
            "train_only_metrics": {"p0i_gate": {"status": "failed"}},
        }
        validate_p0i_checkpoint(payload, step=400)

        invalid = {**payload, "training_args": {"atom_only_capacity": True}}
        with self.assertRaisesRegex(ValueError, "two required CLI flags"):
            validate_p0i_checkpoint(invalid, step=400)

        invalid = {**payload, "structural_scope": {"atom_only_capacity": True}}
        with self.assertRaisesRegex(ValueError, "canonical cell-slot"):
            validate_p0i_checkpoint(invalid, step=400)

        invalid = {**payload, "train_only_metrics": {"p0h_gate": {}}}
        with self.assertRaisesRegex(ValueError, "stored gate"):
            validate_p0i_checkpoint(invalid, step=400)


if __name__ == "__main__":
    unittest.main()
