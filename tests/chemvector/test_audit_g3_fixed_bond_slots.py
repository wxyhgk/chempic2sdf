import unittest

import torch

from chemvector.audit_g3_fixed_bond_slots import (
    summarize_fixed_predictions,
    validate_p0e_checkpoint,
)
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256
from chemvector.train_g3_hierarchical_graph import DEFAULT_VISION_CHECKPOINT_SHA256


class G3FixedBondSlotAuditTests(unittest.TestCase):
    def test_summary_separates_geometry_pointer_and_type(self) -> None:
        mask = torch.tensor([[True, True, False]])
        target_endpoints = torch.tensor(
            [[
                [[0.10, 0.10], [0.30, 0.10]],
                [[0.40, 0.20], [0.60, 0.20]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]]
        )
        pointer_logits = torch.full((1, 3, 2, 4), -10.0)
        pointer_logits[0, 0, 0, 1] = 10.0
        pointer_logits[0, 0, 1, 0] = 10.0
        pointer_logits[0, 1, 0, 2] = 10.0
        pointer_logits[0, 1, 1, 1] = 10.0
        type_logits = torch.zeros(1, 3, 4)
        type_logits[0, 0, 2] = 10.0
        type_logits[0, 1, 1] = 10.0

        metrics, errors = summarize_fixed_predictions(
            predicted_endpoints=target_endpoints.clone(),
            target_endpoints=target_endpoints,
            target_mask=mask,
            pointer_logits=pointer_logits,
            type_logits=type_logits,
            target_atom_refs=torch.tensor([[[0, 1], [1, 3], [0, 0]]]),
            target_types=torch.tensor([[2, 1, 0]]),
        )

        self.assertEqual(errors, [0.0, 0.0])
        self.assertEqual(metrics["pointer_endpoint_correct"], 3)
        self.assertEqual(metrics["pointer_both_endpoint_correct"], 1)
        self.assertEqual(metrics["type_correct"], 2)
        self.assertEqual(metrics["type_given_pointer_correct"], 1)

    def test_checkpoint_validation_is_strictly_p0e(self) -> None:
        payload = {
            "global_step": 400,
            "structural_scope": {
                "gt_atom_slot_oracle": True,
                "fixed_learned_bond_slots": True,
                "gt_bond_slot_identity_oracle": False,
                "shared_confidence_aware_bond_assignment": False,
            },
            "sample_manifest": {"sha256": OVERFIT32_MANIFEST_SHA256},
            "source_vision_checkpoint_sha256": DEFAULT_VISION_CHECKPOINT_SHA256,
        }
        validate_p0e_checkpoint(payload, step=400)

        invalid = {
            **payload,
            "structural_scope": {
                **payload["structural_scope"],
                "gt_bond_slot_identity_oracle": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_p0e_checkpoint(invalid, step=400)


if __name__ == "__main__":
    unittest.main()
