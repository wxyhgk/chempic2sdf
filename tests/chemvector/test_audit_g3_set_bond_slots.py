import unittest

import torch

from chemvector.audit_g3_set_bond_slots import (
    order_queries_by_target,
    validate_p0f_checkpoint,
)
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256
from chemvector.train_g3_hierarchical_graph import DEFAULT_VISION_CHECKPOINT_SHA256


class G3SetBondSlotAuditTests(unittest.TestCase):
    def test_assignment_is_reordered_by_canonical_target(self) -> None:
        ordered = order_queries_by_target(
            [(torch.tensor([7, 3]), torch.tensor([1, 0]))],
            torch.tensor([[True, True, False]]),
        )

        self.assertEqual(ordered.tolist(), [[3, 7, 0]])

    def test_checkpoint_validation_is_strictly_p0f(self) -> None:
        payload = {
            "global_step": 400,
            "structural_scope": {
                "gt_atom_slot_oracle": True,
                "set_learned_bond_slots": True,
                "fixed_learned_bond_slots": False,
                "gt_bond_slot_identity_oracle": False,
                "shared_confidence_aware_bond_assignment": False,
            },
            "sample_manifest": {"sha256": OVERFIT32_MANIFEST_SHA256},
            "source_vision_checkpoint_sha256": DEFAULT_VISION_CHECKPOINT_SHA256,
        }
        validate_p0f_checkpoint(payload, step=400)

        invalid = {
            **payload,
            "structural_scope": {
                **payload["structural_scope"],
                "fixed_learned_bond_slots": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_p0f_checkpoint(invalid, step=400)


if __name__ == "__main__":
    unittest.main()
