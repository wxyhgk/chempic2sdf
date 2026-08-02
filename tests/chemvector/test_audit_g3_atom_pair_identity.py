import unittest

import torch

from chemvector.audit_g3_atom_pair_identity import (
    atom_pair_identity_sha256,
    validate_p0g_checkpoint,
)
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256
from chemvector.train_g3_hierarchical_graph import DEFAULT_VISION_CHECKPOINT_SHA256


class G3AtomPairIdentityAuditTests(unittest.TestCase):
    def test_identity_hash_includes_sample_and_unordered_pair(self) -> None:
        first = atom_pair_identity_sha256(
            ["a", "b"],
            torch.tensor([[0, 1], [0, 2]]),
            torch.tensor([0, 1]),
        )
        second = atom_pair_identity_sha256(
            ["a", "b"],
            torch.tensor([[0, 1], [0, 3]]),
            torch.tensor([0, 1]),
        )

        self.assertNotEqual(first, second)

    def test_checkpoint_validation_is_strictly_p0g(self) -> None:
        payload = {
            "global_step": 400,
            "structural_scope": {
                "gt_atom_slot_oracle": True,
                "direct_gt_atom_pair_bonds": True,
                "fixed_learned_bond_slots": False,
                "set_learned_bond_slots": False,
                "gt_bond_slot_identity_oracle": False,
                "shared_confidence_aware_bond_assignment": False,
            },
            "sample_manifest": {"sha256": OVERFIT32_MANIFEST_SHA256},
            "source_vision_checkpoint_sha256": DEFAULT_VISION_CHECKPOINT_SHA256,
        }
        validate_p0g_checkpoint(payload, step=400)

        invalid = {
            **payload,
            "structural_scope": {
                **payload["structural_scope"],
                "set_learned_bond_slots": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_p0g_checkpoint(invalid, step=400)


if __name__ == "__main__":
    unittest.main()
