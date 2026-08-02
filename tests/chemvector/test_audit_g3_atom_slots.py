import unittest

import torch

from chemvector.audit_g3_atom_slots import matched_atom_count, validate_p0_checkpoint
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256
from chemvector.train_g3_hierarchical_graph import DEFAULT_VISION_CHECKPOINT_SHA256


class G3AtomSlotAuditTests(unittest.TestCase):
    def test_matching_is_global_and_radius_limited(self) -> None:
        predicted = torch.tensor([[0.10, 0.10], [0.80, 0.80]])
        expected = torch.tensor([[0.805, 0.80], [0.10, 0.105]])

        matched = matched_atom_count(
            predicted,
            expected,
            radius_pixels=1.0,
            image_size=100.0,
        )

        self.assertEqual(matched, 2)

    def test_checkpoint_validation_rejects_oracle_protocol(self) -> None:
        payload = {
            "global_step": 400,
            "structural_scope": {"gt_atom_slot_oracle": False},
            "sample_manifest": {"sha256": OVERFIT32_MANIFEST_SHA256},
            "source_vision_checkpoint_sha256": DEFAULT_VISION_CHECKPOINT_SHA256,
        }
        validate_p0_checkpoint(payload, step=400)
        payload["structural_scope"]["gt_atom_slot_oracle"] = True
        with self.assertRaisesRegex(ValueError, "original predicted-Atom P0"):
            validate_p0_checkpoint(payload, step=400)


if __name__ == "__main__":
    unittest.main()
