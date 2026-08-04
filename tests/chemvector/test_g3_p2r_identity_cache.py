"""Tests for the frozen P2-J query-identity Action Cache."""

import copy
import unittest

import torch

from chemvector.training.atoms.train_g3_p2r_p2j_identity_existence_capacity64 import (
    action_cache_digest,
    fixed_identity_existence_loss,
    validate_action_cache,
)
from chemvector.training.atoms.train_g3_p2s_fixed_identity_listwise_existence_capacity64 import (
    smooth_fixed_identity_listwise_loss,
)


class G3P2RIdentityCacheTests(unittest.TestCase):
    @staticmethod
    def payload() -> dict:
        entries = [{
            "domain": "easy32",
            "sample_key": "sample_0001_rdkit",
            "positive_queries": 2,
            "query_mask": [1, 0, 1] + [0] * 125,
        }]
        return {
            "entries": entries,
            "entries_sha256": action_cache_digest(entries),
            "positive_queries": 2,
        }

    def test_action_cache_digest_and_mask_validate(self) -> None:
        payload = self.payload()
        lookup = validate_action_cache(payload)
        self.assertEqual(set(lookup), {"sample_0001_rdkit"})
        self.assertEqual(int(lookup["sample_0001_rdkit"].sum()), 2)
        self.assertEqual(payload["entries_sha256"], action_cache_digest(payload["entries"]))

    def test_action_cache_rejects_tampering(self) -> None:
        payload = copy.deepcopy(self.payload())
        payload["entries"][0]["query_mask"][4] = 1
        with self.assertRaisesRegex(ValueError, "digest differs"):
            validate_action_cache(payload)

    def test_fixed_identity_loss_backpropagates_and_rewards_ordering(self) -> None:
        targets = torch.tensor([[True, False, True, False]])
        good = torch.tensor([[4.0, -4.0, 3.0, -3.0]], requires_grad=True)
        bad = -good.detach()
        good_loss = fixed_identity_existence_loss(good, targets)
        bad_loss = fixed_identity_existence_loss(bad, targets)
        self.assertLess(
            float(good_loss["loss"].detach()), float(bad_loss["loss"].detach())
        )
        self.assertEqual(float(good_loss["boundary_exact_images"]), 1.0)
        good_loss["loss"].backward()
        self.assertTrue(torch.isfinite(good.grad).all())

    def test_smooth_listwise_loss_focuses_misordered_queries(self) -> None:
        targets = torch.tensor([[True, False, True, False]])
        ordered = torch.tensor([[4.0, -2.0, 3.0, -3.0]], requires_grad=True)
        violated = torch.tensor([[4.0, 3.5, 3.0, -3.0]])
        ordered_loss = smooth_fixed_identity_listwise_loss(ordered, targets)
        violated_loss = smooth_fixed_identity_listwise_loss(violated, targets)
        self.assertLess(
            float(ordered_loss["listwise_ordering_loss"].detach()),
            float(violated_loss["listwise_ordering_loss"].detach()),
        )
        ordered_loss["loss"].backward()
        self.assertTrue(torch.isfinite(ordered.grad).all())


if __name__ == "__main__":
    unittest.main()
