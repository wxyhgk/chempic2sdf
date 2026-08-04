"""Tests for P2-D representable target construction."""

import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p2d_stride1_top128_cache import (
    representable_target_mask,
)


class P2DStride1Top128CacheTests(unittest.TestCase):
    def test_representable_mask_is_maximum_cardinality_one_to_one(self) -> None:
        candidates = torch.tensor([[0.1, 0.1], [0.101, 0.1], [0.2, 0.2]])
        targets = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.9, 0.9]])
        mask = torch.tensor([True, True, True])
        result = representable_target_mask(candidates, targets, mask)
        self.assertEqual(result.tolist(), [True, True, False])


if __name__ == "__main__":
    unittest.main()
