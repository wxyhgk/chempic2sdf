"""Tests for the P3-E11 ordered Count loss."""

import unittest

import torch

from chemvector.training.atoms.train_g3_p3e11_ordinal_count import (
    ordinal_count_loss,
)


class G3P3E11OrdinalCountTests(unittest.TestCase):
    def test_nearby_expected_count_has_lower_ordinal_penalty(self) -> None:
        target = torch.tensor([30])
        nearby = torch.full((1, 129), -20.0)
        distant = torch.full((1, 129), -20.0)
        nearby[0, 31] = 20.0
        distant[0, 80] = 20.0
        near_loss = ordinal_count_loss(nearby, target)
        far_loss = ordinal_count_loss(distant, target)
        self.assertLess(
            float(near_loss["ordinal_smooth_l1"]),
            float(far_loss["ordinal_smooth_l1"]),
        )

    def test_negative_ordinal_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ordinal_count_loss(
                torch.zeros(1, 129), torch.tensor([1]), ordinal_weight=-1.0
            )


if __name__ == "__main__":
    unittest.main()
