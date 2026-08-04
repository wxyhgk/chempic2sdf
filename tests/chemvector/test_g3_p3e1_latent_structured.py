"""Tests for equivalence-invariant P3-E1 latent structured loss."""

import unittest

import torch

from chemvector.training.atoms.train_g3_p3e1_latent_structured_pilot256 import (
    latent_structured_loss,
)


class G3P3E1LatentStructuredTests(unittest.TestCase):
    def test_equivalent_query_permutation_is_not_fixed(self) -> None:
        output = {
            "existence_logits": torch.tensor([[5.0, 4.0, -2.0]], requires_grad=True),
            "pointer_logits": torch.tensor([[[
                8.0, 0.0, 0.0,
            ], [
                0.0, 8.0, 0.0,
            ], [
                0.0, 0.0, 8.0,
            ]]], requires_grad=True),
        }
        candidates = torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.8, 0.8]]])
        targets = torch.tensor([[[0.1, 0.1], [0.2, 0.2]]])
        mask = torch.tensor([[True, True]])
        loss = latent_structured_loss(output, candidates, targets, mask)
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()
        self.assertTrue(torch.isfinite(output["existence_logits"].grad).all())

    def test_invalid_high_score_produces_positive_loss(self) -> None:
        output = {
            "existence_logits": torch.tensor([[5.0, 4.0, 3.0]], requires_grad=True),
            "pointer_logits": torch.tensor([[[
                8.0, 7.0, 0.0,
            ], [
                8.0, 7.0, 0.0,
            ], [
                0.0, 0.0, 8.0,
            ]]], requires_grad=True),
        }
        candidates = torch.tensor([[[0.1, 0.1], [0.102, 0.1], [0.2, 0.2]]])
        targets = torch.tensor([[[0.1, 0.1], [0.2, 0.2]]])
        mask = torch.tensor([[True, True]])
        loss = latent_structured_loss(output, candidates, targets, mask)
        self.assertGreater(float(loss["loss"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
