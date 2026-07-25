"""Tests for high-recall whole-set candidate selection."""

from __future__ import annotations

import unittest

import torch

from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
    structured_incident_features,
    structured_selector_loss,
)


class StructuredSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StructuredSelectorConfig(
            candidate_dim=16,
            visual_dim=8,
            d_model=16,
            heads=4,
            ffn_dim=32,
            layers=1,
            maximum_candidates=4,
            incident_topk=2,
            residual_scale=4.0,
            role_count=2,
        )
        self.pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)

    def test_zero_residual_preserves_base_ranking(self) -> None:
        model = GroundedStructuredSelector(self.config).eval()
        base = torch.tensor([[3.0, 2.0, 1.0, -30.0]])
        mask = torch.tensor([[True, True, True, False]])

        output = model(
            torch.randn(1, 4, 16),
            base,
            torch.rand(1, 4, 2),
            mask,
            torch.randn(1, 6),
            self.pairs,
            torch.randn(1, 4, self.config.auxiliary_dim),
            torch.randn(1, 4, self.config.visual_dim),
            torch.tensor([[0, 1, 0, 0]]),
        )

        self.assertTrue(torch.equal(output[mask], base[mask]))

    def test_incident_features_and_whole_set_loss_are_finite(self) -> None:
        mask = torch.tensor([[True, True, True, False]])
        edge_logits = torch.tensor([[8.0, -8.0, -8.0, 4.0, -8.0, -8.0]])
        incident = structured_incident_features(
            edge_logits, self.pairs, mask, incident_topk=2
        )
        logits = torch.tensor([[2.0, -1.0, 3.0, -30.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        losses = structured_selector_loss(logits, targets, mask)
        losses["loss"].backward()

        self.assertEqual(incident.shape, (1, 4, 4))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["whole_set_loss"].detach()), 0.0)

    def test_extreme_set_loss_uses_strongest_negative_and_weakest_positive(self) -> None:
        logits = torch.tensor([[3.0, 1.0, 2.0, -4.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)

        losses = structured_selector_loss(
            logits,
            targets,
            mask,
            whole_set_weight=0.0,
            extreme_set_weight=2.0,
            extreme_set_margin=1.0,
        )

        expected = torch.nn.functional.softplus(torch.tensor(2.0))
        self.assertTrue(torch.allclose(losses["extreme_set_loss"], expected))
        losses["loss"].backward()
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
