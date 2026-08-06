"""Tests for additive candidate-occupancy Count."""

import unittest

import torch

from chemvector.components.atoms.objects.g3_poisson_binomial_count import (
    G3PoissonBinomialCount,
    G3PoissonBinomialCountConfig,
    g3_poisson_binomial_count_loss,
)


class G3PoissonBinomialCountTests(unittest.TestCase):
    def test_shapes_distribution_and_gradients(self) -> None:
        model = G3PoissonBinomialCount(
            G3PoissonBinomialCountConfig(
                query_dim=4,
                candidate_dim=6,
                global_dim=8,
                hidden_dim=16,
            )
        )
        output = model(
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 6),
            torch.randn(2, 8),
        )
        self.assertEqual(output["presence_logits"].shape, (2, 5))
        self.assertEqual(output["count_log_probs"].shape, (2, 6))
        self.assertTrue(
            torch.allclose(
                output["count_log_probs"].exp().sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        losses = g3_poisson_binomial_count_loss(output, torch.tensor([2, 3]))
        losses["loss"].backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_candidate_permutation_is_equivariant_and_count_invariant(self) -> None:
        model = G3PoissonBinomialCount().eval()
        query = torch.randn(2, 128, 8)
        candidate = torch.randn(2, 128, 41)
        global_features = torch.randn(2, 256)
        permutation = torch.randperm(128)
        with torch.no_grad():
            expected = model(query, candidate, global_features)
            current = model(query, candidate[:, permutation], global_features)
        self.assertTrue(
            torch.allclose(
                expected["presence_logits"][:, permutation],
                current["presence_logits"],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                expected["count_log_probs"],
                current["count_log_probs"],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_loss_rejects_count_above_candidate_capacity(self) -> None:
        output = {
            "count_log_probs": torch.zeros(1, 4),
            "expected_count": torch.zeros(1),
        }
        with self.assertRaisesRegex(ValueError, "exceeds candidate capacity"):
            g3_poisson_binomial_count_loss(output, torch.tensor([4]))


if __name__ == "__main__":
    unittest.main()
