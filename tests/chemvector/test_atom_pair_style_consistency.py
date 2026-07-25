from __future__ import annotations

import unittest

import torch

from chemvector.atom_pair_style_consistency import (
    AtomPairStyleResidualAdapter,
    flatten_target_atom_pairs,
    fold_style_residual_adapter,
    paired_style_consistency_loss,
)
from chemvector.atom_pair_graph_context import (
    AtomPairGraphContextConfig,
    AtomPairGraphContextHead,
)
from chemvector.atom_pair_bond_group import GROUP_FEATURE_DIM


class AtomPairStyleConsistencyTest(unittest.TestCase):
    def test_folded_adapter_matches_explicit_delta(self) -> None:
        context = AtomPairGraphContextHead(
            AtomPairGraphContextConfig(hidden_dim=8, maximum_atoms=4)
        )
        adapter = AtomPairStyleResidualAdapter(8)
        torch.nn.init.normal_(adapter.delta.weight)
        features = torch.randn(2, GROUP_FEATURE_DIM)
        base = torch.randn(2)
        samples = torch.tensor([0, 0])
        pairs = torch.tensor([[0, 1], [1, 2]])
        original = context(features, base, samples, pairs, batch_size=1)
        folded = fold_style_residual_adapter(context, adapter)
        revised = folded(features, base, samples, pairs, batch_size=1)
        torch.testing.assert_close(
            revised["presence_logits"],
            original["presence_logits"]
            + adapter(original["residual_hidden"]),
        )

    def test_flatten_target_pairs_keeps_rendering_order(self) -> None:
        pairs = flatten_target_atom_pairs(
            [[(0, 1), (1, 2)], [(0, 1)]], device=torch.device("cpu")
        )
        self.assertEqual(pairs.tolist(), [[0, 1], [1, 2], [0, 1]])

    def test_identical_renderings_have_zero_consistency_loss(self) -> None:
        losses = paired_style_consistency_loss(
            torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([[0, 1], [1, 2], [0, 1], [1, 2]]),
            torch.tensor([0, 0]),
            batch_size=2,
            maximum_atoms=4,
        )
        self.assertAlmostEqual(float(losses["edge_consistency_loss"].detach()), 0.0)
        self.assertAlmostEqual(float(losses["count_consistency_loss"].detach()), 0.0)
        self.assertAlmostEqual(float(losses["degree_consistency_loss"].detach()), 0.0)
        self.assertEqual(int(losses["aligned_target_pairs"]), 2)

    def test_different_renderings_produce_gradient(self) -> None:
        logits = torch.tensor([3.0, -3.0, -3.0, 3.0], requires_grad=True)
        losses = paired_style_consistency_loss(
            logits,
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([[0, 1], [1, 2], [0, 1], [1, 2]]),
            torch.tensor([0, 0]),
            batch_size=2,
            maximum_atoms=4,
        )
        total = (
            losses["edge_consistency_loss"]
            + losses["count_consistency_loss"]
            + losses["degree_consistency_loss"]
        )
        self.assertGreater(float(total.detach()), 0.0)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_incomplete_representable_group_is_excluded(self) -> None:
        losses = paired_style_consistency_loss(
            torch.tensor([3.0, -3.0], requires_grad=True),
            torch.tensor([0, 1]),
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([0, 0]),
            batch_size=2,
            maximum_atoms=4,
            valid_samples=torch.tensor([True, False]),
            normalization_counts=torch.tensor([1, 1]),
        )
        self.assertEqual(int(losses["aligned_target_pairs"]), 0)
        self.assertAlmostEqual(
            float(losses["edge_consistency_loss"].detach()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
