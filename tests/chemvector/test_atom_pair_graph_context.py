from __future__ import annotations

import unittest

import torch

from chemvector.atom_pair_bond_group import GROUP_FEATURE_DIM
from chemvector.atom_pair_graph_context import (
    AtomPairGraphContextConfig,
    AtomPairGraphContextHead,
    atom_pair_graph_context_loss,
    flatten_atom_pairs,
    refine_atom_pair_group_output,
)
from chemvector.calibrate_atom_pair_graph_context import (
    evaluate_scale,
    selection_key,
)


class AtomPairGraphContextTest(unittest.TestCase):
    def test_zero_initialization_preserves_base_logits(self) -> None:
        model = AtomPairGraphContextHead(
            AtomPairGraphContextConfig(hidden_dim=16, maximum_atoms=8)
        )
        features = torch.randn(3, GROUP_FEATURE_DIM)
        base = torch.tensor((1.0, -0.5, 0.25))
        output = model(
            features,
            base,
            torch.tensor((0, 0, 0)),
            torch.tensor(((0, 1), (1, 2), (2, 3))),
            batch_size=1,
        )
        torch.testing.assert_close(output["presence_logits"], base)

    def test_group_permutation_is_equivariant(self) -> None:
        torch.manual_seed(7)
        model = AtomPairGraphContextHead(
            AtomPairGraphContextConfig(hidden_dim=16, maximum_atoms=8)
        )
        torch.nn.init.normal_(model.residual[-1].weight)
        features = torch.randn(4, GROUP_FEATURE_DIM)
        base = torch.randn(4)
        samples = torch.tensor((0, 0, 1, 1))
        pairs = torch.tensor(((0, 1), (1, 2), (0, 2), (2, 3)))
        order = torch.tensor((1, 0, 3, 2))
        direct = model(features, base, samples, pairs, batch_size=2)["presence_logits"]
        permuted = model(
            features[order], base[order], samples[order], pairs[order], batch_size=2
        )["presence_logits"]
        torch.testing.assert_close(permuted, direct[order])

    def test_loss_is_finite(self) -> None:
        output = {
            "presence_logits": torch.tensor((1.0, -0.5, 0.25), requires_grad=True),
            "residual_logits": torch.tensor((0.1, -0.1, 0.0), requires_grad=True),
        }
        losses = atom_pair_graph_context_loss(
            output,
            torch.tensor((True, False, True)),
            torch.tensor((0, 0, 0)),
            torch.tensor(((0, 1), (0, 2), (1, 2))),
            batch_size=1,
            maximum_atoms=8,
            positive_margin_weight=0.25,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()

    def test_positive_margin_penalizes_true_logit_below_zero(self) -> None:
        output = {
            "presence_logits": torch.tensor((-0.25, -1.0), requires_grad=True),
            "residual_logits": torch.zeros(2, requires_grad=True),
        }
        losses = atom_pair_graph_context_loss(
            output,
            torch.tensor((True, False)),
            torch.tensor((0, 0)),
            torch.tensor(((0, 1), (0, 2))),
            batch_size=1,
            maximum_atoms=4,
            positive_margin=0.5,
            positive_margin_weight=1.0,
        )

        self.assertAlmostEqual(
            float(losses["positive_margin_loss"].detach()), 0.75
        )

    def test_flatten_atom_pairs_keeps_sample_order(self) -> None:
        pairs = flatten_atom_pairs(
            [[(0, 1), (1, 2)], [(3, 4)]], device=torch.device("cpu")
        )
        self.assertEqual(pairs.tolist(), [[0, 1], [1, 2], [3, 4]])

    def test_refinement_only_replaces_presence_logits(self) -> None:
        model = AtomPairGraphContextHead(
            AtomPairGraphContextConfig(hidden_dim=16, maximum_atoms=8)
        )
        group_output = {
            "presence_logits": torch.tensor((1.0, -0.5)),
            "chemical_logits": torch.randn(2, 4),
        }
        groups = {
            "features": torch.randn(2, GROUP_FEATURE_DIM),
            "sample_indices": torch.tensor((0, 0)),
            "atom_pairs": [[(0, 1), (1, 2)]],
        }

        refined, diagnostics = refine_atom_pair_group_output(
            model, group_output, groups, batch_size=1
        )

        torch.testing.assert_close(
            refined["presence_logits"], group_output["presence_logits"]
        )
        self.assertIs(refined["chemical_logits"], group_output["chemical_logits"])
        self.assertEqual(diagnostics["groups"], 2)

    def test_scale_evaluation_can_recover_a_false_negative(self) -> None:
        row = {
            "base_logits": torch.tensor((1.0, -0.2)),
            "residual_logits": torch.tensor((0.0, 0.4)),
            "targets": torch.tensor((True, True)),
            "target_count": 2,
            "matched_count": 2,
            "positive_count": 2,
            "type_coverage": True,
            "chemical_exact": True,
        }
        records = {"a": [row], "b": [row]}

        low = evaluate_scale(records, 0.0)
        high = evaluate_scale(records, 1.0)

        self.assertEqual(low["pooled_deployed_exact"], 0)
        self.assertEqual(high["pooled_deployed_exact"], 2)
        self.assertGreater(selection_key(high), selection_key(low))

    def test_base_positive_protection_keeps_high_confidence_edge(self) -> None:
        row = {
            "base_logits": torch.tensor((1.0, -1.0)),
            "residual_logits": torch.tensor((-2.0, 0.0)),
            "targets": torch.tensor((True, False)),
            "target_count": 1,
            "matched_count": 1,
            "positive_count": 1,
            "type_coverage": True,
            "chemical_exact": True,
        }
        records = {"a": [row], "b": [row]}

        unprotected = evaluate_scale(records, 1.0)
        protected = evaluate_scale(records, 1.0, 0.5)

        self.assertEqual(unprotected["pooled_deployed_exact"], 0)
        self.assertEqual(protected["pooled_deployed_exact"], 2)


if __name__ == "__main__":
    unittest.main()
