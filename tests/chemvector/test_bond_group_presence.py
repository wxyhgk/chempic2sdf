"""Tests for Bond Group presence selection and geometry labels."""

from __future__ import annotations

import unittest

import torch

from chemvector.bond_group_presence import (
    BondGroupPresenceConfig,
    BondGroupPresenceSelector,
    bond_group_presence_loss,
    build_geometry_equivalence_labels,
)
from chemvector.bond_object import BOND_TYPES, DIRECTIONS


class BondGroupPresenceTests(unittest.TestCase):
    @staticmethod
    def _config(*, set_layers: int = 0) -> BondGroupPresenceConfig:
        return BondGroupPresenceConfig(
            endpoint_local_dim=6,
            endpoint_ink_dim=4,
            object_dim=8,
            endpoint_embedding_dim=8,
            object_embedding_dim=8,
            d_model=16,
            set_layers=set_layers,
            heads=4,
            ffn_dim=32,
        )

    @staticmethod
    def _inputs(queries: int = 5) -> tuple[dict, dict]:
        torch.manual_seed(7)
        base_endpoints = torch.rand(1, queries, 2, 2) * 0.7 + 0.15
        refined = (base_endpoints + torch.randn_like(base_endpoints) * 0.01).clamp(0, 1)
        mask = torch.ones(1, queries, dtype=torch.bool)
        proposals = {
            "base_logits": torch.linspace(-1.0, 1.0, queries).reshape(1, queries),
            "base_endpoints": base_endpoints,
            "mask": mask,
        }
        group = {
            "endpoints": refined,
            "endpoint_local_features": torch.randn(1, queries, 2, 6),
            "endpoint_ink_line_features": torch.randn(1, queries, 2, 4),
            "object_features": torch.randn(1, queries, 8),
            "display_logits": torch.randn(1, queries, len(BOND_TYPES)),
            "chemical_logits": torch.randn(1, queries, len(BOND_TYPES)),
            "direction_logits": torch.randn(1, queries, len(DIRECTIONS)),
            "proposal_mask": mask,
        }
        return group, proposals

    @staticmethod
    def _permuted(values: dict, permutation: torch.Tensor) -> dict:
        count = len(permutation)
        return {
            name: (
                value[:, permutation]
                if isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.size(1) == count
                else value
            )
            for name, value in values.items()
        }

    @staticmethod
    def _label_case() -> tuple[torch.Tensor, torch.Tensor, dict]:
        target_endpoints = torch.tensor(
            [[
                [[0.10, 0.20], [0.30, 0.20]],
                [[0.10, 0.28], [0.30, 0.28]],
                [[0.60, 0.80], [0.80, 0.80]],
            ]]
        )
        refined = torch.tensor(
            [[
                [[0.10, 0.20], [0.30, 0.20]],
                [[0.10, 0.21], [0.30, 0.21]],
                [[0.10, 0.24], [0.30, 0.24]],
                [[0.10, 0.50], [0.30, 0.50]],
                [[0.10, 0.28], [0.30, 0.28]],
            ]]
        )
        candidate_mask = torch.ones(1, 5, dtype=torch.bool)
        targets = {
            "endpoints": target_endpoints,
            "mask": torch.ones(1, 3, dtype=torch.bool),
            "display": torch.tensor([[1, 2, 3]]),
            "chemical": torch.tensor([[0, 1, 2]]),
        }
        return refined, candidate_mask, targets

    def test_zero_initialized_presence_residual_preserves_base_logits(self) -> None:
        selector = BondGroupPresenceSelector(self._config()).eval()
        group, proposals = self._inputs()

        output = selector(group, proposals)

        self.assertTrue(torch.equal(output["keep_logits"], proposals["base_logits"]))
        self.assertEqual(int(torch.count_nonzero(output["presence_residual"])), 0)

    def test_selector_is_equivariant_to_proposal_permutation(self) -> None:
        group, proposals = self._inputs()
        permutation = torch.tensor([3, 0, 4, 1, 2])

        for set_layers in (0, 1):
            with self.subTest(set_layers=set_layers):
                selector = BondGroupPresenceSelector(
                    self._config(set_layers=set_layers)
                ).eval()
                with torch.no_grad():
                    selector.presence_residual.weight.normal_(std=0.1)
                    selector.presence_residual.bias.normal_(std=0.1)
                original = selector(group, proposals)
                shuffled = selector(
                    self._permuted(group, permutation),
                    self._permuted(proposals, permutation),
                )
                self.assertTrue(
                    torch.allclose(
                        shuffled["keep_logits"],
                        original["keep_logits"][:, permutation],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )

    def test_selector_is_invariant_to_endpoint_swap(self) -> None:
        selector = BondGroupPresenceSelector(self._config()).eval()
        with torch.no_grad():
            selector.presence_residual.weight.normal_(std=0.1)
            selector.presence_residual.bias.normal_(std=0.1)
        group, proposals = self._inputs()
        swapped_group = dict(group)
        swapped_proposals = dict(proposals)
        for name in (
            "endpoints",
            "endpoint_local_features",
            "endpoint_ink_line_features",
        ):
            swapped_group[name] = group[name].flip(2)
        swapped_proposals["base_endpoints"] = proposals["base_endpoints"].flip(2)

        original = selector(group, proposals)
        swapped = selector(swapped_group, swapped_proposals)

        self.assertTrue(
            torch.allclose(
                swapped["keep_logits"], original["keep_logits"], atol=1e-5, rtol=1e-5
            )
        )

    def test_geometry_labels_ignore_ambiguous_and_exclude_unrepresentable_targets(self) -> None:
        refined, candidate_mask, targets = self._label_case()

        labels = build_geometry_equivalence_labels(
            refined,
            candidate_mask,
            targets,
            image_size=100.0,
            tolerance_pixels=6.0,
        )

        self.assertEqual(labels["representable_counts"].tolist(), [2])
        self.assertEqual(labels["gt_counts"].tolist(), [3])
        self.assertEqual(labels["target_representable"].tolist(), [[True, True, False]])
        self.assertEqual(labels["target_group_ids"].tolist(), [[0, 1, -1]])
        self.assertEqual(labels["compatible_counts"].tolist(), [[1, 1, 2, 0, 1]])
        self.assertEqual(labels["ambiguous_mask"].tolist(), [[False, False, True, False, False]])
        self.assertEqual(labels["loss_mask"].tolist(), [[True, True, False, True, True]])
        self.assertEqual(labels["group_ids"].tolist(), [[0, 0, -1, 2, 1]])
        self.assertEqual(labels["group_targets"][0, :3].tolist(), [True, True, False])
        self.assertEqual(
            labels["canonical_targets"].tolist(),
            [[True, False, False, False, True]],
        )

    def test_same_target_nonwinner_is_not_an_ordinary_negative(self) -> None:
        refined, candidate_mask, targets = self._label_case()
        labels = build_geometry_equivalence_labels(
            refined, candidate_mask, targets, image_size=100.0
        )

        def negative_loss(nonwinner: float) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.tensor(
                [[0.5, nonwinner, 4.0, -0.4, 0.3]], requires_grad=True
            )
            output = {
                "keep_logits": logits,
                "display_logits": torch.zeros(1, 5, len(BOND_TYPES)),
                "chemical_logits": torch.zeros(1, 5, len(BOND_TYPES)),
            }
            losses = bond_group_presence_loss(
                output,
                labels,
                positive_weight=0.0,
                negative_weight=1.0,
                exactly_one_weight=0.0,
                representative_weight=0.0,
                cardinality_weight=0.0,
                margin_weight=0.0,
            )
            losses["loss"].backward()
            return losses["loss"].detach(), logits.grad.detach()

        low_loss, low_gradient = negative_loss(-8.0)
        high_loss, high_gradient = negative_loss(8.0)

        self.assertTrue(torch.allclose(low_loss, high_loss, atol=1e-7, rtol=0.0))
        self.assertEqual(float(low_gradient[0, 1]), 0.0)
        self.assertEqual(float(high_gradient[0, 1]), 0.0)
        self.assertEqual(float(low_gradient[0, 2]), 0.0)
        self.assertGreater(float(low_gradient[0, 3]), 0.0)

    def test_ambiguous_candidate_can_be_a_canonical_maximum_matching_winner(self) -> None:
        targets = {
            "endpoints": torch.tensor(
                [[
                    [[0.10, 0.20], [0.30, 0.20]],
                    [[0.10, 0.28], [0.30, 0.28]],
                ]]
            ),
            "mask": torch.ones(1, 2, dtype=torch.bool),
            "display": torch.zeros(1, 2, dtype=torch.long),
            "chemical": torch.zeros(1, 2, dtype=torch.long),
        }
        refined = torch.tensor(
            [[
                [[0.10, 0.24], [0.30, 0.24]],
                [[0.10, 0.20], [0.30, 0.20]],
            ]]
        )

        labels = build_geometry_equivalence_labels(
            refined,
            torch.ones(1, 2, dtype=torch.bool),
            targets,
            image_size=100.0,
            tolerance_pixels=6.0,
        )

        self.assertEqual(labels["compatible_counts"].tolist(), [[2, 1]])
        self.assertEqual(labels["representable_counts"].tolist(), [2])
        self.assertEqual(labels["canonical_targets"].sum().item(), 2)
        self.assertTrue(bool(labels["loss_mask"][0, 0]))

    def test_selector_and_equivalence_loss_have_finite_gradients(self) -> None:
        selector = BondGroupPresenceSelector(self._config())
        with torch.no_grad():
            selector.presence_residual.weight.normal_(std=0.05)
        group, proposals = self._inputs()
        refined, candidate_mask, targets = self._label_case()
        group["endpoints"] = refined
        proposals["base_endpoints"] = refined.clone()
        labels = build_geometry_equivalence_labels(
            refined, candidate_mask, targets, image_size=100.0
        )

        losses = bond_group_presence_loss(selector(group, proposals), labels)
        losses["loss"].backward()

        gradients = [
            parameter.grad
            for parameter in selector.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertGreater(
            float(selector.presence_residual.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(selector.endpoint_local_encoder[1].weight.grad.abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
