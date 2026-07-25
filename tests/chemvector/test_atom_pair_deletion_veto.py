from __future__ import annotations

import unittest

import torch

from chemvector.atom_pair_bond_group import GROUP_FEATURE_DIM
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    AtomPairDeletionVetoHead,
    apply_atom_pair_deletion_veto,
    atom_pair_deletion_veto_loss,
    build_atom_pair_relation_features,
)


class AtomPairDeletionVetoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AtomPairDeletionVetoConfig(
            hidden_dim=16,
            maximum_atoms=8,
            image_size=64.0,
            line_samples=9,
        )

    def _relation_features(
        self,
        anchors: torch.Tensor,
        pairs: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        atom_mask = torch.zeros((1, 8), dtype=torch.bool)
        atom_mask[0, : len(anchors)] = True
        padded = torch.zeros((1, 8, 2))
        padded[0, : len(anchors)] = anchors
        atom_logits = torch.full((1, 8), -10.0)
        atom_logits[0, : len(anchors)] = 10.0
        return build_atom_pair_relation_features(
            image[None],
            padded,
            atom_mask,
            atom_logits,
            torch.zeros(len(pairs), dtype=torch.long),
            pairs,
            torch.ones(len(pairs)),
            batch_size=1,
            config=self.config,
        )

    def test_intermediate_vertex_marks_spanning_edge(self) -> None:
        anchors = torch.tensor(((0.15, 0.5), (0.5, 0.5), (0.85, 0.5)))
        pairs = torch.tensor(((0, 1), (1, 2), (0, 2)))
        image = torch.ones((3, 64, 64))
        image[:, 31:34, 8:56] = -1.0

        features = self._relation_features(anchors, pairs, image)

        self.assertLess(float(features[0, 11]), 0.05)
        self.assertGreater(float(features[2, 11]), 0.9)
        self.assertGreater(float(features[2, 15]), 0.7)

    def test_direct_ink_distinguishes_triangle_edge_from_empty_chord(self) -> None:
        anchors = torch.tensor(((0.5, 0.15), (0.2, 0.8), (0.8, 0.8)))
        pair = torch.tensor(((1, 2),))
        empty = torch.ones((3, 64, 64))
        triangle = empty.clone()
        triangle[:, 50:53, 12:52] = -1.0

        empty_features = self._relation_features(anchors, pair, empty)
        triangle_features = self._relation_features(anchors, pair, triangle)

        self.assertLess(float(empty_features[0, 3]), 0.05)
        self.assertGreater(float(triangle_features[0, 3]), 0.7)

    def test_zero_initialized_head_keeps_all_edges(self) -> None:
        model = AtomPairDeletionVetoHead(self.config)
        pairs = torch.tensor(((0, 1), (1, 2), (0, 2)))
        output = model(
            torch.randn(3, GROUP_FEATURE_DIM),
            torch.randn(3, self.config.relation_dim),
            torch.tensor((1.0, 0.5, -0.5)),
            torch.zeros(3, dtype=torch.long),
            pairs,
            batch_size=1,
        )

        self.assertTrue(bool((output["deletion_logits"] < 0.0).all()))

    def test_relation_only_head_ignores_group_features(self) -> None:
        config = AtomPairDeletionVetoConfig(
            hidden_dim=16,
            maximum_atoms=8,
            relation_only=True,
        )
        model = AtomPairDeletionVetoHead(config)
        torch.nn.init.normal_(model.veto[-1].weight)
        relation = torch.randn(2, config.relation_dim)
        base = torch.tensor((1.0, 0.5))
        common = dict(
            relation_features=relation,
            base_logits=base,
            sample_indices=torch.zeros(2, dtype=torch.long),
            atom_pairs=torch.tensor(((0, 1), (1, 2))),
            batch_size=1,
        )
        first = model(torch.randn(2, GROUP_FEATURE_DIM), **common)["deletion_logits"]
        second = model(torch.randn(2, GROUP_FEATURE_DIM), **common)["deletion_logits"]

        torch.testing.assert_close(first, second)

    def test_head_is_equivariant_to_group_permutation(self) -> None:
        torch.manual_seed(7)
        model = AtomPairDeletionVetoHead(self.config)
        torch.nn.init.normal_(model.veto[-1].weight)
        pairs = torch.tensor(((0, 1), (1, 2), (0, 2), (2, 3)))
        group_features = torch.randn(4, GROUP_FEATURE_DIM)
        relation_features = torch.randn(4, self.config.relation_dim)
        base_logits = torch.randn(4)
        order = torch.tensor((2, 0, 3, 1))
        direct = model(
            group_features,
            relation_features,
            base_logits,
            torch.zeros(4, dtype=torch.long),
            pairs,
            batch_size=1,
        )["deletion_logits"]
        permuted = model(
            group_features[order],
            relation_features[order],
            base_logits[order],
            torch.zeros(4, dtype=torch.long),
            pairs[order],
            batch_size=1,
        )["deletion_logits"]

        torch.testing.assert_close(permuted, direct[order])

    def test_veto_only_changes_base_positive_edges(self) -> None:
        refined, deleted = apply_atom_pair_deletion_veto(
            torch.tensor((1.0, -0.5, 0.25)),
            torch.tensor((2.0, 3.0, -1.0)),
            threshold=0.0,
        )

        self.assertEqual(deleted.tolist(), [True, False, False])
        self.assertLess(float(refined[0]), 0.0)
        self.assertEqual(float(refined[1]), -0.5)

    def test_loss_supervises_only_base_positive_edges(self) -> None:
        logits = torch.tensor((1.0, -1.0, 3.0), requires_grad=True)
        losses = atom_pair_deletion_veto_loss(
            logits,
            torch.tensor((False, True, False)),
            torch.tensor((1.0, 1.0, -1.0)),
            torch.zeros(3, dtype=torch.long),
        )

        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertEqual(float(logits.grad[2]), 0.0)

    def test_precision_first_loss_can_reduce_deletion_weight(self) -> None:
        logits = torch.tensor((0.0, 0.0), requires_grad=True)
        common = dict(
            edge_targets=torch.tensor((False, True)),
            base_logits=torch.tensor((1.0, 1.0)),
            sample_indices=torch.zeros(2, dtype=torch.long),
            ranking_weight=0.0,
        )
        balanced = atom_pair_deletion_veto_loss(logits, **common)
        conservative = atom_pair_deletion_veto_loss(
            logits,
            **common,
            deletion_positive_weight=0.5,
        )

        self.assertLess(
            float(conservative["binary_loss"].detach()),
            float(balanced["binary_loss"].detach()),
        )


if __name__ == "__main__":
    unittest.main()
