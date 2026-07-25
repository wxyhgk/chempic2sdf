"""Tests for post-snap grounded query selection."""

from __future__ import annotations

import unittest

import torch

from chemvector.grounded_selector import (
    GroundedQuerySelector,
    GroundedSelectorConfig,
    decode_grounded_queries,
    dense_candidate_targets,
    fuse_dense_candidates_with_queries,
    ground_all_queries,
    grounded_selector_loss,
    grounded_selector_targets,
    sample_candidate_features,
    sample_candidate_ink_patches,
)


class GroundedSelectorTests(unittest.TestCase):
    def test_candidate_ink_patch_preserves_local_two_dimensional_layout(self) -> None:
        pixels = torch.ones(1, 3, 16, 16)
        pixels[:, :, 8, 5:11] = -1.0
        patches = sample_candidate_ink_patches(
            pixels,
            torch.tensor([[[0.5, 0.5]]]),
            torch.ones(1, 1, dtype=torch.bool),
            patch_size=8,
        )
        self.assertEqual(patches.shape, (1, 1, 1, 8, 8))
        self.assertGreater(float(patches.max()), 0.9)
        self.assertLess(float(patches.min()), 0.1)

    def test_candidate_features_are_sampled_at_candidate_centers(self) -> None:
        feature_map = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
        centers = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        mask = torch.tensor([[True, False]])

        sampled = sample_candidate_features(feature_map, centers, mask)

        self.assertTrue(torch.allclose(sampled[0, 0], torch.arange(1.0, 10.0)))
        self.assertEqual(sampled.shape, (1, 2, 9))
        self.assertEqual(float(sampled[0, 1].sum()), 0.0)

    def test_zero_initialized_residual_preserves_base_logits(self) -> None:
        config = GroundedSelectorConfig(
            query_dim=16,
            scalar_dim=10,
            d_model=16,
            heads=4,
            ffn_dim=32,
        )
        selector = GroundedQuerySelector(config).eval()
        base = torch.randn(2, 5)

        actual = selector(
            torch.randn(2, 5, 16),
            torch.randn(2, 5, 10),
            torch.rand(2, 5, 2),
            base,
        )

        self.assertTrue(torch.equal(actual, base))

    def test_all_queries_snap_to_distinct_dense_candidates(self) -> None:
        centers = torch.tensor([[[0.10, 0.10], [0.12, 0.10], [0.80, 0.80]]])
        output = {
            "centers": centers,
            "selection_features": torch.randn(1, 3, 8),
            "presence_logits": torch.zeros(1, 3),
            "ranking_presence_logits": torch.zeros(1, 3),
            "local_presence_logits": torch.zeros(1, 3),
            "selection_context_logits": torch.zeros(1, 3),
            "selection_logits": torch.zeros(1, 3),
            "aromatic_logits": torch.zeros(1, 3),
        }
        dense = [[
            {"x": 0.11, "y": 0.10, "score": 0.9, "aromatic": False},
            {"x": 0.81, "y": 0.80, "score": 0.8, "aromatic": False},
        ]]

        grounded = ground_all_queries(
            output,
            dense,
            image_size=100,
            maximum_distance_pixels=20.0,
        )

        snapped_flags = grounded["scalar_features"][0, :, 6]
        self.assertEqual(int(snapped_flags.sum()), 2)
        snapped_centers = grounded["grounded_centers"][0][snapped_flags.bool()]
        self.assertEqual(len(torch.unique(snapped_centers, dim=0)), 2)

    def test_targets_assign_one_unique_query_per_anchor(self) -> None:
        centers = torch.tensor(
            [[[0.10, 0.10], [0.11, 0.10], [0.80, 0.80], [0.50, 0.50]]]
        )
        anchors = torch.tensor([[[0.10, 0.10], [0.80, 0.80]]])
        mask = torch.ones((1, 2), dtype=torch.bool)

        targets = grounded_selector_targets(centers, anchors, mask)

        self.assertEqual(targets.tolist(), [[1.0, 0.0, 1.0, 0.0]])

    def test_dense_candidates_keep_coordinates_and_receive_query_context(self) -> None:
        output = {
            "centers": torch.tensor([[[0.10, 0.10], [0.80, 0.80]]]),
            "selection_features": torch.randn(1, 2, 8),
            "presence_logits": torch.zeros(1, 2),
            "ranking_presence_logits": torch.zeros(1, 2),
            "local_presence_logits": torch.zeros(1, 2),
            "selection_context_logits": torch.zeros(1, 2),
            "selection_logits": torch.zeros(1, 2),
        }
        dense = [[
            {"x": 0.12, "y": 0.11, "score": 0.9, "aromatic": False},
            {"x": 0.78, "y": 0.81, "score": 0.8, "aromatic": True},
        ]]

        fused = fuse_dense_candidates_with_queries(
            output,
            dense,
            maximum_candidates=4,
            image_size=100,
        )
        targets = dense_candidate_targets(
            fused["grounded_centers"],
            fused["candidate_mask"],
            torch.tensor([[[0.12, 0.11], [0.78, 0.81]]]),
            torch.ones((1, 2), dtype=torch.bool),
        )

        self.assertTrue(
            torch.allclose(
                fused["grounded_centers"][0, :2],
                torch.tensor([[0.12, 0.11], [0.78, 0.81]]),
            )
        )
        self.assertEqual(fused["candidate_mask"].tolist(), [[True, True, False, False]])
        self.assertEqual(targets.tolist(), [[1.0, 1.0, 0.0, 0.0]])

    def test_loss_is_finite_and_updates_logits(self) -> None:
        logits = torch.randn(2, 8, requires_grad=True)
        targets = torch.zeros(2, 8)
        targets[0, :2] = 1.0
        targets[1, :3] = 1.0

        losses = grounded_selector_loss(logits, targets)
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_decode_uses_set_count_or_explicit_count(self) -> None:
        logits = torch.tensor([[8.0, 7.0, -8.0, -9.0]])
        centers = torch.rand(1, 4, 2)
        aromatic = torch.zeros(1, 4)

        inferred = decode_grounded_queries(logits, centers, aromatic)
        explicit = decode_grounded_queries(
            logits,
            centers,
            aromatic,
            counts=torch.tensor([1]),
        )

        self.assertEqual(len(inferred[0]), 2)
        self.assertEqual(len(explicit[0]), 1)


if __name__ == "__main__":
    unittest.main()
