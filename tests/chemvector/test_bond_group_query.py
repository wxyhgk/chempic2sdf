"""Tests for proposal-conditioned Bond group queries."""

from __future__ import annotations

import unittest

import torch

from chemvector.bond_group_query import (
    BondGroupQueryConfig,
    BondGroupQueryDetector,
    bond_group_query_assignments,
    bond_group_query_geometry_loss,
    bond_group_query_loss,
    decode_bond_group_queries,
    extract_dense_bond_proposals,
    sample_proposal_endpoint_patches,
    sample_proposal_ink_line_views,
    sample_proposal_line_bands,
    set_endpoint_local_only_trainable,
)
from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.dense_bond_object import decode_dense_bond_objects


class BondGroupQueryTests(unittest.TestCase):
    @staticmethod
    def _config() -> BondGroupQueryConfig:
        return BondGroupQueryConfig(
            input_channels=8,
            proposal_channels=16,
            d_model=32,
            queries=4,
            layers=1,
            heads=4,
            ffn_dim=64,
            memory_stride=2,
        )

    @classmethod
    def _local_config(cls) -> BondGroupQueryConfig:
        config = cls._config()
        return BondGroupQueryConfig(
            input_channels=config.input_channels,
            proposal_channels=config.proposal_channels,
            d_model=config.d_model,
            queries=config.queries,
            layers=config.layers,
            heads=config.heads,
            ffn_dim=config.ffn_dim,
            memory_stride=config.memory_stride,
            endpoint_delta_scale=config.endpoint_delta_scale,
            endpoint_local_channels=4,
            endpoint_patch_radius_cells=1,
        )

    @classmethod
    def _line_config(cls) -> BondGroupQueryConfig:
        config = cls._local_config()
        return BondGroupQueryConfig(
            input_channels=config.input_channels,
            proposal_channels=config.proposal_channels,
            d_model=config.d_model,
            queries=config.queries,
            layers=config.layers,
            heads=config.heads,
            ffn_dim=config.ffn_dim,
            memory_stride=config.memory_stride,
            endpoint_delta_scale=config.endpoint_delta_scale,
            endpoint_local_channels=config.endpoint_local_channels,
            endpoint_patch_radius_cells=config.endpoint_patch_radius_cells,
            endpoint_line_samples=4,
            endpoint_line_radius_cells=1,
            endpoint_line_feature_dim=8,
        )

    @classmethod
    def _ink_config(cls) -> BondGroupQueryConfig:
        config = cls._local_config()
        return BondGroupQueryConfig(
            input_channels=config.input_channels,
            proposal_channels=config.proposal_channels,
            d_model=config.d_model,
            queries=config.queries,
            layers=config.layers,
            heads=config.heads,
            ffn_dim=config.ffn_dim,
            memory_stride=config.memory_stride,
            endpoint_delta_scale=config.endpoint_delta_scale,
            endpoint_local_channels=config.endpoint_local_channels,
            endpoint_patch_radius_cells=config.endpoint_patch_radius_cells,
            endpoint_ink_line_samples=8,
            endpoint_ink_line_band=5,
            endpoint_ink_line_radius_pixels=2.0,
            endpoint_ink_line_feature_dim=8,
        )

    @staticmethod
    def _dense_output() -> dict[str, torch.Tensor]:
        batch, slots, channels, height, width = 1, 2, 16, 5, 5
        heatmap = torch.full((batch, slots, height, width), -9.0)
        for slot, y, x, logit in (
            (0, 0, 0, 5.0),
            (1, 4, 0, 4.0),
            (0, 4, 4, 3.0),
            (1, 2, 2, 2.0),
            (0, 0, 4, 1.0),
        ):
            heatmap[0, slot, y, x] = logit
        endpoints = torch.zeros(batch, slots, 2, 2, height, width)
        display = torch.zeros(batch, slots, len(BOND_TYPES), height, width)
        chemical = torch.zeros_like(display)
        direction = torch.zeros(batch, slots, len(DIRECTIONS), height, width)
        for slot in range(slots):
            for y in range(height):
                for x in range(width):
                    center_x = (x + 0.5) / width
                    center_y = (y + 0.5) / height
                    endpoints[0, slot, :, :, y, x] = torch.tensor(
                        (
                            (max(0.0, center_x - 0.04), center_y),
                            (min(1.0, center_x + 0.04), center_y),
                        )
                    )
                    display[0, slot, (slot + y + x) % len(BOND_TYPES), y, x] = 4.0
                    chemical[0, slot, (slot + y + x + 1) % len(BOND_TYPES), y, x] = 4.0
                    direction[0, slot, (slot + y + x) % len(DIRECTIONS), y, x] = 4.0
        features = torch.arange(
            batch * channels * height * width, dtype=torch.float32
        ).reshape(batch, channels, height, width)
        return {
            "heatmap_logits": heatmap,
            "object_features": features,
            "endpoints": endpoints,
            "display_logits": display,
            "chemical_logits": chemical,
            "direction_logits": direction,
        }

    @staticmethod
    def _proposals() -> dict[str, torch.Tensor]:
        return {
            "base_logits": torch.tensor([[2.0, 1.0, -1.0, -2.0]]),
            "base_endpoints": torch.tensor(
                [
                    [
                        [[0.20, 0.20], [0.80, 0.20]],
                        [[0.50, 0.10], [0.50, 0.90]],
                        [[0.10, 0.70], [0.40, 0.70]],
                        [[0.60, 0.60], [0.90, 0.90]],
                    ]
                ]
            ),
            "base_display_logits": torch.randn(1, 4, len(BOND_TYPES)),
            "base_chemical_logits": torch.randn(1, 4, len(BOND_TYPES)),
            "base_direction_logits": torch.randn(1, 4, len(DIRECTIONS)),
            "base_features": torch.randn(1, 4, 16),
            "mask": torch.ones(1, 4, dtype=torch.bool),
            "indices": torch.tensor([[[0, 1, 1], [1, 2, 2], [0, 3, 3], [1, 4, 4]]]),
        }

    @staticmethod
    def _targets() -> dict[str, torch.Tensor]:
        return {
            "endpoints": torch.tensor(
                [
                    [
                        [[0.20, 0.20], [0.80, 0.20]],
                        [[0.50, 0.10], [0.50, 0.90]],
                    ]
                ]
            ),
            "bbox": torch.zeros(1, 2, 4),
            "display": torch.tensor([[0, 2]]),
            "chemical": torch.tensor([[1, 3]]),
            "direction": torch.tensor([[0, 1]]),
            "mask": torch.tensor([[True, True]]),
        }

    def test_dense_proposals_match_existing_peak_ranking(self) -> None:
        dense = self._dense_output()
        proposals = extract_dense_bond_proposals(dense, maximum_queries=4)
        decoded = decode_dense_bond_objects(
            dense, threshold=0.0, maximum_objects=4
        )[0]

        self.assertEqual(proposals["indices"].shape, (1, 4, 3))
        for index, item in enumerate(decoded):
            self.assertAlmostEqual(
                float(proposals["base_logits"][0, index].sigmoid()),
                float(item["score"]),
                places=6,
            )
            self.assertTrue(
                torch.equal(proposals["base_endpoints"][0, index], item["endpoints"])
            )
            self.assertTrue(
                torch.equal(proposals["base_features"][0, index], item["feature"])
            )
            self.assertEqual(
                int(proposals["base_display_logits"][0, index].argmax()),
                item["display"],
            )
            self.assertEqual(
                int(proposals["base_chemical_logits"][0, index].argmax()),
                item["chemical"],
            )
            self.assertEqual(
                int(proposals["base_direction_logits"][0, index].argmax()),
                item["direction"],
            )

    def test_zero_initialized_residuals_preserve_dense_outputs_and_decode(self) -> None:
        dense = self._dense_output()
        proposals = extract_dense_bond_proposals(dense, maximum_queries=4)
        detector = BondGroupQueryDetector(self._line_config()).eval()
        output = detector(torch.randn(1, 8, 16, 16), proposals)

        self.assertTrue(torch.equal(output["presence_logits"], proposals["base_logits"]))
        self.assertTrue(torch.equal(output["endpoints"], proposals["base_endpoints"]))
        self.assertTrue(
            torch.equal(output["display_logits"], proposals["base_display_logits"])
        )
        self.assertTrue(
            torch.equal(output["chemical_logits"], proposals["base_chemical_logits"])
        )
        self.assertTrue(
            torch.equal(output["direction_logits"], proposals["base_direction_logits"])
        )
        self.assertFalse(
            any("direction" in name for name, _ in detector.named_parameters())
        )

        expected = decode_dense_bond_objects(
            dense, threshold=0.25, maximum_objects=4
        )[0]
        actual = decode_bond_group_queries(
            output, mode="threshold", threshold=0.25
        )[0]
        self.assertEqual(len(actual), len(expected))
        for first, second in zip(actual, expected):
            self.assertAlmostEqual(first["score"], second["score"], places=6)
            self.assertTrue(torch.equal(first["endpoints"], second["endpoints"]))
            self.assertEqual(first["display"], second["display"])
            self.assertEqual(first["chemical"], second["chemical"])
            self.assertEqual(first["direction"], second["direction"])

    def test_group_queries_are_equivariant_to_proposal_permutation(self) -> None:
        torch.manual_seed(11)
        detector = BondGroupQueryDetector(self._line_config()).eval()
        with torch.no_grad():
            for head in (
                detector.presence_residual,
                detector.endpoint_residual,
                detector.display_residual,
                detector.chemical_residual,
            ):
                head.weight.normal_(std=0.03)
                head.bias.normal_(std=0.03)
            detector.endpoint_local_residual[-1].weight.normal_(std=0.03)
            detector.endpoint_local_residual[-1].bias.normal_(std=0.03)
        features = torch.randn(1, 8, 16, 16)
        proposals = self._proposals()
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = {
            name: (
                value[:, permutation]
                if isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.size(1) == 4
                else value
            )
            for name, value in proposals.items()
        }

        original = detector(features, proposals)
        shuffled = detector(features, permuted)

        for name in (
            "presence_logits",
            "endpoints",
            "display_logits",
            "chemical_logits",
            "direction_logits",
            "object_features",
            "endpoint_features",
            "endpoint_local_features",
            "endpoint_line_features",
        ):
            self.assertTrue(
                torch.allclose(
                    shuffled[name], original[name][:, permutation], atol=1e-5, rtol=1e-5
                ),
                name,
            )

    def test_endpoint_patch_sampling_matches_feature_cell_centers(self) -> None:
        height, width = 5, 6
        feature_map = torch.tensor(
            [[[[10.0 * y + x for x in range(width)] for y in range(height)]]]
        )
        center = torch.tensor([(3.5 / width), (2.5 / height)])
        endpoints = center.reshape(1, 1, 1, 2).repeat(1, 1, 2, 1)
        patches = sample_proposal_endpoint_patches(
            feature_map,
            endpoints,
            torch.ones(1, 1, dtype=torch.bool),
            radius_cells=1,
        )

        expected = torch.tensor(
            [[12.0, 13.0, 14.0], [22.0, 23.0, 24.0], [32.0, 33.0, 34.0]]
        )
        self.assertEqual(patches.shape, (1, 1, 2, 1, 3, 3))
        self.assertTrue(torch.allclose(patches[0, 0, 0, 0], expected, atol=1e-5))
        self.assertTrue(torch.allclose(patches[0, 0, 1, 0], expected, atol=1e-5))

    def test_endpoint_patch_sampling_masks_invalid_queries_at_borders(self) -> None:
        patches = sample_proposal_endpoint_patches(
            torch.randn(1, 3, 5, 5),
            torch.tensor(
                [[[[0.0, 0.0], [1.0, 1.0]], [[0.2, 0.3], [0.7, 0.8]]]]
            ),
            torch.tensor([[True, False]]),
            radius_cells=2,
        )

        self.assertEqual(patches.shape, (1, 2, 2, 3, 5, 5))
        self.assertTrue(bool(torch.isfinite(patches).all()))
        self.assertEqual(int(torch.count_nonzero(patches[:, 1])), 0)

    def test_line_band_sampling_follows_segment_and_perpendicular_cells(self) -> None:
        height, width = 5, 6
        feature_map = torch.tensor(
            [[[[10.0 * y + x for x in range(width)] for y in range(height)]]]
        )
        endpoints = torch.tensor(
            [[[[1.5 / width, 2.5 / height], [4.5 / width, 2.5 / height]]]]
        )
        bands = sample_proposal_line_bands(
            feature_map,
            endpoints,
            torch.ones(1, 1, dtype=torch.bool),
            line_samples=4,
            radius_cells=1,
        )

        expected = torch.tensor(
            [[11.0, 12.0, 13.0, 14.0],
             [21.0, 22.0, 23.0, 24.0],
             [31.0, 32.0, 33.0, 34.0]]
        )
        self.assertEqual(bands.shape, (1, 1, 1, 3, 4))
        self.assertTrue(torch.allclose(bands[0, 0, 0], expected, atol=1e-5))

    def test_line_band_sampling_masks_invalid_queries(self) -> None:
        bands = sample_proposal_line_bands(
            torch.randn(1, 2, 7, 7),
            torch.tensor(
                [[[[0.1, 0.2], [0.9, 0.8]], [[0.2, 0.3], [0.7, 0.6]]]]
            ),
            torch.tensor([[True, False]]),
            line_samples=8,
            radius_cells=2,
        )

        self.assertEqual(bands.shape, (1, 2, 2, 5, 8))
        self.assertTrue(bool(torch.isfinite(bands).all()))
        self.assertEqual(int(torch.count_nonzero(bands[:, 1])), 0)

    def test_raw_ink_line_views_are_endpoint_oriented_and_masked(self) -> None:
        endpoints = torch.tensor(
            [
                [
                    [[0.2, 0.3], [0.8, 0.7]],
                    [[0.3, 0.2], [0.7, 0.8]],
                ]
            ]
        )
        views = sample_proposal_ink_line_views(
            torch.zeros(1, 3, 32, 32),
            endpoints,
            torch.tensor([[True, False]]),
            line_samples=8,
            band_samples=5,
            radius_pixels=2.0,
        )

        self.assertEqual(views.shape, (1, 2, 2, 1, 5, 8))
        self.assertTrue(bool(torch.isfinite(views).all()))
        self.assertTrue(torch.allclose(views[0, 0, 0], torch.full((1, 5, 8), 0.5)))
        self.assertTrue(torch.equal(views[0, 0, 1], views[0, 0, 0].flip((-2, -1))))
        self.assertEqual(int(torch.count_nonzero(views[:, 1])), 0)

    def test_endpoint_local_branch_receives_finite_gradients(self) -> None:
        torch.manual_seed(13)
        detector = BondGroupQueryDetector(self._local_config())
        with torch.no_grad():
            detector.endpoint_local_residual[-1].weight.normal_(std=0.03)
        output = detector(torch.randn(1, 8, 16, 16), self._proposals())
        output["endpoints"].sum().backward()

        projection_gradient = detector.endpoint_local_projection[0].weight.grad
        fusion_gradient = detector.endpoint_local_residual[1].weight.grad
        self.assertIsNotNone(projection_gradient)
        self.assertIsNotNone(fusion_gradient)
        self.assertTrue(bool(torch.isfinite(projection_gradient).all()))
        self.assertTrue(bool(torch.isfinite(fusion_gradient).all()))
        self.assertGreater(float(projection_gradient.abs().sum()), 0.0)
        self.assertGreater(float(fusion_gradient.abs().sum()), 0.0)

    def test_endpoint_local_only_freezes_global_group_query_path(self) -> None:
        detector = BondGroupQueryDetector(self._line_config())
        set_endpoint_local_only_trainable(detector)

        trainable = {
            name for name, parameter in detector.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("endpoint_local_projection.")
                or name.startswith("endpoint_local_residual.")
                or name.startswith("endpoint_line_encoder.")
                for name in trainable
            )
        )
        self.assertFalse(detector.endpoint_residual.weight.requires_grad)

    def test_endpoint_line_encoder_receives_finite_gradients(self) -> None:
        torch.manual_seed(19)
        detector = BondGroupQueryDetector(self._line_config())
        with torch.no_grad():
            detector.endpoint_local_residual[-1].weight.normal_(std=0.03)
        output = detector(torch.randn(1, 8, 16, 16), self._proposals())
        output["endpoints"].sum().backward()

        gradient = detector.endpoint_line_encoder[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_zero_initialized_ink_line_preserves_dense_endpoints(self) -> None:
        detector = BondGroupQueryDetector(self._ink_config()).eval()
        proposals = self._proposals()
        output = detector(
            torch.randn(1, 8, 16, 16),
            proposals,
            pixel_values=torch.randn(1, 3, 64, 64),
        )

        self.assertTrue(torch.equal(output["endpoints"], proposals["base_endpoints"]))

    def test_raw_ink_endpoint_residual_is_endpoint_swap_equivariant(self) -> None:
        torch.manual_seed(29)
        detector = BondGroupQueryDetector(self._ink_config()).eval()
        with torch.no_grad():
            detector.endpoint_ink_line_residual[-1].weight.normal_(std=0.03)
            detector.endpoint_ink_line_residual[-1].bias.normal_(std=0.03)
        features = torch.randn(1, 8, 16, 16)
        pixels = torch.randn(1, 3, 64, 64)
        proposals = self._proposals()
        swapped = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in proposals.items()
        }
        swapped["base_endpoints"] = proposals["base_endpoints"].flip(2)

        original_output = detector(features, proposals, pixel_values=pixels)
        swapped_output = detector(features, swapped, pixel_values=pixels)

        self.assertTrue(
            torch.allclose(
                swapped_output["endpoints"],
                original_output["endpoints"].flip(2),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                swapped_output["endpoint_ink_line_features"],
                original_output["endpoint_ink_line_features"].flip(2),
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_raw_ink_line_encoder_receives_finite_gradients(self) -> None:
        torch.manual_seed(31)
        detector = BondGroupQueryDetector(self._ink_config())
        with torch.no_grad():
            detector.endpoint_ink_line_residual[-1].weight.normal_(std=0.03)
        output = detector(
            torch.randn(1, 8, 16, 16),
            self._proposals(),
            pixel_values=torch.randn(1, 3, 64, 64),
        )
        output["endpoints"].sum().backward()

        gradient = detector.endpoint_ink_line_encoder[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_hungarian_assignment_does_not_use_presence(self) -> None:
        output = {
            "presence_logits": torch.tensor([[-100.0, 100.0, 0.0]]),
            "endpoints": torch.tensor(
                [
                    [
                        [[0.20, 0.20], [0.80, 0.20]],
                        [[0.50, 0.10], [0.50, 0.90]],
                        [[0.10, 0.10], [0.15, 0.15]],
                    ]
                ]
            ),
            "proposal_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        targets = self._targets()
        first = bond_group_query_assignments(output, targets)
        output["presence_logits"] = -output["presence_logits"]
        second = bond_group_query_assignments(output, targets)

        self.assertTrue(torch.equal(first[0][0], second[0][0]))
        self.assertTrue(torch.equal(first[0][1], second[0][1]))
        self.assertEqual(first[0][0].tolist(), [0, 1])

    def test_loss_is_invariant_to_gt_order_and_endpoint_flip(self) -> None:
        torch.manual_seed(17)
        proposals = self._proposals()
        output = {
            "presence_logits": proposals["base_logits"].clone().requires_grad_(),
            "endpoints": proposals["base_endpoints"].clone().requires_grad_(),
            "display_logits": proposals["base_display_logits"].clone().requires_grad_(),
            "chemical_logits": proposals["base_chemical_logits"].clone().requires_grad_(),
            "direction_logits": proposals["base_direction_logits"],
            "proposal_mask": proposals["mask"],
        }
        original = self._targets()
        equivalent = {
            "endpoints": torch.stack(
                (
                    original["endpoints"][:, 1].flip(1),
                    original["endpoints"][:, 0],
                ),
                dim=1,
            ),
            "bbox": original["bbox"][:, [1, 0]],
            "display": original["display"][:, [1, 0]],
            "chemical": original["chemical"][:, [1, 0]],
            "direction": original["direction"][:, [1, 0]],
            "mask": original["mask"][:, [1, 0]],
        }

        first = bond_group_query_loss(output, original)
        second = bond_group_query_loss(output, equivalent)

        self.assertNotIn("direction_loss", first)
        for name in first:
            self.assertTrue(
                torch.allclose(first[name], second[name], atol=1e-6, rtol=0.0),
                name,
            )

    def test_same_center_targets_use_distinct_global_queries_and_backpropagate(self) -> None:
        torch.manual_seed(23)
        detector = BondGroupQueryDetector(self._config())
        proposals = self._proposals()
        proposals["base_endpoints"] = proposals["base_endpoints"].clone()
        proposals["base_endpoints"][0, 0] = torch.tensor(
            [[0.40, 0.50], [0.60, 0.50]]
        )
        proposals["base_endpoints"][0, 1] = torch.tensor(
            [[0.50, 0.40], [0.50, 0.60]]
        )
        targets = self._targets()
        targets["endpoints"] = proposals["base_endpoints"][:, :2].clone()
        output = detector(torch.randn(1, 8, 16, 16), proposals)

        queries, _ = bond_group_query_assignments(output, targets)[0]
        self.assertEqual(len(set(queries.tolist())), 2)
        losses = bond_group_query_loss(output, targets)
        losses["loss"].backward()

        gradients = [
            parameter.grad
            for parameter in detector.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertIsNotNone(detector.endpoint_residual.weight.grad)
        self.assertNotIn("direction_loss", losses)

    def test_assignment_rejects_more_targets_than_queries(self) -> None:
        output = {
            "endpoints": torch.zeros(1, 1, 2, 2),
            "proposal_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        targets = self._targets()
        with self.assertRaisesRegex(ValueError, "exceed"):
            bond_group_query_assignments(output, targets)

    def test_initial_geometry_can_anchor_assignment_after_predictions_move(self) -> None:
        targets = self._targets()
        proposals = self._proposals()
        output = {
            "endpoints": proposals["base_endpoints"].clone(),
            "proposal_mask": proposals["mask"],
        }
        output["endpoints"][0, 0] = proposals["base_endpoints"][0, 1]
        output["endpoints"][0, 1] = proposals["base_endpoints"][0, 0]

        dynamic = bond_group_query_assignments(output, targets)[0]
        anchored = bond_group_query_assignments(
            output,
            targets,
            matching_endpoints=proposals["base_endpoints"],
        )[0]

        self.assertEqual(dynamic[0].tolist(), [0, 1])
        self.assertEqual(dynamic[1].tolist(), [1, 0])
        self.assertEqual(anchored[0].tolist(), [0, 1])
        self.assertEqual(anchored[1].tolist(), [0, 1])

    def test_geometry_only_loss_excludes_count_presence_and_types(self) -> None:
        proposals = self._proposals()
        output = {
            "endpoints": proposals["base_endpoints"].clone().requires_grad_(),
            "proposal_mask": proposals["mask"],
        }
        losses = bond_group_query_geometry_loss(
            output,
            self._targets(),
            matching_endpoints=proposals["base_endpoints"],
            target_weights=torch.tensor([[16.0, 16.0]]),
        )

        self.assertEqual(
            set(losses), {"loss", "endpoint_loss", "endpoint_barrier_loss"}
        )
        losses["loss"].backward()
        self.assertTrue(bool(torch.isfinite(output["endpoints"].grad).all()))


if __name__ == "__main__":
    unittest.main()
