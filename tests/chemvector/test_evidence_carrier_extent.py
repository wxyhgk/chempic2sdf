from __future__ import annotations

import unittest

import torch

from chemvector.evidence_carrier_extent import (
    EvidenceCarrierExtentConfig,
    EvidenceCarrierExtentPredictor,
    canonicalize_carrier_segments,
    clip_carrier_lines_to_image,
    decode_evidence_carrier_extent,
    decode_ordered_extent_bins,
    evidence_carrier_extent_loss,
    extract_evidence_carrier_strips,
)


def _config(*, extent_bins: int = 9) -> EvidenceCarrierExtentConfig:
    return EvidenceCarrierExtentConfig(
        longitudinal_samples=33,
        band_samples=9,
        band_radius_pixels=2.0,
        normal_offset_radius_pixels=2.0,
        normal_offset_step_pixels=1.0,
        extent_bins=extent_bins,
        hidden_dim=16,
        dropout=0.0,
    )


def _horizontal_image(y: int, *, height: int = 32, width: int = 40) -> torch.Tensor:
    pixels = torch.ones((1, 3, height, width))
    pixels[:, :, y, 4:36] = -1.0
    return pixels


class EvidenceCarrierExtentTest(unittest.TestCase):
    def test_config_validates_discrete_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "odd"):
            EvidenceCarrierExtentConfig(band_samples=8)
        with self.assertRaisesRegex(ValueError, "extent bins"):
            EvidenceCarrierExtentConfig(extent_bins=1)
        with self.assertRaisesRegex(ValueError, "integer number"):
            EvidenceCarrierExtentConfig(
                normal_offset_radius_pixels=2.0,
                normal_offset_step_pixels=1.5,
            )

        config = _config()
        self.assertEqual(config.normal_offset_bins, 5)
        torch.testing.assert_close(
            config.normal_offset_values(),
            torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
        )
        torch.testing.assert_close(
            config.extent_fraction_values()[[0, -1]],
            torch.tensor([0.0, 1.0]),
        )
        default = EvidenceCarrierExtentConfig()
        self.assertEqual(default.longitudinal_samples, 385)
        self.assertEqual(default.extent_bins, 385)
        self.assertEqual(default.normal_offset_bins, 17)
        self.assertEqual(default.normal_offset_step_pixels, 0.5)
        diagonal_bin_pixels = (2.0**0.5 * 767.0) / (default.extent_bins - 1)
        self.assertLess(diagonal_bin_pixels, 3.0)

    def test_canonical_geometry_is_invariant_to_endpoint_order(self) -> None:
        segments = torch.tensor(
            [[[[8.0, 4.0], [32.0, 16.0]], [[12.0, 18.0], [12.0, 2.0]]]]
        )
        mask = torch.ones((1, 2), dtype=torch.bool)
        direct = canonicalize_carrier_segments(segments, mask)
        reversed_geometry = canonicalize_carrier_segments(segments.flip(-2), mask)
        for actual, expected in zip(direct, reversed_geometry, strict=True):
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

        canonical, _, axes, _ = direct
        self.assertTrue(bool((axes[..., 0] >= 0.0).all()))
        self.assertGreater(float(axes[0, 1, 1]), 0.0)
        torch.testing.assert_close(canonical[0, 1, 0], torch.tensor([12.0, 2.0]))

    def test_infinite_line_is_clipped_to_complete_image_intersection(self) -> None:
        centers = torch.tensor([[[20.0, 10.0]]])
        axis = torch.tensor([[[2.0, 1.0]]])
        axis = axis / axis.norm(dim=-1, keepdim=True)
        clipped, intersects = clip_carrier_lines_to_image(
            centers,
            axis,
            torch.ones((1, 1), dtype=torch.bool),
            image_height=21,
            image_width=41,
        )

        self.assertTrue(bool(intersects[0, 0]))
        torch.testing.assert_close(
            clipped,
            torch.tensor([[[[0.0, 0.0], [40.0, 20.0]]]]),
            atol=2e-6,
            rtol=0.0,
        )

    def test_horizontal_and_vertical_clips_use_exact_pixel_center_bounds(self) -> None:
        centers = torch.tensor([[[7.0, 5.0], [8.0, 4.0]]])
        axes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        clipped, intersects = clip_carrier_lines_to_image(
            centers,
            axes,
            torch.ones((1, 2), dtype=torch.bool),
            image_height=11,
            image_width=17,
        )

        self.assertTrue(bool(intersects.all()))
        torch.testing.assert_close(
            clipped,
            torch.tensor(
                [[[[0.0, 5.0], [16.0, 5.0]], [[8.0, 0.0], [8.0, 10.0]]]]
            ),
            atol=0.0,
            rtol=0.0,
        )

    def test_normalized_and_pixel_coordinates_sample_the_same_strip(self) -> None:
        pixels = _horizontal_image(10)
        pixel_segments = torch.tensor([[[[8.0, 10.0], [30.0, 10.0]]]])
        normalized_segments = pixel_segments / torch.tensor([39.0, 31.0])
        mask = torch.ones((1, 1), dtype=torch.bool)

        pixel_geometry = extract_evidence_carrier_strips(
            pixels,
            pixel_segments,
            mask,
            _config(),
            coordinate_mode="pixels",
        )
        normalized_geometry = extract_evidence_carrier_strips(
            pixels,
            normalized_segments,
            mask,
            _config(),
            coordinate_mode="normalized",
        )

        for name in (
            "oriented_strips",
            "canonical_segments_pixels",
            "carrier_centers_pixels",
            "carrier_axes",
            "carrier_normals",
            "clipped_segments_pixels",
            "anchor_interval_fractions",
            "anchor_support_mask",
            "anchor_endpoint_maps",
        ):
            torch.testing.assert_close(
                getattr(pixel_geometry, name),
                getattr(normalized_geometry, name),
                atol=1e-6,
                rtol=1e-6,
            )

    def test_forward_is_endpoint_order_invariant_and_masks_padding(self) -> None:
        torch.manual_seed(20260722)
        model = EvidenceCarrierExtentPredictor(_config()).eval()
        pixels = _horizontal_image(10)
        segments = torch.tensor(
            [[[[8.0, 10.0], [30.0, 10.0]], [[0.0, 0.0], [0.0, 0.0]]]]
        )
        mask = torch.tensor([[True, False]])
        changed_padding = segments.clone()
        changed_padding[0, 1] = torch.tensor([[1000.0, -500.0], [12.0, 900.0]])

        with torch.no_grad():
            direct = model(pixels, segments, mask, coordinate_mode="pixels")
            reversed_output = model(
                pixels,
                segments.flip(-2),
                mask,
                coordinate_mode="pixels",
            )
            changed = model(
                pixels,
                changed_padding,
                mask,
                coordinate_mode="pixels",
            )

        for name in (
            "keep_logits",
            "normal_offset_logits",
            "left_extent_logits",
            "right_extent_logits",
            "carrier_features",
            "oriented_strips",
            "anchor_support_mask",
            "anchor_endpoint_maps",
            "carrier_input_strips",
        ):
            torch.testing.assert_close(
                direct[name],
                reversed_output[name],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                direct[name],
                changed[name],
                atol=0.0,
                rtol=0.0,
            )
        self.assertEqual(float(direct["keep_logits"][0, 1]), 0.0)
        self.assertEqual(int(torch.count_nonzero(direct["oriented_strips"][:, 1])), 0)

    def test_collinear_carriers_are_disambiguated_by_relative_anchor_channels(self) -> None:
        torch.manual_seed(20260723)
        config = EvidenceCarrierExtentConfig(
            longitudinal_samples=65,
            band_samples=9,
            band_radius_pixels=2.0,
            normal_offset_radius_pixels=2.0,
            normal_offset_step_pixels=1.0,
            extent_bins=65,
            hidden_dim=16,
            dropout=0.0,
        )
        model = EvidenceCarrierExtentPredictor(config).eval()
        pixels = torch.ones((1, 3, 25, 65))
        pixels[:, :, 12, 6:21] = -1.0
        pixels[:, :, 12, 42:57] = -1.0
        carriers = torch.tensor(
            [[[[6.0, 12.0], [20.0, 12.0]], [[42.0, 12.0], [56.0, 12.0]]]]
        )
        mask = torch.ones((1, 2), dtype=torch.bool)

        with torch.no_grad():
            output = model(pixels, carriers, mask, coordinate_mode="pixels")
            reversed_output = model(
                pixels,
                carriers.flip(-2),
                mask,
                coordinate_mode="pixels",
            )

        # The raster evidence is identical because both candidates define the
        # same infinite line.  Only the candidate-relative anchor may separate
        # these two learning examples.
        torch.testing.assert_close(
            output["oriented_strips"][:, 0],
            output["oriented_strips"][:, 1],
            atol=0.0,
            rtol=0.0,
        )
        self.assertFalse(
            torch.equal(
                output["anchor_interval_fractions"][:, 0],
                output["anchor_interval_fractions"][:, 1],
            )
        )
        self.assertGreater(
            float(
                (
                    output["carrier_input_strips"][:, 0]
                    - output["carrier_input_strips"][:, 1]
                ).abs().max()
            ),
            0.1,
        )
        self.assertGreater(
            float(
                (
                    output["longitudinal_features"][:, 0]
                    - output["longitudinal_features"][:, 1]
                ).abs().max()
            ),
            1e-5,
        )
        extent_difference = torch.maximum(
            (
                output["left_extent_logits"][:, 0]
                - output["left_extent_logits"][:, 1]
            ).abs().max(),
            (
                output["right_extent_logits"][:, 0]
                - output["right_extent_logits"][:, 1]
            ).abs().max(),
        )
        self.assertGreater(float(extent_difference), 1e-6)

        for name in (
            "anchor_interval_fractions",
            "anchor_support_mask",
            "anchor_endpoint_maps",
            "carrier_input_strips",
            "longitudinal_features",
            "left_extent_logits",
            "right_extent_logits",
        ):
            torch.testing.assert_close(
                output[name],
                reversed_output[name],
                atol=0.0,
                rtol=0.0,
            )

    def test_candidate_relative_strip_and_prediction_translate_together(self) -> None:
        torch.manual_seed(19)
        model = EvidenceCarrierExtentPredictor(_config()).eval()
        first_pixels = _horizontal_image(10)
        second_pixels = _horizontal_image(15)
        first_segment = torch.tensor([[[[8.0, 10.0], [30.0, 10.0]]]])
        second_segment = first_segment + torch.tensor([0.0, 5.0])
        mask = torch.ones((1, 1), dtype=torch.bool)

        with torch.no_grad():
            first = model(
                first_pixels,
                first_segment,
                mask,
                coordinate_mode="pixels",
            )
            second = model(
                second_pixels,
                second_segment,
                mask,
                coordinate_mode="pixels",
            )
        torch.testing.assert_close(
            first["oriented_strips"],
            second["oriented_strips"],
            atol=2e-6,
            rtol=0.0,
        )
        for name in (
            "keep_logits",
            "normal_offset_logits",
            "left_extent_logits",
            "right_extent_logits",
        ):
            torch.testing.assert_close(first[name], second[name], atol=2e-6, rtol=0.0)

        first_decoded = decode_evidence_carrier_extent(first, keep_threshold=0.0)
        second_decoded = decode_evidence_carrier_extent(second, keep_threshold=0.0)
        displacement = (
            second_decoded["refined_segments_pixels"]
            - first_decoded["refined_segments_pixels"]
        )
        torch.testing.assert_close(
            displacement,
            torch.tensor([[[[0.0, 5.0], [0.0, 5.0]]]]),
            atol=2e-5,
            rtol=0.0,
        )

    def test_adjacent_endpoint_evidence_survives_longitudinal_feature_path(self) -> None:
        torch.manual_seed(41)
        config = EvidenceCarrierExtentConfig(
            longitudinal_samples=65,
            band_samples=9,
            band_radius_pixels=2.0,
            normal_offset_radius_pixels=2.0,
            normal_offset_step_pixels=1.0,
            extent_bins=65,
            hidden_dim=16,
            dropout=0.0,
        )
        model = EvidenceCarrierExtentPredictor(config).eval()
        first = torch.ones((1, 3, 24, 65))
        second = first.clone()
        first[:, :, 12, 8:40] = -1.0
        second[:, :, 12, 8:42] = -1.0
        carrier = torch.tensor([[[[10.0, 12.0], [35.0, 12.0]]]])
        mask = torch.ones((1, 1), dtype=torch.bool)

        with torch.no_grad():
            first_output = model(first, carrier, mask, coordinate_mode="pixels")
            second_output = model(second, carrier, mask, coordinate_mode="pixels")

        self.assertEqual(
            first_output["longitudinal_features"].shape,
            (1, 1, 16, 65),
        )
        feature_difference = (
            first_output["longitudinal_features"]
            - second_output["longitudinal_features"]
        ).abs().max()
        left_logit_difference = (
            first_output["left_extent_logits"]
            - second_output["left_extent_logits"]
        ).abs().max()
        right_logit_difference = (
            first_output["right_extent_logits"]
            - second_output["right_extent_logits"]
        ).abs().max()
        self.assertGreater(float(feature_difference), 1e-5)
        self.assertGreater(float(left_logit_difference), 1e-6)
        self.assertGreater(float(right_logit_difference), 1e-6)

    def test_ordered_decode_rejects_independent_crossed_extents(self) -> None:
        left = torch.tensor([[[0.0, 8.0, 0.0, 0.0, 20.0]]])
        right = torch.tensor([[[0.0, 10.0, 0.0, 0.0, 7.0]]])
        left_bin, right_bin = decode_ordered_extent_bins(left, right)

        self.assertEqual(int(left_bin[0, 0]), 1)
        self.assertEqual(int(right_bin[0, 0]), 4)
        self.assertTrue(bool((left_bin < right_bin).all()))

    def test_decode_offsets_reclips_and_reconstructs_bounded_line(self) -> None:
        model = EvidenceCarrierExtentPredictor(_config(extent_bins=5)).eval()
        pixels = _horizontal_image(10)
        segment = torch.tensor([[[[8.0, 10.0], [30.0, 10.0]]]])
        mask = torch.ones((1, 1), dtype=torch.bool)
        with torch.no_grad():
            output = model(pixels, segment, mask, coordinate_mode="pixels")
        output["keep_logits"] = torch.tensor([[20.0]])
        output["normal_offset_logits"] = torch.tensor(
            [[[-5.0, -5.0, -5.0, -5.0, 10.0]]]
        )
        output["left_extent_logits"] = torch.tensor(
            [[[0.0, 8.0, 0.0, 0.0, 20.0]]]
        )
        output["right_extent_logits"] = torch.tensor(
            [[[0.0, 10.0, 0.0, 0.0, 7.0]]]
        )

        decoded = decode_evidence_carrier_extent(output)

        self.assertEqual(int(decoded["normal_offset_bins"][0, 0]), 4)
        self.assertEqual(int(decoded["left_extent_bins"][0, 0]), 1)
        self.assertEqual(int(decoded["right_extent_bins"][0, 0]), 4)
        torch.testing.assert_close(
            decoded["offset_clipped_segments_pixels"],
            torch.tensor([[[[0.0, 12.0], [39.0, 12.0]]]]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            decoded["refined_segments_pixels"],
            torch.tensor([[[[9.75, 12.0], [39.0, 12.0]]]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_loss_masks_padding_and_backpropagates_every_head(self) -> None:
        torch.manual_seed(31)
        model = EvidenceCarrierExtentPredictor(_config(extent_bins=5))
        pixels = _horizontal_image(10)
        segments = torch.tensor(
            [[[[8.0, 10.0], [30.0, 10.0]], [[4.0, 20.0], [20.0, 20.0]]]]
        )
        mask = torch.tensor([[True, True]])
        output = model(pixels, segments, mask, coordinate_mode="pixels")
        losses = evidence_carrier_extent_loss(
            output,
            torch.tensor([[True, False]]),
            torch.tensor([[2, 999]]),
            torch.tensor([[1, 999]]),
            torch.tensor([[4, 999]]),
        )
        losses["loss"].backward()

        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        self.assertEqual(int(losses["active_carrier_count"]), 2)
        self.assertEqual(int(losses["positive_carrier_count"]), 1)
        for head in (
            model.keep_head,
            model.normal_offset_head,
            model.left_extent_head,
            model.right_extent_head,
        ):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)

        with self.assertRaisesRegex(ValueError, "left < right"):
            evidence_carrier_extent_loss(
                output,
                torch.tensor([[True, False]]),
                torch.tensor([[2, 0]]),
                torch.tensor([[3, 0]]),
                torch.tensor([[3, 1]]),
            )

    def test_loss_honors_ignore_masks_and_equivalence_weights(self) -> None:
        model = EvidenceCarrierExtentPredictor(_config(extent_bins=5)).eval()
        pixels = _horizontal_image(10)
        segments = torch.tensor(
            [
                [
                    [[8.0, 10.0], [30.0, 10.0]],
                    [[4.0, 15.0], [20.0, 15.0]],
                    [[5.0, 20.0], [25.0, 20.0]],
                ]
            ]
        )
        mask = torch.ones((1, 3), dtype=torch.bool)
        with torch.no_grad():
            output = model(pixels, segments, mask, coordinate_mode="pixels")
        targets = {
            "keep_labels": torch.tensor([[1, -1, 0]]),
            "normal_offset_bin": torch.tensor([[2, 999, 999]]),
            "extent_left_bin": torch.tensor([[1, 999, 999]]),
            "extent_right_bin": torch.tensor([[4, 999, 999]]),
        }
        options = {
            "keep_loss_mask": torch.tensor([[True, True, False]]),
            "geometry_loss_mask": torch.tensor([[True, True, False]]),
            "equivalence_weight": torch.tensor([[0.25, 50.0, 70.0]]),
        }
        baseline = evidence_carrier_extent_loss(output, **targets, **options)

        changed = dict(output)
        for name in (
            "keep_logits",
            "normal_offset_logits",
            "left_extent_logits",
            "right_extent_logits",
        ):
            changed[name] = output[name].clone()
            changed[name][:, 1:] += 100.0
        modified = evidence_carrier_extent_loss(changed, **targets, **options)

        for name in (
            "loss",
            "keep_loss",
            "normal_offset_loss",
            "left_extent_loss",
            "right_extent_loss",
        ):
            torch.testing.assert_close(baseline[name], modified[name])
        self.assertEqual(int(baseline["keep_supervision_count"]), 1)
        self.assertEqual(int(baseline["geometry_supervision_count"]), 1)
        self.assertAlmostEqual(float(baseline["keep_supervision_weight"]), 0.25)
        self.assertAlmostEqual(float(baseline["geometry_supervision_weight"]), 0.25)

    def test_zero_carrier_padding_supports_decode_and_backward(self) -> None:
        model = EvidenceCarrierExtentPredictor(_config())
        pixels = torch.ones((2, 3, 16, 16))
        segments = torch.empty((2, 0, 2, 2))
        mask = torch.empty((2, 0), dtype=torch.bool)

        output = model(pixels, segments, mask)
        self.assertEqual(output["keep_logits"].shape, (2, 0))
        self.assertEqual(output["normal_offset_logits"].shape, (2, 0, 5))
        self.assertEqual(output["left_extent_logits"].shape, (2, 0, 9))
        self.assertEqual(output["oriented_strips"].shape, (2, 0, 3, 9, 33))
        self.assertEqual(output["anchor_support_mask"].shape, (2, 0, 1, 9, 33))
        self.assertEqual(output["anchor_endpoint_maps"].shape, (2, 0, 2, 9, 33))
        self.assertEqual(output["carrier_input_strips"].shape, (2, 0, 6, 9, 33))
        decoded = decode_evidence_carrier_extent(output)
        self.assertEqual(decoded["refined_segments_pixels"].shape, (2, 0, 2, 2))

        empty_targets = torch.empty((2, 0), dtype=torch.long)
        losses = evidence_carrier_extent_loss(
            output,
            torch.empty((2, 0), dtype=torch.bool),
            empty_targets,
            empty_targets,
            empty_targets,
        )
        losses["loss"].backward()
        self.assertEqual(float(losses["loss"].detach()), 0.0)
        self.assertIsNotNone(model.keep_head.weight.grad)

    def test_input_validation_is_strict(self) -> None:
        config = _config()
        pixels = torch.ones((1, 3, 16, 16))
        segment = torch.tensor([[[[0.2, 0.2], [0.8, 0.8]]]])
        mask = torch.ones((1, 1), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "shape"):
            extract_evidence_carrier_strips(
                pixels[:, :1],
                segment,
                mask,
                config,
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            extract_evidence_carrier_strips(
                pixels,
                segment,
                mask.float(),
                config,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            extract_evidence_carrier_strips(
                pixels,
                segment + 1.0,
                mask,
                config,
            )
        with self.assertRaisesRegex(ValueError, "zero-length"):
            extract_evidence_carrier_strips(
                pixels,
                torch.tensor([[[[0.4, 0.4], [0.4, 0.4]]]]),
                mask,
                config,
            )


if __name__ == "__main__":
    unittest.main()
