from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from chemvector.evidence_carrier_stroke_pairing import (
    StrokeBoundaryPairingConfig,
    pair_stroke_boundaries,
    stroke_boundary_pair_geometry,
)


def black_stroke(*, y: int = 50, width: int = 3) -> Image.Image:
    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).line((10, y, 90, y), fill=0, width=width)
    return image


class EvidenceCarrierStrokePairingTest(unittest.TestCase):
    def test_dark_strip_between_boundaries_is_compatible(self) -> None:
        geometry = stroke_boundary_pair_geometry(
            ((10.0, 48.5), (90.0, 48.5)),
            ((10.0, 51.5), (90.0, 51.5)),
            black_stroke(),
        )

        self.assertTrue(geometry.compatible)
        self.assertAlmostEqual(geometry.normal_separation_pixels, 3.0)
        self.assertEqual(geometry.shorter_interval_overlap, 1.0)
        self.assertGreaterEqual(geometry.interior_ink_occupancy, 0.8)

    def test_white_gap_between_distinct_strokes_is_rejected(self) -> None:
        image = Image.new("L", (100, 100), 255)
        draw = ImageDraw.Draw(image)
        draw.line((10, 45, 90, 45), fill=0, width=2)
        draw.line((10, 53, 90, 53), fill=0, width=2)
        config = StrokeBoundaryPairingConfig(
            maximum_normal_separation_pixels=8.0
        )

        geometry = stroke_boundary_pair_geometry(
            ((10.0, 47.0), (90.0, 47.0)),
            ((10.0, 51.0), (90.0, 51.0)),
            image,
            config=config,
        )

        self.assertLess(geometry.interior_ink_occupancy, 0.8)
        self.assertFalse(geometry.compatible)

    def test_pair_replaces_two_boundaries_and_retains_unpaired_anchor(self) -> None:
        segments = np.asarray(
            [
                ((10.0, 48.5), (90.0, 48.5)),
                ((10.0, 51.5), (90.0, 51.5)),
                ((20.0, 80.0), (60.0, 80.0)),
            ]
        )

        result = pair_stroke_boundaries(segments, black_stroke())

        self.assertEqual(len(result.selected_pairs), 1)
        self.assertEqual(len(result.carriers), 2)
        paired = next(carrier for carrier in result.carriers if carrier.paired)
        self.assertEqual(paired.source_indices, (0, 1))
        np.testing.assert_allclose(
            np.asarray(paired.segment_pixels),
            np.asarray(((10.0, 50.0), (90.0, 50.0))),
            atol=1e-6,
        )
        unpaired = next(carrier for carrier in result.carriers if not carrier.paired)
        self.assertEqual(unpaired.source_indices, (2,))

    def test_centerline_uses_only_observed_common_overlap_extent(self) -> None:
        segments = np.asarray(
            [
                ((10.0, 48.5), (90.0, 48.5)),
                ((30.0, 51.5), (80.0, 51.5)),
            ]
        )
        config = StrokeBoundaryPairingConfig(
            minimum_shorter_interval_overlap=0.70,
        )

        result = pair_stroke_boundaries(segments, black_stroke(), config=config)

        self.assertEqual(len(result.carriers), 1)
        np.testing.assert_allclose(
            result.segments_pixels()[0],
            np.asarray(((30.0, 50.0), (80.0, 50.0))),
            atol=1e-6,
        )

    def test_mutual_then_greedy_is_order_deterministic(self) -> None:
        segments = np.asarray(
            [
                ((10.0, 48.5), (90.0, 48.5)),
                ((10.0, 51.5), (90.0, 51.5)),
                ((12.0, 51.7), (88.0, 51.7)),
            ]
        )
        first = pair_stroke_boundaries(segments, black_stroke())
        second = pair_stroke_boundaries(segments.copy(), black_stroke())

        self.assertEqual(first, second)
        self.assertEqual(len(first.selected_pairs), 1)

    def test_equivalent_pair_tie_break_is_input_permutation_invariant(self) -> None:
        image = black_stroke(y=50, width=5)
        segments = np.asarray(
            [
                ((10.0, 49.0), (90.0, 49.0)),
                ((10.0, 50.0), (90.0, 50.0)),
                ((10.0, 51.0), (90.0, 51.0)),
            ]
        )
        config = StrokeBoundaryPairingConfig(
            reference_stroke_width_pixels=1.0,
        )

        forward = pair_stroke_boundaries(segments, image, config=config)
        reverse = pair_stroke_boundaries(segments[::-1].copy(), image, config=config)

        np.testing.assert_allclose(
            forward.segments_pixels(),
            reverse.segments_pixels(),
            atol=1e-9,
        )
        forward_sources = sorted(
            tuple(
                sorted(
                    tuple(segments[index].reshape(-1))
                    for index in (pair.first_index, pair.second_index)
                )
            )
            for pair in forward.selected_pairs
        )
        reverse_sources = sorted(
            tuple(
                sorted(
                    tuple(segments[::-1][index].reshape(-1))
                    for index in (pair.first_index, pair.second_index)
                )
            )
            for pair in reverse.selected_pairs
        )
        self.assertEqual(forward_sources, reverse_sources)


if __name__ == "__main__":
    unittest.main()
