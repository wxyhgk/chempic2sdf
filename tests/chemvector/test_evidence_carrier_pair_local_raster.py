import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from chemvector.evidence_carrier_pair_local_raster import (
    PairLocalRasterConfig,
    dequantize_pair_local_rasters,
    extract_pair_local_rasters,
    quantize_pair_local_rasters,
)


class PairLocalRasterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (96, 64), "white")
        draw = ImageDraw.Draw(self.image)
        draw.line((18, 29, 78, 29), fill="black", width=1)
        draw.line((18, 33, 78, 33), fill="black", width=1)
        self.segments = np.asarray(
            [
                [[18.0, 29.0], [78.0, 29.0]],
                [[18.0, 33.0], [78.0, 33.0]],
            ]
        )
        self.pairs = np.asarray([[0, 1]])
        self.centerlines = np.asarray([[[18.0, 31.0], [78.0, 31.0]]])

    def test_shape_range_and_candidate_mask(self) -> None:
        crops = extract_pair_local_rasters(
            self.image,
            self.segments,
            self.pairs,
            pair_centerlines_pixels=self.centerlines,
        )
        self.assertEqual(crops.shape, (1, 2, 24, 48))
        self.assertEqual(crops.dtype, torch.float32)
        self.assertGreater(float(crops[0, 0].max()), 0.75)
        self.assertGreater(float(crops[0, 0].sum()), 20.0)
        self.assertGreater(float(crops[0, 1].max()), 0.9)
        self.assertGreater(int((crops[0, 1] > 0.5).sum()), 40)
        self.assertGreaterEqual(float(crops.min()), 0.0)
        self.assertLessEqual(float(crops.max()), 1.0)

    def test_candidate_order_and_endpoint_order_are_invariant(self) -> None:
        original = extract_pair_local_rasters(
            self.image,
            self.segments,
            self.pairs,
            pair_centerlines_pixels=self.centerlines,
        )
        reversed_segments = self.segments[::-1, ::-1].copy()
        reversed_centerline = self.centerlines[:, ::-1].copy()
        transformed = extract_pair_local_rasters(
            self.image,
            reversed_segments,
            self.pairs,
            pair_centerlines_pixels=reversed_centerline,
        )
        torch.testing.assert_close(original, transformed, rtol=0.0, atol=1e-6)

    def test_complete_image_crossing_remains_visible_outside_candidate_mask(self) -> None:
        baseline = extract_pair_local_rasters(
            self.image,
            self.segments,
            self.pairs,
            pair_centerlines_pixels=self.centerlines,
        )
        crossed = self.image.copy()
        ImageDraw.Draw(crossed).line((48, 12, 48, 52), fill="black", width=2)
        candidate = extract_pair_local_rasters(
            crossed,
            self.segments,
            self.pairs,
            pair_centerlines_pixels=self.centerlines,
        )
        ink_delta = candidate[0, 0] - baseline[0, 0]
        self.assertGreater(float(ink_delta.clamp_min(0.0).sum()), 5.0)
        torch.testing.assert_close(candidate[:, 1], baseline[:, 1])

    def test_white_padding_and_empty_pair_output(self) -> None:
        image = Image.new("L", (32, 32), 255)
        segments = np.asarray(
            [
                [[-8.0, 2.0], [16.0, 2.0]],
                [[-8.0, 5.0], [16.0, 5.0]],
            ]
        )
        crops = extract_pair_local_rasters(image, segments, [[0, 1]])
        self.assertEqual(float(crops[0, 0].max()), 0.0)
        empty = extract_pair_local_rasters(image, segments, [])
        self.assertEqual(empty.shape, (0, 2, 24, 48))

    def test_uint8_cache_round_trip(self) -> None:
        crops = extract_pair_local_rasters(
            self.image,
            self.segments,
            self.pairs,
            pair_centerlines_pixels=self.centerlines,
        )
        encoded = quantize_pair_local_rasters(crops)
        self.assertEqual(encoded.dtype, torch.uint8)
        decoded = dequantize_pair_local_rasters(encoded)
        torch.testing.assert_close(crops, decoded, rtol=0.0, atol=1.0 / 255.0)

    def test_rejects_invalid_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            extract_pair_local_rasters(self.image, self.segments, [[0, 0]])
        with self.assertRaisesRegex(ValueError, "exceed"):
            extract_pair_local_rasters(self.image, self.segments, [[0, 2]])
        with self.assertRaisesRegex(ValueError, "width"):
            PairLocalRasterConfig(width=1)


if __name__ == "__main__":
    unittest.main()
