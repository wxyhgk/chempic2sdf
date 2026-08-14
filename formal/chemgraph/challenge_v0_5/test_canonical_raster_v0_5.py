#!/usr/bin/env python3
"""Contract tests for the v0.5 canonical raster preprocessor."""

from __future__ import annotations

import unittest

try:
    from .canonical_raster_v0_5 import (
        BACKGROUND, CANVAS_SIZE, CanonicalRasterError, decode_canonical_png,
        embed_canonical, foreground_bbox, integer_shift, preprocess_gray,
    )
    from ..challenge_v0_4.rasterizer_v0_4 import encode_canonical_png
except ImportError:
    from canonical_raster_v0_5 import (  # type: ignore
        BACKGROUND, CANVAS_SIZE, CanonicalRasterError, decode_canonical_png,
        embed_canonical, foreground_bbox, integer_shift, preprocess_gray,
    )
    from challenge_v0_4.rasterizer_v0_4 import encode_canonical_png  # type: ignore


def raster(points: list[tuple[int, int, int]]) -> bytes:
    gray = bytearray([BACKGROUND]) * (CANVAS_SIZE * CANVAS_SIZE)
    for x, y, value in points:
        gray[y * CANVAS_SIZE + x] = value
    return bytes(gray)


class CanonicalRasterTests(unittest.TestCase):
    def test_strict_png_roundtrip(self) -> None:
        gray = raster([(300, 350, 0), (301, 350, 127)])
        png = encode_canonical_png(gray)
        self.assertEqual(decode_canonical_png(png), gray)
        with self.assertRaisesRegex(CanonicalRasterError, "png_trailing_bytes"):
            decode_canonical_png(png + b"x")

    def test_empty_and_border_fail(self) -> None:
        empty = bytes([BACKGROUND]) * (CANVAS_SIZE * CANVAS_SIZE)
        with self.assertRaisesRegex(CanonicalRasterError, "empty_foreground"):
            preprocess_gray(empty)
        with self.assertRaisesRegex(CanonicalRasterError, "foreground_border_too_small"):
            preprocess_gray(raster([(0, 10, 0)]))

    def test_even_crop_uses_floor_center(self) -> None:
        canonical = preprocess_gray(raster([(100, 200, 0), (101, 200, 64)]))
        self.assertEqual(foreground_bbox(canonical), (383, 383, 384, 383))

    def test_odd_crop_center(self) -> None:
        canonical = preprocess_gray(raster([(100, 200, 0), (102, 202, 64)]))
        self.assertEqual(foreground_bbox(canonical), (382, 382, 384, 384))

    def test_translation_invariance_at_all_extremes(self) -> None:
        centered = raster([(300, 330, 0), (301, 331, 64), (460, 440, 127)])
        expected = preprocess_gray(centered)
        for dx in (-32, 0, 32):
            for dy in (-32, 0, 32):
                shifted = integer_shift(centered, dx, dy)
                self.assertEqual(preprocess_gray(shifted), expected)

    def test_typed_canonical_identity(self) -> None:
        canonical = preprocess_gray(raster([(300, 330, 0), (460, 440, 127)]))
        self.assertEqual(preprocess_gray(embed_canonical(canonical)), canonical)


if __name__ == "__main__":
    unittest.main()
