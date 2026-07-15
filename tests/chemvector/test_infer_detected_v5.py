"""Tests for external-image preparation used by detected inference."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from chemvector.image_io import prepare_input_image


class DetectedInferencePreparationTests(unittest.TestCase):
    def test_frame_is_removed_and_content_aspect_ratio_is_preserved(self) -> None:
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 199, 99), outline="black", width=2)
        draw.line((50, 40, 150, 60), fill="black", width=2)
        prepared = prepare_input_image(image, image_size=256, content_size=200)
        array = np.asarray(prepared)
        dark_y, dark_x = np.where(array.min(axis=2) < 160)
        self.assertEqual(prepared.size, (256, 256))
        self.assertGreater(dark_x.min(), 0)
        self.assertGreater(dark_y.min(), 0)
        self.assertLess(dark_x.max(), 255)
        self.assertLess(dark_y.max(), 255)
        self.assertGreater(dark_x.max() - dark_x.min(), dark_y.max() - dark_y.min())

    def test_content_size_must_fit_canvas(self) -> None:
        with self.assertRaisesRegex(ValueError, "content_size"):
            prepare_input_image(Image.new("RGB", (10, 10)), image_size=32, content_size=33)

    def test_edge_to_edge_connected_content_is_not_removed_as_a_frame(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.line((0, 40, 50, 0), fill="black", width=2)
        draw.line((50, 0, 99, 40), fill="black", width=2)
        draw.line((99, 40, 50, 79), fill="black", width=2)
        draw.line((50, 79, 0, 40), fill="black", width=2)

        prepared = prepare_input_image(image, image_size=128, content_size=100)

        self.assertGreater((np.asarray(prepared).min(axis=2) < 160).sum(), 200)

    def test_transparent_pixels_are_composited_on_white(self) -> None:
        image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        prepared = prepare_input_image(image, image_size=32, content_size=20)
        self.assertTrue(np.all(np.asarray(prepared) == 255))


if __name__ == "__main__":
    unittest.main()
