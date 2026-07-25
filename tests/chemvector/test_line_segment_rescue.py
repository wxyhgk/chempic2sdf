from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from chemvector.line_segment_rescue import (
    LineSegmentRescueConfig,
    attach_segments_to_atom_pairs,
    detect_atom_pair_line_candidates,
    detect_raster_line_segments,
)


class LineSegmentRescueTest(unittest.TestCase):
    def test_long_segment_has_no_maximum_length_gate(self) -> None:
        candidates = attach_segments_to_atom_pairs(
            np.asarray([((10.0, 50.0), (490.0, 50.0))]),
            np.asarray(((10.0, 50.0), (490.0, 50.0))),
        )

        self.assertEqual([item.atom_pair for item in candidates], [(0, 1)])
        self.assertAlmostEqual(candidates[0].maximum_support_length_pixels, 480.0)

    def test_two_stroke_boundaries_form_one_semantic_atom_pair(self) -> None:
        candidates = attach_segments_to_atom_pairs(
            np.asarray(
                [
                    ((10.0, 49.0), (90.0, 49.0)),
                    ((10.0, 51.0), (90.0, 51.0)),
                ]
            ),
            np.asarray(((10.0, 50.0), (90.0, 50.0))),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].atom_pair, (0, 1))
        self.assertEqual(candidates[0].support_segment_indices, (0, 1))

    def test_endpoint_gap_allows_text_anchor_without_length_prior(self) -> None:
        candidates = attach_segments_to_atom_pairs(
            np.asarray([((22.0, 50.0), (178.0, 50.0))]),
            np.asarray(((10.0, 50.0), (190.0, 50.0))),
        )

        self.assertEqual([item.atom_pair for item in candidates], [(0, 1)])

    def test_far_endpoint_and_wrong_angle_are_rejected(self) -> None:
        anchors = np.asarray(((10.0, 50.0), (90.0, 50.0), (90.0, 90.0)))
        far = attach_segments_to_atom_pairs(
            np.asarray([((30.0, 50.0), (70.0, 50.0))]), anchors
        )
        angled = attach_segments_to_atom_pairs(
            np.asarray([((10.0, 50.0), (90.0, 90.0))]),
            anchors,
            atom_mask=np.asarray((True, True, False)),
        )

        self.assertEqual(far, ())
        self.assertEqual(angled, ())

    def test_atom_mask_preserves_original_atom_indices(self) -> None:
        candidates = attach_segments_to_atom_pairs(
            np.asarray([((10.0, 50.0), (90.0, 50.0))]),
            np.asarray(((10.0, 50.0), (50.0, 90.0), (90.0, 50.0))),
            atom_mask=np.asarray((True, False, True)),
        )

        self.assertEqual([item.atom_pair for item in candidates], [(0, 2)])

    def test_detector_ignores_canvas_border_and_keeps_long_bond(self) -> None:
        image = Image.new("L", (240, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 239, 119), outline=0, width=1)
        draw.line((20, 60, 220, 60), fill=0, width=2)

        segments = detect_raster_line_segments(image)
        candidates = detect_atom_pair_line_candidates(
            image,
            np.asarray(((20.0, 60.0), (220.0, 60.0))),
            config=LineSegmentRescueConfig(maximum_endpoint_distance_pixels=18.0),
        )

        self.assertTrue(any(np.linalg.norm(item[1] - item[0]) > 180 for item in segments))
        self.assertEqual([item.atom_pair for item in candidates], [(0, 1)])


if __name__ == "__main__":
    unittest.main()
