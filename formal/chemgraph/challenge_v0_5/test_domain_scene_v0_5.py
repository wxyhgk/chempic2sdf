#!/usr/bin/env python3
"""Domain adapter and known observation-collision regression tests."""

from __future__ import annotations

import unittest

try:
    from .canonical_raster_v0_5 import preprocess_gray, require_white_border
    from .domain_scene_v0_5 import (
        decode_motif_ordinal, domain_motif, render_and_verify_padded_certificate,
        render_centered_gray,
    )
except ImportError:
    from canonical_raster_v0_5 import preprocess_gray, require_white_border  # type: ignore
    from domain_scene_v0_5 import (  # type: ignore
        decode_motif_ordinal, domain_motif, render_and_verify_padded_certificate,
        render_centered_gray,
    )


class DomainSceneTests(unittest.TestCase):
    def test_all_motif_ordinals_construct(self) -> None:
        for ordinal in range(52):
            choice = decode_motif_ordinal(ordinal)
            motif = domain_motif(85, ordinal, 5)
            self.assertTrue(motif.primitives, choice)
            self.assertTrue(motif.interpretations, choice)

    def test_centered_public_extrema_keep_frozen_border(self) -> None:
        for motif_ordinal in (0, 2, 3, 24, 25, 36, 47, 51):
            gray, _ = render_centered_gray(
                half_length=110, motif_ordinal=motif_ordinal, width_px=9,
                cap_index=2, antialias_index=2, reflection_index=2,
                rotation_index=3, scale_index=6,
            )
            require_white_border(gray, 33)

    def test_padded_clipping_certificate(self) -> None:
        raw, _ = render_and_verify_padded_certificate(
            half_length=110, motif_ordinal=2, width_px=9,
            cap_index=2, antialias_index=2, reflection_index=0,
            rotation_index=1, scale_index=6,
        )
        require_white_border(raw, 33)

    def test_s8_c4_canonical_collision_in_eight_styles(self) -> None:
        for width_px in (2, 4, 6, 8):
            for cap_index in (0, 2):
                s8, s8_motif = render_centered_gray(
                    half_length=62, motif_ordinal=11, width_px=width_px,
                    cap_index=cap_index, antialias_index=0, reflection_index=0,
                    rotation_index=0, scale_index=2,
                )
                c4, c4_motif = render_centered_gray(
                    half_length=93, motif_ordinal=48, width_px=width_px,
                    cap_index=cap_index, antialias_index=0, reflection_index=0,
                    rotation_index=4, scale_index=0,
                )
                self.assertEqual(preprocess_gray(s8), preprocess_gray(c4))
                self.assertEqual(len(s8_motif.interpretations), 1)
                self.assertEqual(len(c4_motif.interpretations), 2)


if __name__ == "__main__":
    unittest.main()
