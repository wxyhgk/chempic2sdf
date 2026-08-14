#!/usr/bin/env python3
"""Bind one frozen v0.5 domain tuple to the v0.4 motif and rasterizer."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ..challenge_v0_4.generate_pack_v0_4 import (
        AA_PROFILES, CAPS, Q, REFLECTIONS, ROTATIONS, SCALES, Motif,
        Transform, build_motif,
    )
    from ..challenge_v0_4.rasterizer_v0_4 import LineStyle, Segment, rasterize_gray
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from challenge_v0_4.generate_pack_v0_4 import (  # type: ignore
        AA_PROFILES, CAPS, Q, REFLECTIONS, ROTATIONS, SCALES, Motif,
        Transform, build_motif,
    )
    from challenge_v0_4.rasterizer_v0_4 import (  # type: ignore
        LineStyle, Segment, rasterize_gray,
    )


@dataclass(frozen=True)
class MotifChoice:
    family: str
    variant_kind: str
    variant_value: int | None


def decode_motif_ordinal(ordinal: int) -> MotifChoice:
    fixed = {
        0: "s1", 1: "s2", 2: "s3", 3: "s4", 6: "s6", 7: "s7", 25: "c1",
    }
    if ordinal in fixed:
        return MotifChoice(fixed[ordinal], "none", None)
    if 4 <= ordinal <= 5:
        return MotifChoice("s5", "shape_index", ordinal - 4)
    if 8 <= ordinal <= 24:
        return MotifChoice("s8", "gap", ordinal + 4)
    if 26 <= ordinal <= 36:
        return MotifChoice("c2", "spacing_delta", ordinal - 26)
    if 37 <= ordinal <= 47:
        return MotifChoice("c3", "spacing_delta", ordinal - 37)
    if 48 <= ordinal <= 51:
        return MotifChoice("c4", "glyph_index", ordinal - 48)
    raise ValueError(f"motif_ordinal_out_of_domain:{ordinal}")


class FrozenMotifRng:
    """Substitute the v0.5 explicit axes for the two v0.4 random draws."""

    def __init__(self, half_length: int, choice: MotifChoice, width_px: int):
        if not 60 <= half_length <= 110:
            raise ValueError("half_length_out_of_domain")
        if not 1 <= width_px <= 9:
            raise ValueError("width_out_of_domain")
        self.half_length = half_length
        self.choice = choice
        self.width_px = width_px
        self.length_consumed = False

    def randint(self, lower: int, upper: int) -> int:
        if not self.length_consumed:
            if (lower, upper) != (120, 220):
                raise AssertionError(f"unexpected_length_axis:{lower}:{upper}")
            self.length_consumed = True
            return 2 * self.half_length
        if self.choice.variant_kind == "gap" and (lower, upper) == (12, 28):
            return int(self.choice.variant_value)
        if self.choice.variant_kind == "spacing_delta":
            expected_lower = 2 * (self.width_px + 3)
            if (lower, upper) != (expected_lower, expected_lower + 10):
                raise AssertionError(f"unexpected_spacing_axis:{lower}:{upper}")
            return expected_lower + int(self.choice.variant_value)
        raise AssertionError(f"unexpected_randint:{self.choice}:{lower}:{upper}")

    def randbelow(self, upper: int) -> int:
        if self.choice.variant_kind in ("shape_index", "glyph_index"):
            expected = 2 if self.choice.variant_kind == "shape_index" else 4
            if upper != expected:
                raise AssertionError(f"unexpected_variant_cardinality:{upper}:{expected}")
            return int(self.choice.variant_value)
        raise AssertionError(f"unexpected_randbelow:{self.choice}:{upper}")


def domain_motif(half_length: int, motif_ordinal: int, width_px: int) -> Motif:
    choice = decode_motif_ordinal(motif_ordinal)
    return build_motif(
        choice.family,
        FrozenMotifRng(half_length, choice, width_px),  # type: ignore[arg-type]
        width_px,
    )


def centered_transform(
    reflection_index: int,
    rotation_index: int,
    scale_index: int,
    center_px: int = 384,
) -> Transform:
    try:
        reflection = REFLECTIONS[reflection_index]
        cosine, sine = ROTATIONS[rotation_index]
        scale = SCALES[scale_index]
    except IndexError as error:
        raise ValueError("transform_index_out_of_domain") from error
    center = center_px * Q
    return Transform(reflection, cosine, sine, scale, center, center)


def domain_style(width_px: int, cap_index: int, antialias_index: int) -> LineStyle:
    if not 1 <= width_px <= 9:
        raise ValueError("width_out_of_domain")
    try:
        cap = CAPS[cap_index]
        antialias = AA_PROFILES[antialias_index]
    except IndexError as error:
        raise ValueError("style_index_out_of_domain") from error
    return LineStyle(width_px * Q, cap, antialias, "round")


def render_centered_gray(
    *,
    half_length: int,
    motif_ordinal: int,
    width_px: int,
    cap_index: int,
    antialias_index: int,
    reflection_index: int,
    rotation_index: int,
    scale_index: int,
    canvas_size: int = 768,
    center_px: int = 384,
) -> tuple[bytes, Motif]:
    motif = domain_motif(half_length, motif_ordinal, width_px)
    transform = centered_transform(
        reflection_index, rotation_index, scale_index, center_px=center_px,
    )
    segments = []
    for index, primitive in enumerate(motif.primitives):
        start = transform.point(primitive.start)
        end = transform.point(primitive.end)
        segments.append(Segment(f"p-{index:06d}", start[0], start[1], end[0], end[1]))
    style = domain_style(width_px, cap_index, antialias_index)
    return rasterize_gray(segments, style, width=canvas_size, height=canvas_size), motif


def verify_padded_clipping_certificate(raw: bytes, padded: bytes) -> None:
    if len(raw) != 768 * 768 or len(padded) != 1024 * 1024:
        raise ValueError("clipping_certificate_shape_mismatch")
    white_top = bytes([255]) * (128 * 1024)
    if padded[:128 * 1024] != white_top or padded[896 * 1024:] != white_top:
        raise ValueError("clipping_certificate_foreground_outside_vertical")
    white_side = bytes([255]) * 128
    for y in range(768):
        padded_row = padded[(y + 128) * 1024:(y + 129) * 1024]
        if padded_row[:128] != white_side or padded_row[896:] != white_side:
            raise ValueError("clipping_certificate_foreground_outside_horizontal")
        raw_row = raw[y * 768:(y + 1) * 768]
        if padded_row[128:896] != raw_row:
            raise ValueError("clipping_certificate_embedded_bytes_differ")


def render_and_verify_padded_certificate(**axes: int) -> tuple[bytes, Motif]:
    raw, motif = render_centered_gray(**axes)
    padded, _ = render_centered_gray(
        **axes, canvas_size=1024, center_px=512,
    )
    verify_padded_clipping_certificate(raw, padded)
    return raw, motif
