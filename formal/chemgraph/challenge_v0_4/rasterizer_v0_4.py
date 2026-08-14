#!/usr/bin/env python3
"""Deterministic Q16 line rasterizer and canonical grayscale PNG encoder.

This module intentionally depends only on the Python standard library.  It uses
integer geometry throughout; no platform drawing library or floating-point
trigonometry is involved.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import binascii
import hashlib
import math
import struct
import zlib

Q = 1 << 16
CANVAS_SIZE = 768
AA_SAMPLES = {
    "binary_1x": 1,
    "box_2x2": 2,
    "box_4x4": 4,
}


@dataclass(frozen=True)
class Segment:
    segment_id: str
    x0_q16: int
    y0_q16: int
    x1_q16: int
    y1_q16: int


@dataclass(frozen=True)
class LineStyle:
    width_q16: int
    cap: str = "round"
    antialias_profile: str = "box_4x4"
    join: str = "round"


def round_half_even(numerator: int, denominator: int) -> int:
    """Round a non-negative rational exactly using ties-to-even."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("round_half_even expects numerator >= 0 and denominator > 0")
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def _inside_segment(sample_x: int, sample_y: int, segment: Segment, style: LineStyle) -> bool:
    ax, ay = segment.x0_q16, segment.y0_q16
    bx, by = segment.x1_q16, segment.y1_q16
    vx, vy = bx - ax, by - ay
    wx, wy = sample_x - ax, sample_y - ay
    length_sq = vx * vx + vy * vy
    if length_sq == 0:
        return False
    radius = style.width_q16 // 2
    radius_sq = radius * radius
    projection = wx * vx + wy * vy
    cross = wx * vy - wy * vx

    if style.cap == "butt":
        return 0 <= projection <= length_sq and cross * cross <= radius_sq * length_sq
    if style.cap == "square":
        length = math.isqrt(length_sq)
        extension = radius * length
        return (-extension <= projection <= length_sq + extension and
                cross * cross <= radius_sq * length_sq)
    if style.cap != "round":
        raise ValueError(f"unsupported cap: {style.cap}")
    if projection < 0:
        return wx * wx + wy * wy <= radius_sq
    if projection > length_sq:
        dx, dy = sample_x - bx, sample_y - by
        return dx * dx + dy * dy <= radius_sq
    return cross * cross <= radius_sq * length_sq


def _pixel_bounds(segment: Segment, style: LineStyle, width: int, height: int) -> tuple[int, int, int, int]:
    padding = (style.width_q16 + Q - 1) // Q + 2
    min_x = max(0, min(segment.x0_q16, segment.x1_q16) // Q - padding)
    max_x = min(width - 1, max(segment.x0_q16, segment.x1_q16) // Q + padding)
    min_y = max(0, min(segment.y0_q16, segment.y1_q16) // Q - padding)
    max_y = min(height - 1, max(segment.y0_q16, segment.y1_q16) // Q + padding)
    return min_x, max_x, min_y, max_y


def _shared_join_points(segments: list[Segment]) -> list[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for segment in segments:
        for point in ((segment.x0_q16, segment.y0_q16),
                      (segment.x1_q16, segment.y1_q16)):
            counts[point] = counts.get(point, 0) + 1
    return sorted(point for point, count in counts.items() if count >= 2)


def _inside_join_disk(sample_x: int, sample_y: int, center: tuple[int, int], radius: int) -> bool:
    dx, dy = sample_x - center[0], sample_y - center[1]
    return dx * dx + dy * dy <= radius * radius


def _join_bounds(center: tuple[int, int], style: LineStyle,
                 width: int, height: int) -> tuple[int, int, int, int]:
    padding = (style.width_q16 + Q - 1) // Q + 2
    x, y = center
    return (max(0, x // Q - padding), min(width - 1, x // Q + padding),
            max(0, y // Q - padding), min(height - 1, y // Q + padding))


def rasterize_support(
    segments: list[Segment],
    style: LineStyle,
    width: int = CANVAS_SIZE,
    height: int = CANVAS_SIZE,
) -> set[int]:
    """Return pixels with non-zero foreground coverage.

    This uses the exact sampling rule used by :func:`rasterize_gray`, but only
    retains a sparse support set.  It is used by the collision acceptance gate
    and avoids allocating a full image for every individual lane.
    """
    sample_count = AA_SAMPLES.get(style.antialias_profile)
    if sample_count is None:
        raise ValueError(f"unsupported AA profile: {style.antialias_profile}")
    if not (1 <= style.width_q16 <= 64 * Q):
        raise ValueError("line width outside supported range")
    if style.join != "round":
        raise ValueError(f"unsupported join: {style.join}")
    offsets = [((2 * index + 1) * Q) // (2 * sample_count) for index in range(sample_count)]
    support: set[int] = set()
    for segment in segments:
        min_x, max_x, min_y, max_y = _pixel_bounds(segment, style, width, height)
        for pixel_y in range(min_y, max_y + 1):
            base_y = pixel_y * Q
            row = pixel_y * width
            for pixel_x in range(min_x, max_x + 1):
                base_x = pixel_x * Q
                covered = False
                for offset_y in offsets:
                    sample_y = base_y + offset_y
                    for offset_x in offsets:
                        if _inside_segment(base_x + offset_x, sample_y, segment, style):
                            covered = True
                            break
                    if covered:
                        break
                if covered:
                    support.add(row + pixel_x)
    radius = style.width_q16 // 2
    for center in _shared_join_points(segments):
        min_x, max_x, min_y, max_y = _join_bounds(center, style, width, height)
        for pixel_y in range(min_y, max_y + 1):
            base_y = pixel_y * Q
            row = pixel_y * width
            for pixel_x in range(min_x, max_x + 1):
                base_x = pixel_x * Q
                if any(_inside_join_disk(base_x + offset_x, base_y + offset_y,
                                         center, radius)
                       for offset_y in offsets for offset_x in offsets):
                    support.add(row + pixel_x)
    return support


def rasterize_gray(
    segments: list[Segment],
    style: LineStyle,
    width: int = CANVAS_SIZE,
    height: int = CANVAS_SIZE,
) -> bytes:
    """Rasterize the union of line segments to black-on-white grayscale bytes."""
    sample_count = AA_SAMPLES.get(style.antialias_profile)
    if sample_count is None:
        raise ValueError(f"unsupported AA profile: {style.antialias_profile}")
    if not (1 <= style.width_q16 <= 64 * Q):
        raise ValueError("line width outside supported range")
    if style.join != "round":
        raise ValueError(f"unsupported join: {style.join}")
    total_samples = sample_count * sample_count
    masks = array("Q", [0]) * (width * height)
    offsets = [((2 * index + 1) * Q) // (2 * sample_count) for index in range(sample_count)]

    for segment in segments:
        min_x, max_x, min_y, max_y = _pixel_bounds(segment, style, width, height)
        for pixel_y in range(min_y, max_y + 1):
            row = pixel_y * width
            base_y = pixel_y * Q
            for pixel_x in range(min_x, max_x + 1):
                index = row + pixel_x
                mask = masks[index]
                base_x = pixel_x * Q
                bit = 1
                for offset_y in offsets:
                    sample_y = base_y + offset_y
                    for offset_x in offsets:
                        if not mask & bit and _inside_segment(
                            base_x + offset_x, sample_y, segment, style
                        ):
                            mask |= bit
                        bit <<= 1
                masks[index] = mask

    radius = style.width_q16 // 2
    for center in _shared_join_points(segments):
        min_x, max_x, min_y, max_y = _join_bounds(center, style, width, height)
        for pixel_y in range(min_y, max_y + 1):
            row = pixel_y * width
            base_y = pixel_y * Q
            for pixel_x in range(min_x, max_x + 1):
                index = row + pixel_x
                mask = masks[index]
                base_x = pixel_x * Q
                bit = 1
                for offset_y in offsets:
                    for offset_x in offsets:
                        if not mask & bit and _inside_join_disk(
                            base_x + offset_x, base_y + offset_y, center, radius
                        ):
                            mask |= bit
                        bit <<= 1
                masks[index] = mask

    gray = bytearray(width * height)
    target = 0
    for mask in masks:
        covered = mask.bit_count()
        value = 255 - round_half_even(255 * covered, total_samples)
        gray[target] = value
        target += 1
    return bytes(gray)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def encode_canonical_png(gray: bytes, width: int = CANVAS_SIZE, height: int = CANVAS_SIZE) -> bytes:
    expected = width * height
    if len(gray) != expected:
        raise ValueError(f"expected {expected} grayscale bytes, got {len(gray)}")
    stride = width
    scanlines = bytearray((stride + 1) * height)
    for row in range(height):
        output_start = row * (stride + 1)
        input_start = row * stride
        scanlines[output_start] = 0
        scanlines[output_start + 1:output_start + 1 + stride] = gray[input_start:input_start + stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    compressed = zlib.compress(bytes(scanlines), level=9)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) +
            _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b""))


def rasterize_png(
    segments: list[Segment],
    style: LineStyle,
    width: int = CANVAS_SIZE,
    height: int = CANVAS_SIZE,
) -> tuple[bytes, str]:
    gray = rasterize_gray(segments, style, width, height)
    png = encode_canonical_png(gray, width, height)
    return png, hashlib.sha256(gray).hexdigest()


def png_chunk_types(png: bytes) -> list[str]:
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid PNG signature")
    cursor = 8
    chunks: list[str] = []
    while cursor < len(png):
        if cursor + 12 > len(png):
            raise ValueError("truncated PNG chunk")
        size = struct.unpack(">I", png[cursor:cursor + 4])[0]
        kind = png[cursor + 4:cursor + 8]
        end = cursor + 12 + size
        if end > len(png):
            raise ValueError("truncated PNG payload")
        expected_crc = struct.unpack(">I", png[end - 4:end])[0]
        actual_crc = binascii.crc32(kind)
        actual_crc = binascii.crc32(png[cursor + 8:end - 4], actual_crc) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ValueError("PNG CRC mismatch")
        chunks.append(kind.decode("ascii"))
        cursor = end
        if kind == b"IEND":
            break
    if cursor != len(png):
        raise ValueError("trailing PNG bytes")
    return chunks
