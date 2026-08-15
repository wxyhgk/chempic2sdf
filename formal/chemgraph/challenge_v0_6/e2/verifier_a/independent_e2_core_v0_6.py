#!/usr/bin/env python3
"""Independent stdlib implementation of the frozen v0.6 E2 semantics."""

from __future__ import annotations

from dataclasses import dataclass
import binascii
import hashlib
import itertools
import json
import math
import struct
import zlib


Q16 = 1 << 16
RAW_SIZE = 768
PADDED_SIZE = 1024
BACKGROUND = 255
PARENT_CONTRACT_SHA256 = (
    "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65"
)
V0_6_CONTRACT_SHA256 = (
    "e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402"
)
PARENT_CARDINALITY = 36_088_416
CAPS = ("butt", "round", "square")
ANTIALIAS_PROFILES = ("binary_1x", "box_2x2", "box_4x4")
REFLECTIONS = ("none", "x", "y")
ROTATIONS = (
    ((1, 1), (0, 1)),
    ((3, 5), (4, 5)),
    ((5, 13), (12, 13)),
    ((8, 17), (15, 17)),
    ((0, 1), (1, 1)),
    ((-3, 5), (4, 5)),
    ((5, 13), (-12, 13)),
    ((-8, 17), (-15, 17)),
)
SCALES = ((1, 2), (5, 8), (3, 4), (1, 1), (5, 4), (3, 2), (2, 1))
WIDTHS = (2, 4, 6, 8)
E2_CAPS = ("butt", "square")
TRANSLATION_CHECKS = ((-32, -32), (-32, 32), (0, 0), (32, -32), (32, 32))


class IndependentVerificationError(ValueError):
    """The producer artifact disagrees with independently reconstructed truth."""


@dataclass(frozen=True)
class Segment:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class SideComputation:
    tuple_record: dict[str, object]
    family: str
    interpretation_index: int
    raw_gray: bytes
    padded_gray: bytes
    canonical_gray: bytes
    canonical_png: bytes
    padded_certificate: dict[str, object]
    graph: dict[str, object]


@dataclass(frozen=True)
class PairComputation:
    cap: str
    width_px: int
    left: SideComputation
    right: SideComputation


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + suffix
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _div_round_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise IndependentVerificationError("nonpositive_transform_denominator")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = 2 * remainder
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return sign * quotient


def _transform_point(
    point: tuple[int, int],
    reflection: str,
    rotation_index: int,
    scale_index: int,
    center_px: int,
) -> tuple[int, int]:
    x, y = point
    if reflection == "x":
        y = -y
    elif reflection == "y":
        x = -x
    elif reflection != "none":
        raise IndependentVerificationError("reflection_out_of_domain")
    try:
        cosine, sine = ROTATIONS[rotation_index]
        scale = SCALES[scale_index]
    except IndexError as error:
        raise IndependentVerificationError("transform_index_out_of_domain") from error
    common = cosine[1] * sine[1]
    rotated_x = _div_round_signed(
        x * cosine[0] * sine[1] - y * sine[0] * cosine[1], common
    )
    rotated_y = _div_round_signed(
        x * sine[0] * cosine[1] + y * cosine[0] * sine[1], common
    )
    scaled_x = _div_round_signed(rotated_x * scale[0], scale[1])
    scaled_y = _div_round_signed(rotated_y * scale[0], scale[1])
    center = center_px * Q16
    return scaled_x + center, scaled_y + center


def _source_segments(record: dict[str, object]) -> tuple[str, tuple[Segment, ...]]:
    half = _require_int(record, "half_length")
    motif = _require_int(record, "motif_ordinal")
    if motif == 11:
        gap = 15
        points = ((-half, 0), (-gap, 0), (gap, -half), (gap, half))
        return "s8", (
            Segment(*(coordinate * Q16 for point in points[:2] for coordinate in point)),
            Segment(*(coordinate * Q16 for point in points[2:] for coordinate in point)),
        )
    if motif == 48:
        return "c4", (
            Segment(-half * Q16, -80 * Q16, half * Q16, -80 * Q16),
            Segment(0, -35 * Q16, 0, 35 * Q16),
        )
    raise IndependentVerificationError("motif_outside_frozen_e2_plan")


def _transformed_segments(record: dict[str, object], canvas_size: int) -> tuple[str, list[Segment]]:
    family, source = _source_segments(record)
    reflection = _require_str(record, "reflection")
    rotation = _require_int(record, "rotation_index")
    scale = _require_int(record, "scale_index")
    center = canvas_size // 2
    transformed = []
    for segment in source:
        start = _transform_point(
            (segment.x0, segment.y0), reflection, rotation, scale, center
        )
        end = _transform_point(
            (segment.x1, segment.y1), reflection, rotation, scale, center
        )
        transformed.append(Segment(start[0], start[1], end[0], end[1]))
    return family, transformed


def _inside_segment(sample_x: int, sample_y: int, segment: Segment, width_q16: int, cap: str) -> bool:
    vx = segment.x1 - segment.x0
    vy = segment.y1 - segment.y0
    wx = sample_x - segment.x0
    wy = sample_y - segment.y0
    length_sq = vx * vx + vy * vy
    if length_sq == 0:
        return False
    radius = width_q16 // 2
    radius_sq = radius * radius
    projection = wx * vx + wy * vy
    cross = wx * vy - wy * vx
    if cap == "butt":
        return 0 <= projection <= length_sq and cross * cross <= radius_sq * length_sq
    if cap == "square":
        extension = radius * math.isqrt(length_sq)
        return (
            -extension <= projection <= length_sq + extension
            and cross * cross <= radius_sq * length_sq
        )
    if cap == "round":
        if projection < 0:
            return wx * wx + wy * wy <= radius_sq
        if projection > length_sq:
            dx = sample_x - segment.x1
            dy = sample_y - segment.y1
            return dx * dx + dy * dy <= radius_sq
        return cross * cross <= radius_sq * length_sq
    raise IndependentVerificationError("cap_out_of_domain")


def _pixel_bounds(segment: Segment, width_q16: int, size: int) -> tuple[int, int, int, int]:
    padding = (width_q16 + Q16 - 1) // Q16 + 2
    return (
        max(0, min(segment.x0, segment.x1) // Q16 - padding),
        min(size - 1, max(segment.x0, segment.x1) // Q16 + padding),
        max(0, min(segment.y0, segment.y1) // Q16 - padding),
        min(size - 1, max(segment.y0, segment.y1) // Q16 + padding),
    )


def _shared_endpoints(segments: list[Segment]) -> list[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for segment in segments:
        for point in ((segment.x0, segment.y0), (segment.x1, segment.y1)):
            counts[point] = counts.get(point, 0) + 1
    return sorted(point for point, count in counts.items() if count >= 2)


def _rasterize_binary(segments: list[Segment], width_px: int, cap: str, size: int) -> bytes:
    if width_px not in range(1, 10) or cap not in CAPS:
        raise IndependentVerificationError("style_out_of_parent_domain")
    width_q16 = width_px * Q16
    gray = bytearray([BACKGROUND]) * (size * size)
    offset = Q16 // 2
    for segment in segments:
        min_x, max_x, min_y, max_y = _pixel_bounds(segment, width_q16, size)
        for pixel_y in range(min_y, max_y + 1):
            sample_y = pixel_y * Q16 + offset
            row = pixel_y * size
            for pixel_x in range(min_x, max_x + 1):
                if _inside_segment(
                    pixel_x * Q16 + offset, sample_y, segment, width_q16, cap
                ):
                    gray[row + pixel_x] = 0
    radius = width_q16 // 2
    padding = (width_q16 + Q16 - 1) // Q16 + 2
    for center_x, center_y in _shared_endpoints(segments):
        min_x = max(0, center_x // Q16 - padding)
        max_x = min(size - 1, center_x // Q16 + padding)
        min_y = max(0, center_y // Q16 - padding)
        max_y = min(size - 1, center_y // Q16 + padding)
        for pixel_y in range(min_y, max_y + 1):
            dy = pixel_y * Q16 + offset - center_y
            row = pixel_y * size
            for pixel_x in range(min_x, max_x + 1):
                dx = pixel_x * Q16 + offset - center_x
                if dx * dx + dy * dy <= radius * radius:
                    gray[row + pixel_x] = 0
    return bytes(gray)


def _foreground_bbox(gray: bytes, size: int) -> tuple[int, int, int, int]:
    if len(gray) != size * size:
        raise IndependentVerificationError("gray_length_mismatch")
    min_x, min_y, max_x, max_y = size, size, -1, -1
    for y in range(size):
        row = gray[y * size:(y + 1) * size]
        left_trimmed = row.lstrip(b"\xff")
        if not left_trimmed:
            continue
        left = size - len(left_trimmed)
        right = len(row.rstrip(b"\xff")) - 1
        min_x = min(min_x, left)
        min_y = min(min_y, y)
        max_x = max(max_x, right)
        max_y = y
    if max_x < 0:
        raise IndependentVerificationError("empty_foreground")
    return min_x, min_y, max_x, max_y


def _margins(gray: bytes, size: int) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = _foreground_bbox(gray, size)
    return min_x, min_y, size - 1 - max_x, size - 1 - max_y


def preprocess_gray(gray: bytes) -> bytes:
    min_x, min_y, max_x, max_y = _foreground_bbox(gray, RAW_SIZE)
    if min(min_x, min_y, RAW_SIZE - 1 - max_x, RAW_SIZE - 1 - max_y) < 1:
        raise IndependentVerificationError("raw_white_border_below_one")
    crop_width = max_x - min_x + 1
    crop_height = max_y - min_y + 1
    paste_left = (RAW_SIZE - crop_width) // 2
    paste_top = (RAW_SIZE - crop_height) // 2
    output = bytearray([BACKGROUND]) * (RAW_SIZE * RAW_SIZE)
    for source_y in range(min_y, max_y + 1):
        source = source_y * RAW_SIZE + min_x
        target_y = paste_top + source_y - min_y
        target = target_y * RAW_SIZE + paste_left
        output[target:target + crop_width] = gray[source:source + crop_width]
    return bytes(output)


def _integer_shift(gray: bytes, dx: int, dy: int) -> bytes:
    if len(gray) != RAW_SIZE * RAW_SIZE or not (-32 <= dx <= 32 and -32 <= dy <= 32):
        raise IndependentVerificationError("translation_input_invalid")
    output = bytearray([BACKGROUND]) * len(gray)
    source_x0 = max(0, -dx)
    source_x1 = min(RAW_SIZE, RAW_SIZE - dx)
    width = source_x1 - source_x0
    for source_y in range(max(0, -dy), min(RAW_SIZE, RAW_SIZE - dy)):
        source = source_y * RAW_SIZE + source_x0
        target = (source_y + dy) * RAW_SIZE + source_x0 + dx
        output[target:target + width] = gray[source:source + width]
    return bytes(output)


def _verify_translation_preprocess(raw: bytes, canonical: bytes) -> None:
    margins = _margins(raw, RAW_SIZE)
    if min(margins) < 33:
        raise IndependentVerificationError("centered_white_border_below_33")
    for dx, dy in TRANSLATION_CHECKS:
        if preprocess_gray(_integer_shift(raw, dx, dy)) != canonical:
            raise IndependentVerificationError(f"translation_preprocess_changed:{dx}:{dy}")


def _verify_padded(raw: bytes, padded: bytes) -> dict[str, object]:
    if len(raw) != RAW_SIZE * RAW_SIZE or len(padded) != PADDED_SIZE * PADDED_SIZE:
        raise IndependentVerificationError("padded_shape_mismatch")
    white_top = bytes([BACKGROUND]) * (128 * PADDED_SIZE)
    if padded[:128 * PADDED_SIZE] != white_top or padded[896 * PADDED_SIZE:] != white_top:
        raise IndependentVerificationError("padded_vertical_clipping")
    white_side = bytes([BACKGROUND]) * 128
    for y in range(RAW_SIZE):
        padded_row = padded[(y + 128) * PADDED_SIZE:(y + 129) * PADDED_SIZE]
        if padded_row[:128] != white_side or padded_row[896:] != white_side:
            raise IndependentVerificationError("padded_horizontal_clipping")
        raw_row = raw[y * RAW_SIZE:(y + 1) * RAW_SIZE]
        if padded_row[128:896] != raw_row:
            raise IndependentVerificationError("padded_embedded_raw_mismatch")
    return {
        "embedded_raw_full_bytes_equal": True,
        "foreground_bbox": list(_foreground_bbox(padded, PADDED_SIZE)),
        "offset": [128, 128],
        "outer_white": True,
        "passed": True,
    }


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(payload, binascii.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def encode_canonical_png(gray: bytes) -> bytes:
    if len(gray) != RAW_SIZE * RAW_SIZE:
        raise IndependentVerificationError("canonical_png_gray_length")
    scanlines = bytearray((RAW_SIZE + 1) * RAW_SIZE)
    for row in range(RAW_SIZE):
        source = row * RAW_SIZE
        target = row * (RAW_SIZE + 1)
        scanlines[target] = 0
        scanlines[target + 1:target + 1 + RAW_SIZE] = gray[source:source + RAW_SIZE]
    header = struct.pack(">IIBBBBB", RAW_SIZE, RAW_SIZE, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _canonical_graph(nodes: tuple[tuple[str, int], ...], edges: tuple[tuple[int, int, int], ...]) -> dict[str, object]:
    best: bytes | None = None
    best_value: dict[str, object] | None = None
    for ordering in itertools.permutations(range(len(nodes))):
        inverse = {old: new for new, old in enumerate(ordering)}
        value = {
            "edges": sorted(
                [
                    min(inverse[left], inverse[right]),
                    max(inverse[left], inverse[right]),
                    order,
                ]
                for left, right, order in edges
            ),
            "nodes": [[nodes[old][0], nodes[old][1]] for old in ordering],
        }
        encoded = canonical_json_bytes(value, newline=False)
        if best is None or encoded < best:
            best, best_value = encoded, value
    if best is None or best_value is None:
        raise IndependentVerificationError("empty_graph")
    return {
        "byte_length": len(best),
        "canonical_bytes_ascii": best.decode("ascii"),
        "edge_count": len(edges),
        "node_count": len(nodes),
        "sha256": sha256(best),
    }


def _selected_graph(family: str, interpretation_index: int) -> dict[str, object]:
    if interpretation_index != 0:
        raise IndependentVerificationError("unexpected_interpretation_index")
    if family == "s8":
        nodes = (("C", 0),) * 4
        edges = ((0, 1, 1), (2, 3, 1))
        decode_kind = "strict_graph"
    elif family == "c4":
        nodes = (("C", 0), ("C", 0), ("TEXT_STROKE", 0))
        edges = ((0, 1, 1),)
        decode_kind = "stroke_role_ambiguity"
    else:
        raise IndependentVerificationError("unknown_e2_family")
    graph = _canonical_graph(nodes, edges)
    graph["decode_kind"] = decode_kind
    return graph


def _axis_values(record: dict[str, object]) -> tuple[object, ...]:
    return (
        _require_int(record, "half_length"),
        _require_int(record, "motif_ordinal"),
        _require_int(record, "width_px"),
        _require_str(record, "cap"),
        _require_str(record, "antialias_profile"),
        _require_str(record, "reflection"),
        _require_int(record, "rotation_index"),
        _require_int(record, "scale_index"),
    )


def encode_tuple(record: dict[str, object]) -> int:
    axes = (
        tuple(range(60, 111)),
        tuple(range(52)),
        tuple(range(1, 10)),
        CAPS,
        ANTIALIAS_PROFILES,
        REFLECTIONS,
        tuple(range(8)),
        tuple(range(7)),
    )
    encoded = 0
    for value, allowed in zip(_axis_values(record), axes):
        try:
            index = allowed.index(value)
        except ValueError as error:
            raise IndependentVerificationError("tuple_axis_out_of_domain") from error
        encoded = encoded * len(allowed) + index
    if not 0 <= encoded < PARENT_CARDINALITY:
        raise IndependentVerificationError("tuple_id_out_of_parent_domain")
    if _require_int(record, "tuple_id") != encoded:
        raise IndependentVerificationError("tuple_id_mixed_radix_mismatch")
    return encoded


def _tuple_record(cap: str, width_px: int, side: str) -> dict[str, object]:
    if cap not in E2_CAPS or width_px not in WIDTHS:
        raise IndependentVerificationError("pair_outside_frozen_plan")
    if side == "left":
        record: dict[str, object] = {
            "tuple_id": 0,
            "half_length": 62,
            "motif_ordinal": 11,
            "width_px": width_px,
            "cap": cap,
            "antialias_profile": "binary_1x",
            "reflection": "none",
            "rotation_index": 0,
            "scale_index": 2,
        }
    elif side == "right":
        record = {
            "tuple_id": 0,
            "half_length": 93,
            "motif_ordinal": 48,
            "width_px": width_px,
            "cap": cap,
            "antialias_profile": "binary_1x",
            "reflection": "none",
            "rotation_index": 4,
            "scale_index": 0,
        }
    else:
        raise IndependentVerificationError("invalid_pair_side")
    record["tuple_id"] = encode_tuple({**record, "tuple_id": _encode_axes_without_id(record)})
    return record


def _encode_axes_without_id(record: dict[str, object]) -> int:
    axes = (
        tuple(range(60, 111)), tuple(range(52)), tuple(range(1, 10)), CAPS,
        ANTIALIAS_PROFILES, REFLECTIONS, tuple(range(8)), tuple(range(7)),
    )
    encoded = 0
    for value, allowed in zip(_axis_values(record), axes):
        encoded = encoded * len(allowed) + allowed.index(value)
    return encoded


def _compute_side(record: dict[str, object], interpretation_index: int) -> SideComputation:
    encode_tuple(record)
    family, raw_segments = _transformed_segments(record, RAW_SIZE)
    padded_family, padded_segments = _transformed_segments(record, PADDED_SIZE)
    if family != padded_family:
        raise IndependentVerificationError("family_changed_by_canvas")
    width_px = _require_int(record, "width_px")
    cap = _require_str(record, "cap")
    raw = _rasterize_binary(raw_segments, width_px, cap, RAW_SIZE)
    padded = _rasterize_binary(padded_segments, width_px, cap, PADDED_SIZE)
    certificate = _verify_padded(raw, padded)
    canonical = preprocess_gray(raw)
    _verify_translation_preprocess(raw, canonical)
    png = encode_canonical_png(canonical)
    return SideComputation(
        tuple_record=record,
        family=family,
        interpretation_index=interpretation_index,
        raw_gray=raw,
        padded_gray=padded,
        canonical_gray=canonical,
        canonical_png=png,
        padded_certificate=certificate,
        graph=_selected_graph(family, interpretation_index),
    )


def compute_pair(cap: str, width_px: int) -> PairComputation:
    left = _compute_side(_tuple_record(cap, width_px, "left"), 0)
    right = _compute_side(_tuple_record(cap, width_px, "right"), 0)
    if left.tuple_record["tuple_id"] == right.tuple_record["tuple_id"]:
        raise IndependentVerificationError("cross_tuple_ids_equal")
    if left.raw_gray == right.raw_gray:
        raise IndependentVerificationError("raw_gray_unexpectedly_equal")
    if left.canonical_gray != right.canonical_gray:
        raise IndependentVerificationError("canonical_gray_collision_missing")
    if left.canonical_png != right.canonical_png:
        raise IndependentVerificationError("canonical_png_collision_missing")
    if left.graph["canonical_bytes_ascii"] == right.graph["canonical_bytes_ascii"]:
        raise IndependentVerificationError("selected_graphs_equivalent")
    return PairComputation(cap=cap, width_px=width_px, left=left, right=right)


def all_expected_pairs() -> list[PairComputation]:
    return [compute_pair(cap, width) for cap in E2_CAPS for width in WIDTHS]


def _require_int(value: dict[str, object], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise IndependentVerificationError(f"expected_integer:{key}")
    return result


def _require_str(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise IndependentVerificationError(f"expected_string:{key}")
    return result
