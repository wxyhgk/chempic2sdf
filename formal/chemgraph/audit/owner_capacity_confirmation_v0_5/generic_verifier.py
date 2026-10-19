"""Independent verifier for owner-capacity L artifacts.

This module is intentionally stdlib-only.  It does not import any source
builder, candidate constructor, shared candidate helpers, or the fragment and
support compilers used by the producer.
"""

from __future__ import annotations

import argparse
import binascii
import bisect
import hashlib
import json
import struct
import zlib
from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


WIDTH = 768
HEIGHT = 768
GEOMETRY_SCHEMA = "chemgraph-fragment-root-v0.4-research"
SUPPORT_SCHEMA = "chemgraph-fragment-support-partition-v0.4-research"
OBSERVATION_SCHEMA = "owner-candidate-r0-observation-v0.5-research"
CANDIDATE_SCHEMA = "chemgraph-owner-candidate-domain-v0.5-r0-research"
WITNESS_SCHEMA = "owner-capacity-feasible-world-v0.5"
PROTOCOL_COMMIT = "5a8bdd9"
FROZEN_PLAN_SHA256 = "06366e83731f1ca9dee116f995da14dd5a7e64c68f45e98f2a847c0e423bcd89"
FROZEN_MANIFEST_SHA256 = "a06489212869743c414d8f87729c4ff6d3836f7bec3b7c5ef8af908ad7410118"
FROZEN_EXCLUSION_SHA256 = "48e7918c9f940fc38e924764ea8e60d8d1b987cd638bdc931ee6acbc32b55a7d"

LIMIT_KEYS = {
    "max_adjacency_edges",
    "max_ambiguous_variables",
    "max_atomic_fragments",
    "max_carriers",
    "max_constraints",
    "max_domain_classes",
    "max_intersections",
    "max_output_bytes",
    "max_owner_options",
    "max_pixel_variables",
    "max_raw_assignments",
    "max_total_rle_runs",
}
END_TO_END_AXES = {
    "max_carriers",
    "max_pixel_variables",
    "max_total_rle_runs",
}
CANDIDATE_KEYS = {
    "adjacency_edges",
    "constraints",
    "counts",
    "domain_classes",
    "evidence_memberships",
    "limits",
    "metrics",
    "owner_options",
    "pixel_variables",
    "residual_reasons",
    "schema_version",
    "section_hashes",
    "source_binding",
    "stage_trace",
    "status",
    "training_authorized",
}
SECTION_ORDER = [
    "source_binding",
    "limits",
    "counts",
    "owner_options",
    "evidence_memberships",
    "domain_classes",
    "pixel_variables",
    "adjacency_edges",
    "constraints",
    "residual_reasons",
    "metrics",
]
STAGE_ORDER = [
    "source_binding",
    "owner_options",
    "evidence_memberships",
    "domain_classes",
    "pixel_variables",
    "adjacency_edges",
    "constraints",
    "metrics",
]


class GenericCapacityVerificationError(ValueError):
    """Closed verification failure."""


def _fail(code: str) -> None:
    raise GenericCapacityVerificationError(code)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_int(value: Any, low: int, high: int, code: str) -> int:
    if not _is_int(value) or not low <= value <= high:
        _fail(code)
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON_DUPLICATE_KEY")
        result[key] = value
    return result


def load_canonical_json(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GenericCapacityVerificationError(f"UTF8_INVALID:{label}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=lambda _: _fail("JSON_NONFINITE_NUMBER"),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GenericCapacityVerificationError(f"JSON_INVALID:{label}") from error
    if raw != canonical_bytes(value):
        _fail(f"JSON_NONCANONICAL:{label}")
    return value


def _framed_items_sha256(items: list[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(len(items).to_bytes(4, "big"))
    for item in items:
        encoded = canonical_bytes(item)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> list[int]:
    if numerator < 0 or denominator <= 0:
        _fail("RATIO_INVALID")
    value = Fraction(numerator, denominator)
    return [value.numerator, value.denominator]


def _decode_runs(
    runs: Any,
    expected_count: Any | None = None,
    *,
    allow_empty: bool = False,
) -> set[tuple[int, int]]:
    if not isinstance(runs, list) or (not allow_empty and not runs):
        _fail("RLE_RUNS_INVALID")
    pixels: set[tuple[int, int]] = set()
    previous: tuple[int, int, int] | None = None
    for run in runs:
        if not (
            isinstance(run, list)
            and len(run) == 3
            and all(_is_int(item) for item in run)
        ):
            _fail("RLE_RUN_INVALID")
        row, start, length = run
        if not (0 <= row < HEIGHT and 0 <= start < WIDTH and 1 <= length <= WIDTH - start):
            _fail("RLE_RUN_OUT_OF_RANGE")
        if previous is not None:
            old_row, old_start, old_length = previous
            if (row, start) <= (old_row, old_start):
                _fail("RLE_ORDER_NONCANONICAL")
            if row == old_row and start <= old_start + old_length:
                _fail("RLE_NOT_MAXIMAL")
        for column in range(start, start + length):
            pixel = row, column
            if pixel in pixels:
                _fail("RLE_PIXEL_DUPLICATE")
            pixels.add(pixel)
        previous = row, start, length
    if expected_count is not None:
        if not _is_int(expected_count) or len(pixels) != expected_count:
            _fail("RLE_PIXEL_COUNT_MISMATCH")
    return pixels


def _encode_runs(pixels: Iterable[tuple[int, int]]) -> list[list[int]]:
    ordered = sorted(set(pixels))
    if not ordered:
        return []
    row, start = ordered[0]
    end = start + 1
    result: list[list[int]] = []
    for next_row, column in ordered[1:]:
        if next_row == row and column == end:
            end += 1
            continue
        result.append([row, start, end - start])
        row, start, end = next_row, column, column + 1
    result.append([row, start, end - start])
    return result


def _rle_sha256(runs: list[list[int]]) -> str:
    digest = hashlib.sha256()
    digest.update(len(runs).to_bytes(4, "big"))
    for row, start, length in runs:
        digest.update(row.to_bytes(4, "big"))
        digest.update(start.to_bytes(4, "big"))
        digest.update(length.to_bytes(4, "big"))
    return digest.hexdigest()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body))


def _support_union(parent_supports: list[dict[str, Any]]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for support in parent_supports:
        result.update(_decode_runs(support["support_runs"], support["support_pixel_count"]))
    return result


def render_support_png(parent_supports: list[dict[str, Any]]) -> bytes:
    foreground = _support_union(parent_supports)
    rows = bytearray()
    for row in range(HEIGHT):
        rows.append(0)
        rows.extend(0 if (row, column) in foreground else 255 for column in range(WIDTH))
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + b"".join(
        (
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def decode_support_png(raw: bytes) -> set[tuple[int, int]]:
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        _fail("PNG_SIGNATURE_INVALID")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(raw):
        if offset + 12 > len(raw):
            _fail("PNG_TRUNCATED")
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        kind = raw[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(raw):
            _fail("PNG_TRUNCATED")
        payload = raw[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", raw[offset + 8 + length : end])[0]
        if crc != binascii.crc32(kind + payload):
            _fail("PNG_CRC_INVALID")
        chunks.append((kind, payload))
        offset = end
    if [kind for kind, _ in chunks] != [b"IHDR", b"IDAT", b"IEND"]:
        _fail("PNG_CHUNK_SEQUENCE_INVALID")
    if chunks[2][1] != b"":
        _fail("PNG_IEND_INVALID")
    if chunks[0][1] != struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0):
        _fail("PNG_IHDR_INVALID")
    try:
        scanlines = zlib.decompress(chunks[1][1])
    except zlib.error as error:
        raise GenericCapacityVerificationError("PNG_DEFLATE_INVALID") from error
    if len(scanlines) != HEIGHT * (WIDTH + 1):
        _fail("PNG_SCANLINE_LENGTH_INVALID")
    foreground: set[tuple[int, int]] = set()
    for row in range(HEIGHT):
        begin = row * (WIDTH + 1)
        if scanlines[begin] != 0:
            _fail("PNG_FILTER_INVALID")
        pixels = scanlines[begin + 1 : begin + 1 + WIDTH]
        for column, value in enumerate(pixels):
            if value not in (0, 255):
                _fail("PNG_PIXEL_VALUE_INVALID")
            if value == 0:
                foreground.add((row, column))
    return foreground


def _point(value: Any) -> tuple[int, int]:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_int(item) and 0 <= item <= 12_288 for item in value)
    ):
        _fail("GEOMETRY_POINT_INVALID")
    return value[0], value[1]


def _sub(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] - right[0], left[1] - right[1]


def _cross(left: tuple[int, int], right: tuple[int, int]) -> int:
    return left[0] * right[1] - left[1] * right[0]


def _rat(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _rat_point(
    begin: tuple[int, int], direction: tuple[int, int], parameter: Fraction
) -> list[list[int]]:
    return [
        _rat(Fraction(begin[0]) + parameter * direction[0]),
        _rat(Fraction(begin[1]) + parameter * direction[1]),
    ]


def _positive_collinear_overlap(
    left_begin: tuple[int, int],
    left_end: tuple[int, int],
    right_begin: tuple[int, int],
    right_end: tuple[int, int],
) -> bool:
    axis = 0 if abs(left_end[0] - left_begin[0]) >= abs(left_end[1] - left_begin[1]) else 1
    lower = max(
        min(left_begin[axis], left_end[axis]), min(right_begin[axis], right_end[axis])
    )
    upper = min(
        max(left_begin[axis], left_end[axis]), max(right_begin[axis], right_end[axis])
    )
    return lower < upper


def _canonicalize_observation_source(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict) or set(observation) != {
        "geometry_root",
        "parent_supports",
        "png_sha256",
        "renderer_sha256",
        "schema_version",
        "support_input_sha256",
        "training_authorized",
    }:
        _fail("OBSERVATION_NOT_CLOSED")
    if observation["schema_version"] != OBSERVATION_SCHEMA:
        _fail("OBSERVATION_SCHEMA_INVALID")
    if observation["training_authorized"] is not False:
        _fail("OBSERVATION_TRAINING_AUTHORIZED")
    for field in ("png_sha256", "renderer_sha256", "support_input_sha256"):
        if not _is_sha256(observation[field]):
            _fail(f"OBSERVATION_HASH_INVALID:{field}")
    geometry_root = observation["geometry_root"]
    supports = observation["parent_supports"]
    if not isinstance(geometry_root, dict) or set(geometry_root) != {
        "base_strokes",
        "schema_version",
        "training_authorized",
    }:
        _fail("GEOMETRY_ROOT_NOT_CLOSED")
    if geometry_root["schema_version"] != GEOMETRY_SCHEMA:
        _fail("GEOMETRY_SCHEMA_INVALID")
    if geometry_root["training_authorized"] is not False:
        _fail("GEOMETRY_TRAINING_AUTHORIZED")
    strokes = geometry_root["base_strokes"]
    if not isinstance(strokes, list) or not strokes or not isinstance(supports, list):
        _fail("SOURCE_FIBERS_INVALID")
    support_by_id: dict[int, dict[str, Any]] = {}
    for support in supports:
        if not isinstance(support, dict) or set(support) != {
            "base_stroke_id",
            "support_pixel_count",
            "support_rle_sha256",
            "support_runs",
        }:
            _fail("PARENT_SUPPORT_NOT_CLOSED")
        old_id = support["base_stroke_id"]
        if not _is_int(old_id) or old_id in support_by_id:
            _fail("PARENT_SUPPORT_ID_INVALID")
        support_by_id[old_id] = support
    records: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    seen_ids: set[int] = set()
    seen_boundary_ids: set[int] = set()
    for stroke in strokes:
        if not isinstance(stroke, dict) or set(stroke) != {
            "boundary_member_ids",
            "id",
            "segment_q4",
        }:
            _fail("BASE_STROKE_NOT_CLOSED")
        old_id = stroke["id"]
        members = stroke["boundary_member_ids"]
        if not _is_int(old_id) or old_id in seen_ids:
            _fail("BASE_STROKE_ID_INVALID")
        if not (
            isinstance(members, list)
            and len(members) == 1
            and _is_int(members[0])
            and members[0] not in seen_boundary_ids
        ):
            _fail("BOUNDARY_MEMBER_INVALID")
        segment = stroke["segment_q4"]
        if not isinstance(segment, list) or len(segment) != 2:
            _fail("SEGMENT_INVALID")
        begin, end = sorted((_point(segment[0]), _point(segment[1])))
        if begin == end or old_id not in support_by_id:
            _fail("SEGMENT_OR_SUPPORT_INVALID")
        records.append((begin, end, old_id))
        seen_ids.add(old_id)
        seen_boundary_ids.add(members[0])
    if seen_ids != set(support_by_id):
        _fail("PARENT_SUPPORT_FIBER_INCOMPLETE")
    records.sort()
    if len({(begin, end) for begin, end, _ in records}) != len(records):
        _fail("DUPLICATE_SEGMENT")
    normalized_strokes: list[dict[str, Any]] = []
    normalized_supports: list[dict[str, Any]] = []
    for new_id, (begin, end, old_id) in enumerate(records):
        support = support_by_id[old_id]
        normalized_strokes.append(
            {
                "boundary_member_ids": [new_id],
                "id": new_id,
                "segment_q4": [list(begin), list(end)],
            }
        )
        normalized_supports.append(
            {
                "base_stroke_id": new_id,
                "support_pixel_count": support["support_pixel_count"],
                "support_rle_sha256": support["support_rle_sha256"],
                "support_runs": support["support_runs"],
            }
        )
    return {
        "geometry_root": {
            "base_strokes": normalized_strokes,
            "schema_version": GEOMETRY_SCHEMA,
            "training_authorized": False,
        },
        "parent_supports": normalized_supports,
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }


def _validate_canonical_root(root: Any) -> list[dict[str, Any]]:
    if not isinstance(root, dict) or set(root) != {
        "base_strokes",
        "schema_version",
        "training_authorized",
    }:
        _fail("ROOT_NOT_CLOSED")
    if root["schema_version"] != GEOMETRY_SCHEMA or root["training_authorized"] is not False:
        _fail("ROOT_HEADER_INVALID")
    strokes = root["base_strokes"]
    if not isinstance(strokes, list) or not strokes or len(strokes) > 128:
        _fail("BASE_STROKES_INVALID")
    result: list[dict[str, Any]] = []
    keys: list[tuple[Any, ...]] = []
    boundary_ids: set[int] = set()
    for index, stroke in enumerate(strokes):
        if not isinstance(stroke, dict) or set(stroke) != {
            "boundary_member_ids",
            "id",
            "segment_q4",
        }:
            _fail("BASE_STROKE_NOT_CLOSED")
        if stroke["id"] != index or not _is_int(stroke["id"]):
            _fail("BASE_STROKE_ID_NONCANONICAL")
        members = stroke["boundary_member_ids"]
        if not (
            isinstance(members, list)
            and 1 <= len(members) <= 2
            and all(_is_int(item) and 0 <= item <= 127 for item in members)
            and members == sorted(set(members))
            and not boundary_ids.intersection(members)
        ):
            _fail("BOUNDARY_MEMBERS_NONCANONICAL")
        boundary_ids.update(members)
        segment = stroke["segment_q4"]
        if not isinstance(segment, list) or len(segment) != 2:
            _fail("SEGMENT_INVALID")
        begin, end = _point(segment[0]), _point(segment[1])
        if not begin < end:
            _fail("SEGMENT_ENDPOINTS_NONCANONICAL")
        keys.append((begin, end, tuple(members)))
        result.append(
            {
                "id": index,
                "boundary_member_ids": members,
                "begin": begin,
                "end": end,
                "direction": _sub(end, begin),
            }
        )
    if keys != sorted(set(keys)):
        _fail("BASE_STROKE_ORDER_NONCANONICAL")
    return result


def compile_geometry(root: dict[str, Any]) -> dict[str, Any]:
    strokes = _validate_canonical_root(root)
    point_members: dict[tuple[Fraction, Fraction], set[int]] = defaultdict(set)
    for left_index, left in enumerate(strokes):
        p, r = left["begin"], left["direction"]
        for right in strokes[left_index + 1 :]:
            q, s = right["begin"], right["direction"]
            denominator = _cross(r, s)
            q_minus_p = _sub(q, p)
            if denominator == 0:
                if _cross(q_minus_p, r) == 0:
                    if _positive_collinear_overlap(p, left["end"], q, right["end"]):
                        _fail("COLLINEAR_OVERLAP_UNSUPPORTED")
                    shared = {p, left["end"]}.intersection({q, right["end"]})
                    if len(shared) == 1:
                        point = next(iter(shared))
                        point_members[(Fraction(point[0]), Fraction(point[1]))].update(
                            (left["id"], right["id"])
                        )
                continue
            t = Fraction(_cross(q_minus_p, s), denominator)
            u = Fraction(_cross(q_minus_p, r), denominator)
            if 0 <= t <= 1 and 0 <= u <= 1:
                point_members[
                    (Fraction(p[0]) + t * r[0], Fraction(p[1]) + t * r[1])
                ].update((left["id"], right["id"]))
    points = sorted(point_members)
    if len(points) > 128:
        _fail("INTERSECTIONS_OUTSIDE_UPSTREAM_CAPACITY")
    intersections: list[dict[str, Any]] = []
    point_to_region: dict[tuple[Fraction, Fraction], int] = {}
    for region_id, point in enumerate(points):
        point_to_region[point] = region_id
        intersections.append(
            {
                "id": region_id,
                "point_q4_exact": [_rat(point[0]), _rat(point[1])],
                "base_stroke_ids": sorted(point_members[point]),
            }
        )
    cuts: dict[int, list[tuple[Fraction, int]]] = defaultdict(list)
    for point, region_id in point_to_region.items():
        for stroke_id in sorted(point_members[point]):
            stroke = strokes[stroke_id]
            begin, direction = stroke["begin"], stroke["direction"]
            parameter = (
                (point[0] - begin[0]) / direction[0]
                if direction[0] != 0
                else (point[1] - begin[1]) / direction[1]
            )
            if 0 < parameter < 1:
                cuts[stroke_id].append((parameter, region_id))
    split_sites: list[dict[str, Any]] = []
    for stroke in strokes:
        for parameter, region_id in sorted(set(cuts[stroke["id"]])):
            split_sites.append(
                {
                    "id": len(split_sites),
                    "base_stroke_id": stroke["id"],
                    "intersection_id": region_id,
                    "t": _rat(parameter),
                    "point_q4_exact": _rat_point(
                        stroke["begin"], stroke["direction"], parameter
                    ),
                }
            )
    fragments: list[dict[str, Any]] = []
    for stroke in strokes:
        stroke_cuts = sorted(set(cuts[stroke["id"]]))
        begin_point = (Fraction(stroke["begin"][0]), Fraction(stroke["begin"][1]))
        end_point = (Fraction(stroke["end"][0]), Fraction(stroke["end"][1]))
        knots = [
            (Fraction(0), point_to_region.get(begin_point)),
            *stroke_cuts,
            (Fraction(1), point_to_region.get(end_point)),
        ]
        for (begin_t, begin_region), (end_t, end_region) in zip(knots, knots[1:]):
            fragments.append(
                {
                    "id": len(fragments),
                    "base_stroke_id": stroke["id"],
                    "boundary_member_ids": stroke["boundary_member_ids"],
                    "begin_t": _rat(begin_t),
                    "end_t": _rat(end_t),
                    "begin_point_q4_exact": _rat_point(
                        stroke["begin"], stroke["direction"], begin_t
                    ),
                    "end_point_q4_exact": _rat_point(
                        stroke["begin"], stroke["direction"], end_t
                    ),
                    "begin_intersection_id": begin_region,
                    "end_intersection_id": end_region,
                }
            )
    if len(fragments) > 512:
        _fail("ATOMIC_FRAGMENTS_OUTSIDE_UPSTREAM_CAPACITY")
    return {
        "schema_version": "chemgraph-fragment-root-compiled-v0.4-research",
        "training_authorized": False,
        "source_root_sha256": _sha256(canonical_bytes(root)),
        "base_stroke_count": len(strokes),
        "intersection_count": len(intersections),
        "split_site_count": len(split_sites),
        "atomic_fragment_count": len(fragments),
        "intersections": intersections,
        "split_sites": split_sites,
        "atomic_fragments": fragments,
    }


def _fraction(value: Any) -> Fraction:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_int(item) for item in value)
        and value[1] > 0
    ):
        _fail("RATIONAL_INVALID")
    result = Fraction(value[0], value[1])
    if value != [result.numerator, result.denominator]:
        _fail("RATIONAL_NONCANONICAL")
    return result


def compile_support_partition(support_input: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(support_input, dict) or set(support_input) != {
        "geometry_root",
        "parent_supports",
        "schema_version",
        "training_authorized",
    }:
        _fail("SUPPORT_INPUT_NOT_CLOSED")
    if support_input["schema_version"] != SUPPORT_SCHEMA:
        _fail("SUPPORT_SCHEMA_INVALID")
    if support_input["training_authorized"] is not False:
        _fail("SUPPORT_TRAINING_AUTHORIZED")
    root = support_input["geometry_root"]
    strokes = _validate_canonical_root(root)
    parent_supports = support_input["parent_supports"]
    if not isinstance(parent_supports, list) or len(parent_supports) != len(strokes):
        _fail("PARENT_SUPPORT_FIBER_INCOMPLETE")
    parent_pixels: list[set[tuple[int, int]]] = []
    total_memberships = 0
    total_runs = 0
    for index, support in enumerate(parent_supports):
        if not isinstance(support, dict) or set(support) != {
            "base_stroke_id",
            "support_pixel_count",
            "support_rle_sha256",
            "support_runs",
        }:
            _fail("PARENT_SUPPORT_NOT_CLOSED")
        if not _is_int(support["base_stroke_id"]) or support["base_stroke_id"] != index:
            _fail("PARENT_SUPPORT_ID_NONCANONICAL")
        pixels = _decode_runs(support["support_runs"], support["support_pixel_count"])
        if support["support_rle_sha256"] != _rle_sha256(support["support_runs"]):
            _fail("PARENT_SUPPORT_RLE_HASH_MISMATCH")
        total_memberships += len(pixels)
        total_runs += len(support["support_runs"])
        if len(support["support_runs"]) > 100_000:
            _fail("PARENT_SUPPORT_RUNS_OUTSIDE_UPSTREAM_CAPACITY")
        parent_pixels.append(pixels)
    if total_memberships > 2_000_000 or total_runs > 200_000:
        _fail("PARENT_SUPPORT_TOTAL_OUTSIDE_UPSTREAM_CAPACITY")
    geometry = compile_geometry(root)
    fragments_by_stroke: dict[int, list[dict[str, Any]]] = {
        stroke["id"]: [] for stroke in strokes
    }
    for fragment in geometry["atomic_fragments"]:
        fragments_by_stroke[fragment["base_stroke_id"]].append(fragment)
    child_pixels: dict[int, set[tuple[int, int]]] = {
        fragment["id"]: set() for fragment in geometry["atomic_fragments"]
    }
    for stroke in strokes:
        fragments = fragments_by_stroke[stroke["id"]]
        cut_values = [_fraction(fragment["end_t"]) for fragment in fragments[:-1]]
        begin, direction = stroke["begin"], stroke["direction"]
        norm_squared = direction[0] ** 2 + direction[1] ** 2
        for row, column in parent_pixels[stroke["id"]]:
            pixel_q4 = 16 * column + 8, 16 * row + 8
            offset = pixel_q4[0] - begin[0], pixel_q4[1] - begin[1]
            projected = Fraction(
                offset[0] * direction[0] + offset[1] * direction[1], norm_squared
            )
            parameter = min(Fraction(1), max(Fraction(0), projected))
            child_index = bisect.bisect_right(cut_values, parameter)
            child_pixels[fragments[child_index]["id"]].add((row, column))
    child_supports: list[dict[str, Any]] = []
    child_run_count = 0
    for fragment in geometry["atomic_fragments"]:
        pixels = child_pixels[fragment["id"]]
        if not pixels:
            _fail("EMPTY_FRAGMENT_SUPPORT")
        runs = _encode_runs(pixels)
        child_run_count += len(runs)
        child_supports.append(
            {
                "atomic_fragment_id": fragment["id"],
                "base_stroke_id": fragment["base_stroke_id"],
                "support_pixel_count": len(pixels),
                "support_rle_sha256": _rle_sha256(runs),
                "support_runs": runs,
            }
        )
    if child_run_count > 200_000:
        _fail("CHILD_SUPPORT_RUNS_OUTSIDE_UPSTREAM_CAPACITY")
    pixel_carriers: dict[tuple[int, int], list[int]] = {}
    for stroke_id, pixels in enumerate(parent_pixels):
        for pixel in pixels:
            pixel_carriers.setdefault(pixel, []).append(stroke_id)
    overlaps = [
        {"pixel": [row, column], "base_stroke_ids": carrier_ids}
        for (row, column), carrier_ids in sorted(pixel_carriers.items())
        if len(carrier_ids) > 1
    ]
    if len(overlaps) > 100_000:
        _fail("CROSS_CARRIER_OVERLAPS_OUTSIDE_UPSTREAM_CAPACITY")
    result = {
        "schema_version": "chemgraph-fragment-support-compiled-v0.4-research",
        "training_authorized": False,
        "source_input_sha256": _sha256(canonical_bytes(support_input)),
        "geometry_root_sha256": geometry["source_root_sha256"],
        "parent_support_count": len(parent_pixels),
        "parent_membership_count": total_memberships,
        "parent_support_run_count_total": total_runs,
        "atomic_fragment_count": len(child_supports),
        "child_membership_count": sum(item["support_pixel_count"] for item in child_supports),
        "child_support_run_count_total": child_run_count,
        "unique_parent_pixel_count": len(pixel_carriers),
        "cross_carrier_overlap_pixel_count": len(overlaps),
        "cross_carrier_overlaps": overlaps,
        "child_supports": child_supports,
    }
    if len(canonical_bytes(result)) > 67_108_864:
        _fail("SUPPORT_OUTPUT_BYTES_OUTSIDE_UPSTREAM_CAPACITY")
    return result


def _anchor_pixel(point: list[list[int]]) -> list[int]:
    x, y = _fraction(point[0]), _fraction(point[1])
    row = y.numerator // (16 * y.denominator)
    column = x.numerator // (16 * x.denominator)
    if not (0 <= row < HEIGHT and 0 <= column < WIDTH):
        _fail("INTERSECTION_ANCHOR_OUTSIDE_RASTER")
    return [row, column]


def rebuild_expected_artifact(
    observation: dict[str, Any], limits: dict[str, Any], constructor_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(limits) != LIMIT_KEYS or any(
        not _is_int(value) or value < 0 for value in limits.values()
    ):
        _fail("LIMITS_INVALID")
    if not _is_sha256(constructor_sha256):
        _fail("CONSTRUCTOR_SHA_INVALID")
    raw_support_input = {
        "geometry_root": observation.get("geometry_root"),
        "parent_supports": observation.get("parent_supports"),
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }
    if observation.get("support_input_sha256") != _sha256(canonical_bytes(raw_support_input)):
        _fail("SUPPORT_INPUT_HASH_MISMATCH")
    support_input = _canonicalize_observation_source(observation)
    geometry = compile_geometry(support_input["geometry_root"])
    support = compile_support_partition(support_input)
    derived_capacity = {
        "max_carriers": geometry["base_stroke_count"],
        "max_intersections": geometry["intersection_count"],
        "max_atomic_fragments": geometry["atomic_fragment_count"],
        "max_total_rle_runs": support["child_support_run_count_total"],
    }
    for name, value in derived_capacity.items():
        if value > limits[name]:
            _fail(f"L_SOURCE_CAPACITY_EXCEEDED:{name}")
    if geometry["intersection_count"] != 2:
        _fail("L_SOURCE_NOT_FIXED_R0")
    source_binding = {
        "compiled_fragment_root_sha256": _sha256(canonical_bytes(geometry)),
        "compiled_support_partition_sha256": _sha256(canonical_bytes(support)),
        "constructor_protocol_commit": PROTOCOL_COMMIT,
        "geometry_root_sha256": geometry["source_root_sha256"],
        "observation_sha256": _sha256(canonical_bytes(observation)),
        "png_sha256": observation["png_sha256"],
        "renderer_sha256": observation["renderer_sha256"],
        "support_input_sha256": observation["support_input_sha256"],
    }
    owner_options = [
        {
            "anchor_pixel": _anchor_pixel(intersection["point_q4_exact"]),
            "geometry_intersection_id": intersection["id"],
            "id": index,
            "incident_carrier_ids": intersection["base_stroke_ids"],
            "kind": "intersection",
        }
        for index, intersection in enumerate(geometry["intersections"])
    ]
    if len(owner_options) > limits["max_owner_options"]:
        _fail("L_SOURCE_CAPACITY_EXCEEDED:max_owner_options")
    fragment_at: dict[tuple[int, tuple[int, int]], int] = {}
    for child in support["child_supports"]:
        fragment_id = child["atomic_fragment_id"]
        carrier_id = child["base_stroke_id"]
        for pixel in _decode_runs(child["support_runs"], child["support_pixel_count"]):
            key = carrier_id, pixel
            if key in fragment_at:
                _fail("FRAGMENT_SUPPORT_NOT_PARTITION")
            fragment_at[key] = fragment_id
    overlaps = support["cross_carrier_overlaps"]
    if len(overlaps) > limits["max_pixel_variables"]:
        _fail("L_SOURCE_CAPACITY_EXCEEDED:max_pixel_variables")
    evidence: list[dict[str, Any]] = []
    domains: list[tuple[int, ...]] = []
    for variable_id, record in enumerate(overlaps):
        pixel = tuple(record["pixel"])
        memberships = []
        for carrier_id in record["base_stroke_ids"]:
            key = carrier_id, pixel
            if key not in fragment_at:
                _fail("FRAGMENT_MEMBERSHIP_NOT_UNIQUE")
            memberships.append(
                {"atomic_fragment_id": fragment_at[key], "carrier_id": carrier_id}
            )
        evidence.append(
            {
                "carrier_ids": record["base_stroke_ids"],
                "fragment_memberships": memberships,
                "id": variable_id,
                "pixel": record["pixel"],
            }
        )
        carrier_set = set(record["base_stroke_ids"])
        domain = tuple(
            option["id"]
            for option in owner_options
            if set(option["incident_carrier_ids"]).issubset(carrier_set)
        )
        if not domain:
            _fail("UNOWNED_OVERLAP")
        domains.append(domain)
    unique_domains = sorted(set(domains))
    if len(unique_domains) > limits["max_domain_classes"]:
        _fail("L_SOURCE_CAPACITY_EXCEEDED:max_domain_classes")
    domain_classes = [
        {"id": index, "owner_option_ids": list(domain)}
        for index, domain in enumerate(unique_domains)
    ]
    domain_to_class = {domain: index for index, domain in enumerate(unique_domains)}
    variables = [
        {
            "domain_class_id": domain_to_class[domains[index]],
            "evidence_membership_id": index,
            "id": index,
            "pixel": record["pixel"],
        }
        for index, record in enumerate(overlaps)
    ]
    pixel_to_variable = {
        tuple(record["pixel"]): index for index, record in enumerate(overlaps)
    }
    edges: list[dict[str, Any]] = []
    for left_id, record in enumerate(overlaps):
        row, column = record["pixel"]
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                right_id = pixel_to_variable.get((row + row_delta, column + column_delta))
                if right_id is not None and left_id < right_id:
                    edges.append(
                        {
                            "id": len(edges),
                            "left_variable_id": left_id,
                            "right_variable_id": right_id,
                        }
                    )
    if len(edges) > limits["max_adjacency_edges"]:
        _fail("L_SOURCE_CAPACITY_EXCEEDED:max_adjacency_edges")
    constraints: list[dict[str, Any]] = [
        {"id": 0, "kind": "exact_one", "variable_ids": list(range(len(variables)))}
    ]
    for option in owner_options:
        seed = pixel_to_variable.get(tuple(option["anchor_pixel"]))
        if seed is None or option["id"] not in domains[seed]:
            _fail("SEED_OWNER_UNAVAILABLE")
        eligible = [index for index, domain in enumerate(domains) if option["id"] in domain]
        eligible_set = set(eligible)
        eligible_edges = [
            edge["id"]
            for edge in edges
            if edge["left_variable_id"] in eligible_set
            and edge["right_variable_id"] in eligible_set
        ]
        constraints.extend(
            [
                {
                    "id": len(constraints),
                    "kind": "seed_owner",
                    "owner_option_id": option["id"],
                    "variable_id": seed,
                },
                {
                    "adjacency_edge_ids": eligible_edges,
                    "eligible_variable_ids": eligible,
                    "id": len(constraints) + 1,
                    "kind": "connected_to_seed",
                    "owner_option_id": option["id"],
                    "seed_variable_id": seed,
                },
                {
                    "eligible_variable_ids": eligible,
                    "id": len(constraints) + 2,
                    "kind": "option_incidence",
                    "owner_option_id": option["id"],
                },
            ]
        )
    if len(constraints) > limits["max_constraints"]:
        _fail("L_SOURCE_CAPACITY_EXCEEDED:max_constraints")
    ambiguous_count = sum(len(domain) > 1 for domain in domains)
    if ambiguous_count > limits["max_ambiguous_variables"]:
        _fail("L_SOURCE_CAPACITY_EXCEEDED:max_ambiguous_variables")
    raw_count = 1
    for domain in domains:
        raw_count *= len(domain)
        if raw_count > limits["max_raw_assignments"]:
            _fail("L_SOURCE_CAPACITY_EXCEEDED:max_raw_assignments")
    universal = len(variables) * len(owner_options)
    memberships = sum(len(domain) for domain in domains)
    eliminated = universal - memberships
    eligible_count = sum(
        set(option["incident_carrier_ids"]).issubset(item["carrier_ids"])
        for item in evidence
        for option in owner_options
    )
    ineligible_count = universal - eligible_count
    metrics = {
        "baseline_id": "all_and_only_known_intersections_per_variable-v0.5-r0",
        "candidate_option_membership_count": memberships,
        "domain_size_histogram": [
            {
                "domain_size": size,
                "variable_count": sum(len(domain) == size for domain in domains),
            }
            for size in sorted({len(domain) for domain in domains})
        ],
        "eligible_option_count": eligible_count,
        "eligible_option_recall": _ratio(memberships, eligible_count),
        "eliminated_option_membership_count": eliminated,
        "eliminated_option_ratio": _ratio(eliminated, universal),
        "eliminated_ineligible_option_count": eliminated,
        "ineligible_option_count": ineligible_count,
        "ineligible_option_elimination_recall": _ratio(eliminated, ineligible_count),
        "max_domain_size": max(len(domain) for domain in domains),
        "mean_domain_size": _ratio(memberships, len(domains)),
        "multi_intersection_ratio": _ratio(ambiguous_count, len(domains)),
        "raw_cartesian_assignment_count": raw_count,
        "retained_eligible_option_count": memberships,
        "residual_containing_ratio": [0, 1],
        "residual_only_ratio": [0, 1],
        "singleton_ratio": _ratio(len(domains) - ambiguous_count, len(domains)),
        "universal_option_membership_count": universal,
    }
    counts = {
        "adjacency_edge_count": len(edges),
        "ambiguous_variable_count": ambiguous_count,
        "atomic_fragment_count": geometry["atomic_fragment_count"],
        "carrier_count": geometry["base_stroke_count"],
        "constraint_count": len(constraints),
        "domain_class_count": len(domain_classes),
        "evidence_membership_count": len(evidence),
        "owner_option_count": len(owner_options),
        "pixel_variable_count": len(variables),
        "residual_reason_count": 0,
    }
    section_items: dict[str, list[Any]] = {
        "source_binding": [source_binding],
        "limits": [limits],
        "counts": [counts],
        "owner_options": owner_options,
        "evidence_memberships": evidence,
        "domain_classes": domain_classes,
        "pixel_variables": variables,
        "adjacency_edges": edges,
        "constraints": constraints,
        "residual_reasons": [],
        "metrics": [metrics],
    }
    section_hashes = [
        {
            "item_count": len(section_items[name]),
            "section": name,
            "sha256": _framed_items_sha256(section_items[name]),
        }
        for name in SECTION_ORDER
    ]
    digest_by_section = {item["section"]: item["sha256"] for item in section_hashes}
    previous = source_binding["observation_sha256"]
    stage_trace: list[dict[str, Any]] = []
    for index, name in enumerate(STAGE_ORDER):
        output = digest_by_section[name]
        stage_trace.append(
            {
                "component_sha256": constructor_sha256,
                "index": index,
                "input_sha256": previous,
                "output_sha256": output,
                "record_count": len(section_items[name]),
                "stage": name,
            }
        )
        previous = output
    artifact = {
        "adjacency_edges": edges,
        "constraints": constraints,
        "counts": counts,
        "domain_classes": domain_classes,
        "evidence_memberships": evidence,
        "limits": limits,
        "metrics": metrics,
        "owner_options": owner_options,
        "pixel_variables": variables,
        "residual_reasons": [],
        "schema_version": CANDIDATE_SCHEMA,
        "section_hashes": section_hashes,
        "source_binding": source_binding,
        "stage_trace": stage_trace,
        "status": "candidate_domain",
        "training_authorized": False,
    }
    if len(canonical_bytes(artifact)) > limits["max_output_bytes"]:
        _fail("L_SOURCE_CAPACITY_EXCEEDED:max_output_bytes")
    context = {
        "domains": domains,
        "geometry": geometry,
        "support": support,
        "support_input": support_input,
    }
    return artifact, context


def validate_candidate_envelope(
    candidate: Any,
    candidate_raw: bytes,
    limits: dict[str, Any],
    constructor_sha256: str,
) -> None:
    if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_KEYS:
        _fail("CANDIDATE_NOT_CLOSED")
    if candidate_raw != canonical_bytes(candidate):
        _fail("CANDIDATE_NONCANONICAL")
    if candidate["schema_version"] != CANDIDATE_SCHEMA:
        _fail("CANDIDATE_SCHEMA_INVALID")
    if candidate["status"] != "candidate_domain" or candidate["training_authorized"] is not False:
        _fail("CANDIDATE_STATUS_INVALID")
    if candidate["limits"] != limits:
        _fail("CANDIDATE_LIMITS_MISMATCH")
    if len(candidate_raw) > limits["max_output_bytes"]:
        _fail("CANDIDATE_OUTPUT_BYTES_EXCEEDED")
    for name in (
        "owner_options",
        "evidence_memberships",
        "domain_classes",
        "pixel_variables",
        "adjacency_edges",
        "constraints",
        "residual_reasons",
        "section_hashes",
        "stage_trace",
    ):
        if not isinstance(candidate[name], list):
            _fail(f"CANDIDATE_ARRAY_INVALID:{name}")
    if candidate["residual_reasons"] != []:
        _fail("CANDIDATE_RESIDUAL_FORBIDDEN")
    for name in ("counts", "metrics", "source_binding"):
        if not isinstance(candidate[name], dict):
            _fail(f"CANDIDATE_OBJECT_INVALID:{name}")
    for field in (
        "compiled_fragment_root_sha256",
        "compiled_support_partition_sha256",
        "geometry_root_sha256",
        "observation_sha256",
        "png_sha256",
        "renderer_sha256",
        "support_input_sha256",
    ):
        if not _is_sha256(candidate["source_binding"].get(field)):
            _fail(f"SOURCE_BINDING_HASH_INVALID:{field}")
    sections = {
        "source_binding": [candidate["source_binding"]],
        "limits": [candidate["limits"]],
        "counts": [candidate["counts"]],
        "owner_options": candidate["owner_options"],
        "evidence_memberships": candidate["evidence_memberships"],
        "domain_classes": candidate["domain_classes"],
        "pixel_variables": candidate["pixel_variables"],
        "adjacency_edges": candidate["adjacency_edges"],
        "constraints": candidate["constraints"],
        "residual_reasons": candidate["residual_reasons"],
        "metrics": [candidate["metrics"]],
    }
    expected_hashes = [
        {
            "item_count": len(sections[name]),
            "section": name,
            "sha256": _framed_items_sha256(sections[name]),
        }
        for name in SECTION_ORDER
    ]
    if candidate["section_hashes"] != expected_hashes:
        _fail("CANDIDATE_SECTION_HASH_MISMATCH")
    section_digest = {item["section"]: item["sha256"] for item in expected_hashes}
    previous = candidate["source_binding"]["observation_sha256"]
    expected_trace = []
    for index, name in enumerate(STAGE_ORDER):
        output = section_digest[name]
        expected_trace.append(
            {
                "component_sha256": constructor_sha256,
                "index": index,
                "input_sha256": previous,
                "output_sha256": output,
                "record_count": len(sections[name]),
                "stage": name,
            }
        )
        previous = output
    if candidate["stage_trace"] != expected_trace:
        _fail("CANDIDATE_STAGE_TRACE_MISMATCH")


def verify_feasible_witness(
    candidate: dict[str, Any],
    domains: list[tuple[int, ...]],
    witness: Any,
    witness_raw: bytes,
    *,
    axis: str,
    observation_sha256: str,
    source_sha256: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    if not isinstance(witness, dict) or set(witness) != {
        "ambiguous_overrides",
        "artifact_sha256",
        "assignment_mode",
        "axis",
        "observation_sha256",
        "schema_version",
        "side",
        "source_sha256",
        "training_authorized",
    }:
        _fail("FEASIBLE_WITNESS_NOT_CLOSED")
    if witness_raw != canonical_bytes(witness):
        _fail("FEASIBLE_WITNESS_NONCANONICAL")
    if (
        witness["schema_version"] != WITNESS_SCHEMA
        or witness["assignment_mode"] != "singleton_plus_overrides"
        or witness["side"] != "L"
        or witness["training_authorized"] is not False
    ):
        _fail("FEASIBLE_WITNESS_HEADER_INVALID")
    expected_bindings = {
        "axis": axis,
        "observation_sha256": observation_sha256,
        "source_sha256": source_sha256,
        "artifact_sha256": artifact_sha256,
    }
    if any(witness[name] != value for name, value in expected_bindings.items()):
        _fail("FEASIBLE_WITNESS_BINDING_MISMATCH")
    overrides = witness["ambiguous_overrides"]
    if not isinstance(overrides, list):
        _fail("FEASIBLE_WITNESS_OVERRIDES_INVALID")
    override_by_variable: dict[int, int] = {}
    for item in overrides:
        if not isinstance(item, dict) or set(item) != {"owner_option_id", "variable_id"}:
            _fail("FEASIBLE_WITNESS_OVERRIDE_NOT_CLOSED")
        variable_id, owner_id = item["variable_id"], item["owner_option_id"]
        if not _is_int(variable_id) or not _is_int(owner_id) or variable_id in override_by_variable:
            _fail("FEASIBLE_WITNESS_OVERRIDE_INVALID")
        override_by_variable[variable_id] = owner_id
    ambiguous_ids = {index for index, domain in enumerate(domains) if len(domain) > 1}
    if set(override_by_variable) != ambiguous_ids:
        _fail("FEASIBLE_WITNESS_OVERRIDE_COVERAGE_MISMATCH")
    assignment: list[int] = []
    for variable_id, domain in enumerate(domains):
        owner_id = domain[0] if len(domain) == 1 else override_by_variable[variable_id]
        if owner_id not in domain:
            _fail("FEASIBLE_WITNESS_OWNER_OUTSIDE_DOMAIN")
        assignment.append(owner_id)
    edge_by_id = {edge["id"]: edge for edge in candidate["adjacency_edges"]}
    if set(edge_by_id) != set(range(len(edge_by_id))):
        _fail("FEASIBLE_WITNESS_EDGE_INDEX_INVALID")
    seed_constraints: dict[int, dict[str, Any]] = {}
    connectivity_constraints: dict[int, dict[str, Any]] = {}
    incidence_constraints: dict[int, dict[str, Any]] = {}
    for constraint in candidate["constraints"]:
        kind = constraint["kind"]
        if kind == "exact_one":
            if constraint != {
                "id": 0,
                "kind": "exact_one",
                "variable_ids": list(range(len(domains))),
            }:
                _fail("FEASIBLE_WITNESS_EXACT_ONE_INVALID")
        elif kind == "seed_owner":
            seed_constraints[constraint["owner_option_id"]] = constraint
        elif kind == "connected_to_seed":
            connectivity_constraints[constraint["owner_option_id"]] = constraint
        elif kind == "option_incidence":
            incidence_constraints[constraint["owner_option_id"]] = constraint
        else:
            _fail("FEASIBLE_WITNESS_CONSTRAINT_KIND_INVALID")
    owner_ids = {option["id"] for option in candidate["owner_options"]}
    if not (
        set(seed_constraints) == owner_ids
        and set(connectivity_constraints) == owner_ids
        and set(incidence_constraints) == owner_ids
    ):
        _fail("FEASIBLE_WITNESS_CONSTRAINT_COVERAGE_INVALID")
    for owner_id in sorted(owner_ids):
        fiber = {index for index, value in enumerate(assignment) if value == owner_id}
        seed = seed_constraints[owner_id]["variable_id"]
        if seed not in fiber or connectivity_constraints[owner_id]["seed_variable_id"] != seed:
            _fail("FEASIBLE_WITNESS_SEED_INVALID")
        eligible = set(incidence_constraints[owner_id]["eligible_variable_ids"])
        if not fiber.issubset(eligible):
            _fail("FEASIBLE_WITNESS_INCIDENCE_INVALID")
        adjacency = {variable_id: set() for variable_id in fiber}
        for edge_id in connectivity_constraints[owner_id]["adjacency_edge_ids"]:
            edge = edge_by_id.get(edge_id)
            if edge is None:
                _fail("FEASIBLE_WITNESS_EDGE_REFERENCE_INVALID")
            left, right = edge["left_variable_id"], edge["right_variable_id"]
            if left in fiber and right in fiber:
                adjacency[left].add(right)
                adjacency[right].add(left)
        seen = {seed}
        queue = deque([seed])
        while queue:
            variable_id = queue.popleft()
            for neighbor in adjacency[variable_id]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if seen != fiber:
            _fail("FEASIBLE_WITNESS_FIBER_DISCONNECTED")
    return {
        "ambiguous_override_count": len(overrides),
        "assignment_materialized_count": len(assignment),
        "feasible_owner_count": len(owner_ids),
        "witness_sha256": _sha256(witness_raw),
    }


def _measurement_layers(
    observation: dict[str, Any],
    normalized: dict[str, Any],
    geometry: dict[str, Any],
    support: dict[str, Any],
) -> dict[str, int]:
    raw_strokes = observation["geometry_root"]["base_strokes"]
    canonical_strokes = normalized["geometry_root"]["base_strokes"]
    raw_parent_runs = sum(len(item["support_runs"]) for item in observation["parent_supports"])
    child_runs = support["child_support_run_count_total"]
    raw_memberships = sum(item["support_pixel_count"] for item in observation["parent_supports"])
    overlaps = support["cross_carrier_overlap_pixel_count"]
    return {
        "carrier_canonical_distinct": len(canonical_strokes),
        "carrier_compiled": geometry["base_stroke_count"],
        "carrier_deduplicated": len(canonical_strokes),
        "carrier_raw": len(raw_strokes),
        "intersection_canonical_distinct": geometry["intersection_count"],
        "intersection_compiled": geometry["intersection_count"],
        "intersection_deduplicated": geometry["intersection_count"],
        "intersection_raw": geometry["intersection_count"],
        "pixel_canonical_distinct_memberships": raw_memberships,
        "pixel_compiled_overlap_variables": overlaps,
        "pixel_deduplicated_overlap_variables": overlaps,
        "pixel_raw_memberships": raw_memberships,
        "rle_canonical_distinct_parent": raw_parent_runs,
        "rle_compiled_child": child_runs,
        "rle_deduplicated_child": child_runs,
        "rle_raw_parent": raw_parent_runs,
    }


def verify_l_artifact(
    *,
    axis: str,
    plan_raw: bytes,
    expected_plan_sha256: str,
    manifest_raw: bytes,
    exclusion_raw: bytes,
    expected_exclusion_sha256: str,
    observation_raw: bytes,
    candidate_raw: bytes,
    witness_raw: bytes,
    constructor_raw: bytes,
    renderer_raw: bytes,
    png_raw: bytes,
) -> dict[str, Any]:
    if axis not in END_TO_END_AXES:
        _fail("AXIS_NOT_END_TO_END_L")
    if expected_plan_sha256 != FROZEN_PLAN_SHA256:
        _fail("PLAN_EXPECTATION_NOT_FROZEN")
    if _sha256(plan_raw) != FROZEN_PLAN_SHA256:
        _fail("PLAN_SHA_MISMATCH")
    if expected_exclusion_sha256 != FROZEN_EXCLUSION_SHA256:
        _fail("EXCLUSION_EXPECTATION_NOT_FROZEN")
    if _sha256(exclusion_raw) != FROZEN_EXCLUSION_SHA256:
        _fail("EXCLUSION_SHA_MISMATCH")
    plan = load_canonical_json(plan_raw, "axis_plan")
    manifest = load_canonical_json(manifest_raw, "source_manifest")
    observation = load_canonical_json(observation_raw, "observation")
    candidate = load_canonical_json(candidate_raw, "candidate")
    witness = load_canonical_json(witness_raw, "feasible_witness")
    if plan.get("training_authorized") is not False or manifest.get("training_authorized") is not False:
        _fail("PLAN_OR_MANIFEST_TRAINING_AUTHORIZED")
    if not (
        _sha256(manifest_raw)
        == plan.get("source_manifest_sha256")
        == FROZEN_MANIFEST_SHA256
    ):
        _fail("SOURCE_MANIFEST_SHA_MISMATCH")
    limits = plan.get("limits")
    if not isinstance(limits, dict) or set(limits) != LIMIT_KEYS:
        _fail("PLAN_LIMITS_INVALID")
    constructor_sha = _sha256(constructor_raw)
    renderer_sha = _sha256(renderer_raw)
    if plan.get("constructor", {}).get("sha256") != constructor_sha:
        _fail("CONSTRUCTOR_IDENTITY_MISMATCH")
    if plan.get("capacity_support_renderer", {}).get("sha256") != renderer_sha:
        _fail("RENDERER_IDENTITY_MISMATCH")
    axis_records = [item for item in plan.get("axes", []) if item.get("axis") == axis]
    if len(axis_records) != 1 or axis_records[0].get("category") != "reachable_end_to_end":
        _fail("PLAN_AXIS_RECORD_INVALID")
    records = [
        item
        for item in manifest.get("records", [])
        if item.get("axis") == axis and item.get("side") == "L"
    ]
    if len(records) != 1:
        _fail("MANIFEST_L_RECORD_INVALID")
    record = records[0]
    source_sha = _sha256(observation_raw)
    boundary = axis_records[0]["boundary"]
    if not (
        source_sha == record.get("canonical_source_sha256")
        == record.get("raw_source_sha256")
        == boundary.get("L_source_sha256")
    ):
        _fail("L_SOURCE_SHA_MISMATCH")
    if observation.get("renderer_sha256") != renderer_sha:
        _fail("OBSERVATION_RENDERER_SHA_MISMATCH")
    if not (
        record.get("renderer_binding") is True
        and record.get("png_binding") is True
        and record.get("geometry_renderer_binding") is False
        and record.get("renderer_sha256") == renderer_sha
    ):
        _fail("MANIFEST_RENDERER_BINDING_MISMATCH")
    if _sha256(png_raw) != observation.get("png_sha256") or record.get("png_sha256") != _sha256(
        png_raw
    ):
        _fail("OBSERVATION_PNG_SHA_MISMATCH")
    expected_png = render_support_png(observation["parent_supports"])
    if png_raw != expected_png:
        _fail("PNG_SUPPORT_UNION_BYTES_MISMATCH")
    if decode_support_png(png_raw) != _support_union(observation["parent_supports"]):
        _fail("PNG_SUPPORT_UNION_PIXELS_MISMATCH")
    expected, context = rebuild_expected_artifact(observation, limits, constructor_sha)
    validate_candidate_envelope(candidate, candidate_raw, limits, constructor_sha)
    if candidate != expected:
        for field in sorted(CANDIDATE_KEYS):
            if candidate.get(field) != expected.get(field):
                _fail(f"CANDIDATE_REBUILD_MISMATCH:{field}")
        _fail("CANDIDATE_REBUILD_MISMATCH")
    count_vector = {
        "max_adjacency_edges": expected["counts"]["adjacency_edge_count"],
        "max_ambiguous_variables": expected["counts"]["ambiguous_variable_count"],
        "max_atomic_fragments": expected["counts"]["atomic_fragment_count"],
        "max_carriers": expected["counts"]["carrier_count"],
        "max_constraints": expected["counts"]["constraint_count"],
        "max_domain_classes": expected["counts"]["domain_class_count"],
        "max_intersections": context["geometry"]["intersection_count"],
        "max_output_bytes": len(candidate_raw),
        "max_owner_options": expected["counts"]["owner_option_count"],
        "max_pixel_variables": expected["counts"]["pixel_variable_count"],
        "max_raw_assignments": expected["metrics"]["raw_cartesian_assignment_count"],
        "max_total_rle_runs": context["support"]["child_support_run_count_total"],
    }
    declared_counts = record.get("count_vector")
    count_kinds = record.get("count_kinds")
    if not isinstance(declared_counts, dict) or not isinstance(count_kinds, dict):
        _fail("MANIFEST_COUNT_VECTOR_INVALID")
    for name, actual in count_vector.items():
        kind = count_kinds.get(name)
        declared = declared_counts.get(name)
        if kind == "exact" and actual != declared:
            _fail(f"MANIFEST_EXACT_COUNT_MISMATCH:{name}")
        if kind in {"universal_upper_bound", "pre_gate_upper_bound"} and actual > declared:
            _fail(f"MANIFEST_COUNT_BOUND_EXCEEDED:{name}")
    if record.get("target_count") != count_vector[axis] or boundary.get("L") != count_vector[axis]:
        _fail("MANIFEST_TARGET_COUNT_MISMATCH")
    normalized = context["support_input"]
    layers = _measurement_layers(
        observation, normalized, context["geometry"], context["support"]
    )
    if record.get("measurement_layers") != layers:
        _fail("MANIFEST_MEASUREMENT_LAYERS_MISMATCH")
    if record.get("support_union_pixel_count") != len(_support_union(observation["parent_supports"])):
        _fail("MANIFEST_SUPPORT_UNION_COUNT_MISMATCH")
    feasible = verify_feasible_witness(
        candidate,
        context["domains"],
        witness,
        witness_raw,
        axis=axis,
        observation_sha256=_sha256(canonical_bytes(observation)),
        source_sha256=source_sha,
        artifact_sha256=_sha256(candidate_raw),
    )
    return {
        "artifact_sha256": _sha256(candidate_raw),
        "axis": axis,
        "checks": [
            "L_ARTIFACT_BYTE_EXACT",
            "SOURCE_DERIVATION_EXACT",
            "FEASIBLE_WORLD_WITNESSED",
        ],
        "exclusion_sha256": expected_exclusion_sha256,
        "exact_count_keys": sorted(
            name for name, kind in count_kinds.items() if kind == "exact"
        ),
        "feasible_world": feasible,
        "first_failure_gate": None,
        "nonexact_count_keys": sorted(
            name for name, kind in count_kinds.items() if kind != "exact"
        ),
        "observation_sha256": source_sha,
        "predecessor_gates_recomputed_pass": True,
        "recomputed_count_vector": count_vector,
        "schema_version": "owner-capacity-generic-verification-v0.5",
        "side": "L",
        "status": "passed",
        "target_count_recomputed_exact": True,
        "training_authorized": False,
    }


def verify_rejection_source(
    *,
    axis: str,
    side: str,
    expected_error: str,
    plan_raw: bytes,
    manifest_raw: bytes,
    exclusion_raw: bytes,
    observation_raw: bytes,
    renderer_raw: bytes,
    png_raw: bytes,
) -> dict[str, Any]:
    """Independently recompute the first gate for a preregistered rejection source."""
    allowed = {
        ("max_carriers", "H"): ("CAPACITY_EXCEEDED:max_carriers", "max_carriers", 0),
        ("max_intersections", "L"): (
            "R0_REQUIRES_TWO_INTERSECTIONS", "R0_REQUIRES_TWO_INTERSECTIONS", 4
        ),
        ("max_intersections", "H"): (
            "CAPACITY_EXCEEDED:max_intersections", "max_intersections", 1
        ),
        ("max_pixel_variables", "H"): (
            "CAPACITY_EXCEEDED:max_pixel_variables", "max_pixel_variables", 6
        ),
        ("max_total_rle_runs", "H"): (
            "CAPACITY_EXCEEDED:max_total_rle_runs", "max_total_rle_runs", 3
        ),
    }
    if (axis, side) not in allowed or expected_error != allowed[(axis, side)][0]:
        _fail("REJECTION_CASE_NOT_ALLOWLISTED")
    if _sha256(plan_raw) != FROZEN_PLAN_SHA256:
        _fail("PLAN_SHA_MISMATCH")
    if _sha256(manifest_raw) != FROZEN_MANIFEST_SHA256:
        _fail("SOURCE_MANIFEST_SHA_MISMATCH")
    if _sha256(exclusion_raw) != FROZEN_EXCLUSION_SHA256:
        _fail("EXCLUSION_SHA_MISMATCH")
    plan = load_canonical_json(plan_raw, "axis_plan")
    manifest = load_canonical_json(manifest_raw, "source_manifest")
    observation = load_canonical_json(observation_raw, "observation")
    if plan.get("training_authorized") is not False or manifest.get("training_authorized") is not False:
        _fail("PLAN_OR_MANIFEST_TRAINING_AUTHORIZED")
    limits = plan.get("limits")
    if not isinstance(limits, dict) or set(limits) != LIMIT_KEYS:
        _fail("PLAN_LIMITS_INVALID")
    axis_records = [item for item in plan.get("axes", []) if item.get("axis") == axis]
    records = [
        item for item in manifest.get("records", [])
        if item.get("axis") == axis and item.get("side") == side
    ]
    if len(axis_records) != 1 or len(records) != 1:
        _fail("REJECTION_PLAN_OR_MANIFEST_RECORD_INVALID")
    axis_record = axis_records[0]
    record = records[0]
    source_sha = _sha256(observation_raw)
    boundary = axis_record.get("boundary", {})
    if not (
        source_sha == record.get("raw_source_sha256")
        == record.get("canonical_source_sha256")
        == boundary.get(f"{side}_source_sha256")
    ):
        _fail("REJECTION_SOURCE_SHA_MISMATCH")
    renderer_sha = _sha256(renderer_raw)
    if not (
        observation.get("renderer_sha256") == renderer_sha
        and record.get("renderer_sha256") == renderer_sha
        and record.get("renderer_binding") is True
        and record.get("png_binding") is True
        and record.get("geometry_renderer_binding") is False
    ):
        _fail("REJECTION_RENDERER_BINDING_MISMATCH")
    if not (
        _sha256(png_raw) == observation.get("png_sha256") == record.get("png_sha256")
        and png_raw == render_support_png(observation["parent_supports"])
        and decode_support_png(png_raw) == _support_union(observation["parent_supports"])
    ):
        _fail("REJECTION_PNG_BINDING_MISMATCH")
    raw_support_input = {
        "geometry_root": observation.get("geometry_root"),
        "parent_supports": observation.get("parent_supports"),
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }
    if observation.get("support_input_sha256") != _sha256(canonical_bytes(raw_support_input)):
        _fail("SUPPORT_INPUT_HASH_MISMATCH")
    support_input = _canonicalize_observation_source(observation)
    geometry = compile_geometry(support_input["geometry_root"])
    support = compile_support_partition(support_input)
    values = {
        "max_carriers": geometry["base_stroke_count"],
        "max_intersections": geometry["intersection_count"],
        "max_atomic_fragments": geometry["atomic_fragment_count"],
        "max_total_rle_runs": support["child_support_run_count_total"],
        "max_owner_options": geometry["intersection_count"],
        "max_pixel_variables": len(support["cross_carrier_overlaps"]),
    }
    ordered_gates: list[tuple[str, int]] = [
        ("max_carriers", 0),
        ("max_intersections", 1),
        ("max_atomic_fragments", 2),
        ("max_total_rle_runs", 3),
    ]
    first_failure: str | None = None
    first_failure_index: int | None = None
    recomputed: dict[str, int] = {}
    for name, index in ordered_gates:
        recomputed[name] = values[name]
        if values[name] > limits[name]:
            first_failure = name
            first_failure_index = index
            break
    if first_failure is None and values["max_intersections"] != 2:
        first_failure = "R0_REQUIRES_TWO_INTERSECTIONS"
        first_failure_index = 4
    if first_failure is None:
        for name, index in (("max_owner_options", 5), ("max_pixel_variables", 6)):
            recomputed[name] = values[name]
            if values[name] > limits[name]:
                first_failure = name
                first_failure_index = index
                break
    declared_failure = allowed[(axis, side)][1]
    declared_index = allowed[(axis, side)][2]
    if first_failure != declared_failure or first_failure_index != declared_index:
        _fail("REJECTION_FIRST_FAILURE_MISMATCH")
    target = recomputed.get(axis)
    if not (
        target == record.get("target_count")
        and target == boundary.get(side)
    ):
        _fail("REJECTION_TARGET_COUNT_MISMATCH")
    count_kinds = record.get("count_kinds")
    count_vector = record.get("count_vector")
    if not isinstance(count_kinds, dict) or not isinstance(count_vector, dict):
        _fail("REJECTION_MANIFEST_COUNTS_INVALID")
    for name, value in recomputed.items():
        if count_kinds.get(name) == "exact" and count_vector.get(name) != value:
            _fail(f"REJECTION_MANIFEST_EXACT_COUNT_MISMATCH:{name}")
    layers = _measurement_layers(
        observation, support_input, geometry, support
    )
    if record.get("measurement_layers") != layers:
        _fail("REJECTION_MEASUREMENT_LAYERS_MISMATCH")
    return {
        "axis": axis,
        "exact_count_keys": sorted(recomputed),
        "first_failure_gate": first_failure,
        "first_failure_index": first_failure_index,
        "nonexact_count_keys": sorted(LIMIT_KEYS - set(recomputed)),
        "observation_sha256": source_sha,
        "predecessor_gates_recomputed_pass": True,
        "recomputed_count_vector": recomputed,
        "schema_version": "owner-capacity-generic-rejection-verification-v0.5",
        "side": side,
        "status": "passed",
        "target_count_recomputed_exact": True,
        "training_authorized": False,
    }


def _read(path: str) -> bytes:
    return Path(path).read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=sorted(END_TO_END_AXES))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exclusion", required=True)
    parser.add_argument("--expected-exclusion-sha256", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--feasible-witness", required=True)
    parser.add_argument("--constructor", required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--png", required=True)
    args = parser.parse_args()
    result = verify_l_artifact(
        axis=args.axis,
        plan_raw=_read(args.plan),
        expected_plan_sha256=args.expected_plan_sha256,
        manifest_raw=_read(args.manifest),
        exclusion_raw=_read(args.exclusion),
        expected_exclusion_sha256=args.expected_exclusion_sha256,
        observation_raw=_read(args.observation),
        candidate_raw=_read(args.candidate),
        witness_raw=_read(args.feasible_witness),
        constructor_raw=_read(args.constructor),
        renderer_raw=_read(args.renderer),
        png_raw=_read(args.png),
    )
    print(canonical_bytes(result).decode("ascii"), end="")


if __name__ == "__main__":
    main()
