"""Deterministically partition parent raster support across atomic fragments."""

from __future__ import annotations

import bisect
import hashlib
from fractions import Fraction
from typing import Any

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    FragmentRootError,
    _canonical_bytes,
    _matches_strict_json_shape,
    compile_fragment_root,
    validate_root,
)


SCHEMA_VERSION = "chemgraph-fragment-support-partition-v0.4-research"
MAX_SUPPORT_PIXELS = 768 * 768
MAX_SUPPORT_RUNS = 100_000
MAX_TOTAL_SUPPORT_MEMBERSHIPS = 2_000_000
MAX_TOTAL_SUPPORT_RUNS = 200_000
MAX_CROSS_CARRIER_OVERLAPS = 100_000
MAX_OUTPUT_BYTES = 67_108_864


class SupportPartitionError(ValueError):
    """Closed failure raised before a partial support document is returned."""


def _fail(code: str) -> None:
    raise SupportPartitionError(code)


def _decode_runs(value: Any, expected_count: Any) -> set[tuple[int, int]]:
    if type(expected_count) is not int or not 1 <= expected_count <= MAX_SUPPORT_PIXELS:
        _fail("INVALID_SUPPORT_PIXEL_COUNT")
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SUPPORT_RUNS:
        _fail("INVALID_SUPPORT_RUNS")
    pixels: set[tuple[int, int]] = set()
    previous: tuple[int, int, int] | None = None
    for run in value:
        if not (
            isinstance(run, list)
            and len(run) == 3
            and all(type(item) is int for item in run)
        ):
            _fail("INVALID_SUPPORT_RUN")
        row, start, length = run
        if not (0 <= row < 768 and 0 <= start < 768 and 1 <= length <= 768 - start):
            _fail("SUPPORT_RUN_OUT_OF_RANGE")
        if previous is not None:
            previous_row, previous_start, previous_length = previous
            if (row, start) <= (previous_row, previous_start):
                _fail("SUPPORT_RUN_ORDER_NONCANONICAL")
            if row == previous_row and start <= previous_start + previous_length:
                _fail("SUPPORT_RUN_NOT_MAXIMAL")
        pixels.update((row, column) for column in range(start, start + length))
        previous = (row, start, length)
    if len(pixels) != expected_count:
        _fail("SUPPORT_PIXEL_COUNT_MISMATCH")
    return pixels


def _encode_runs(pixels: set[tuple[int, int]]) -> list[list[int]]:
    if not pixels:
        return []
    ordered = sorted(pixels)
    runs: list[list[int]] = []
    row, start = ordered[0]
    end = start + 1
    for next_row, column in ordered[1:]:
        if next_row == row and column == end:
            end += 1
            continue
        runs.append([row, start, end - start])
        row, start, end = next_row, column, column + 1
    runs.append([row, start, end - start])
    return runs


def _rle_sha256(runs: list[list[int]]) -> str:
    digest = hashlib.sha256()
    digest.update(len(runs).to_bytes(4, "big"))
    for row, start, length in runs:
        digest.update(row.to_bytes(4, "big"))
        digest.update(start.to_bytes(4, "big"))
        digest.update(length.to_bytes(4, "big"))
    return digest.hexdigest()


def _fraction(value: Any) -> Fraction:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(type(item) is int for item in value)
        and value[1] > 0
    ):
        _fail("INVALID_RATIONAL")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        _fail("RATIONAL_NONCANONICAL")
    return result


def _validate_input(document: Any) -> tuple[dict[str, Any], list[set[tuple[int, int]]]]:
    if not isinstance(document, dict) or set(document) != {
        "geometry_root", "parent_supports", "schema_version", "training_authorized"
    }:
        _fail("INPUT_NOT_CLOSED")
    if document["schema_version"] != SCHEMA_VERSION:
        _fail("SCHEMA_VERSION_MISMATCH")
    if document["training_authorized"] is not False:
        _fail("TRAINING_NOT_AUTHORIZED")
    geometry_root = document["geometry_root"]
    try:
        strokes = validate_root(geometry_root)
    except FragmentRootError as error:
        raise SupportPartitionError(f"GEOMETRY_ROOT_INVALID:{error}") from error
    supports = document["parent_supports"]
    if not isinstance(supports, list) or len(supports) != len(strokes):
        _fail("PARENT_SUPPORT_FIBER_INCOMPLETE")
    decoded: list[set[tuple[int, int]]] = []
    total_memberships = 0
    total_runs = 0
    for index, support in enumerate(supports):
        if not isinstance(support, dict) or set(support) != {
            "base_stroke_id", "support_pixel_count", "support_rle_sha256", "support_runs"
        }:
            _fail("PARENT_SUPPORT_NOT_CLOSED")
        if type(support["base_stroke_id"]) is not int or support["base_stroke_id"] != index:
            _fail("PARENT_SUPPORT_ID_NONCANONICAL")
        count = support["support_pixel_count"]
        runs = support["support_runs"]
        if type(count) is int:
            total_memberships += count
            if total_memberships > MAX_TOTAL_SUPPORT_MEMBERSHIPS:
                _fail("TOTAL_SUPPORT_MEMBERSHIPS_OUTSIDE_CAPACITY")
        if isinstance(runs, list):
            total_runs += len(runs)
            if total_runs > MAX_TOTAL_SUPPORT_RUNS:
                _fail("TOTAL_SUPPORT_RUNS_OUTSIDE_CAPACITY")
        pixels = _decode_runs(runs, count)
        if support["support_rle_sha256"] != _rle_sha256(runs):
            _fail("PARENT_SUPPORT_RLE_HASH_MISMATCH")
        decoded.append(pixels)
    return geometry_root, decoded


def compile_support_partition(document: Any) -> dict[str, Any]:
    geometry_root, parent_pixels = _validate_input(document)
    geometry = compile_fragment_root(geometry_root)
    strokes = validate_root(geometry_root)
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
        begin = stroke["begin"]
        direction = stroke["direction"]
        norm_squared = direction[0] ** 2 + direction[1] ** 2
        for row, column in parent_pixels[stroke["id"]]:
            pixel_q4 = (16 * column + 8, 16 * row + 8)
            offset = (pixel_q4[0] - begin[0], pixel_q4[1] - begin[1])
            projected = Fraction(
                offset[0] * direction[0] + offset[1] * direction[1],
                norm_squared,
            )
            clamped = min(Fraction(1), max(Fraction(0), projected))
            child_index = bisect.bisect_right(cut_values, clamped)
            child_pixels[fragments[child_index]["id"]].add((row, column))

    child_supports: list[dict[str, Any]] = []
    child_run_count_total = 0
    for fragment in geometry["atomic_fragments"]:
        pixels = child_pixels[fragment["id"]]
        if not pixels:
            _fail("EMPTY_FRAGMENT_SUPPORT")
        runs = _encode_runs(pixels)
        child_run_count_total += len(runs)
        if child_run_count_total > MAX_TOTAL_SUPPORT_RUNS:
            _fail("CHILD_SUPPORT_RUNS_OUTSIDE_CAPACITY")
        child_supports.append({
            "atomic_fragment_id": fragment["id"],
            "base_stroke_id": fragment["base_stroke_id"],
            "support_pixel_count": len(pixels),
            "support_rle_sha256": _rle_sha256(runs),
            "support_runs": runs,
        })

    pixel_carriers: dict[tuple[int, int], list[int]] = {}
    for stroke_id, pixels in enumerate(parent_pixels):
        for pixel in pixels:
            pixel_carriers.setdefault(pixel, []).append(stroke_id)
    overlaps = [
        {"pixel": [row, column], "base_stroke_ids": carrier_ids}
        for (row, column), carrier_ids in sorted(pixel_carriers.items())
        if len(carrier_ids) > 1
    ]
    if len(overlaps) > MAX_CROSS_CARRIER_OVERLAPS:
        _fail("CROSS_CARRIER_OVERLAPS_OUTSIDE_CAPACITY")

    result = {
        "schema_version": "chemgraph-fragment-support-compiled-v0.4-research",
        "training_authorized": False,
        "source_input_sha256": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
        "geometry_root_sha256": geometry["source_root_sha256"],
        "parent_support_count": len(parent_pixels),
        "parent_membership_count": sum(len(pixels) for pixels in parent_pixels),
        "parent_support_run_count_total": sum(
            len(support["support_runs"]) for support in document["parent_supports"]
        ),
        "atomic_fragment_count": len(child_supports),
        "child_membership_count": sum(item["support_pixel_count"] for item in child_supports),
        "child_support_run_count_total": child_run_count_total,
        "unique_parent_pixel_count": len(pixel_carriers),
        "cross_carrier_overlap_pixel_count": len(overlaps),
        "cross_carrier_overlaps": overlaps,
        "child_supports": child_supports,
    }
    if len(_canonical_bytes(result)) > MAX_OUTPUT_BYTES:
        _fail("OUTPUT_BYTES_OUTSIDE_CAPACITY")
    return result


def compare_declared_partition(document: Any, declared: Any) -> dict[str, Any]:
    expected = compile_support_partition(document)
    if not _matches_strict_json_shape(declared, expected):
        return {"status": "INVALID", "exact": False}
    declared_supports = declared["child_supports"]
    declared_by_id: dict[int, set[tuple[int, int]]] = {}
    try:
        for index, support in enumerate(declared_supports):
            if support["atomic_fragment_id"] != index:
                return {"status": "CHILD_SUPPORT_ID_NONCANONICAL", "exact": False}
            declared_by_id[index] = _decode_runs(
                support["support_runs"], support["support_pixel_count"]
            )
    except (SupportPartitionError, KeyError, TypeError):
        return {"status": "INVALID", "exact": False}
    expected_by_id = {
        support["atomic_fragment_id"]: _decode_runs(
            support["support_runs"], support["support_pixel_count"]
        )
        for support in expected["child_supports"]
    }
    if declared_by_id != expected_by_id:
        missing = sum(len(expected_by_id[key] - declared_by_id.get(key, set())) for key in expected_by_id)
        extra = sum(len(declared_by_id.get(key, set()) - expected_by_id.get(key, set())) for key in declared_by_id)
        return {
            "status": "SUPPORT_PARTITION_MISMATCH",
            "exact": False,
            "missing_pixel_count": missing,
            "extra_pixel_count": extra,
        }
    if _canonical_bytes(declared) != _canonical_bytes(expected):
        return {"status": "SUPPORT_PARTITION_NONCANONICAL", "exact": False}
    return {"status": "SUPPORT_PARTITION_EXACT", "exact": True}
