"""Build closed synthetic-support observations without running the R0 constructor."""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    compile_fragment_root,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    SCHEMA_VERSION as SUPPORT_SCHEMA,
    _rle_sha256,
    compile_support_partition,
)
from formal.chemgraph.audit.owner_candidate_domain_v0_5.build_observation import (
    build_observation,
)
from formal.chemgraph.audit.owner_candidate_domain_v0_5.common import (
    canonical_bytes,
    encode_runs,
    sha256,
)

from .capacity_png import render_support_png


ROOT = Path(__file__).resolve().parent
AXIS_SIDES = {
    "max_carriers": {"L": 8, "H": 9},
    "max_intersections": {"L": 8, "H": 9},
    "max_pixel_variables": {"L": 4096, "H": 4097},
    "max_total_rle_runs": {"L": 65_536, "H": 65_537},
}


def _support(old_id: int, pixels: Iterable[tuple[int, int]]) -> dict[str, Any]:
    pixel_set = set(pixels)
    runs = encode_runs(pixel_set)
    return {
        "base_stroke_id": old_id,
        "support_pixel_count": len(pixel_set),
        "support_rle_sha256": _rle_sha256(runs),
        "support_runs": runs,
    }


def _finish(
    geometry_root: dict[str, Any], parent_supports: list[dict[str, Any]]
) -> dict[str, Any]:
    frozen = build_observation()
    png_bytes = render_support_png(parent_supports)
    renderer_sha = sha256(ROOT.joinpath("capacity_png.py").read_bytes())
    support_input = {
        "geometry_root": geometry_root,
        "parent_supports": parent_supports,
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }
    return {
        "geometry_root": geometry_root,
        "parent_supports": parent_supports,
        "png_sha256": sha256(png_bytes),
        "renderer_sha256": renderer_sha,
        "schema_version": frozen["schema_version"],
        "support_input_sha256": sha256(canonical_bytes(support_input)),
        "training_authorized": False,
    }


def _carrier_source(count: int) -> dict[str, Any]:
    observation = copy.deepcopy(build_observation())
    strokes = observation["geometry_root"]["base_strokes"]
    supports = observation["parent_supports"]
    for old_id in range(len(strokes), count):
        offset = old_id - 3
        y = 100 + 100 * offset
        strokes.append({
            "boundary_member_ids": [old_id],
            "id": old_id,
            "segment_q4": [[11_000, y], [12_000, y]],
        })
        supports.append(_support(old_id, {(700, 700 + offset)}))
    return _finish(observation["geometry_root"], supports)


def _canonical_geometry(segments: list[list[list[int]]]) -> dict[str, Any]:
    ordered = sorted(segments)
    return {
        "base_strokes": [
            {"boundary_member_ids": [index], "id": index, "segment_q4": segment}
            for index, segment in enumerate(ordered)
        ],
        "schema_version": "chemgraph-fragment-root-v0.4-research",
        "training_authorized": False,
    }


def _nearest_pixel_center(value: Fraction) -> int:
    lower = (value - 8) // 16
    candidates = (lower, lower + 1)
    valid = [candidate for candidate in candidates if 0 <= candidate < 768]
    return min(valid, key=lambda candidate: abs(Fraction(16 * candidate + 8) - value))


def _minimal_fragment_supports(geometry_root: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = compile_fragment_root(geometry_root)
    strokes = {item["id"]: item for item in geometry_root["base_strokes"]}
    pixels_by_stroke: dict[int, set[tuple[int, int]]] = {
        item["id"]: set() for item in geometry_root["base_strokes"]
    }
    for fragment in geometry["atomic_fragments"]:
        stroke = strokes[fragment["base_stroke_id"]]
        begin = Fraction(*fragment["begin_t"])
        end = Fraction(*fragment["end_t"])
        midpoint = (begin + end) / 2
        (x0, y0), (x1, y1) = stroke["segment_q4"]
        x = Fraction(x0) + midpoint * (x1 - x0)
        y = Fraction(y0) + midpoint * (y1 - y0)
        pixel = (_nearest_pixel_center(y), _nearest_pixel_center(x))
        if pixel in pixels_by_stroke[stroke["id"]]:
            raise RuntimeError("MINIMAL_FRAGMENT_SUPPORT_COLLISION")
        pixels_by_stroke[stroke["id"]].add(pixel)
    supports = [
        _support(stroke_id, pixels_by_stroke[stroke_id])
        for stroke_id in range(len(strokes))
    ]
    compiled = compile_support_partition({
        "geometry_root": geometry_root,
        "parent_supports": supports,
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    })
    if compiled["atomic_fragment_count"] != geometry["atomic_fragment_count"]:
        raise RuntimeError("MINIMAL_FRAGMENT_SUPPORT_INCOMPLETE")
    return supports


def _intersection_source(count: int) -> dict[str, Any]:
    if count == 8:
        horizontal_count, vertical_count = 2, 4
    elif count == 9:
        horizontal_count, vertical_count = 3, 3
    else:
        raise ValueError("INTERSECTION_BOUNDARY_UNSUPPORTED")
    low, high = 50 * 16 + 8, 650 * 16 + 8
    horizontal_y = [200 * 16 + 8 + 1600 * index for index in range(horizontal_count)]
    vertical_x = [200 * 16 + 8 + 1600 * index for index in range(vertical_count)]
    segments = [
        [[low, y], [high, y]] for y in horizontal_y
    ] + [
        [[x, low], [x, high]] for x in vertical_x
    ]
    geometry_root = _canonical_geometry(segments)
    supports = _minimal_fragment_supports(geometry_root)
    return _finish(geometry_root, supports)


def _isolated_pixels(count: int, *, start_row: int = 100) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for row in range(start_row, 768):
        for column in range(0, 768, 2):
            pixels.add((row, column))
            if len(pixels) == count:
                return pixels
    raise ValueError("ISOLATED_PIXEL_CAPACITY_EXHAUSTED")


def _rle_source(target_child_runs: int) -> dict[str, Any]:
    observation = copy.deepcopy(build_observation())
    base_support_input = {
        "geometry_root": observation["geometry_root"],
        "parent_supports": observation["parent_supports"],
        "schema_version": SUPPORT_SCHEMA,
        "training_authorized": False,
    }
    base_runs = compile_support_partition(base_support_input)["child_support_run_count_total"]
    extra_runs = target_child_runs - base_runs
    if extra_runs <= 0:
        raise ValueError("RLE_TARGET_TOO_SMALL")
    old_id = len(observation["geometry_root"]["base_strokes"])
    observation["geometry_root"]["base_strokes"].append({
        "boundary_member_ids": [old_id],
        "id": old_id,
        "segment_q4": [[11_000, 1_000], [12_000, 1_000]],
    })
    observation["parent_supports"].append(
        _support(old_id, _isolated_pixels(extra_runs))
    )
    return _finish(observation["geometry_root"], observation["parent_supports"])


def _seeded_strip(
    seed: tuple[int, int], count: int, columns: range
) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for row in range(seed[0], 768):
        for column in columns:
            pixels.add((row, column))
            if len(pixels) == count:
                if seed not in pixels:
                    raise RuntimeError("SEEDED_STRIP_MISSES_SEED")
                return pixels
    raise ValueError("SEEDED_STRIP_CAPACITY_EXHAUSTED")


def _pixel_variable_source(target: int) -> dict[str, Any]:
    observation = build_observation()
    left_count = target - target // 2
    right_count = target // 2
    left = _seeded_strip((20, 18), left_count, range(18, -1, -1))
    right = _seeded_strip((20, 22), right_count, range(22, 54))
    if left & right:
        raise RuntimeError("PIXEL_OWNER_REGIONS_OVERLAP")
    supports = [
        _support(0, left | right),
        _support(1, left),
        _support(2, right),
    ]
    return _finish(observation["geometry_root"], supports)


def build_source(axis: str, side: str) -> dict[str, Any]:
    try:
        target = AXIS_SIDES[axis][side]
    except KeyError as error:
        raise ValueError("UNKNOWN_AXIS_SIDE") from error
    if axis == "max_carriers":
        return _carrier_source(target)
    if axis == "max_intersections":
        return _intersection_source(target)
    if axis == "max_total_rle_runs":
        return _rle_source(target)
    if axis == "max_pixel_variables":
        return _pixel_variable_source(target)
    raise AssertionError("UNREACHABLE_AXIS_DISPATCH")


def source_bytes(axis: str, side: str) -> bytes:
    return canonical_bytes(build_source(axis, side))
