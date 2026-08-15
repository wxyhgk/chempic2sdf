"""Exact geometry-only fragment constructor for v0.4 research fixtures."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from typing import Any


SCHEMA_VERSION = "chemgraph-fragment-root-v0.4-research"
MAX_BASE_STROKES = 128
MAX_INTERSECTIONS = 128
MAX_ATOMIC_FRAGMENTS = 512


class FragmentRootError(ValueError):
    """Closed failure raised before any partial root is returned."""


def _fail(code: str) -> None:
    raise FragmentRootError(code)


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _point(value: Any) -> tuple[int, int]:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(_integer(item, 0, 12288) for item in value)
    ):
        _fail("INVALID_POINT_Q4")
    return value[0], value[1]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _matches_strict_json_shape(value: Any, exemplar: Any) -> bool:
    """Check closed keys and native JSON scalar types without Python numeric coercion."""
    if isinstance(exemplar, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(exemplar)
            and all(_matches_strict_json_shape(value[key], exemplar[key]) for key in exemplar)
        )
    if isinstance(exemplar, list):
        if not isinstance(value, list):
            return False
        if not exemplar:
            return not value
        return all(
            any(_matches_strict_json_shape(item, candidate) for candidate in exemplar)
            for item in value
        )
    return type(value) is type(exemplar)


def _rat(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _rat_point(
    begin: tuple[int, int], direction: tuple[int, int], t: Fraction,
) -> list[list[int]]:
    return [
        _rat(Fraction(begin[0]) + t * direction[0]),
        _rat(Fraction(begin[1]) + t * direction[1]),
    ]


def _cross(left: tuple[int, int], right: tuple[int, int]) -> int:
    return left[0] * right[1] - left[1] * right[0]


def _sub(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] - right[0], left[1] - right[1]


def _positive_collinear_overlap(
    p: tuple[int, int], p2: tuple[int, int], q: tuple[int, int], q2: tuple[int, int],
) -> bool:
    axis = 0 if abs(p2[0] - p[0]) >= abs(p2[1] - p[1]) else 1
    left = max(min(p[axis], p2[axis]), min(q[axis], q2[axis]))
    right = min(max(p[axis], p2[axis]), max(q[axis], q2[axis]))
    return left < right


def validate_root(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict) or set(document) != {
        "base_strokes", "schema_version", "training_authorized"
    }:
        _fail("ROOT_NOT_CLOSED")
    if document["schema_version"] != SCHEMA_VERSION:
        _fail("SCHEMA_VERSION_MISMATCH")
    if document["training_authorized"] is not False:
        _fail("TRAINING_NOT_AUTHORIZED")
    strokes = document["base_strokes"]
    if not isinstance(strokes, list) or not strokes:
        _fail("BASE_STROKES_EMPTY_OR_INVALID")
    if len(strokes) > MAX_BASE_STROKES:
        _fail("BASE_STROKES_OUTSIDE_CAPACITY")

    result: list[dict[str, Any]] = []
    canonical_keys: list[tuple[Any, ...]] = []
    used_boundary_ids: set[int] = set()
    for index, stroke in enumerate(strokes):
        if not isinstance(stroke, dict) or set(stroke) != {
            "boundary_member_ids", "id", "segment_q4"
        }:
            _fail("BASE_STROKE_NOT_CLOSED")
        if type(stroke["id"]) is not int or stroke["id"] != index:
            _fail("BASE_STROKE_ID_NONCANONICAL")
        members = stroke["boundary_member_ids"]
        if not (
            isinstance(members, list)
            and 1 <= len(members) <= 2
            and all(_integer(item, 0, 127) for item in members)
            and members == sorted(set(members))
        ):
            _fail("BOUNDARY_MEMBERS_NONCANONICAL")
        if used_boundary_ids.intersection(members):
            _fail("BOUNDARY_MEMBER_REUSED")
        used_boundary_ids.update(members)
        segment = stroke["segment_q4"]
        if not isinstance(segment, list) or len(segment) != 2:
            _fail("INVALID_SEGMENT_Q4")
        begin, end = _point(segment[0]), _point(segment[1])
        if not begin < end:
            _fail("SEGMENT_ENDPOINTS_NONCANONICAL")
        key = (begin, end, tuple(members))
        canonical_keys.append(key)
        result.append({
            "id": index,
            "boundary_member_ids": members,
            "begin": begin,
            "end": end,
            "direction": _sub(end, begin),
        })
    if canonical_keys != sorted(set(canonical_keys)):
        _fail("BASE_STROKE_ORDER_NONCANONICAL")
    return result


def compile_fragment_root(document: Any) -> dict[str, Any]:
    strokes = validate_root(document)
    point_members: dict[tuple[Fraction, Fraction], set[int]] = defaultdict(set)

    for left_index, left in enumerate(strokes):
        p, r = left["begin"], left["direction"]
        for right in strokes[left_index + 1:]:
            q, s = right["begin"], right["direction"]
            denominator = _cross(r, s)
            q_minus_p = _sub(q, p)
            if denominator == 0:
                if _cross(q_minus_p, r) == 0:
                    if _positive_collinear_overlap(p, left["end"], q, right["end"]):
                        _fail("COLLINEAR_OVERLAP_UNSUPPORTED")
                    shared_endpoints = {p, left["end"]}.intersection({q, right["end"]})
                    if len(shared_endpoints) == 1:
                        endpoint = next(iter(shared_endpoints))
                        point_members[(Fraction(endpoint[0]), Fraction(endpoint[1]))].update(
                            (left["id"], right["id"])
                        )
                continue
            t = Fraction(_cross(q_minus_p, s), denominator)
            u = Fraction(_cross(q_minus_p, r), denominator)
            if 0 <= t <= 1 and 0 <= u <= 1:
                x = Fraction(p[0]) + t * r[0]
                y = Fraction(p[1]) + t * r[1]
                point_members[(x, y)].update((left["id"], right["id"]))

    ordered_points = sorted(point_members)
    if len(ordered_points) > MAX_INTERSECTIONS:
        _fail("INTERSECTIONS_OUTSIDE_CAPACITY")
    intersections: list[dict[str, Any]] = []
    point_to_region: dict[tuple[Fraction, Fraction], int] = {}
    for region_id, point in enumerate(ordered_points):
        members = sorted(point_members[point])
        point_to_region[point] = region_id
        intersections.append({
            "id": region_id,
            "point_q4_exact": [_rat(point[0]), _rat(point[1])],
            "base_stroke_ids": members,
        })

    cuts_by_stroke: dict[int, list[tuple[Fraction, int]]] = defaultdict(list)
    for point, region_id in point_to_region.items():
        for stroke_id in sorted(point_members[point]):
            stroke = strokes[stroke_id]
            p, r = stroke["begin"], stroke["direction"]
            if r[0] != 0:
                t = (point[0] - p[0]) / r[0]
            else:
                t = (point[1] - p[1]) / r[1]
            if 0 < t < 1:
                cuts_by_stroke[stroke_id].append((t, region_id))

    split_sites: list[dict[str, Any]] = []
    cut_ids: dict[tuple[int, Fraction], int] = {}
    for stroke in strokes:
        unique_cuts = sorted(set(cuts_by_stroke[stroke["id"]]))
        for t, region_id in unique_cuts:
            cut_id = len(split_sites)
            cut_ids[(stroke["id"], t)] = cut_id
            split_sites.append({
                "id": cut_id,
                "base_stroke_id": stroke["id"],
                "intersection_id": region_id,
                "t": _rat(t),
                "point_q4_exact": _rat_point(stroke["begin"], stroke["direction"], t),
            })

    expected_fragment_count = sum(len(cuts_by_stroke[stroke["id"]]) + 1 for stroke in strokes)
    if expected_fragment_count > MAX_ATOMIC_FRAGMENTS:
        _fail("ATOMIC_FRAGMENTS_OUTSIDE_CAPACITY")
    fragments: list[dict[str, Any]] = []
    for stroke in strokes:
        cuts = sorted(set(cuts_by_stroke[stroke["id"]]))
        begin_point = (Fraction(stroke["begin"][0]), Fraction(stroke["begin"][1]))
        end_point = (Fraction(stroke["end"][0]), Fraction(stroke["end"][1]))
        knots = [
            (Fraction(0), point_to_region.get(begin_point)),
            *cuts,
            (Fraction(1), point_to_region.get(end_point)),
        ]
        for (begin_t, begin_region), (end_t, end_region) in zip(knots, knots[1:]):
            fragments.append({
                "id": len(fragments),
                "base_stroke_id": stroke["id"],
                "boundary_member_ids": stroke["boundary_member_ids"],
                "begin_t": _rat(begin_t),
                "end_t": _rat(end_t),
                "begin_point_q4_exact": _rat_point(stroke["begin"], stroke["direction"], begin_t),
                "end_point_q4_exact": _rat_point(stroke["begin"], stroke["direction"], end_t),
                "begin_intersection_id": begin_region,
                "end_intersection_id": end_region,
            })

    return {
        "schema_version": "chemgraph-fragment-root-compiled-v0.4-research",
        "training_authorized": False,
        "source_root_sha256": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
        "base_stroke_count": len(strokes),
        "intersection_count": len(intersections),
        "split_site_count": len(split_sites),
        "atomic_fragment_count": len(fragments),
        "intersections": intersections,
        "split_sites": split_sites,
        "atomic_fragments": fragments,
    }


def compare_declared_root(document: Any, declared: Any) -> dict[str, Any]:
    expected = compile_fragment_root(document)
    if not _matches_strict_json_shape(declared, expected):
        return {"status": "INVALID", "exact": False}
    expected_bodies = [{key: value for key, value in item.items() if key != "id"}
                       for item in expected["atomic_fragments"]]
    declared_items = declared.get("atomic_fragments")
    if not isinstance(declared_items, list) or any(not isinstance(item, dict) for item in declared_items):
        return {"status": "INVALID", "exact": False}
    declared_ids = [item.get("id") for item in declared_items]
    if any(type(value) is not int for value in declared_ids):
        return {"status": "INVALID", "exact": False}
    if len(declared_ids) != len(set(declared_ids)):
        return {"status": "FRAGMENT_ID_CONFLICT", "exact": False}
    declared_bodies = [{key: value for key, value in item.items() if key != "id"}
                       for item in declared_items]
    try:
        expected_set = {_canonical_bytes(item) for item in expected_bodies}
        declared_body_bytes = [_canonical_bytes(item) for item in declared_bodies]
    except (TypeError, ValueError):
        return {"status": "INVALID", "exact": False}
    declared_set = set(declared_body_bytes)
    if len(declared_body_bytes) != len(declared_set):
        return {"status": "DUPLICATE_FRAGMENT_BODY", "exact": False}
    if expected_set != declared_set:
        return {
            "status": "FRAGMENT_UNIVERSE_INCOMPLETE",
            "exact": False,
            "missing_count": len(expected_set - declared_set),
            "extra_count": len(declared_set - expected_set),
        }
    if _canonical_bytes(declared) != _canonical_bytes(expected):
        return {
            "status": "FRAGMENT_UNIVERSE_NONCANONICAL",
            "exact": False,
            "missing_count": 0,
            "extra_count": 0,
        }
    return {"status": "FRAGMENT_UNIVERSE_EXACT", "exact": True}
