"""Analyze 8-connected overlap components without claiming semantic ownership."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from typing import Any

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    compile_fragment_root,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    compile_support_partition,
)


class IntersectionOwnershipError(ValueError):
    """Closed structural failure in the component probe."""


def _exact_coordinate(value: Any) -> Fraction:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(type(item) is int for item in value)
        and value[1] > 0
    ):
        raise IntersectionOwnershipError("INVALID_EXACT_INTERSECTION_POINT")
    coordinate = Fraction(value[0], value[1])
    if [coordinate.numerator, coordinate.denominator] != value:
        raise IntersectionOwnershipError("NONCANONICAL_EXACT_INTERSECTION_POINT")
    return coordinate


def _anchor_pixel(point_q4_exact: Any) -> tuple[int, int]:
    if not isinstance(point_q4_exact, list) or len(point_q4_exact) != 2:
        raise IntersectionOwnershipError("INVALID_EXACT_INTERSECTION_POINT")
    x = _exact_coordinate(point_q4_exact[0])
    y = _exact_coordinate(point_q4_exact[1])
    column = x.numerator // (16 * x.denominator)
    row = y.numerator // (16 * y.denominator)
    if not (0 <= row < 768 and 0 <= column < 768):
        raise IntersectionOwnershipError("INTERSECTION_ANCHOR_OUTSIDE_RASTER")
    return row, column


def _components(pixels: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    unseen = set(pixels)
    result: list[list[tuple[int, int]]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        component = [start]
        while queue:
            row, column = queue.popleft()
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor = row + row_delta, column + column_delta
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
        result.append(sorted(component))
    return sorted(result, key=lambda component: component[0])


def analyze_overlap_components(support_input: Any) -> dict[str, Any]:
    support = compile_support_partition(support_input)
    geometry = compile_fragment_root(support_input["geometry_root"])
    overlap_fibers = {
        tuple(record["pixel"]): record["base_stroke_ids"]
        for record in support["cross_carrier_overlaps"]
    }
    components = _components(set(overlap_fibers))
    anchors = {
        intersection["id"]: _anchor_pixel(intersection["point_q4_exact"])
        for intersection in geometry["intersections"]
    }
    anchor_to_ids: dict[tuple[int, int], list[int]] = {}
    for intersection_id, pixel in anchors.items():
        anchor_to_ids.setdefault(pixel, []).append(intersection_id)

    regions: list[dict[str, Any]] = []
    unexplained_components = 0
    merged_components = 0
    fiber_mismatches = 0
    covered_intersections: set[int] = set()
    for component_id, component in enumerate(components):
        intersection_ids = sorted({
            intersection_id
            for pixel in component
            for intersection_id in anchor_to_ids.get(pixel, [])
        })
        if not intersection_ids:
            unexplained_components += 1
            continue
        if len(intersection_ids) != 1:
            merged_components += 1
            continue
        intersection_id = intersection_ids[0]
        covered_intersections.add(intersection_id)
        expected_carriers = geometry["intersections"][intersection_id]["base_stroke_ids"]
        actual_carriers = sorted({
            carrier_id for pixel in component for carrier_id in overlap_fibers[pixel]
        })
        if actual_carriers != expected_carriers:
            fiber_mismatches += 1
            continue
        regions.append({
            "id": len(regions),
            "component_id": component_id,
            "intersection_id": intersection_id,
            "anchor_pixel": list(anchors[intersection_id]),
            "base_stroke_ids": actual_carriers,
            "support_pixel_count": len(component),
            "support_pixels": [list(pixel) for pixel in component],
            "pixel_carrier_fibers": [
                {"pixel": list(pixel), "base_stroke_ids": overlap_fibers[pixel]}
                for pixel in component
            ],
        })

    missing_intersections = len(set(anchors) - covered_intersections)
    if merged_components:
        status = "MERGED_INTERSECTION_COMPONENTS"
    elif unexplained_components:
        status = "UNEXPLAINED_OVERLAP_COMPONENTS"
    elif missing_intersections:
        status = "INTERSECTIONS_WITHOUT_OVERLAP_ANCHOR"
    elif fiber_mismatches:
        status = "INTERSECTION_CARRIER_FIBER_MISMATCH"
    else:
        status = "STRUCTURALLY_EXACT_NOT_SEMANTICALLY_IDENTIFIED"
    return {
        "status": status,
        "semantic_identifiability_proven": False,
        "component_count": len(components),
        "intersection_count": len(anchors),
        "region_count": len(regions),
        "unexplained_component_count": unexplained_components,
        "merged_component_count": merged_components,
        "missing_intersection_count": missing_intersections,
        "carrier_fiber_mismatch_count": fiber_mismatches,
        "regions": regions,
    }
