"""Enumerate local intersection-support partitions from actual raster supports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from formal.chemgraph.challenge_v0_4.rasterizer_v0_4 import (
    CANVAS_SIZE,
    Q,
    LineStyle,
    Segment,
    encode_canonical_png,
    rasterize_gray,
    rasterize_support,
)


PHASES = ("edge", "center")
WIDTHS = (2, 3, 4, 5, 6)
CAPS = ("butt", "round", "square")
AA_PROFILES = ("binary_1x", "box_2x2", "box_4x4")
MAX_AMBIGUOUS_PIXELS = 20
RENDERER_PATH = Path(__file__).resolve().parents[2] / "challenge_v0_4" / "rasterizer_v0_4.py"


class OwnershipRealizationError(ValueError):
    """Closed failure for an exploratory raster configuration."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_runs(pixels: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(pixels))
    if not ordered:
        return []
    first_row, first_column = divmod(ordered[0], CANVAS_SIZE)
    runs: list[list[int]] = []
    row, start, end = first_row, first_column, first_column + 1
    for pixel in ordered[1:]:
        next_row, column = divmod(pixel, CANVAS_SIZE)
        if next_row == row and column == end:
            end += 1
            continue
        runs.append([row, start, end - start])
        row, start, end = next_row, column, column + 1
    runs.append([row, start, end - start])
    return runs


def segments_for_phase(phase: str) -> list[Segment]:
    if phase not in PHASES:
        raise OwnershipRealizationError("PHASE_OUTSIDE_PREREGISTRATION")
    offset = 0 if phase == "edge" else Q // 2
    return [
        Segment("H", 10 * Q + offset, 20 * Q + offset, 30 * Q + offset, 20 * Q + offset),
        Segment("V1", 18 * Q + offset, 10 * Q + offset, 18 * Q + offset, 20 * Q + offset),
        Segment("V2", 22 * Q + offset, 20 * Q + offset, 22 * Q + offset, 30 * Q + offset),
    ]


def _connected(pixels: set[int]) -> bool:
    if not pixels:
        return False
    unseen = set(pixels)
    start = min(unseen)
    unseen.remove(start)
    stack = [start]
    while stack:
        pixel = stack.pop()
        row, column = divmod(pixel, CANVAS_SIZE)
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                next_row, next_column = row + row_delta, column + column_delta
                if not (0 <= next_row < CANVAS_SIZE and 0 <= next_column < CANVAS_SIZE):
                    continue
                neighbor = next_row * CANVAS_SIZE + next_column
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return not unseen


def _partition_is_locally_admissible(
    universe: set[int], left: set[int], right: set[int], seeds: tuple[int, int],
    fibers: dict[int, tuple[int, ...]],
) -> bool:
    return (
        seeds[0] in left
        and seeds[1] in right
        and not left.intersection(right)
        and left.union(right) == universe
        and _connected(left)
        and _connected(right)
        and all({0, 1}.issubset(fibers[pixel]) for pixel in left)
        and all({0, 2}.issubset(fibers[pixel]) for pixel in right)
    )


def analyze_configuration(
    phase: str, width_pixels: int, cap: str, antialias_profile: str,
    *, include_witness: bool = False,
) -> dict[str, Any]:
    if width_pixels not in WIDTHS or cap not in CAPS or antialias_profile not in AA_PROFILES:
        raise OwnershipRealizationError("CONFIGURATION_OUTSIDE_PREREGISTRATION")
    segments = segments_for_phase(phase)
    style = LineStyle(
        width_q16=width_pixels * Q,
        cap=cap,
        antialias_profile=antialias_profile,
    )
    supports = [rasterize_support([segment], style) for segment in segments]
    foreground = set().union(*supports)
    fibers = {
        pixel: tuple(index for index, support in enumerate(supports) if pixel in support)
        for pixel in foreground
    }
    cross_overlap = {pixel for pixel, carrier_ids in fibers.items() if len(carrier_ids) > 1}
    universe = cross_overlap
    left_eligible = {
        pixel for pixel in universe if {0, 1}.issubset(fibers[pixel])
    }
    right_eligible = {
        pixel for pixel in universe if {0, 2}.issubset(fibers[pixel])
    }
    unowned = universe - left_eligible - right_eligible
    left_fixed = {
        pixel for pixel in left_eligible if pixel not in right_eligible
    }
    right_fixed = {
        pixel for pixel in right_eligible if pixel not in left_eligible
    }
    ambiguous = left_eligible.intersection(right_eligible)
    ambiguity_within_capacity = len(ambiguous) <= MAX_AMBIGUOUS_PIXELS

    left_seed = 20 * CANVAS_SIZE + 18
    right_seed = 20 * CANVAS_SIZE + 22
    seeds = left_seed, right_seed
    seeds_available = left_seed in left_eligible and right_seed in right_eligible

    ambiguous_order = sorted(ambiguous)
    locally_admissible_count = 0
    first_partition: tuple[set[int], set[int]] | None = None
    second_partition: tuple[set[int], set[int]] | None = None
    assignment_count = (
        0 if unowned or not seeds_available or not ambiguity_within_capacity
        else 1 << len(ambiguous_order)
    )
    for assignment in range(assignment_count):
        left = left_fixed | {
            pixel for index, pixel in enumerate(ambiguous_order)
            if assignment & (1 << index)
        }
        right = right_fixed | (ambiguous - left)
        if _partition_is_locally_admissible(universe, left, right, seeds, fibers):
            locally_admissible_count += 1
            if first_partition is None:
                first_partition = left, right
            elif second_partition is None:
                second_partition = left, right

    reported_count: int | None = (
        locally_admissible_count if ambiguity_within_capacity else None
    )

    gray = rasterize_gray(segments, style)
    png = encode_canonical_png(gray)
    nonwhite = {index for index, value in enumerate(gray) if value != 255}
    result: dict[str, Any] = {
        "antialias_profile": antialias_profile,
        "ambiguity_within_capacity": ambiguity_within_capacity,
        "cap": cap,
        "cross_carrier_overlap_pixel_count": len(cross_overlap),
        "foreground_pixel_count": len(foreground),
        "enumeration_status": (
            "ENUMERATED" if ambiguity_within_capacity else "OVER_CAPACITY"
        ),
        "locally_admissible_partition_count": reported_count,
        "left_fixed_pixel_count": len(left_fixed),
        "ownership_universe_pixel_count": len(universe),
        "phase": phase,
        "png_sha256": sha256(png),
        "raw_gray_sha256": sha256(gray),
        "unowned_overlap_pixel_count": len(unowned),
        "right_fixed_pixel_count": len(right_fixed),
        "seeds_available": seeds_available,
        "support_union_exact": foreground == nonwhite,
        "triple_overlap_pixel_count": len(ambiguous),
        "width_pixels": width_pixels,
    }
    if include_witness:
        if first_partition is None or second_partition is None:
            raise OwnershipRealizationError("REPRESENTATIVE_HAS_FEWER_THAN_TWO_WORLDS")
        result["carrier_supports"] = [
            {
                "carrier_id": segment.segment_id,
                "pixel_count": len(support),
                "support_runs": encode_runs(support),
            }
            for segment, support in zip(segments, supports, strict=True)
        ]
        result["ownership_sets"] = {
            "ambiguous_runs": encode_runs(ambiguous),
            "left_fixed_runs": encode_runs(left_fixed),
            "ownership_universe_runs": encode_runs(universe),
            "unowned_overlap_runs": encode_runs(unowned),
            "right_fixed_runs": encode_runs(right_fixed),
        }
        result["partition_annotations"] = [
            {
                "left_region_runs": encode_runs(left),
                "right_region_runs": encode_runs(right),
            }
            for left, right in (first_partition, second_partition)
        ]
        result["png_bytes"] = png
    return result


def build_matrix() -> dict[str, Any]:
    configurations = [
        analyze_configuration(phase, width, cap, aa)
        for phase in PHASES
        for width in WIDTHS
        for cap in CAPS
        for aa in AA_PROFILES
    ]
    return {
        "configuration_count": len(configurations),
        "configurations": configurations,
        "multiple_locally_admissible_configuration_count": sum(
            item["support_union_exact"]
            and isinstance(item["locally_admissible_partition_count"], int)
            and item["locally_admissible_partition_count"] >= 2
            for item in configurations
        ),
        "max_enumerated_locally_admissible_partition_count": max(
            item["locally_admissible_partition_count"]
            for item in configurations
            if isinstance(item["locally_admissible_partition_count"], int)
        ),
        "renderer_sha256": sha256(RENDERER_PATH.read_bytes()),
        "schema_version": "png-ownership-realization-v0.5-research",
        "status": "GO" if any(
            item["support_union_exact"]
            and isinstance(item["locally_admissible_partition_count"], int)
            and item["locally_admissible_partition_count"] >= 2
            for item in configurations
        ) else "NO_GO",
        "training_authorized": False,
    }


def representative_witness() -> tuple[dict[str, Any], bytes]:
    result = analyze_configuration("edge", 5, "round", "box_4x4", include_witness=True)
    png = result.pop("png_bytes")
    return {
        "configuration": result,
        "schema_version": "png-ownership-realization-witness-v0.5-research",
        "training_authorized": False,
    }, png
