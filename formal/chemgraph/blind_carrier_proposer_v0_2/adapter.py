"""Adapt chemistry-free segment/group evidence into Blind Carrier v0.2 JSON.

This module deliberately has no image extractor, model, filesystem access, or
chemical dependency. It accepts an already extracted physical representation
and either emits one canonical v0.2 interpretation or a closed-schema reject.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable, Sequence


PointQ4 = tuple[int, int]
SupportRun = tuple[int, int, int]
AA_MARGIN_Q4 = 16
GEOMETRY_SAMPLE_STEP_Q4 = 16
SCHEMA_VERSION = "chemgraph-blind-carrier-prediction-v0.2"


@dataclass(frozen=True, slots=True)
class VisibleSegment:
    """One straight physical stroke with explicit foreground ownership support."""

    id: int
    begin_q4: PointQ4
    end_q4: PointQ4
    width_q4: int
    support_runs: tuple[SupportRun, ...]


@dataclass(frozen=True, slots=True)
class TerminalHint:
    """Opaque upstream terminal identity plus a point used only for orientation."""

    id: int
    point_q4: PointQ4


@dataclass(frozen=True, slots=True)
class SemanticBondGroup:
    """One semantic bond represented by one to three physical lane segments."""

    segment_ids: tuple[int, ...]
    begin_hint: TerminalHint
    end_hint: TerminalHint


class _Reject(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _reject(reason: str) -> bytes:
    return _canonical_bytes({
        "reason": reason,
        "schema_version": SCHEMA_VERSION,
        "status": "reject",
        "training_authorized": False,
    })


def _is_int(value: object) -> bool:
    return type(value) is int


def _point_in_capacity(point: PointQ4) -> bool:
    return (
        isinstance(point, tuple)
        and len(point) == 2
        and all(_is_int(value) and 0 <= value <= 12288 for value in point)
    )


def _carrier_radius_q4(segment: VisibleSegment) -> int:
    return (segment.width_q4 + 1) // 2 + AA_MARGIN_Q4


def _decode_support(runs: Sequence[SupportRun]) -> set[tuple[int, int]]:
    if not runs or len(runs) > 100000:
        raise _Reject("outside_capacity")
    pixels: set[tuple[int, int]] = set()
    for run in runs:
        if (
            not isinstance(run, tuple)
            or len(run) != 3
            or not all(_is_int(value) for value in run)
        ):
            raise _Reject("unsupported_geometry")
        row, start, length = run
        if not (0 <= row <= 767 and 0 <= start <= 767 and 1 <= length <= 768):
            raise _Reject("outside_capacity")
        if start + length > 768:
            raise _Reject("outside_capacity")
        run_pixels = {(column, row) for column in range(start, start + length)}
        if pixels & run_pixels:
            raise _Reject("unsupported_geometry")
        pixels.update(run_pixels)
    if not (1 <= len(pixels) <= 589824):
        raise _Reject("outside_capacity")
    return pixels


def _encode_support(pixels: set[tuple[int, int]]) -> list[list[int]]:
    by_row: dict[int, list[int]] = {}
    for column, row in sorted(pixels, key=lambda item: (item[1], item[0])):
        by_row.setdefault(row, []).append(column)
    runs: list[list[int]] = []
    for row in sorted(by_row):
        columns = by_row[row]
        start = previous = columns[0]
        for column in columns[1:]:
            if column != previous + 1:
                runs.append([row, start, previous - start + 1])
                start = column
            previous = column
        runs.append([row, start, previous - start + 1])
    return runs


def _point_near_line_q4(
    point: PointQ4, begin: PointQ4, end: PointQ4, radius_q4: int,
) -> bool:
    px, py = point
    ax, ay = begin
    bx, by = end
    dx, dy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return wx * wx + wy * wy <= radius_q4 * radius_q4
    projection = wx * dx + wy * dy
    if projection <= 0:
        return wx * wx + wy * wy <= radius_q4 * radius_q4
    if projection >= length_squared:
        ex, ey = px - bx, py - by
        return ex * ex + ey * ey <= radius_q4 * radius_q4
    cross = wx * dy - wy * dx
    return cross * cross <= radius_q4 * radius_q4 * length_squared


def _support_near_point_q4(
    point: PointQ4, support: set[tuple[int, int]], radius_q4: int,
) -> bool:
    center_x, center_y = point
    pixel_radius = (radius_q4 + 15) // 16 + 1
    base_x, base_y = center_x // 16, center_y // 16
    radius_squared = radius_q4 * radius_q4
    for row in range(max(0, base_y - pixel_radius), min(767, base_y + pixel_radius) + 1):
        for column in range(max(0, base_x - pixel_radius), min(767, base_x + pixel_radius) + 1):
            if (column, row) not in support:
                continue
            dx = 16 * column + 8 - center_x
            dy = 16 * row + 8 - center_y
            if dx * dx + dy * dy <= radius_squared:
                return True
    return False


def _geometry_has_support(
    segment: VisibleSegment, support: set[tuple[int, int]],
) -> bool:
    radius = _carrier_radius_q4(segment)
    for column, row in support:
        center = (16 * column + 8, 16 * row + 8)
        if not _point_near_line_q4(center, segment.begin_q4, segment.end_q4, radius):
            return False
    if not _support_near_point_q4(segment.begin_q4, support, radius):
        return False
    if not _support_near_point_q4(segment.end_q4, support, radius):
        return False
    dx = segment.end_q4[0] - segment.begin_q4[0]
    dy = segment.end_q4[1] - segment.begin_q4[1]
    length_q4 = math.isqrt(dx * dx + dy * dy)
    steps = max(1, (length_q4 + GEOMETRY_SAMPLE_STEP_Q4 - 1) // GEOMETRY_SAMPLE_STEP_Q4)
    for index in range(steps + 1):
        sample = (
            (segment.begin_q4[0] * (steps - index) + segment.end_q4[0] * index + steps // 2) // steps,
            (segment.begin_q4[1] * (steps - index) + segment.end_q4[1] * index + steps // 2) // steps,
        )
        if not _support_near_point_q4(sample, support, radius):
            return False
    return True


def _squared_distance(left: PointQ4, right: PointQ4) -> int:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    return dx * dx + dy * dy


def _orient_segment(
    segment: VisibleSegment, begin_hint: TerminalHint, end_hint: TerminalHint,
) -> tuple[int, int]:
    direct = (
        _squared_distance(segment.begin_q4, begin_hint.point_q4)
        + _squared_distance(segment.end_q4, end_hint.point_q4)
    )
    reverse = (
        _squared_distance(segment.end_q4, begin_hint.point_q4)
        + _squared_distance(segment.begin_q4, end_hint.point_q4)
    )
    if direct == reverse:
        raise _Reject("unsupported_geometry")
    return (0, 1) if direct < reverse else (1, 0)


def _disks_connected(
    references: Sequence[tuple[VisibleSegment, int]],
) -> bool:
    if not references:
        return False
    visited = {0}
    pending = [0]
    while pending:
        left_index = pending.pop()
        left_segment, left_endpoint = references[left_index]
        left_point = left_segment.begin_q4 if left_endpoint == 0 else left_segment.end_q4
        left_radius = _carrier_radius_q4(left_segment)
        for right_index, (right_segment, right_endpoint) in enumerate(references):
            if right_index in visited:
                continue
            right_point = right_segment.begin_q4 if right_endpoint == 0 else right_segment.end_q4
            radius = left_radius + _carrier_radius_q4(right_segment)
            if _squared_distance(left_point, right_point) <= radius * radius:
                visited.add(right_index)
                pending.append(right_index)
    return len(visited) == len(references)


def _parallel_carrier_tubes_overlap(left: VisibleSegment, right: VisibleSegment) -> bool:
    left_dx = left.end_q4[0] - left.begin_q4[0]
    left_dy = left.end_q4[1] - left.begin_q4[1]
    right_dx = right.end_q4[0] - right.begin_q4[0]
    right_dy = right.end_q4[1] - right.begin_q4[1]
    if left_dx * right_dy - left_dy * right_dx != 0:
        return False
    length_squared = left_dx * left_dx + left_dy * left_dy
    offset_x = right.begin_q4[0] - left.begin_q4[0]
    offset_y = right.begin_q4[1] - left.begin_q4[1]
    right_begin_projection = offset_x * left_dx + offset_y * left_dy
    right_end_projection = (
        (right.end_q4[0] - left.begin_q4[0]) * left_dx
        + (right.end_q4[1] - left.begin_q4[1]) * left_dy
    )
    projection_min, projection_max = sorted((right_begin_projection, right_end_projection))
    if max(0, projection_min) >= min(length_squared, projection_max):
        return False
    cross = left_dx * offset_y - left_dy * offset_x
    minimum_separation = (
        (left.width_q4 + 1) // 2
        + (right.width_q4 + 1) // 2
        + AA_MARGIN_Q4
    )
    return cross * cross < minimum_separation * minimum_separation * length_squared


def _same_forward_direction(
    left: VisibleSegment, left_orientation: tuple[int, int],
    right: VisibleSegment, right_orientation: tuple[int, int],
) -> bool:
    left_points = (left.begin_q4, left.end_q4)
    right_points = (right.begin_q4, right.end_q4)
    left_dx = left_points[left_orientation[1]][0] - left_points[left_orientation[0]][0]
    left_dy = left_points[left_orientation[1]][1] - left_points[left_orientation[0]][1]
    right_dx = right_points[right_orientation[1]][0] - right_points[right_orientation[0]][0]
    right_dy = right_points[right_orientation[1]][1] - right_points[right_orientation[0]][1]
    return (
        left_dx * right_dy - left_dy * right_dx == 0
        and left_dx * right_dx + left_dy * right_dy > 0
    )


def _rounded_mean(points: Sequence[PointQ4]) -> list[int]:
    count = len(points)
    return [
        (sum(point[axis] for point in points) + count // 2) // count
        for axis in (0, 1)
    ]


def _validate_hint(hint: TerminalHint) -> None:
    if not _is_int(hint.id) or not (0 <= hint.id <= 65535):
        raise _Reject("outside_capacity")
    if not _point_in_capacity(hint.point_q4):
        raise _Reject("outside_capacity")


def _build_prediction(
    visible_segments: Iterable[VisibleSegment],
    semantic_groups: Iterable[SemanticBondGroup],
) -> bytes:
    segments = list(visible_segments)
    groups = list(semantic_groups)
    if not segments or not groups:
        raise _Reject("no_supported_carrier")
    # This adapter emits exactly one lane per physical segment.  The frozen
    # prediction schema permits at most 512 lanes even though it permits more
    # segment records for future multi-segment lanes.
    if len(segments) > 512 or len(groups) > 512:
        raise _Reject("outside_capacity")

    input_ids = [segment.id for segment in segments]
    if (
        any(not _is_int(segment_id) or segment_id < 0 for segment_id in input_ids)
        or len(input_ids) != len(set(input_ids))
    ):
        raise _Reject("unsupported_geometry")
    segments.sort(key=lambda segment: segment.id)
    segment_by_input_id = {segment.id: segment for segment in segments}
    output_id_by_input_id = {
        segment.id: output_id for output_id, segment in enumerate(segments)
    }

    hint_by_id: dict[int, TerminalHint] = {}
    hint_id_by_point: dict[PointQ4, int] = {}
    segment_group_count = {segment.id: 0 for segment in segments}
    terminal_ref_count: dict[int, int] = {}
    terminal_segment_union: dict[int, set[int]] = {}
    prepared_groups: list[tuple[SemanticBondGroup, tuple[int, ...]]] = []
    seen_hint_pairs: set[tuple[int, int]] = set()
    for raw_group in groups:
        _validate_hint(raw_group.begin_hint)
        _validate_hint(raw_group.end_hint)
        if raw_group.begin_hint.id == raw_group.end_hint.id:
            raise _Reject("unsupported_geometry")
        begin_hint, end_hint = sorted(
            (raw_group.begin_hint, raw_group.end_hint), key=lambda hint: hint.point_q4,
        )
        group = SemanticBondGroup(raw_group.segment_ids, begin_hint, end_hint)
        for hint in (begin_hint, end_hint):
            previous = hint_by_id.get(hint.id)
            if previous is not None and previous.point_q4 != hint.point_q4:
                raise _Reject("unsupported_geometry")
            point_owner = hint_id_by_point.get(hint.point_q4)
            if point_owner is not None and point_owner != hint.id:
                raise _Reject("unsupported_geometry")
            hint_by_id[hint.id] = hint
            hint_id_by_point[hint.point_q4] = hint.id
        if not (1 <= len(group.segment_ids) <= 3):
            raise _Reject("outside_capacity")
        if any(not _is_int(segment_id) for segment_id in group.segment_ids):
            raise _Reject("unsupported_geometry")
        if len(group.segment_ids) != len(set(group.segment_ids)):
            raise _Reject("unsupported_geometry")
        if set(group.segment_ids) - set(segment_by_input_id):
            raise _Reject("unsupported_geometry")
        unordered_pair = (begin_hint.id, end_hint.id)
        if unordered_pair in seen_hint_pairs:
            raise _Reject("unsupported_geometry")
        seen_hint_pairs.add(unordered_pair)
        ordered_ids = tuple(sorted(group.segment_ids, key=output_id_by_input_id.__getitem__))
        for segment_id in ordered_ids:
            segment_group_count[segment_id] += 1
        for hint in (begin_hint, end_hint):
            terminal_ref_count[hint.id] = terminal_ref_count.get(hint.id, 0) + len(ordered_ids)
            terminal_segment_union.setdefault(hint.id, set()).update(ordered_ids)
            if terminal_ref_count[hint.id] > 64 or len(terminal_segment_union[hint.id]) > 64:
                raise _Reject("outside_capacity")
        prepared_groups.append((group, ordered_ids))

    if any(count != 1 for count in segment_group_count.values()):
        raise _Reject("unsupported_geometry")
    if len(hint_by_id) > 512:
        raise _Reject("outside_capacity")

    support_by_input_id: dict[int, set[tuple[int, int]]] = {}
    for segment in segments:
        if (
            not _point_in_capacity(segment.begin_q4)
            or not _point_in_capacity(segment.end_q4)
            or segment.begin_q4 == segment.end_q4
        ):
            raise _Reject("unsupported_geometry")
        if not _is_int(segment.width_q4) or not (1 <= segment.width_q4 <= 1024):
            raise _Reject("outside_capacity")
        support = _decode_support(segment.support_runs)
        if not _geometry_has_support(segment, support):
            raise _Reject("no_supported_carrier")
        support_by_input_id[segment.id] = support

    for segment in segments:
        support = support_by_input_id[segment.id]
        other_support = set().union(*(
            support_by_input_id[other.id]
            for other in segments if other.id != segment.id
        )) if len(segments) > 1 else set()
        if not support - other_support:
            raise _Reject("unsupported_geometry")

    for left_index, left in enumerate(segments):
        for right in segments[left_index + 1:]:
            if _parallel_carrier_tubes_overlap(left, right):
                raise _Reject("unsupported_geometry")

    normalized_groups: list[
        tuple[SemanticBondGroup, tuple[int, ...], dict[int, tuple[int, int]]]
    ] = []
    for group, ordered_ids in prepared_groups:
        orientations = {
            segment_id: _orient_segment(
                segment_by_input_id[segment_id], group.begin_hint, group.end_hint,
            )
            for segment_id in ordered_ids
        }
        for left_index, left_id in enumerate(ordered_ids):
            left = segment_by_input_id[left_id]
            for right_id in ordered_ids[left_index + 1:]:
                right = segment_by_input_id[right_id]
                if support_by_input_id[left_id] & support_by_input_id[right_id]:
                    raise _Reject("unsupported_geometry")
                if not _same_forward_direction(
                    left, orientations[left_id], right, orientations[right_id],
                ):
                    raise _Reject("unsupported_geometry")
        begin_refs = [
            (segment_by_input_id[segment_id], orientations[segment_id][0])
            for segment_id in ordered_ids
        ]
        end_refs = [
            (segment_by_input_id[segment_id], orientations[segment_id][1])
            for segment_id in ordered_ids
        ]
        if not _disks_connected(begin_refs) or not _disks_connected(end_refs):
            raise _Reject("unsupported_geometry")
        normalized_groups.append((group, ordered_ids, orientations))

    normalized_groups.sort(key=lambda item: (
        item[0].begin_hint.point_q4,
        item[0].end_hint.point_q4,
        tuple(output_id_by_input_id[segment_id] for segment_id in item[1]),
    ))
    alignment_by_hint: dict[int, list[tuple[int, int]]] = {
        hint_id: [] for hint_id in hint_by_id
    }
    for group, ordered_ids, orientations in normalized_groups:
        for segment_id in ordered_ids:
            output_id = output_id_by_input_id[segment_id]
            begin_endpoint, end_endpoint = orientations[segment_id]
            alignment_by_hint[group.begin_hint.id].append((output_id, begin_endpoint))
            alignment_by_hint[group.end_hint.id].append((output_id, end_endpoint))

    for hint_id, references in alignment_by_hint.items():
        physical_refs = [
            (
                segments[output_segment_id],
                endpoint_index,
            )
            for output_segment_id, endpoint_index in references
        ]
        if not _disks_connected(physical_refs):
            raise _Reject("unsupported_geometry")

    # A single-interpretation output cannot decide that overlapping endpoint
    # regions are distinct junctions.  Doing so would let tiny hint offsets
    # split one visible endpoint into disconnected graph vertices.
    hint_ids = sorted(alignment_by_hint)
    for left_index, left_hint_id in enumerate(hint_ids):
        for right_hint_id in hint_ids[left_index + 1:]:
            for left_segment_id, left_endpoint in alignment_by_hint[left_hint_id]:
                left_segment = segments[left_segment_id]
                left_point = (
                    left_segment.begin_q4 if left_endpoint == 0
                    else left_segment.end_q4
                )
                left_radius = _carrier_radius_q4(left_segment)
                for right_segment_id, right_endpoint in alignment_by_hint[right_hint_id]:
                    right_segment = segments[right_segment_id]
                    right_point = (
                        right_segment.begin_q4 if right_endpoint == 0
                        else right_segment.end_q4
                    )
                    radius = left_radius + _carrier_radius_q4(right_segment)
                    if _squared_distance(left_point, right_point) <= radius * radius:
                        raise _Reject("unsupported_geometry")

    output_segments = []
    for output_id, segment in enumerate(segments):
        support = support_by_input_id[segment.id]
        output_segments.append({
            "id": output_id,
            "polyline_q4": [list(segment.begin_q4), list(segment.end_q4)],
            "support_pixel_count": len(support),
            "support_runs": _encode_support(support),
            "width_q4": segment.width_q4,
        })

    terminal_records: list[tuple[int, list[int], list[tuple[int, int]]]] = []
    for hint_id in hint_by_id:
        references = sorted(set(alignment_by_hint[hint_id]))
        if len(references) != len(alignment_by_hint[hint_id]):
            raise _Reject("unsupported_geometry")
        points = [
            segments[segment_id].begin_q4 if endpoint_index == 0
            else segments[segment_id].end_q4
            for segment_id, endpoint_index in references
        ]
        terminal_records.append((hint_id, _rounded_mean(points), references))
    terminal_records.sort(key=lambda item: (tuple(item[1]), tuple(item[2])))
    terminal_id_by_hint = {
        hint_id: terminal_id
        for terminal_id, (hint_id, _center, _references) in enumerate(terminal_records)
    }
    terminals = []
    for terminal_id, (_hint_id, center, references) in enumerate(terminal_records):
        terminals.append({
            "center_q4": center,
            "endpoint_alignment": [
                {"endpoint_index": endpoint_index, "segment_id": segment_id}
                for segment_id, endpoint_index in references
            ],
            "id": terminal_id,
            "support_segment_ids": sorted({segment_id for segment_id, _ in references}),
        })

    lanes = []
    for output_id, segment in enumerate(segments):
        owner_group = next(
            item for item in normalized_groups if segment.id in item[1]
        )
        group, _, orientations = owner_group
        lanes.append({
            "begin_terminal_id": terminal_id_by_hint[group.begin_hint.id],
            "end_terminal_id": terminal_id_by_hint[group.end_hint.id],
            "id": output_id,
            "segment_ids": [output_id],
        })

    bond_groups = sorted(normalized_groups, key=lambda item: (
        terminal_id_by_hint[item[0].begin_hint.id],
        terminal_id_by_hint[item[0].end_hint.id],
        tuple(output_id_by_input_id[segment_id] for segment_id in item[1]),
    ))
    bonds = []
    for bond_id, (group, ordered_ids, _orientations) in enumerate(bond_groups):
        begin_terminal_id = terminal_id_by_hint[group.begin_hint.id]
        end_terminal_id = terminal_id_by_hint[group.end_hint.id]
        lane_ids = sorted(output_id_by_input_id[segment_id] for segment_id in ordered_ids)
        bonds.append({
            "begin_terminal_id": begin_terminal_id,
            "end_terminal_id": end_terminal_id,
            "id": bond_id,
            "lane_endpoint_alignment": [
                {
                    "begin_terminal_id": begin_terminal_id,
                    "end_terminal_id": end_terminal_id,
                    "lane_id": lane_id,
                }
                for lane_id in lane_ids
            ],
            "lane_ids": lane_ids,
            "order": len(lane_ids),
        })

    prediction = {
        "bond_hypotheses": bonds,
        "image": {
            "coordinate_format": "Q4-pixel",
            "height": 768,
            "width": 768,
        },
        "interpretations": [{
            "id": 0,
            "ownership": [
                {
                    "owner_id": output_id,
                    "owner_kind": "bond_lane",
                    "segment_id": output_id,
                }
                for output_id in range(len(segments))
            ],
            "selected_bond_ids": list(range(len(bonds))),
        }],
        "lane_hypotheses": lanes,
        "schema_version": SCHEMA_VERSION,
        "segments": output_segments,
        "status": "ok",
        "terminals": terminals,
        "training_authorized": False,
    }
    return _canonical_bytes(prediction)


def propose_prediction(
    visible_segments: Iterable[VisibleSegment],
    semantic_groups: Iterable[SemanticBondGroup],
) -> bytes:
    """Return canonical v0.2 JSON bytes; invalid evidence becomes a closed reject."""

    try:
        return _build_prediction(visible_segments, semantic_groups)
    except _Reject as error:
        return _reject(error.reason)
