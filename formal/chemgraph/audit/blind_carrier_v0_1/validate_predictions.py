#!/usr/bin/env python3
"""Validate blind carrier predictions structurally, without GT or scoring."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from formal.chemgraph.audit.blind_carrier_v0_1._common import (  # type: ignore
        DocumentError,
        PREDICTION_SCHEMA_SHA256,
        StructuredArgumentParser,
        ValidationIssue,
        emit_report,
        error_report,
        load_frozen_schema,
        parse_canonical_json,
        schema_issues,
        success_report,
    )
    from formal.chemgraph.audit.blind_carrier_v0_1.validate_inbox import _validated_inbox  # type: ignore
else:
    from ._common import (
        DocumentError,
        PREDICTION_SCHEMA_SHA256,
        StructuredArgumentParser,
        ValidationIssue,
        emit_report,
        error_report,
        load_frozen_schema,
        parse_canonical_json,
        schema_issues,
        success_report,
    )
    from .validate_inbox import _validated_inbox


FORBIDDEN_CHEMICAL_FIELDS = {
    "atom", "atomic_number", "atoms", "chemical_symbol", "element", "elements",
    "ground_truth", "gt", "inchi", "inchikey", "label", "molblock", "molfile",
    "sdf", "smiles",
}

AA_MARGIN_Q4 = 16
GEOMETRY_SAMPLE_STEP_Q4 = 16


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code, path, message)


def _forbidden_field_issues(value: Any, path: str = "$") -> list[ValidationIssue]:
    issues = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key.lower() in FORBIDDEN_CHEMICAL_FIELDS:
                issues.append(_issue(
                    "forbidden_chemical_field", child_path,
                    "element, molecule, SDF, and ground-truth fields are forbidden",
                ))
            issues.extend(_forbidden_field_issues(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_forbidden_field_issues(child, f"{path}/{index}"))
    return issues


def _contiguous_id_issues(values: list[dict[str, Any]], path: str) -> list[ValidationIssue]:
    actual = [value["id"] for value in values]
    expected = list(range(len(values)))
    if actual != expected:
        return [_issue("noncontiguous_ids", path, f"expected {expected}, got {actual}")]
    return []


def _sorted_unique(values: list[int]) -> bool:
    return values == sorted(set(values))


def _carrier_radius_q4(segment: dict[str, Any]) -> int:
    return (segment["width_q4"] + 1) // 2 + AA_MARGIN_Q4


def _point_near_line_segment_q4(
    point: tuple[int, int], begin: list[int], end: list[int], radius_q4: int,
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


def _point_near_polyline_q4(
    point: tuple[int, int], polyline: list[list[int]], radius_q4: int,
) -> bool:
    return any(
        _point_near_line_segment_q4(point, begin, end, radius_q4)
        for begin, end in zip(polyline, polyline[1:])
    )


def _support_near_point_q4(
    point: tuple[int, int], support: set[tuple[int, int]], radius_q4: int,
) -> bool:
    center_x, center_y = point
    pixel_radius = (radius_q4 + 15) // 16 + 1
    base_x, base_y = center_x // 16, center_y // 16
    radius_squared = radius_q4 * radius_q4
    for y in range(max(0, base_y - pixel_radius), min(767, base_y + pixel_radius) + 1):
        for x in range(max(0, base_x - pixel_radius), min(767, base_x + pixel_radius) + 1):
            if (x, y) not in support:
                continue
            dx, dy = 16 * x + 8 - center_x, 16 * y + 8 - center_y
            if dx * dx + dy * dy <= radius_squared:
                return True
    return False


def _geometry_support_issues(
    segment: dict[str, Any], support: set[tuple[int, int]], path: str,
) -> list[ValidationIssue]:
    issues = []
    polyline = segment["polyline_q4"]
    radius_q4 = _carrier_radius_q4(segment)
    for pixel in sorted(support):
        center = (16 * pixel[0] + 8, 16 * pixel[1] + 8)
        if not _point_near_polyline_q4(center, polyline, radius_q4):
            issues.append(_issue(
                "support_far_from_geometry", f"{path}/support_runs",
                f"support pixel {pixel} lies outside the carrier radius",
            ))
            break
    for vertex_index, vertex in enumerate(polyline):
        if not _support_near_point_q4(tuple(vertex), support, radius_q4):
            issues.append(_issue(
                "polyline_vertex_unsupported", f"{path}/polyline_q4/{vertex_index}",
                "polyline vertex has no nearby support pixel",
            ))
    for segment_index, (begin, end) in enumerate(zip(polyline, polyline[1:])):
        dx, dy = end[0] - begin[0], end[1] - begin[1]
        length_q4 = math.isqrt(dx * dx + dy * dy)
        steps = max(1, (length_q4 + GEOMETRY_SAMPLE_STEP_Q4 - 1) // GEOMETRY_SAMPLE_STEP_Q4)
        for sample_index in range(steps + 1):
            sample = (
                (begin[0] * (steps - sample_index) + end[0] * sample_index + steps // 2) // steps,
                (begin[1] * (steps - sample_index) + end[1] * sample_index + steps // 2) // steps,
            )
            if not _support_near_point_q4(sample, support, radius_q4):
                issues.append(_issue(
                    "polyline_segment_unsupported", f"{path}/polyline_q4/{segment_index}",
                    f"sample {sample_index}/{steps} has no nearby support pixel",
                ))
                break
    return issues


def _terminal_near_segment_endpoint(terminal: dict[str, Any], segment: dict[str, Any]) -> bool:
    center = tuple(terminal["center_q4"])
    radius = _carrier_radius_q4(segment)
    endpoints = (segment["polyline_q4"][0], segment["polyline_q4"][-1])
    return any(
        (center[0] - endpoint[0]) ** 2 + (center[1] - endpoint[1]) ** 2 <= radius * radius
        for endpoint in endpoints
    )


def _parallel_carrier_tubes_overlap_q4(
    left_begin: list[int], left_end: list[int], left_width_q4: int,
    right_begin: list[int], right_end: list[int], right_width_q4: int,
) -> bool:
    left_dx, left_dy = left_end[0] - left_begin[0], left_end[1] - left_begin[1]
    right_dx, right_dy = right_end[0] - right_begin[0], right_end[1] - right_begin[1]
    if left_dx * right_dy - left_dy * right_dx != 0:
        return False

    length_squared = left_dx * left_dx + left_dy * left_dy
    offset_x = right_begin[0] - left_begin[0]
    offset_y = right_begin[1] - left_begin[1]

    right_begin_projection = offset_x * left_dx + offset_y * left_dy
    right_end_projection = (
        (right_end[0] - left_begin[0]) * left_dx
        + (right_end[1] - left_begin[1]) * left_dy
    )
    right_projection_min, right_projection_max = sorted(
        (right_begin_projection, right_end_projection),
    )
    if max(0, right_projection_min) >= min(length_squared, right_projection_max):
        return False

    cross = left_dx * offset_y - left_dy * offset_x
    minimum_separation_q4 = (
        (left_width_q4 + 1) // 2
        + (right_width_q4 + 1) // 2
        + AA_MARGIN_Q4
    )
    return cross * cross < minimum_separation_q4 * minimum_separation_q4 * length_squared


def _lanes_have_overlaid_geometry(
    left: dict[str, Any], right: dict[str, Any], segment_by_id: dict[int, dict[str, Any]],
) -> bool:
    left_pieces = [
        (begin, end, segment_by_id[segment_id]["width_q4"])
        for segment_id in left["segment_ids"] if segment_id in segment_by_id
        for begin, end in zip(
            segment_by_id[segment_id]["polyline_q4"],
            segment_by_id[segment_id]["polyline_q4"][1:],
        )
    ]
    right_pieces = [
        (begin, end, segment_by_id[segment_id]["width_q4"])
        for segment_id in right["segment_ids"] if segment_id in segment_by_id
        for begin, end in zip(
            segment_by_id[segment_id]["polyline_q4"],
            segment_by_id[segment_id]["polyline_q4"][1:],
        )
    ]
    return any(
        _parallel_carrier_tubes_overlap_q4(
            left_begin, left_end, left_width,
            right_begin, right_end, right_width,
        )
        for left_begin, left_end, left_width in left_pieces
        for right_begin, right_end, right_width in right_pieces
    )


def _structural_issues(
    prediction: dict[str, Any], foreground: set[tuple[int, int]],
) -> list[ValidationIssue]:
    if prediction["status"] == "reject":
        return []

    issues: list[ValidationIssue] = []
    segments = prediction["segments"]
    terminals = prediction["terminals"]
    lanes = prediction["lane_hypotheses"]
    bonds = prediction["bond_hypotheses"]
    interpretations = prediction["interpretations"]
    for values, path in (
        (segments, "$/segments"),
        (terminals, "$/terminals"),
        (lanes, "$/lane_hypotheses"),
        (bonds, "$/bond_hypotheses"),
        (interpretations, "$/interpretations"),
    ):
        issues.extend(_contiguous_id_issues(values, path))

    segment_ids = {segment["id"] for segment in segments}
    terminal_ids = {terminal["id"] for terminal in terminals}
    lane_ids = {lane["id"] for lane in lanes}
    bond_ids = {bond["id"] for bond in bonds}
    segment_by_id = {segment["id"]: segment for segment in segments}
    terminal_by_id = {terminal["id"]: terminal for terminal in terminals}
    lane_by_id = {lane["id"]: lane for lane in lanes}
    bond_by_id = {bond["id"]: bond for bond in bonds}

    support_by_segment: dict[int, set[tuple[int, int]]] = {}
    for index, segment in enumerate(segments):
        path = f"$/segments/{index}"
        points = segment["polyline_q4"]
        if any(point == next_point for point, next_point in zip(points, points[1:])):
            issues.append(_issue("zero_length_polyline_step", f"{path}/polyline_q4", "adjacent Q4 points repeat"))
        runs = segment["support_runs"]
        if runs != sorted(runs):
            issues.append(_issue("unsorted_support_runs", f"{path}/support_runs", "RLE runs must be sorted"))
        counted = 0
        support_pixels: set[tuple[int, int]] = set()
        previous_by_row: dict[int, tuple[int, int]] = {}
        for run_index, (row, start, length) in enumerate(runs):
            run_path = f"{path}/support_runs/{run_index}"
            previous = previous_by_row.get(row)
            if previous is not None and start < previous[0] + previous[1]:
                issues.append(_issue("overlapping_support_runs", run_path, "RLE runs overlap"))
            previous_by_row[row] = (start, length)
            counted += length
            support_pixels.update((column, row) for column in range(start, start + length))
            missing = next(
                ((column, row) for column in range(start, start + length)
                 if (column, row) not in foreground),
                None,
            )
            if missing is not None:
                issues.append(_issue(
                    "support_not_foreground", run_path,
                    f"support references background pixel {missing}",
                ))
        if counted != segment["support_pixel_count"]:
            issues.append(_issue(
                "support_count_mismatch", f"{path}/support_pixel_count",
                f"declared {segment['support_pixel_count']}, decoded {counted}",
            ))
        support_by_segment[segment["id"]] = support_pixels
        issues.extend(_geometry_support_issues(segment, support_pixels, path))

    for left_index, left in enumerate(segments):
        left_support = support_by_segment[left["id"]]
        other_support = set().union(*(
            support_by_segment[item["id"]] for item in segments if item["id"] != left["id"]
        )) if len(segments) > 1 else set()
        if not left_support - other_support:
            issues.append(_issue(
                "segment_has_no_exclusive_support", f"$/segments/{left_index}/support_runs",
                "segment must retain at least one exclusive foreground pixel",
            ))
        for right_index in range(left_index + 1, len(segments)):
            right = segments[right_index]
            if left_support == support_by_segment[right["id"]]:
                issues.append(_issue(
                    "duplicate_segment_support", f"$/segments/{right_index}/support_runs",
                    f"support set duplicates segment {left['id']}",
                ))

    for index, terminal in enumerate(terminals):
        path = f"$/terminals/{index}/support_segment_ids"
        references = terminal["support_segment_ids"]
        if not _sorted_unique(references):
            issues.append(_issue("terminal_refs_not_sorted_unique", path, "terminal segment refs must be sorted unique"))
        unknown = sorted(set(references) - segment_ids)
        if unknown:
            issues.append(_issue("unknown_terminal_segment", path, f"unknown segment IDs: {unknown}"))
        for segment_id in references:
            if segment_id in segment_by_id and not _terminal_near_segment_endpoint(
                terminal, segment_by_id[segment_id],
            ):
                issues.append(_issue(
                    "terminal_not_near_segment_endpoint", path,
                    f"terminal is not near an endpoint of segment {segment_id}",
                ))

    for index, lane in enumerate(lanes):
        path = f"$/lane_hypotheses/{index}"
        references = lane["segment_ids"]
        if not _sorted_unique(references):
            issues.append(_issue("lane_segments_not_sorted_unique", f"{path}/segment_ids", "lane segment union must be sorted unique"))
        unknown = sorted(set(references) - segment_ids)
        if unknown:
            issues.append(_issue("unknown_lane_segment", f"{path}/segment_ids", f"unknown segment IDs: {unknown}"))
        begin_id, end_id = lane["begin_terminal_id"], lane["end_terminal_id"]
        if begin_id == end_id:
            issues.append(_issue("lane_self_loop", path, "lane endpoints must differ"))
        unknown_terminals = sorted({begin_id, end_id} - terminal_ids)
        if unknown_terminals:
            issues.append(_issue("unknown_lane_terminal", path, f"unknown terminal IDs: {unknown_terminals}"))
        else:
            begin_support = set(terminal_by_id[begin_id]["support_segment_ids"])
            end_support = set(terminal_by_id[end_id]["support_segment_ids"])
            if not set(references) & begin_support:
                issues.append(_issue("lane_begin_terminal_unsupported", path, "begin terminal shares no lane segment"))
            if not set(references) & end_support:
                issues.append(_issue("lane_end_terminal_unsupported", path, "end terminal shares no lane segment"))
            begin_matches = any(
                segment_id in segment_by_id
                and _terminal_near_segment_endpoint(terminal_by_id[begin_id], segment_by_id[segment_id])
                for segment_id in references
            )
            end_matches = any(
                segment_id in segment_by_id
                and _terminal_near_segment_endpoint(terminal_by_id[end_id], segment_by_id[segment_id])
                for segment_id in references
            )
            if not begin_matches:
                issues.append(_issue("lane_begin_geometry_mismatch", path, "begin terminal does not match lane geometry"))
            if not end_matches:
                issues.append(_issue("lane_end_geometry_mismatch", path, "end terminal does not match lane geometry"))

    for index, bond in enumerate(bonds):
        path = f"$/bond_hypotheses/{index}"
        references = bond["lane_ids"]
        if not _sorted_unique(references):
            issues.append(_issue("bond_lanes_not_sorted_unique", f"{path}/lane_ids", "bond lane IDs must be sorted unique"))
        if len(references) != bond["order"]:
            issues.append(_issue("bond_lane_count_mismatch", path, "lane count must equal bond order"))
        unknown = sorted(set(references) - lane_ids)
        if unknown:
            issues.append(_issue("unknown_bond_lane", f"{path}/lane_ids", f"unknown lane IDs: {unknown}"))
        begin_id, end_id = bond["begin_terminal_id"], bond["end_terminal_id"]
        if begin_id == end_id:
            issues.append(_issue("bond_self_loop", path, "bond endpoints must differ"))
        if not {begin_id, end_id} <= terminal_ids:
            issues.append(_issue("unknown_bond_terminal", path, "bond references an unknown terminal"))
        alignment = bond["lane_endpoint_alignment"]
        alignment_ids = [item["lane_id"] for item in alignment]
        if not _sorted_unique(alignment_ids) or alignment_ids != references:
            issues.append(_issue("bond_lane_union_mismatch", f"{path}/lane_endpoint_alignment", "alignment lane union must exactly equal sorted lane_ids"))
        for alignment_index, item in enumerate(alignment):
            lane_id = item["lane_id"]
            alignment_path = f"{path}/lane_endpoint_alignment/{alignment_index}"
            if lane_id not in lane_by_id:
                issues.append(_issue("unknown_alignment_lane", alignment_path, "alignment references unknown lane"))
                continue
            lane = lane_by_id[lane_id]
            expected = (begin_id, end_id)
            actual = (item["begin_terminal_id"], item["end_terminal_id"])
            lane_endpoints = (lane["begin_terminal_id"], lane["end_terminal_id"])
            if actual != expected or lane_endpoints != expected:
                issues.append(_issue("bond_lane_endpoint_mismatch", alignment_path, "all lanes must share ordered bond endpoints"))
        known_lanes = [lane_by_id[lane_id] for lane_id in references if lane_id in lane_by_id]
        for left_index, left_lane in enumerate(known_lanes):
            for right_lane in known_lanes[left_index + 1:]:
                left_support = set().union(*(
                    support_by_segment.get(segment_id, set())
                    for segment_id in left_lane["segment_ids"]
                ))
                right_support = set().union(*(
                    support_by_segment.get(segment_id, set())
                    for segment_id in right_lane["segment_ids"]
                ))
                if left_support & right_support:
                    issues.append(_issue(
                        "bond_lane_support_overlap", path,
                        f"lanes {left_lane['id']} and {right_lane['id']} share foreground support",
                    ))
                if _lanes_have_overlaid_geometry(left_lane, right_lane, segment_by_id):
                    issues.append(_issue(
                        "bond_lane_geometry_overlaid", path,
                        f"lanes {left_lane['id']} and {right_lane['id']} have overlapping parallel carrier tubes",
                    ))

    for index, interpretation in enumerate(interpretations):
        path = f"$/interpretations/{index}"
        selected = interpretation["selected_bond_ids"]
        if not _sorted_unique(selected):
            issues.append(_issue("selected_bonds_not_sorted_unique", f"{path}/selected_bond_ids", "selected bonds must be sorted unique"))
        unknown_bonds = sorted(set(selected) - bond_ids)
        if unknown_bonds:
            issues.append(_issue("unknown_selected_bond", f"{path}/selected_bond_ids", f"unknown bond IDs: {unknown_bonds}"))
        ownership = interpretation["ownership"]
        ownership_ids = [item["segment_id"] for item in ownership]
        if ownership_ids != sorted(ownership_ids):
            issues.append(_issue("ownership_not_sorted", f"{path}/ownership", "ownership must be sorted by segment_id"))
        if len(ownership_ids) != len(set(ownership_ids)) or set(ownership_ids) != segment_ids:
            issues.append(_issue("ownership_not_exact", f"{path}/ownership", "every segment must have exactly one owner"))
        owner_by_segment = {item["segment_id"]: item for item in ownership}

        selected_lanes: set[int] = set()
        lane_use_count: dict[int, int] = {}
        for bond_id in selected:
            if bond_id not in bond_by_id:
                continue
            for lane_id in bond_by_id[bond_id]["lane_ids"]:
                lane_use_count[lane_id] = lane_use_count.get(lane_id, 0) + 1
                selected_lanes.add(lane_id)
        reused = sorted(lane_id for lane_id, count in lane_use_count.items() if count != 1)
        if reused:
            issues.append(_issue("selected_lane_reused", path, f"selected bonds reuse lanes: {reused}"))

        selected_lane_support = {
            lane_id: set().union(*(
                support_by_segment.get(segment_id, set())
                for segment_id in lane_by_id[lane_id]["segment_ids"]
            ))
            for lane_id in selected_lanes if lane_id in lane_by_id
        }
        for lane_id, lane_support in selected_lane_support.items():
            other_lane_support = set().union(*(
                support for other_id, support in selected_lane_support.items()
                if other_id != lane_id
            )) if len(selected_lane_support) > 1 else set()
            if not lane_support - other_lane_support:
                issues.append(_issue(
                    "selected_lane_has_no_exclusive_support", path,
                    f"selected lane {lane_id} has no exclusive foreground support",
                ))

        for lane_id in sorted(selected_lanes):
            if lane_id not in lane_by_id:
                continue
            for segment_id in lane_by_id[lane_id]["segment_ids"]:
                owner = owner_by_segment.get(segment_id)
                if owner is None or owner["owner_kind"] != "bond_lane" or owner["owner_id"] != lane_id:
                    issues.append(_issue(
                        "selected_bond_ownership_mismatch", path,
                        f"segment {segment_id} must be owned by selected lane {lane_id}",
                    ))
        for owner_index, owner in enumerate(ownership):
            if owner["owner_kind"] != "bond_lane":
                continue
            lane_id = owner["owner_id"]
            segment_id = owner["segment_id"]
            if lane_id not in lane_by_id:
                issues.append(_issue("unknown_ownership_lane", f"{path}/ownership/{owner_index}", "bond_lane owner does not exist"))
            elif lane_id not in selected_lanes or segment_id not in lane_by_id[lane_id]["segment_ids"]:
                issues.append(_issue("unselected_lane_ownership", f"{path}/ownership/{owner_index}", "bond_lane ownership must match a selected bond lane"))

    return issues


def validate_prediction_bytes(prediction_raw: bytes, inbox_raw: bytes) -> dict[str, Any]:
    try:
        inbox = _validated_inbox(inbox_raw)
        prediction = parse_canonical_json(prediction_raw)
        schema = load_frozen_schema("prediction.schema.json", PREDICTION_SCHEMA_SHA256)
    except DocumentError as error:
        return error_report([error.issue])
    issues = schema_issues(prediction, schema)
    issues.extend(_forbidden_field_issues(prediction))
    if not issues:
        issues.extend(_structural_issues(prediction, inbox.foreground))
    if issues:
        return error_report(issues)
    counts = {
        "bond_count": len(prediction.get("bond_hypotheses", [])),
        "interpretation_count": len(prediction.get("interpretations", [])),
        "lane_count": len(prediction.get("lane_hypotheses", [])),
        "segment_count": len(prediction.get("segments", [])),
        "terminal_count": len(prediction.get("terminals", [])),
    }
    return success_report(
        "blind_carrier_prediction_v0.1",
        prediction_sha256=hashlib.sha256(prediction_raw).hexdigest(),
        prediction_status=prediction["status"],
        schema_sha256=PREDICTION_SCHEMA_SHA256,
        **counts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = StructuredArgumentParser()
    parser.add_argument("prediction", help="canonical prediction JSON path")
    parser.add_argument("--inbox", required=True, help="paired anonymous inbox JSON path")
    args = parser.parse_args(argv)
    try:
        prediction_raw = Path(args.prediction).read_bytes()
        inbox_raw = Path(args.inbox).read_bytes()
    except OSError as error:
        return emit_report(error_report([
            ValidationIssue("input_read_failed", "$", str(error)),
        ]), error=True)
    report = validate_prediction_bytes(prediction_raw, inbox_raw)
    return emit_report(report, error=report["status"] == "INVALID")


if __name__ == "__main__":
    raise SystemExit(main())
