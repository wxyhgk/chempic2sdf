#!/usr/bin/env python3
"""Validate v0.2 terminal regions while retaining all v0.1 carrier gates."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from formal.chemgraph.audit.blind_carrier_v0_1 import validate_predictions as v01  # type: ignore
    from formal.chemgraph.audit.blind_carrier_v0_2._common import (  # type: ignore
        DocumentError, PREDICTION_SCHEMA_SHA256, StructuredArgumentParser,
        ValidationIssue, emit_report, error_report, load_frozen_schema,
        parse_canonical_json, schema_issues, success_report,
    )
    from formal.chemgraph.audit.blind_carrier_v0_2.validate_inbox import _validated_inbox  # type: ignore
else:
    from formal.chemgraph.audit.blind_carrier_v0_1 import validate_predictions as v01
    from ._common import (
        DocumentError, PREDICTION_SCHEMA_SHA256, StructuredArgumentParser,
        ValidationIssue, emit_report, error_report, load_frozen_schema,
        parse_canonical_json, schema_issues, success_report,
    )
    from .validate_inbox import _validated_inbox


LEGACY_POINT_TERMINAL_CODES = {
    "terminal_not_near_segment_endpoint",
    "lane_begin_geometry_mismatch",
    "lane_end_geometry_mismatch",
}


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code, path, message)


def _rounded_mean(points: list[list[int]]) -> list[int]:
    count = len(points)
    return [
        (sum(point[axis] for point in points) + count // 2) // count
        for axis in (0, 1)
    ]


def _alignment_key(item: dict[str, int]) -> tuple[int, int]:
    return int(item["segment_id"]), int(item["endpoint_index"])


def _endpoint_record(
    reference: tuple[int, int], segment_by_id: dict[int, dict[str, Any]],
) -> tuple[list[int], int] | None:
    segment_id, endpoint_index = reference
    segment = segment_by_id.get(segment_id)
    if segment is None:
        return None
    point = segment["polyline_q4"][0 if endpoint_index == 0 else -1]
    return point, v01._carrier_radius_q4(segment)


def _disks_connected(
    references: set[tuple[int, int]], segment_by_id: dict[int, dict[str, Any]],
) -> bool:
    if not references:
        return False
    ordered = sorted(references)
    records = [_endpoint_record(reference, segment_by_id) for reference in ordered]
    if any(record is None for record in records):
        return False
    visited = {0}
    pending = [0]
    while pending:
        left = pending.pop()
        left_point, left_radius = records[left]  # type: ignore[misc]
        for right, record in enumerate(records):
            if right in visited:
                continue
            right_point, right_radius = record  # type: ignore[misc]
            dx = left_point[0] - right_point[0]
            dy = left_point[1] - right_point[1]
            radius = left_radius + right_radius
            if dx * dx + dy * dy <= radius * radius:
                visited.add(right)
                pending.append(right)
    return len(visited) == len(ordered)


def _terminal_region_issues(prediction: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    segments = prediction["segments"]
    terminals = prediction["terminals"]
    lanes = prediction["lane_hypotheses"]
    bonds = prediction["bond_hypotheses"]
    segment_by_id = {segment["id"]: segment for segment in segments}
    terminal_by_id = {terminal["id"]: terminal for terminal in terminals}
    lane_by_id = {lane["id"]: lane for lane in lanes}
    refs_by_terminal: dict[int, set[tuple[int, int]]] = {}

    for index, terminal in enumerate(terminals):
        path = f"$/terminals/{index}"
        alignment = terminal["endpoint_alignment"]
        keys = [_alignment_key(item) for item in alignment]
        if keys != sorted(set(keys)):
            issues.append(_issue(
                "terminal_alignment_not_sorted_unique", f"{path}/endpoint_alignment",
                "endpoint alignment must be sorted and unique",
            ))
        references = set(keys)
        refs_by_terminal[terminal["id"]] = references
        unknown = sorted(segment_id for segment_id, _ in references if segment_id not in segment_by_id)
        if unknown:
            issues.append(_issue(
                "unknown_terminal_endpoint_segment", f"{path}/endpoint_alignment",
                f"unknown segment IDs: {unknown}",
            ))
            continue
        support_union = sorted({segment_id for segment_id, _ in references})
        if terminal["support_segment_ids"] != support_union:
            issues.append(_issue(
                "terminal_endpoint_union_mismatch", path,
                "support_segment_ids must exactly equal endpoint alignment segment union",
            ))
        points = [
            segment_by_id[segment_id]["polyline_q4"][0 if endpoint_index == 0 else -1]
            for segment_id, endpoint_index in keys
        ]
        if points and terminal["center_q4"] != _rounded_mean(points):
            issues.append(_issue(
                "terminal_center_mismatch", f"{path}/center_q4",
                "terminal center must equal the deterministic half-up rounded endpoint mean",
            ))
        if references and not _disks_connected(references, segment_by_id):
            issues.append(_issue(
                "terminal_region_disconnected", f"{path}/endpoint_alignment",
                "aligned endpoint carrier disks must form a connected intersection graph",
            ))

    lane_endpoint_refs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for index, lane in enumerate(lanes):
        path = f"$/lane_hypotheses/{index}"
        if len(lane["segment_ids"]) != 1:
            issues.append(_issue(
                "lane_not_single_segment", f"{path}/segment_ids",
                "Blind Carrier v0.2 supports exactly one segment per lane",
            ))
            continue
        segment_id = lane["segment_ids"][0]
        begin_refs = {
            reference for reference in refs_by_terminal.get(lane["begin_terminal_id"], set())
            if reference[0] == segment_id
        }
        end_refs = {
            reference for reference in refs_by_terminal.get(lane["end_terminal_id"], set())
            if reference[0] == segment_id
        }
        if len(begin_refs) != 1 or len(end_refs) != 1:
            issues.append(_issue(
                "lane_terminal_endpoint_mismatch", path,
                "one-segment lane requires exactly one aligned endpoint at each terminal",
            ))
            continue
        begin_ref = next(iter(begin_refs))
        end_ref = next(iter(end_refs))
        if {begin_ref, end_ref} != {(segment_id, 0), (segment_id, 1)}:
            issues.append(_issue(
                "lane_endpoint_indices_not_distinct", path,
                "lane begin/end must align the two distinct segment endpoints",
            ))
            continue
        begin_terminal = terminal_by_id.get(lane["begin_terminal_id"])
        end_terminal = terminal_by_id.get(lane["end_terminal_id"])
        if (begin_terminal is not None and end_terminal is not None
                and begin_terminal["center_q4"] == end_terminal["center_q4"]):
            issues.append(_issue(
                "lane_terminal_centers_equal", path,
                "lane begin/end terminal centers must differ",
            ))
        lane_endpoint_refs[lane["id"]] = (begin_ref, end_ref)

    for index, bond in enumerate(bonds):
        path = f"$/bond_hypotheses/{index}"
        known_lanes = [lane_by_id[lane_id] for lane_id in bond["lane_ids"] if lane_id in lane_by_id]
        if len(known_lanes) != len(bond["lane_ids"]):
            continue
        segment_ids = {
            lane["segment_ids"][0] for lane in known_lanes if len(lane["segment_ids"]) == 1
        }
        if len(segment_ids) != len(known_lanes):
            continue
        for side, terminal_id, endpoint_position in (
            ("begin", bond["begin_terminal_id"], 0),
            ("end", bond["end_terminal_id"], 1),
        ):
            expected = {
                lane_endpoint_refs[lane["id"]][endpoint_position]
                for lane in known_lanes if lane["id"] in lane_endpoint_refs
            }
            actual = {
                reference for reference in refs_by_terminal.get(terminal_id, set())
                if reference[0] in segment_ids
            }
            if actual != expected or len(expected) != len(known_lanes):
                issues.append(_issue(
                    "bond_terminal_endpoint_union_mismatch", path,
                    f"{side} terminal must align exactly one endpoint from every bond lane",
                ))
            elif not _disks_connected(expected, segment_by_id):
                issues.append(_issue(
                    "bond_terminal_region_disconnected", path,
                    f"{side} bond-lane endpoint disks are disconnected",
                ))
        begin = terminal_by_id.get(bond["begin_terminal_id"])
        end = terminal_by_id.get(bond["end_terminal_id"])
        if begin is not None and end is not None and begin["center_q4"] == end["center_q4"]:
            issues.append(_issue(
                "bond_terminal_centers_equal", path,
                "bond begin/end terminal centers must differ",
            ))

    bond_by_id = {bond["id"]: bond for bond in bonds}
    for index, interpretation in enumerate(prediction["interpretations"]):
        path = f"$/interpretations/{index}"
        selected_lanes = {
            lane_id
            for bond_id in interpretation["selected_bond_ids"] if bond_id in bond_by_id
            for lane_id in bond_by_id[bond_id]["lane_ids"]
        }
        selected_segments = {
            lane_by_id[lane_id]["segment_ids"][0]
            for lane_id in selected_lanes
            if lane_id in lane_by_id and len(lane_by_id[lane_id]["segment_ids"]) == 1
        }
        selected_terminal_ids = {
            terminal_id
            for lane_id in selected_lanes if lane_id in lane_by_id
            for terminal_id in (
                lane_by_id[lane_id]["begin_terminal_id"],
                lane_by_id[lane_id]["end_terminal_id"],
            )
        }
        owner: dict[tuple[int, int], int] = {}
        for terminal_id in sorted(selected_terminal_ids):
            references = {
                reference for reference in refs_by_terminal.get(terminal_id, set())
                if reference[0] in selected_segments
            }
            if not _disks_connected(references, segment_by_id):
                issues.append(_issue(
                    "selected_terminal_region_disconnected", path,
                    f"selected terminal {terminal_id} cannot use unselected segments as a bridge",
                ))
            for reference in references:
                if reference in owner and owner[reference] != terminal_id:
                    issues.append(_issue(
                        "selected_endpoint_reused", path,
                        f"endpoint {reference} belongs to multiple selected terminals",
                    ))
                owner[reference] = terminal_id
    return issues


def _structural_issues(
    prediction: dict[str, Any], foreground: set[tuple[int, int]],
) -> list[ValidationIssue]:
    if prediction["status"] == "reject":
        return []
    legacy = [
        issue for issue in v01._structural_issues(prediction, foreground)
        if issue.code not in LEGACY_POINT_TERMINAL_CODES
    ]
    return [*legacy, *_terminal_region_issues(prediction)]


def validate_prediction_bytes(prediction_raw: bytes, inbox_raw: bytes) -> dict[str, Any]:
    try:
        inbox = _validated_inbox(inbox_raw)
        prediction = parse_canonical_json(prediction_raw)
        schema = load_frozen_schema("prediction.schema.json", PREDICTION_SCHEMA_SHA256)
    except DocumentError as error:
        return error_report([error.issue])
    issues = schema_issues(prediction, schema)
    issues.extend(v01._forbidden_field_issues(prediction))
    if not issues:
        issues.extend(_structural_issues(prediction, inbox.foreground))
    if issues:
        return error_report(issues)
    return success_report(
        "blind_carrier_prediction_v0.2",
        bond_count=len(prediction.get("bond_hypotheses", [])),
        interpretation_count=len(prediction.get("interpretations", [])),
        lane_count=len(prediction.get("lane_hypotheses", [])),
        prediction_sha256=hashlib.sha256(prediction_raw).hexdigest(),
        prediction_status=prediction["status"],
        schema_sha256=PREDICTION_SCHEMA_SHA256,
        segment_count=len(prediction.get("segments", [])),
        terminal_count=len(prediction.get("terminals", [])),
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
