#!/usr/bin/env python3
"""Validate v0.3 candidate provenance without claiming enumerator completeness."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from formal.chemgraph.audit.blind_carrier_v0_1 import validate_predictions as v01

from ._common import (
    CANDIDATE_SCHEMA_SHA256,
    DocumentError,
    PROTOCOL_SHA256,
    StructuredArgumentParser,
    ValidationIssue,
    canonical_json_bytes,
    emit_report,
    error_report,
    framed_sha256,
    load_frozen_schema,
    parse_canonical_json,
    schema_issues,
    success_report,
)
from .validate_input import validated_input


MAX_OUTPUT_BYTES = 67_108_864
P1A_SHA256 = "490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74"
ATOM_SOURCE_SHA256 = "e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac"
VISION_SHA256 = "1eede2e814efc0c440c26eba88bc8e9ed86e0911d7727b770a992c1ea2671c6a"
DIRECT_SHA256 = "e0d769b457fdbcac2d405fa234ccd108c80640db63cb35a1d8ed36446eea0614"

FORBIDDEN_FIELDS = {
    "element", "ground_truth", "gt", "inchi", "inchikey", "label",
    "molblock", "molfile", "sdf", "selected_interpretation", "smiles",
    "unique_graph", "winner",
}


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code, path, message)


def _forbidden_issues(value: Any, path: str = "$") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key.lower() in FORBIDDEN_FIELDS:
                issues.append(_issue(
                    "forbidden_semantic_field", child_path,
                    "chemical identity, ground truth, winner and unique graph fields are forbidden",
                ))
            issues.extend(_forbidden_issues(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_forbidden_issues(child, f"{path}/{index}"))
    return issues


def _contiguous_ids(records: list[dict[str, Any]], path: str) -> list[ValidationIssue]:
    actual = [record["id"] for record in records]
    expected = list(range(len(records)))
    if actual == expected:
        return []
    return [_issue("noncontiguous_ids", path, f"expected {expected}, got {actual}")]


def _sorted_unique(values: list[int]) -> bool:
    return values == sorted(set(values))


def _decode_runs(
    runs: list[list[int]], path: str,
) -> tuple[set[tuple[int, int]], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    pixels: set[tuple[int, int]] = set()
    previous: tuple[int, int] | None = None
    for index, run in enumerate(runs):
        row, start, length = run
        key = (row, start)
        if previous is not None and key <= previous:
            issues.append(_issue(
                "rle_not_strictly_sorted", f"{path}/{index}",
                "RLE runs must be strictly sorted by row and start column",
            ))
        previous = key
        if start + length > 768:
            issues.append(_issue(
                "rle_run_outside_canvas", f"{path}/{index}",
                "RLE run extends past column 767",
            ))
            continue
        current = {(column, row) for column in range(start, start + length)}
        if pixels & current:
            issues.append(_issue(
                "rle_overlap", f"{path}/{index}", "RLE runs overlap",
            ))
        pixels.update(current)
    return pixels, issues


def _support(
    record: dict[str, Any], runs_key: str, count_key: str, path: str,
) -> tuple[set[tuple[int, int]], list[ValidationIssue]]:
    pixels, issues = _decode_runs(record[runs_key], f"{path}/{runs_key}")
    if len(pixels) != record[count_key]:
        issues.append(_issue(
            "support_count_mismatch", f"{path}/{count_key}",
            f"declared {record[count_key]}, decoded {len(pixels)}",
        ))
    return pixels, issues


def _point(record: dict[str, Any], endpoint_index: int) -> tuple[int, int]:
    point = record["segment_q4"][endpoint_index]
    return int(point[0]), int(point[1])


def _distance_floor(left: tuple[int, int], right: tuple[int, int]) -> int:
    return math.isqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def _squared_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _angle_millidegrees(
    origin: tuple[int, int], left: tuple[int, int], right: tuple[int, int],
) -> int:
    left_vector = (left[0] - origin[0], left[1] - origin[1])
    right_vector = (right[0] - origin[0], right[1] - origin[1])
    left_norm = math.hypot(*left_vector)
    right_norm = math.hypot(*right_vector)
    if left_norm == 0 or right_norm == 0:
        return 0
    cosine = (
        left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1]
    ) / (left_norm * right_norm)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    angle = min(angle, 180.0 - angle)
    return math.floor(angle * 1000 + 0.5)


def _compatible_pair_geometry(
    left: dict[str, Any], right: dict[str, Any],
) -> dict[str, Any]:
    left_begin, left_end = (tuple(point) for point in left["segment_q4"])
    right_begin, right_end = (tuple(point) for point in right["segment_q4"])
    left_vector = (left_end[0] - left_begin[0], left_end[1] - left_begin[1])
    right_vector = (right_end[0] - right_begin[0], right_end[1] - right_begin[1])
    angle = _angle_millidegrees((0, 0), left_vector, right_vector)
    centerline = [
        [(left_begin[axis] + right_begin[axis] + 1) // 2 for axis in range(2)],
        [(left_end[axis] + right_end[axis] + 1) // 2 for axis in range(2)],
    ]
    endpoint_distance_sum = (
        _distance_floor(left_begin, right_begin)
        + _distance_floor(left_end, right_end)
    )
    separation = (endpoint_distance_sum + 1) // 2
    dominant_axis = 0 if abs(left_vector[0]) >= abs(left_vector[1]) else 1
    left_interval = sorted((left_begin[dominant_axis], left_end[dominant_axis]))
    right_interval = sorted((right_begin[dominant_axis], right_end[dominant_axis]))
    overlap_axis = max(0, min(left_interval[1], right_interval[1]) - max(left_interval[0], right_interval[0]))
    dominant_span = abs(left_vector[dominant_axis])
    left_length = _distance_floor(left_begin, left_end)
    right_length = _distance_floor(right_begin, right_end)
    overlap = (
        (overlap_axis * left_length + dominant_span // 2) // dominant_span
        if dominant_span else 0
    )
    shorter_length = min(left_length, right_length)
    shorter_overlap_ppm = (
        min(1_000_000, (overlap * 1_000_000 + shorter_length // 2) // shorter_length)
        if shorter_length else 0
    )
    return {
        "canonical_centerline_q4": centerline,
        "observed_angle_mdeg": angle,
        "overlap_q4": overlap,
        "separation_q4": separation,
        "shorter_overlap_ppm": shorter_overlap_ppm,
    }


def _point_near_segment_q4(
    point: tuple[int, int], begin: tuple[int, int], end: tuple[int, int],
    radius_q4: int,
) -> bool:
    vector = (end[0] - begin[0], end[1] - begin[1])
    offset = (point[0] - begin[0], point[1] - begin[1])
    length_squared = vector[0] ** 2 + vector[1] ** 2
    if length_squared == 0:
        return _squared_distance(point, begin) <= radius_q4 ** 2
    projection = offset[0] * vector[0] + offset[1] * vector[1]
    if projection <= 0:
        return _squared_distance(point, begin) <= radius_q4 ** 2
    if projection >= length_squared:
        return _squared_distance(point, end) <= radius_q4 ** 2
    cross = offset[0] * vector[1] - offset[1] * vector[0]
    return cross ** 2 <= radius_q4 ** 2 * length_squared


def _complete_white_span_support(
    begin: tuple[int, int], end: tuple[int, int],
    foreground: set[tuple[int, int]], radius_q4: int,
) -> set[tuple[int, int]]:
    minimum_x = max(0, (min(begin[0], end[0]) - radius_q4) // 16)
    maximum_x = min(767, (max(begin[0], end[0]) + radius_q4) // 16)
    minimum_y = max(0, (min(begin[1], end[1]) - radius_q4) // 16)
    maximum_y = min(767, (max(begin[1], end[1]) + radius_q4) // 16)
    return {
        (x, y)
        for y in range(minimum_y, maximum_y + 1)
        for x in range(minimum_x, maximum_x + 1)
        if (x, y) not in foreground
        and _point_near_segment_q4(
            (16 * x + 8, 16 * y + 8), begin, end, radius_q4,
        )
    }


def _rle_sha256(runs: list[list[int]]) -> str:
    digest = hashlib.sha256()
    digest.update(len(runs).to_bytes(4, "big"))
    for row, start, length in runs:
        digest.update(row.to_bytes(4, "big"))
        digest.update(start.to_bytes(4, "big"))
        digest.update(length.to_bytes(4, "big"))
    return digest.hexdigest()


STAGE_NAMES = (
    "preprocess",
    "semantic_ink",
    "boundary_detection",
    "boundary_compatibility",
    "boundary_matching",
    "atom_proposals",
    "direct_pairs",
    "stroke_ownership",
    "endpoint_atom_relations",
    "interpretations",
    "graph_equivalence",
    "ambiguity_taxonomy",
)

TAXONOMY_KIND_BY_CODE = {
    "A03": "boundary_matching",
    "A08": "stroke_ownership",
    "A10": "crossing_vs_junction",
    "A11": "touch_vs_crossing",
    "A12": "glyph_vs_bond",
    "A13": "double_vs_parallel_single",
    "A15": "endpoint_atom_attachment",
    "A16": "lane_grouping",
}


def _stage_chain_sha256(stages: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for stage in stages:
        name = stage["stage"].encode("ascii")
        digest.update(stage["index"].to_bytes(4, "big"))
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(stage["component_sha256"]))
        digest.update(bytes.fromhex(stage["input_sha256"]))
        digest.update(bytes.fromhex(stage["output_sha256"]))
        digest.update(stage["record_count"].to_bytes(8, "big"))
    return digest.hexdigest()


def _spatial_graph_value(interpretation: dict[str, Any]) -> dict[str, Any]:
    atoms = sorted(
        interpretation["atoms"],
        key=lambda atom: (
            tuple(atom["anchor_q4"]), atom["kind"],
        ),
    )
    atom_id = {atom["id"]: index for index, atom in enumerate(atoms)}
    graph_atoms = [
        {"anchor_q4": atom["anchor_q4"], "kind": atom["kind"]}
        for atom in atoms
    ]
    graph_bonds = sorted(
        [{
            "atom_pair": sorted(atom_id[value] for value in bond["atom_pair"]),
            "order": bond["order"],
        }
        for bond in interpretation["bonds"]
        if all(value in atom_id for value in bond["atom_pair"])
        ],
        key=lambda bond: (bond["atom_pair"], bond["order"]),
    )
    return {"atoms": graph_atoms, "bonds": graph_bonds}


def _spatial_graph_sha256(interpretation: dict[str, Any]) -> str:
    return framed_sha256([_spatial_graph_value(interpretation)])


def _mask_issues(
    document: dict[str, Any], png_sha256: str, foreground: set[tuple[int, int]],
) -> tuple[dict[str, set[tuple[int, int]]], list[ValidationIssue]]:
    image = document["image"]
    issues: list[ValidationIssue] = []
    if image["png_sha256"] != png_sha256:
        issues.append(_issue(
            "image_digest_mismatch", "$/image/png_sha256",
            "candidate set is not bound to the input PNG bytes",
        ))
    decoded: dict[str, set[tuple[int, int]]] = {}
    for name in ("final_foreground", "bond_mask", "text_mask"):
        pixels, current = _support(
            image, f"{name}_runs", f"{name}_pixel_count", "$/image",
        )
        decoded[name] = pixels
        issues.extend(current)
    if decoded["final_foreground"] != foreground:
        issues.append(_issue(
            "foreground_artifact_mismatch", "$/image/final_foreground_runs",
            "final foreground RLE must exactly equal decoded non-white PNG pixels",
        ))
    for name in ("bond_mask", "text_mask"):
        if not decoded[name] <= foreground:
            issues.append(_issue(
                "semantic_mask_outside_foreground", f"$/image/{name}_runs",
                "semantic masks may contain only final-raster foreground pixels",
            ))
    for name in ("final_foreground", "bond_mask", "text_mask"):
        if image[f"{name}_rle_sha256"] != _rle_sha256(image[f"{name}_runs"]):
            issues.append(_issue(
                "mask_digest_mismatch", f"$/image/{name}_rle_sha256",
                f"{name} digest does not bind its complete RLE array",
            ))
    return decoded, issues


def _interpretation_issues(
    interpretation: dict[str, Any], masks: dict[str, set[tuple[int, int]]],
    png_sha256: str,
    boundary_universe: dict[int, dict[str, Any]],
    compatible_pairs: dict[int, dict[str, Any]],
    atom_proposals: dict[int, dict[str, Any]],
    direct_pairs: dict[int, dict[str, Any]],
    thresholds: dict[str, int],
    path: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    collections = (
        "boundaries", "stroke_fragments", "atoms", "ports",
        "endpoint_atom_relations", "lanes", "bonds", "intersection_regions",
        "white_gap_certificates", "lane_separation_certificates",
    )
    for name in collections:
        issues.extend(_contiguous_ids(interpretation[name], f"{path}/{name}"))

    boundaries = {item["id"]: item for item in interpretation["boundaries"]}
    fragments = {item["id"]: item for item in interpretation["stroke_fragments"]}
    atoms = {item["id"]: item for item in interpretation["atoms"]}
    ports = {item["id"]: item for item in interpretation["ports"]}
    lanes = {item["id"]: item for item in interpretation["lanes"]}
    bonds = {item["id"]: item for item in interpretation["bonds"]}
    regions = {item["id"]: item for item in interpretation["intersection_regions"]}
    gap_certificates = {
        item["id"]: item for item in interpretation["white_gap_certificates"]
    }

    boundary_support: dict[int, tuple[set[tuple[int, int]], set[tuple[int, int]]]] = {}
    source_lsd_indices: set[int] = set()
    for index, boundary in enumerate(interpretation["boundaries"]):
        current_path = f"{path}/boundaries/{index}"
        if boundary["segment_q4"][0] >= boundary["segment_q4"][1]:
            issues.append(_issue(
                "boundary_not_canonical", f"{current_path}/segment_q4",
                "boundary endpoints must be strictly lexicographically increasing",
            ))
        final_pixels, current = _support(
            boundary, "final_foreground_support_runs",
            "final_foreground_support_pixel_count", current_path,
        )
        issues.extend(current)
        bond_pixels, current = _support(
            boundary, "bond_mask_support_runs", "bond_mask_support_pixel_count",
            current_path,
        )
        issues.extend(current)
        if not final_pixels <= masks["final_foreground"]:
            issues.append(_issue(
                "boundary_support_outside_foreground", current_path,
                "boundary final support must belong to input foreground",
            ))
        if not bond_pixels <= masks["bond_mask"]:
            issues.append(_issue(
                "boundary_support_outside_bond_mask", current_path,
                "boundary bond support must belong to the complete bond mask",
            ))
        if not bond_pixels <= final_pixels:
            issues.append(_issue(
                "boundary_bond_support_not_foreground", current_path,
                "one boundary's bond-mask support must be a subset of its own final support",
            ))
        segment = {
            "polyline_q4": boundary["segment_q4"],
            "width_q4": thresholds["boundary_support_width_q4"],
        }
        issues.extend(v01._geometry_support_issues(segment, final_pixels, current_path))
        if boundary["source_lsd_index"] in source_lsd_indices:
            issues.append(_issue(
                "source_lsd_index_reused", f"{current_path}/source_lsd_index",
                "one raw LSD record cannot name two root boundaries",
            ))
        source_lsd_indices.add(boundary["source_lsd_index"])
        boundary_support[boundary["id"]] = (final_pixels, bond_pixels)

    if interpretation["boundaries"] != list(boundary_universe.values()):
        issues.append(_issue(
            "interpretation_boundary_universe_mismatch", f"{path}/boundaries",
            "every interpretation must repeat the frozen raw-boundary universe exactly",
        ))
    matching = interpretation["boundary_matching"]
    selected_pair_ids = matching["compatible_pair_ids"]
    if not _sorted_unique(selected_pair_ids) or any(value not in compatible_pairs for value in selected_pair_ids):
        issues.append(_issue(
            "compatible_pair_reference_invalid", f"{path}/boundary_matching/compatible_pair_ids",
            "matching pair IDs must be sorted, unique and present in the compatible-pair universe",
        ))
    consumed: set[int] = set()
    pair_members: set[tuple[int, ...]] = set()
    pair_by_members: dict[tuple[int, ...], dict[str, Any]] = {}
    for index, pair_id in enumerate(selected_pair_ids):
        if pair_id not in compatible_pairs:
            continue
        pair = compatible_pairs[pair_id]
        ids = pair["boundary_ids"]
        pair_path = f"{path}/boundary_matching/compatible_pair_ids/{index}"
        if not _sorted_unique(ids):
            issues.append(_issue(
                "boundary_pair_not_sorted_unique", f"{pair_path}/boundary_ids",
                "boundary pair IDs must be strictly sorted",
            ))
        if any(value not in boundaries for value in ids):
            issues.append(_issue(
                "unknown_boundary", f"{pair_path}/boundary_ids",
                "boundary pairing references an unknown boundary",
            ))
        if consumed & set(ids):
            issues.append(_issue(
                "boundary_reused_in_matching", pair_path,
                "one matching cannot consume a boundary twice",
            ))
        consumed.update(ids)
        pair_members.add(tuple(ids))
        pair_by_members[tuple(ids)] = pair
    unmatched = matching["unmatched_boundary_ids"]
    if not _sorted_unique(unmatched):
        issues.append(_issue(
            "unmatched_not_sorted_unique", f"{path}/boundary_matching/unmatched_boundary_ids",
            "unmatched boundary IDs must be strictly sorted",
        ))
    if consumed & set(unmatched) or consumed | set(unmatched) != set(boundaries):
        issues.append(_issue(
            "boundary_matching_incomplete", f"{path}/boundary_matching",
            "pairings and unmatched IDs must partition every raw boundary exactly once",
        ))

    fragment_support: dict[int, set[tuple[int, int]]] = {}
    fragment_by_boundary: dict[int, list[int]] = {}
    for index, fragment in enumerate(interpretation["stroke_fragments"]):
        current_path = f"{path}/stroke_fragments/{index}"
        member_ids = fragment["boundary_member_ids"]
        if not _sorted_unique(member_ids) or any(value not in boundaries for value in member_ids):
            issues.append(_issue(
                "fragment_boundary_reference_invalid", f"{current_path}/boundary_member_ids",
                "fragment boundary IDs must be sorted, unique and known",
            ))
        if fragment["segment_q4"][0] >= fragment["segment_q4"][1]:
            issues.append(_issue(
                "fragment_not_canonical", f"{current_path}/segment_q4",
                "fragment endpoints must be strictly lexicographically increasing",
            ))
        derivation = fragment["derivation"]
        if derivation == "paired_boundaries" and tuple(member_ids) not in pair_members:
            issues.append(_issue(
                "fragment_pair_not_in_matching", current_path,
                "paired fragment must use one emitted boundary pair",
            ))
        elif derivation == "paired_boundaries" and fragment["segment_q4"] != pair_by_members[tuple(member_ids)]["canonical_centerline_q4"]:
            issues.append(_issue(
                "paired_fragment_centerline_mismatch", current_path,
                "paired-boundary fragment must use the compatible pair's canonical centerline",
            ))
        if derivation == "unmatched_boundary" and (
            len(member_ids) != 1 or member_ids[0] not in unmatched
        ):
            issues.append(_issue(
                "fragment_unmatched_derivation_invalid", current_path,
                "unmatched fragment must use exactly one unmatched boundary",
            ))
        elif (
            derivation == "unmatched_boundary"
            and member_ids
            and member_ids[0] in boundaries
            and fragment["segment_q4"] != boundaries[member_ids[0]]["segment_q4"]
        ):
            issues.append(_issue(
                "unmatched_fragment_segment_mismatch", current_path,
                "an unmatched-boundary fragment must repeat that boundary segment",
            ))
        pixels, current = _support(
            fragment, "support_runs", "support_pixel_count", current_path,
        )
        issues.extend(current)
        if not pixels <= masks["final_foreground"]:
            issues.append(_issue(
                "fragment_support_outside_foreground", current_path,
                "fragment support must belong to final foreground",
            ))
        known_member_ids = [value for value in member_ids if value in boundary_support]
        member_support = set().union(*(boundary_support[value][0] for value in known_member_ids)) if known_member_ids else set()
        if len(known_member_ids) != len(member_ids) or not pixels <= member_support:
            issues.append(_issue(
                "fragment_support_outside_boundaries", current_path,
                "fragment support must be contained in the final support of its member boundaries",
            ))
        segment = {
            "polyline_q4": fragment["segment_q4"],
            "width_q4": fragment["width_q4"],
        }
        if pixels:
            issues.extend(v01._geometry_support_issues(segment, pixels, current_path))
        fragment_support[fragment["id"]] = pixels
        for boundary_id in member_ids:
            fragment_by_boundary.setdefault(boundary_id, []).append(fragment["id"])

    for boundary_id in boundaries:
        member_fragments = fragment_by_boundary.get(boundary_id, [])
        if not member_fragments:
            issues.append(_issue(
                "boundary_without_fragment", f"{path}/boundaries/{boundary_id}",
                "every root boundary must produce at least one local fragment",
            ))
        elif len(member_fragments) > 1 and any(
            fragments[fragment_id]["derivation"] != "split_at_intersection"
            for fragment_id in member_fragments
        ):
            issues.append(_issue(
                "boundary_fragment_fiber_not_split", f"{path}/boundaries/{boundary_id}",
                "a boundary may produce multiple fragments only through named intersection splitting",
            ))

    region_support: dict[int, set[tuple[int, int]]] = {}
    for index, region in enumerate(interpretation["intersection_regions"]):
        current_path = f"{path}/intersection_regions/{index}"
        for key, known in (
            ("boundary_member_ids", boundaries),
            ("stroke_fragment_member_ids", fragments),
            ("port_member_ids", ports),
        ):
            values = region[key]
            if not _sorted_unique(values) or any(value not in known for value in values):
                issues.append(_issue(
                    "intersection_reference_invalid", f"{current_path}/{key}",
                    "intersection references must be sorted, unique and known",
                ))
        pixels, current = _support(
            region, "support_runs", "support_pixel_count", current_path,
        )
        issues.extend(current)
        if not pixels <= masks["final_foreground"]:
            issues.append(_issue(
                "intersection_support_outside_foreground", current_path,
                "intersection support must belong to final foreground",
            ))
        region_support[region["id"]] = pixels
        member_fragment_ids = [
            value for value in region["stroke_fragment_member_ids"] if value in fragment_support
        ]
        member_pixels = set().union(*(fragment_support[value] for value in member_fragment_ids)) if member_fragment_ids else set()
        if (
            len(member_fragment_ids) != len(region["stroke_fragment_member_ids"])
            or not pixels <= member_pixels
            or any(not (fragment_support[value] & pixels) for value in member_fragment_ids)
        ):
            issues.append(_issue(
                "intersection_support_member_mismatch", current_path,
                "intersection support must lie in, and touch every, named fragment support",
            ))
        expected_boundary_ids = sorted({
            boundary_id
            for fragment_id in member_fragment_ids
            for boundary_id in fragments[fragment_id]["boundary_member_ids"]
        })
        if region["boundary_member_ids"] != expected_boundary_ids:
            issues.append(_issue(
                "intersection_boundary_union_mismatch", current_path,
                "intersection boundary IDs must equal the union induced by member fragments",
            ))
        center_pixel = (region["center_q4"][0] // 16, region["center_q4"][1] // 16)
        if center_pixel not in pixels:
            issues.append(_issue(
                "intersection_center_outside_support", current_path,
                "intersection center must quantize into its named raster support",
            ))
        expected_port_ids = sorted({
            port_id
            for port_id, port in ports.items()
            for ref in port["fragment_endpoint_refs"]
            if ref["fragment_id"] in member_fragment_ids
            and ref["fragment_id"] in fragments
            and (
                _point(fragments[ref["fragment_id"]], ref["endpoint_index"])[0] // 16,
                _point(fragments[ref["fragment_id"]], ref["endpoint_index"])[1] // 16,
            ) in pixels
        })
        if region["port_member_ids"] != expected_port_ids:
            issues.append(_issue(
                "intersection_port_union_mismatch", current_path,
                "intersection port IDs must equal ports whose referenced endpoints lie in region support",
            ))
        if (
            (region["kind"] == "crossing" and expected_port_ids)
            or (region["kind"] == "junction" and not expected_port_ids)
        ):
            issues.append(_issue(
                "intersection_kind_port_mismatch", current_path,
                "crossing has no connecting port; junction requires at least one connecting port",
            ))
    fragment_ids = sorted(fragment_support)
    for left_index, left_id in enumerate(fragment_ids):
        for right_id in fragment_ids[left_index + 1:]:
            pair_allowed_pixels = set().union(*(
                region_support[region["id"]]
                for region in interpretation["intersection_regions"]
                if left_id in region["stroke_fragment_member_ids"] and right_id in region["stroke_fragment_member_ids"]
            )) if any(
                left_id in region["stroke_fragment_member_ids"] and right_id in region["stroke_fragment_member_ids"]
                for region in interpretation["intersection_regions"]
            ) else set()
            illegal = (fragment_support[left_id] & fragment_support[right_id]) - pair_allowed_pixels
            if illegal:
                issues.append(_issue(
                    "pixel_owner_not_unique", f"{path}/stroke_fragments",
                    f"fragments {left_id} and {right_id} share pixels outside a named intersection",
                ))
    split_fragment_ids = {
        fragment["id"]
        for fragment in interpretation["stroke_fragments"]
        if fragment["derivation"] == "split_at_intersection"
    }
    named_split_ids = {
        fragment_id
        for region in interpretation["intersection_regions"]
        for fragment_id in region["stroke_fragment_member_ids"]
    }
    if not split_fragment_ids <= named_split_ids:
        issues.append(_issue(
            "unnamed_intersection_split", f"{path}/stroke_fragments",
            "every split_at_intersection fragment must belong to a named intersection region",
        ))

    ownership = interpretation["ownership"]
    owned_ids = [item["fragment_id"] for item in ownership]
    if owned_ids != sorted(set(owned_ids)) or set(owned_ids) != set(fragments):
        issues.append(_issue(
            "ownership_not_exact", f"{path}/ownership",
            "ownership must contain exactly one canonical record per fragment",
        ))
    owner_by_fragment = {item["fragment_id"]: item for item in ownership}
    for fragment_id, owner in owner_by_fragment.items():
        fragment = fragments.get(fragment_id)
        if fragment is None:
            continue
        pixels = fragment_support[fragment_id]
        if owner["owner_kind"] == "bond_lane" and not pixels <= masks["bond_mask"]:
            issues.append(_issue(
                "bond_owner_outside_bond_mask", f"{path}/ownership/{fragment_id}",
                "bond-lane fragment support must be in the complete bond mask",
            ))
        if owner["owner_kind"] in {"text_glyph", "charge"} and not pixels <= masks["text_mask"]:
            issues.append(_issue(
                "text_owner_outside_text_mask", f"{path}/ownership/{fragment_id}",
                "text/charge fragment support must be in the complete text mask",
            ))
        expected_primitive = {"bond_lane": "straight", "text_glyph": "glyph", "charge": "charge"}.get(owner["owner_kind"])
        if expected_primitive is not None and fragment["primitive_kind"] != expected_primitive:
            issues.append(_issue(
                "owner_primitive_mismatch", f"{path}/ownership/{fragment_id}",
                "ownership kind and primitive kind must agree",
            ))

    atom_source_ids = [atom["source_proposal_id"] for atom in interpretation["atoms"]]
    if len(atom_source_ids) != len(set(atom_source_ids)):
        issues.append(_issue(
            "atom_proposal_reused", f"{path}/atoms",
            "one interpretation cannot instantiate one Atom proposal twice",
        ))
    atom_anchors = [tuple(atom["anchor_q4"]) for atom in interpretation["atoms"]]
    if len(atom_anchors) != len(set(atom_anchors)):
        issues.append(_issue(
            "duplicate_atom_anchor", f"{path}/atoms",
            "one spatial interpretation cannot contain two indistinguishable Atom anchors",
        ))

    for index, atom in enumerate(interpretation["atoms"]):
        current_path = f"{path}/atoms/{index}"
        if not _sorted_unique(atom["port_ids"]):
            issues.append(_issue("atom_ports_not_sorted", f"{current_path}/port_ids", "port IDs must be sorted unique"))
        if not _sorted_unique(atom["glyph_fragment_ids"]):
            issues.append(_issue("atom_glyphs_not_sorted", f"{current_path}/glyph_fragment_ids", "glyph fragment IDs must be sorted unique"))
        if any(value not in fragments for value in atom["glyph_fragment_ids"]):
            issues.append(_issue("unknown_atom_glyph", current_path, "atom references an unknown glyph fragment"))
        if atom["kind"] == "implicit_vertex" and atom["glyph_fragment_ids"]:
            issues.append(_issue("implicit_atom_has_glyph", current_path, "implicit atom cannot own glyph fragments"))
        if atom["kind"] == "explicit_label" and not atom["glyph_fragment_ids"]:
            issues.append(_issue("explicit_atom_missing_glyph", current_path, "explicit atom requires visible glyph provenance"))
        proposal = atom_proposals.get(atom["source_proposal_id"])
        if proposal is None:
            issues.append(_issue("unknown_atom_proposal", current_path, "atom references an unknown proposal-universe member"))
        else:
            if (
                atom["anchor_q4"] != proposal["anchor_q4"]
                or atom["source_cell_q4"] != proposal["source_cell_q4"]
                or atom["presence_score_q24"] != proposal["heatmap_score_q24"]
            ):
                issues.append(_issue("atom_proposal_mismatch", current_path, "local atom geometry and score must match its proposal-universe member"))
            if proposal["set_decision"] != "active":
                issues.append(_issue(
                    "inactive_atom_proposal_used", current_path,
                    "an interpretation may instantiate only active Atom proposals",
                ))

        for fragment_id in atom["glyph_fragment_ids"]:
            owner = owner_by_fragment.get(fragment_id)
            if owner is None or owner["owner_kind"] != "text_glyph" or owner["owner_id"] != atom["id"]:
                issues.append(_issue(
                    "atom_glyph_owner_mismatch", current_path,
                    "every Atom glyph fragment must be owned by that explicit Atom",
                ))

    relation_by_ref: dict[tuple[int, int], list[dict[str, Any]]] = {}
    relation_keys: set[tuple[int, int, int, int]] = set()
    for index, relation in enumerate(interpretation["endpoint_atom_relations"]):
        current_path = f"{path}/endpoint_atom_relations/{index}"
        ref = relation["endpoint_ref"]
        ref_key = (ref["fragment_id"], ref["endpoint_index"])
        if ref["fragment_id"] not in fragments or relation["atom_id"] not in atoms or relation["port_id"] not in ports:
            issues.append(_issue("endpoint_atom_unknown_reference", current_path, "relation references unknown fragment, atom or port"))
            continue
        port = ports[relation["port_id"]]
        if port["atom_id"] != relation["atom_id"]:
            issues.append(_issue("endpoint_atom_port_mismatch", current_path, "relation atom must own relation port"))
        actual_endpoint = _point(fragments[ref["fragment_id"]], ref["endpoint_index"])
        opposite_endpoint = _point(fragments[ref["fragment_id"]], 1 - ref["endpoint_index"])
        evidence = relation["geometry_evidence"]
        atom_anchor = tuple(atoms[relation["atom_id"]]["anchor_q4"])
        outward = (
            actual_endpoint[0] - opposite_endpoint[0],
            actual_endpoint[1] - opposite_endpoint[1],
        )
        attachment = (
            atom_anchor[0] - actual_endpoint[0],
            atom_anchor[1] - actual_endpoint[1],
        )
        outward_dot_q8 = outward[0] * attachment[0] + outward[1] * attachment[1]
        absolute_cross_q8 = abs(outward[0] * attachment[1] - outward[1] * attachment[0])
        if tuple(evidence["endpoint_q4"]) != actual_endpoint or tuple(evidence["atom_anchor_q4"]) != atom_anchor:
            issues.append(_issue("endpoint_geometry_mismatch", current_path, "geometry evidence must repeat actual endpoint and atom anchor"))
        if evidence["distance_q4"] != _distance_floor(actual_endpoint, atom_anchor):
            issues.append(_issue("endpoint_distance_mismatch", current_path, "distance_q4 must be recomputed from endpoint and atom anchor"))
        if (
            evidence["outward_dot_q8"] != outward_dot_q8
            or evidence["absolute_cross_q8"] != absolute_cross_q8
        ):
            issues.append(_issue(
                "endpoint_direction_evidence_mismatch", current_path,
                "outward dot and absolute cross must be recomputed exactly in Q8",
            ))
        expected_angle = _angle_millidegrees(actual_endpoint, opposite_endpoint, atom_anchor)
        if evidence["angle_millidegrees"] != expected_angle:
            issues.append(_issue(
                "endpoint_angle_mismatch", current_path,
                f"angle_millidegrees must be recomputed from fragment direction and Atom anchor; expected {expected_angle}",
            ))
        if _squared_distance(actual_endpoint, atom_anchor) > evidence["attachment_radius_q4"] ** 2:
            issues.append(_issue("endpoint_outside_attachment_radius", current_path, "endpoint lies outside claimed attachment radius"))
        if (
            evidence["attachment_radius_q4"] != thresholds["attach_max_endpoint_distance_q4"]
            or _squared_distance(actual_endpoint, atom_anchor) > thresholds["attach_max_endpoint_distance_q4"] ** 2
            or evidence["angle_millidegrees"] > thresholds["attach_max_angle_millidegrees"]
            or outward_dot_q8 < 0
        ):
            issues.append(_issue(
                "endpoint_attachment_predicate_failed", current_path,
                "endpoint relation must satisfy the frozen distance and angle gate",
            ))
        key = (*ref_key, relation["atom_id"], relation["port_id"])
        if key in relation_keys:
            issues.append(_issue("duplicate_endpoint_atom_relation", current_path, "relation is duplicated"))
        relation_keys.add(key)
        relation_by_ref.setdefault(ref_key, []).append(relation)

    for ref_key, rows in relation_by_ref.items():
        if len(rows) > 8:
            issues.append(_issue(
                "endpoint_alternative_capacity_exceeded", f"{path}/endpoint_atom_relations",
                f"endpoint {ref_key} has {len(rows)} alternatives; maximum is 8",
            ))

    for index, port in enumerate(interpretation["ports"]):
        current_path = f"{path}/ports/{index}"
        refs = [(item["fragment_id"], item["endpoint_index"]) for item in port["fragment_endpoint_refs"]]
        if refs != sorted(set(refs)):
            issues.append(_issue("port_refs_not_sorted_unique", f"{current_path}/fragment_endpoint_refs", "port endpoint refs must be sorted unique"))
        if port["atom_id"] not in atoms or any(fragment_id not in fragments for fragment_id, _ in refs):
            issues.append(_issue("port_unknown_reference", current_path, "port references unknown atom or fragment"))
            continue
        if port["radius_q4"] > thresholds["attach_max_endpoint_distance_q4"]:
            issues.append(_issue(
                "port_radius_outside_attachment_gate", current_path,
                "port radius cannot exceed the frozen endpoint attachment radius",
            ))
        support_ids = sorted({fragment_id for fragment_id, _ in refs})
        if port["support_fragment_ids"] != support_ids:
            issues.append(_issue("port_support_union_mismatch", current_path, "support_fragment_ids must equal endpoint fragment union"))
        for ref in refs:
            if not any(row["port_id"] == port["id"] and row["atom_id"] == port["atom_id"] for row in relation_by_ref.get(ref, [])):
                issues.append(_issue("port_relation_missing", current_path, f"port endpoint {ref} lacks matching endpoint-atom relation"))
            endpoint = _point(fragments[ref[0]], ref[1])
            if _squared_distance(endpoint, tuple(port["center_q4"])) > port["radius_q4"] ** 2:
                issues.append(_issue("endpoint_port_disjoint", current_path, "aligned endpoint lies outside port region"))
        mirrored = sorted(
            (ref, row["id"])
            for ref in refs
            for row in relation_by_ref.get(ref, [])
            if row["port_id"] == port["id"] and row["atom_id"] == port["atom_id"]
        )
        if len(mirrored) != len(refs):
            issues.append(_issue(
                "port_relation_not_bijective", current_path,
                "each port endpoint must have exactly one mirrored endpoint-Atom relation",
            ))
    ports_by_atom = {
        atom_id: sorted(port_id for port_id, port in ports.items() if port["atom_id"] == atom_id)
        for atom_id in atoms
    }
    for atom_id, atom in atoms.items():
        if atom["port_ids"] != ports_by_atom[atom_id]:
            issues.append(_issue("atom_port_union_mismatch", f"{path}/atoms/{atom_id}/port_ids", "atom port_ids must equal all ports owned by that atom"))

    mirrored_relation_ids = {
        row["id"]
        for port in ports.values()
        for ref in ((item["fragment_id"], item["endpoint_index"]) for item in port["fragment_endpoint_refs"])
        for row in relation_by_ref.get(ref, [])
        if row["port_id"] == port["id"] and row["atom_id"] == port["atom_id"]
    }
    if mirrored_relation_ids != set(range(len(interpretation["endpoint_atom_relations"]))):
        issues.append(_issue(
            "endpoint_relation_not_mirrored", f"{path}/endpoint_atom_relations",
            "every relation must be mirrored by exactly one port endpoint reference",
        ))

    for certificate in interpretation["white_gap_certificates"]:
        if certificate["input_png_sha256"] != png_sha256:
            issues.append(_issue("white_gap_wrong_observation", f"{path}/white_gap_certificates/{certificate['id']}", "gap certificate must bind final PNG"))
        pixels, current = _support(certificate, "support_runs", "support_pixel_count", f"{path}/white_gap_certificates/{certificate['id']}")
        issues.extend(current)
        if pixels & masks["final_foreground"]:
            issues.append(_issue("white_gap_not_clear", f"{path}/white_gap_certificates/{certificate['id']}", "white-gap support must be white in final raster"))
        span_begin, span_end = (tuple(point) for point in certificate["span_q4"])
        if any(
            not _point_near_segment_q4(
                (16 * x + 8, 16 * y + 8), span_begin, span_end,
                thresholds["boundary_support_width_q4"],
            )
            for x, y in pixels
        ):
            issues.append(_issue(
                "white_gap_support_far_from_span", f"{path}/white_gap_certificates/{certificate['id']}",
                "every white-gap support pixel must lie near the certified span",
            ))
        expected_white_support = _complete_white_span_support(
            span_begin, span_end, masks["final_foreground"],
            thresholds["boundary_support_width_q4"],
        )
        if pixels != expected_white_support:
            issues.append(_issue(
                "white_gap_support_incomplete", f"{path}/white_gap_certificates/{certificate['id']}",
                "white-gap support must equal the complete deterministic white span neighborhood",
            ))
        if (
            certificate["white_occupancy_ppm"] != 1_000_000
            or certificate["bond_probability_min_q24"] < thresholds["p1a_bond_threshold_q24"]
            or certificate["text_probability_max_q24"] > thresholds["p1a_text_threshold_q24"]
        ):
            issues.append(_issue(
                "white_gap_observation_gate_failed", f"{path}/white_gap_certificates/{certificate['id']}",
                "gap support is white and must satisfy frozen bond/text observation gates",
            ))
        left = fragments.get(certificate["left_fragment_id"])
        right = fragments.get(certificate["right_fragment_id"])
        if left is None or right is None:
            issues.append(_issue(
                "white_gap_unknown_fragment", f"{path}/white_gap_certificates/{certificate['id']}",
                "white-gap certificate references an unknown fragment",
            ))
        else:
            left_point = _point(left, certificate["left_endpoint_index"])
            right_point = _point(right, certificate["right_endpoint_index"])
            distance = _distance_floor(left_point, right_point)
            if (
                certificate["gap_begin_q4"] != list(left_point)
                or certificate["gap_end_q4"] != list(right_point)
                or certificate["span_q4"] != [list(left_point), list(right_point)]
                or certificate["gap_length_q4"] != distance
                or not thresholds["lane_white_gap_min_q4"] <= distance <= thresholds["lane_white_gap_max_q4"]
                or distance > thresholds["white_gap_max_length_q4"]
            ):
                issues.append(_issue(
                    "white_gap_geometry_mismatch", f"{path}/white_gap_certificates/{certificate['id']}",
                    "white-gap endpoints, span and length must match the referenced fragment endpoints and frozen gate",
                ))

    fragment_lane: dict[int, int] = {}
    used_gap_certificates: set[int] = set()
    for index, lane in enumerate(interpretation["lanes"]):
        current_path = f"{path}/lanes/{index}"
        if lane["begin_port_id"] not in ports or lane["end_port_id"] not in ports or lane["begin_port_id"] == lane["end_port_id"]:
            issues.append(_issue("lane_port_reference_invalid", current_path, "lane requires two distinct known ports"))
            continue
        chain = lane["fragment_chain"]
        chain_ids = [item["fragment_id"] for item in chain]
        if len(chain_ids) != len(set(chain_ids)) or any(value not in fragments for value in chain_ids):
            issues.append(_issue("fragment_chain_invalid", current_path, "fragment chain must reference distinct known fragments"))
            continue
        for item in chain:
            if item["begin_endpoint_index"] == item["end_endpoint_index"]:
                issues.append(_issue("fragment_chain_orientation_invalid", current_path, "fragment chain must traverse distinct endpoints"))
            fragment_id = item["fragment_id"]
            if fragment_id in fragment_lane:
                issues.append(_issue("fragment_reused_by_lanes", current_path, "one fragment cannot belong to two lanes"))
            fragment_lane[fragment_id] = lane["id"]
            owner = owner_by_fragment.get(fragment_id)
            if owner is None or owner["owner_kind"] != "bond_lane" or owner["owner_id"] != lane["id"]:
                issues.append(_issue("lane_owner_mismatch", current_path, "every lane fragment must be owned by that bond lane"))
            if fragments[fragment_id]["primitive_kind"] != "straight":
                issues.append(_issue("unsupported_bond_primitive", current_path, "v0.3 ordinary Bond lanes compile only straight primitives"))
        for adjacent_index, (left_item, right_item) in enumerate(zip(chain, chain[1:])):
            left_point = _point(fragments[left_item["fragment_id"]], left_item["end_endpoint_index"])
            right_point = _point(fragments[right_item["fragment_id"]], right_item["begin_endpoint_index"])
            matching_gaps = [
                certificate["id"]
                for certificate in interpretation["white_gap_certificates"]
                if (
                    certificate["left_fragment_id"] == left_item["fragment_id"]
                    and certificate["left_endpoint_index"] == left_item["end_endpoint_index"]
                    and certificate["right_fragment_id"] == right_item["fragment_id"]
                    and certificate["right_endpoint_index"] == right_item["begin_endpoint_index"]
                )
            ]
            if left_point == right_point:
                if matching_gaps:
                    issues.append(_issue(
                        "touching_fragments_have_gap", current_path,
                        f"chain adjacency {adjacent_index} touches and must not use a white-gap certificate",
                    ))
            elif len(matching_gaps) != 1:
                issues.append(_issue(
                    "fragment_chain_gap_missing", current_path,
                    f"chain adjacency {adjacent_index} requires exactly one oriented white-gap certificate",
                ))
        begin_ref = (chain[0]["fragment_id"], chain[0]["begin_endpoint_index"])
        end_ref = (chain[-1]["fragment_id"], chain[-1]["end_endpoint_index"])
        begin_port = ports[lane["begin_port_id"]]
        end_port = ports[lane["end_port_id"]]
        begin_refs = {(item["fragment_id"], item["endpoint_index"]) for item in begin_port["fragment_endpoint_refs"]}
        end_refs = {(item["fragment_id"], item["endpoint_index"]) for item in end_port["fragment_endpoint_refs"]}
        if begin_ref not in begin_refs or end_ref not in end_refs:
            issues.append(_issue("lane_endpoint_port_mismatch", current_path, "lane outer endpoints must align its begin/end ports"))
        gap_ids = lane["white_gap_certificate_ids"]
        if not _sorted_unique(gap_ids) or any(value not in gap_certificates for value in gap_ids):
            issues.append(_issue("lane_gap_reference_invalid", current_path, "lane gap certificate IDs must be sorted, unique and known"))
        for gap_id in gap_ids:
            if gap_id not in gap_certificates:
                continue
            certificate = gap_certificates[gap_id]
            pair = {certificate["left_fragment_id"], certificate["right_fragment_id"]}
            if not pair <= set(chain_ids):
                issues.append(_issue("lane_gap_fragment_mismatch", current_path, "gap certificate fragments must belong to the lane"))
            if gap_id in used_gap_certificates:
                issues.append(_issue("gap_certificate_reused", current_path, "gap certificate cannot belong to two lanes"))
            used_gap_certificates.add(gap_id)
        expected_gap_ids = sorted(
            certificate["id"]
            for left_item, right_item in zip(chain, chain[1:])
            for certificate in interpretation["white_gap_certificates"]
            if (
                certificate["left_fragment_id"] == left_item["fragment_id"]
                and certificate["left_endpoint_index"] == left_item["end_endpoint_index"]
                and certificate["right_fragment_id"] == right_item["fragment_id"]
                and certificate["right_endpoint_index"] == right_item["begin_endpoint_index"]
            )
        )
        if gap_ids != expected_gap_ids:
            issues.append(_issue(
                "lane_gap_set_mismatch", current_path,
                "lane gap IDs must equal exactly its non-touching oriented chain adjacencies",
            ))
    if used_gap_certificates != set(gap_certificates):
        issues.append(_issue("unused_gap_certificate", f"{path}/white_gap_certificates", "every gap certificate must belong to exactly one lane"))
    owned_lane_fragments = {
        fragment_id: owner["owner_id"]
        for fragment_id, owner in owner_by_fragment.items()
        if owner["owner_kind"] == "bond_lane"
    }
    if owned_lane_fragments != fragment_lane:
        issues.append(_issue(
            "bond_lane_ownership_fiber_mismatch", f"{path}/ownership",
            "bond_lane ownership must equal exactly the fragment-to-lane fiber",
        ))

    separation_by_bond: dict[int, list[dict[str, Any]]] = {}
    for certificate in interpretation["lane_separation_certificates"]:
        certificate_path = f"{path}/lane_separation_certificates/{certificate['id']}"
        if certificate["input_png_sha256"] != png_sha256:
            issues.append(_issue("lane_separation_wrong_observation", certificate_path, "lane separation must bind final PNG"))
        pixels, current = _support(certificate, "support_runs", "support_pixel_count", certificate_path)
        issues.extend(current)
        if pixels & masks["final_foreground"]:
            issues.append(_issue("lane_gap_not_white", certificate_path, "lane separation evidence must be white in final raster"))
        span_begin, span_end = (tuple(point) for point in certificate["span_q4"])
        if any(
            not _point_near_segment_q4(
                (16 * x + 8, 16 * y + 8), span_begin, span_end,
                thresholds["boundary_support_width_q4"],
            )
            for x, y in pixels
        ):
            issues.append(_issue(
                "lane_separation_support_far_from_span", certificate_path,
                "lane-separation support must lie near its certified cross-lane span",
            ))
        expected_white_support = _complete_white_span_support(
            span_begin, span_end, masks["final_foreground"],
            thresholds["boundary_support_width_q4"],
        )
        if pixels != expected_white_support:
            issues.append(_issue(
                "lane_separation_support_incomplete", certificate_path,
                "lane-separation support must equal the complete deterministic white span neighborhood",
            ))
        left_lane = lanes.get(certificate["left_lane_id"])
        right_lane = lanes.get(certificate["right_lane_id"])
        if left_lane is None or right_lane is None:
            issues.append(_issue(
                "lane_separation_unknown_lane", certificate_path,
                "lane-separation certificate references an unknown lane",
            ))
        elif any(
            item["fragment_id"] not in fragments
            for lane in (left_lane, right_lane)
            for item in lane["fragment_chain"]
        ):
            issues.append(_issue(
                "lane_separation_unknown_fragment", certificate_path,
                "lane-separation geometry cannot reference unknown fragments",
            ))
        else:
            def outer_endpoints(lane: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
                chain = lane["fragment_chain"]
                return (
                    _point(fragments[chain[0]["fragment_id"]], chain[0]["begin_endpoint_index"]),
                    _point(fragments[chain[-1]["fragment_id"]], chain[-1]["end_endpoint_index"]),
                )

            left_begin, left_end = outer_endpoints(left_lane)
            right_begin, right_end = outer_endpoints(right_lane)
            left_midpoint = tuple((left_begin[axis] + left_end[axis] + 1) // 2 for axis in range(2))
            right_midpoint = tuple((right_begin[axis] + right_end[axis] + 1) // 2 for axis in range(2))
            endpoint_gaps = sorted((
                _distance_floor(left_begin, right_begin),
                _distance_floor(left_end, right_end),
            ))
            if (
                certificate["span_q4"] != [list(left_midpoint), list(right_midpoint)]
                or certificate["observed_minimum_gap_q4"] != endpoint_gaps[0]
                or certificate["observed_maximum_gap_q4"] != endpoint_gaps[1]
            ):
                issues.append(_issue(
                    "lane_separation_geometry_mismatch", certificate_path,
                    "span and min/max gap must be recomputed from the two lane outer endpoints",
                ))
        if (
            certificate["white_occupancy_ppm"] != 1_000_000
            or certificate["observed_minimum_gap_q4"] < thresholds["lane_white_gap_min_q4"]
            or certificate["observed_maximum_gap_q4"] > thresholds["lane_white_gap_max_q4"]
        ):
            issues.append(_issue(
                "lane_separation_observation_gate_failed", certificate_path,
                "lane separation must satisfy frozen white occupancy and distance gates",
            ))
        separation_by_bond.setdefault(certificate["bond_id"], []).append(certificate)

    used_lanes: set[int] = set()
    seen_atom_pairs: set[tuple[int, int]] = set()
    for index, bond in enumerate(interpretation["bonds"]):
        current_path = f"{path}/bonds/{index}"
        pair = tuple(bond["atom_pair"])
        if list(pair) != sorted(set(pair)) or any(value not in atoms for value in pair):
            issues.append(_issue("atom_pair_invalid", f"{current_path}/atom_pair", "AtomPair must be two known strictly increasing atoms"))
            continue
        if pair in seen_atom_pairs:
            issues.append(_issue("duplicate_unordered_pair", current_path, "one interpretation cannot emit duplicate unordered AtomPairs"))
        seen_atom_pairs.add(pair)
        evidence = bond["atom_pair_evidence"]
        direct = direct_pairs.get(evidence["pair_id"])
        source_pair = tuple(sorted(atoms[value]["source_proposal_id"] for value in pair))
        if direct is None:
            issues.append(_issue("unknown_direct_pair", current_path, "Bond evidence references an unknown Direct AtomPair"))
        elif (
            list(source_pair) != direct["atom_proposal_pair"]
            or evidence["presence_decision"] != direct["presence_decision"]
            or evidence["presence_score_q24"] != direct["presence_score_q24"]
        ):
            issues.append(_issue("direct_pair_mismatch", current_path, "Bond AtomPair evidence must exactly match the root Direct-pair universe"))
        if bond["begin_port_id"] not in ports or bond["end_port_id"] not in ports:
            issues.append(_issue("bond_unknown_port", current_path, "bond references an unknown port"))
            continue
        if ports[bond["begin_port_id"]]["atom_id"] != pair[0] or ports[bond["end_port_id"]]["atom_id"] != pair[1]:
            issues.append(_issue("port_atom_mismatch", current_path, "begin/end ports must belong to atom_pair[0]/atom_pair[1]"))
        lane_ids = bond["lane_ids"]
        if not _sorted_unique(lane_ids) or any(value not in lanes for value in lane_ids):
            issues.append(_issue("bond_lane_reference_invalid", current_path, "bond lane IDs must be sorted, unique and known"))
            continue
        if bond["order"] != len(lane_ids):
            issues.append(_issue("order_lane_count_mismatch", current_path, "Bond order must equal physical lane count"))
        alignment = bond["lane_endpoint_alignment"]
        expected_alignment = [
            {"begin_port_id": bond["begin_port_id"], "end_port_id": bond["end_port_id"], "lane_id": lane_id}
            for lane_id in lane_ids
        ]
        if alignment != expected_alignment:
            issues.append(_issue("bond_lane_alignment_mismatch", current_path, "alignment must exactly cover every lane in canonical order"))
        for lane_id in lane_ids:
            if lane_id in used_lanes:
                issues.append(_issue("lane_reused_by_bonds", current_path, "one lane cannot belong to two Bonds"))
            used_lanes.add(lane_id)
            lane = lanes[lane_id]
            if lane["begin_port_id"] != bond["begin_port_id"] or lane["end_port_id"] != bond["end_port_id"]:
                issues.append(_issue("bond_lane_port_mismatch", current_path, "all Bond lanes must share Bond ports"))
        certificates = sorted(separation_by_bond.get(bond["id"], []), key=lambda item: item["adjacent_lane_index"])
        if len(certificates) != max(0, len(lane_ids) - 1):
            issues.append(_issue("lane_separation_certificate_missing", current_path, "order-2/3 Bond requires one certificate per adjacent lane pair"))
        for adjacent, certificate in enumerate(certificates):
            if (
                adjacent + 1 >= len(lane_ids)
                or certificate["adjacent_lane_index"] != adjacent
                or certificate["left_lane_id"] != lane_ids[adjacent]
                or certificate["right_lane_id"] != lane_ids[adjacent + 1]
            ):
                issues.append(_issue("lane_separation_alignment_invalid", current_path, "separation certificate must follow canonical adjacent lane order"))
    if used_lanes != set(lanes):
        issues.append(_issue("lane_bond_fiber_incomplete", f"{path}/bonds", "every lane must belong to exactly one Bond"))
    if set(separation_by_bond) - set(bonds):
        issues.append(_issue("unknown_separation_bond", f"{path}/lane_separation_certificates", "separation certificate references unknown Bond"))
    return issues


def _candidate_issues(
    document: dict[str, Any], input_foreground: set[tuple[int, int]], png_sha256: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(canonical_json_bytes(document)) > MAX_OUTPUT_BYTES:
        issues.append(_issue("output_outside_capacity", "$", "canonical output exceeds 64 MiB"))
    masks, current = _mask_issues(document, png_sha256, input_foreground)
    issues.extend(current)
    provenance = document["provenance"]
    expected_hashes = {
        "p1a_checkpoint_sha256": P1A_SHA256,
        "atom_source_checkpoint_sha256": ATOM_SOURCE_SHA256,
        "vision_checkpoint_sha256": VISION_SHA256,
        "direct_atom_pair_checkpoint_sha256": DIRECT_SHA256,
        "contract_sha256": PROTOCOL_SHA256,
    }
    for key, expected in expected_hashes.items():
        if provenance[key] != expected:
            issues.append(_issue("frozen_provenance_mismatch", f"$/provenance/{key}", f"expected {expected}"))

    boundary_records = document["boundary_universe"]
    pair_records = document["compatible_boundary_pairs"]
    proposal_records = document["atom_proposal_universe"]
    direct_records = document["direct_pair_universe"]
    for name, records in (
        ("boundary_universe", boundary_records),
        ("compatible_boundary_pairs", pair_records),
        ("atom_proposal_universe", proposal_records),
        ("direct_pair_universe", direct_records),
    ):
        issues.extend(_contiguous_ids(records, f"$/{name}"))
    boundary_universe = {item["id"]: item for item in boundary_records}
    compatible_pairs = {item["id"]: item for item in pair_records}
    atom_proposals = {item["id"]: item for item in proposal_records}
    direct_pairs = {item["id"]: item for item in direct_records}
    thresholds = provenance["fixed_thresholds"]
    seen_boundary_pairs: set[tuple[int, int]] = set()
    for index, pair in enumerate(pair_records):
        path = f"$/compatible_boundary_pairs/{index}"
        pair_ids = pair["boundary_ids"]
        if not _sorted_unique(pair_ids) or any(value not in boundary_universe for value in pair_ids):
            issues.append(_issue("compatible_boundary_pair_invalid", path, "compatible pair must reference two known sorted boundaries"))
        pair_key = tuple(pair_ids)
        if pair_key in seen_boundary_pairs:
            issues.append(_issue("duplicate_compatible_boundary_pair", path, "boundary pair appears more than once"))
        seen_boundary_pairs.add(pair_key)
        if all(value in boundary_universe for value in pair_ids):
            expected_geometry = _compatible_pair_geometry(
                boundary_universe[pair_ids[0]], boundary_universe[pair_ids[1]],
            )
            mismatched = [
                key for key, expected in expected_geometry.items()
                if pair[key] != expected
            ]
            if mismatched:
                issues.append(_issue(
                    "compatible_pair_geometry_mismatch", path,
                    f"pair geometry fields must be recomputed from root boundaries: {mismatched}",
                ))
        if (
            pair["observed_angle_mdeg"] > thresholds["pairing_max_angle_millidegrees"]
            or not thresholds["pairing_min_separation_q4"] <= pair["separation_q4"] <= thresholds["pairing_max_separation_q4"]
            or pair["overlap_q4"] < thresholds["pairing_min_overlap_q4"]
            or pair["shorter_overlap_ppm"] < thresholds["pairing_shorter_overlap_ppm"]
            or pair["interior_ink_occupancy_ppm"] < thresholds["pairing_interior_occupancy_ppm"]
            or pair["interior_darkness_q24"] < thresholds["pairing_darkness_threshold_q24"]
        ):
            issues.append(_issue("compatible_pair_predicate_failed", path, "published compatible pair fails frozen observable thresholds"))
        if pair["canonical_centerline_q4"][0] >= pair["canonical_centerline_q4"][1]:
            issues.append(_issue("compatible_centerline_not_canonical", path, "centerline endpoints must be lexicographically increasing"))
    for index, proposal in enumerate(proposal_records):
        expected_decision = (
            "active"
            if (
                proposal["heatmap_score_q24"] >= thresholds["atom_proposal_threshold_q24"]
                and proposal["set_logit_q20"] >= thresholds["atom_set_logit_q20"]
            )
            else "inactive"
        )
        if proposal["set_decision"] != expected_decision:
            issues.append(_issue(
                "atom_proposal_decision_mismatch", f"$/atom_proposal_universe/{index}",
                "set_decision must be recomputed from both frozen Atom gates",
            ))
    active_proposals = [item["id"] for item in proposal_records if item["set_decision"] == "active"]
    expected_direct_pairs = [
        [left, right]
        for left_index, left in enumerate(active_proposals)
        for right in active_proposals[left_index + 1:]
    ]
    actual_direct_pairs = [item["atom_proposal_pair"] for item in direct_records]
    if actual_direct_pairs != expected_direct_pairs:
        issues.append(_issue("direct_pair_universe_incomplete", "$/direct_pair_universe", "Direct-pair universe must exactly enumerate all active proposal pairs"))
    for index, pair in enumerate(direct_records):
        score = pair["presence_score_q24"]
        if score >= thresholds["direct_presence_threshold_q24"]:
            expected_decision = "positive"
        elif score >= thresholds["direct_uncertain_lower_q24"]:
            expected_decision = "uncertain"
        else:
            expected_decision = "negative"
        if pair["presence_decision"] != expected_decision:
            issues.append(_issue(
                "direct_pair_decision_mismatch", f"$/direct_pair_universe/{index}",
                "Direct pair decision must be recomputed from the frozen negative/uncertain/positive bands",
            ))

    interpretations = document["interpretations"]
    issues.extend(_contiguous_ids(interpretations, "$/interpretations"))
    ordering = [canonical_json_bytes({key: value for key, value in item.items() if key != "id"}) for item in interpretations]
    if ordering != sorted(set(ordering)):
        issues.append(_issue("interpretations_not_canonical", "$/interpretations", "interpretations must be sorted and byte-unique after removing ID"))
    for index, interpretation in enumerate(interpretations):
        issues.extend(_interpretation_issues(
            interpretation,
            masks,
            png_sha256,
            boundary_universe,
            compatible_pairs,
            atom_proposals,
            direct_pairs,
            thresholds,
            f"$/interpretations/{index}",
        ))
    if interpretations:
        first_boundaries = canonical_json_bytes(boundary_records)
        for index, interpretation in enumerate(interpretations[1:], 1):
            if canonical_json_bytes(interpretation["boundaries"]) != first_boundaries:
                issues.append(_issue("raw_boundary_domain_changed", f"$/interpretations/{index}/boundaries", "all interpretations must share one frozen raw-boundary domain"))

    actual_graphs: dict[str, list[int]] = {}
    for interpretation in interpretations:
        actual_graphs.setdefault(_spatial_graph_sha256(interpretation), []).append(interpretation["id"])
    expected_classes = [
        {"canonical_spatial_graph_sha256": digest, "id": class_id, "interpretation_ids": ids}
        for class_id, (digest, ids) in enumerate(sorted(actual_graphs.items()))
    ]
    graph_equivalence = document["graph_equivalence"]
    if graph_equivalence["class_count"] != len(expected_classes) or graph_equivalence["graph_classes"] != expected_classes:
        issues.append(_issue("graph_classes_mismatch", "$/graph_equivalence", "graph classes must be recomputed from canonical spatial Atom/Bond graphs"))

    taxonomy = document["ambiguity_taxonomy"]
    issues.extend(_contiguous_ids(taxonomy["entries"], "$/ambiguity_taxonomy/entries"))
    if taxonomy["entry_count"] != len(taxonomy["entries"]):
        issues.append(_issue("ambiguity_count_mismatch", "$/ambiguity_taxonomy/entry_count", "entry_count must equal entries length"))
    class_ids = set(range(len(expected_classes)))
    interpretation_ids = set(range(len(interpretations)))
    if len(expected_classes) > 1 and not taxonomy["entries"]:
        issues.append(_issue("ambiguity_taxonomy_missing", "$/ambiguity_taxonomy/entries", "multiple graph classes require at least one ambiguity entry"))
    object_collections = {
        "boundary": "boundaries",
        "fragment": "stroke_fragments",
        "intersection": "intersection_regions",
        "port": "ports",
        "lane": "lanes",
        "white_gap": "white_gap_certificates",
        "lane_separation": "lane_separation_certificates",
    }
    covered_class_ids: set[int] = set()
    for index, entry in enumerate(taxonomy["entries"]):
        if TAXONOMY_KIND_BY_CODE[entry["taxonomy_code"]] != entry["kind"]:
            issues.append(_issue(
                "ambiguity_code_kind_mismatch", f"$/ambiguity_taxonomy/entries/{index}",
                "taxonomy_code and kind must follow the frozen one-to-one codebook",
            ))
        if not _sorted_unique(entry["graph_class_ids"]) or not set(entry["graph_class_ids"]) <= class_ids:
            issues.append(_issue("ambiguity_graph_class_invalid", f"$/ambiguity_taxonomy/entries/{index}", "ambiguity class IDs must be sorted, unique and known"))
        if not _sorted_unique(entry["interpretation_ids"]) or not set(entry["interpretation_ids"]) <= interpretation_ids:
            issues.append(_issue("ambiguity_interpretation_invalid", f"$/ambiguity_taxonomy/entries/{index}", "ambiguity interpretation IDs must be sorted, unique and known"))
        expected_witnesses = sorted({
            interpretation_id
            for class_id in entry["graph_class_ids"]
            if class_id < len(expected_classes)
            for interpretation_id in expected_classes[class_id]["interpretation_ids"]
        })
        if entry["interpretation_ids"] != expected_witnesses:
            issues.append(_issue(
                "ambiguity_witness_mismatch", f"$/ambiguity_taxonomy/entries/{index}/interpretation_ids",
                "ambiguity interpretation IDs must equal all witnesses of the listed graph classes",
            ))
        covered_class_ids.update(entry["graph_class_ids"])
        for ref in entry["evidence_region_refs"]:
            interpretation_id = ref["interpretation_id"]
            collection = object_collections[ref["object_kind"]]
            if interpretation_id not in interpretation_ids or ref["object_id"] >= len(interpretations[interpretation_id][collection]):
                issues.append(_issue("ambiguity_region_invalid", f"$/ambiguity_taxonomy/entries/{index}", "ambiguity evidence region reference is unknown"))
        if set(ref["interpretation_id"] for ref in entry["evidence_region_refs"]) != set(entry["interpretation_ids"]):
            issues.append(_issue(
                "ambiguity_evidence_witness_mismatch", f"$/ambiguity_taxonomy/entries/{index}/evidence_region_refs",
                "every ambiguity witness interpretation must contribute evidence",
            ))
    if len(expected_classes) > 1 and covered_class_ids != class_ids:
        issues.append(_issue("ambiguity_classes_incomplete", "$/ambiguity_taxonomy/entries", "ambiguity taxonomy must cover every graph class"))

    enumeration = document["enumeration"]
    matching_by_index: dict[int, dict[str, Any]] = {}
    ownership_by_index: dict[int, list[dict[str, Any]]] = {}
    relation_records = []
    used_compatible_pair_keys: set[tuple[int, ...]] = set()
    for interpretation in interpretations:
        matching = interpretation["boundary_matching"]
        index = matching["matching_index"]
        previous = matching_by_index.get(index)
        if previous is not None and previous != matching:
            issues.append(_issue("matching_index_conflict", "$/interpretations", "one matching index names different matchings"))
        matching_by_index[index] = matching
        for pair_id in matching["compatible_pair_ids"]:
            if pair_id in compatible_pairs:
                used_compatible_pair_keys.add(tuple(compatible_pairs[pair_id]["boundary_ids"]))
        ownership_index = interpretation["ownership_hypothesis_index"]
        previous_ownership = ownership_by_index.get(ownership_index)
        if previous_ownership is not None and previous_ownership != interpretation["ownership"]:
            issues.append(_issue("ownership_index_conflict", "$/interpretations", "one ownership index names different ownership arrays"))
        ownership_by_index[ownership_index] = interpretation["ownership"]
        relation_records.append({
            "interpretation_id": interpretation["id"],
            "relations": interpretation["endpoint_atom_relations"],
        })
    if sorted(matching_by_index) != list(range(len(matching_by_index))):
        issues.append(_issue("matching_indices_not_contiguous", "$/interpretations", "matching indices must be contiguous"))
    if sorted(ownership_by_index) != list(range(len(ownership_by_index))):
        issues.append(_issue("ownership_indices_not_contiguous", "$/interpretations", "ownership indices must be contiguous"))
    emitted_sha = framed_sha256(interpretations)
    actual_hashes = {
        "all_boundary_matchings_sha256": framed_sha256([matching_by_index[index] for index in sorted(matching_by_index)]),
        "all_endpoint_atom_relations_sha256": framed_sha256(relation_records),
        "all_interpretations_sha256": emitted_sha,
        "all_ownership_hypotheses_sha256": framed_sha256([ownership_by_index[index] for index in sorted(ownership_by_index)]),
        "candidate_universe_sha256": emitted_sha,
        "emitted_set_sha256": emitted_sha,
    }
    for key, expected in actual_hashes.items():
        if enumeration[key] != expected:
            issues.append(_issue("enumeration_hash_mismatch", f"$/enumeration/{key}", f"expected {expected}"))
    first_boundary_count = len(interpretations[0]["boundaries"]) if interpretations else 0
    actual_counts = {
        "atom_proposal_universe_count": len(proposal_records),
        "boundary_matching_count": len(matching_by_index),
        "compatible_boundary_pair_count": len(pair_records),
        "direct_pair_universe_count": len(direct_records),
        "endpoint_atom_relation_count": sum(len(item["endpoint_atom_relations"]) for item in interpretations),
        "interpretation_count": len(interpretations),
        "ownership_component_count": len({item["fragment_id"] for interpretation in interpretations for item in interpretation["ownership"]}),
        "ownership_hypothesis_count": len(ownership_by_index),
        "raw_boundary_count": len(boundary_records),
    }
    for key, expected in actual_counts.items():
        if enumeration[key] != expected:
            issues.append(_issue("enumeration_count_mismatch", f"$/enumeration/{key}", f"expected {expected}"))

    stage_trace = document["stage_trace"]
    stages = stage_trace["stages"]
    if [stage["index"] for stage in stages] != list(range(len(STAGE_NAMES))):
        issues.append(_issue(
            "stage_indices_invalid", "$/stage_trace/stages",
            "stage indices must be exactly 0 through 11",
        ))
    if [stage["stage"] for stage in stages] != list(STAGE_NAMES):
        issues.append(_issue(
            "stage_order_invalid", "$/stage_trace/stages",
            "stage names must follow the frozen 12-stage order",
        ))

    matching_records = [matching_by_index[index] for index in sorted(matching_by_index)]
    ownership_records = [ownership_by_index[index] for index in sorted(ownership_by_index)]
    stage_outputs = (
        framed_sha256([document["image"]["final_foreground_runs"]]),
        framed_sha256([
            document["image"]["bond_mask_runs"],
            document["image"]["text_mask_runs"],
        ]),
        framed_sha256(boundary_records),
        framed_sha256(pair_records),
        framed_sha256(matching_records),
        framed_sha256(proposal_records),
        framed_sha256(direct_records),
        framed_sha256(ownership_records),
        framed_sha256(relation_records),
        framed_sha256(interpretations),
        framed_sha256([graph_equivalence]),
        framed_sha256([taxonomy]),
    )
    stage_counts = (
        len(masks["final_foreground"]),
        len(masks["bond_mask"] | masks["text_mask"]),
        len(boundary_records),
        len(pair_records),
        len(matching_records),
        len(proposal_records),
        len(direct_records),
        len(ownership_records),
        sum(len(item["relations"]) for item in relation_records),
        len(interpretations),
        len(expected_classes),
        len(taxonomy["entries"]),
    )
    component_hashes = (
        provenance["preprocess"]["sha256"],
        provenance["semantic_ink_extractor"]["sha256"],
        provenance["boundary_extractor"]["sha256"],
        provenance["boundary_pairing"]["sha256"],
        provenance["enumerator_sha256"],
        provenance["atom_source_checkpoint_sha256"],
        provenance["direct_atom_pair_checkpoint_sha256"],
        provenance["enumerator_sha256"],
        provenance["candidate_predicate"]["sha256"],
        provenance["enumerator_sha256"],
        provenance["structural_verifier"]["sha256"],
        provenance["structural_verifier"]["sha256"],
    )
    previous_output = png_sha256
    for index, stage in enumerate(stages):
        if stage["input_sha256"] != previous_output:
            issues.append(_issue(
                "stage_input_chain_mismatch", f"$/stage_trace/stages/{index}/input_sha256",
                f"expected {previous_output}",
            ))
        if stage["output_sha256"] != stage_outputs[index]:
            issues.append(_issue(
                "stage_output_mismatch", f"$/stage_trace/stages/{index}/output_sha256",
                f"expected {stage_outputs[index]}",
            ))
        if stage["component_sha256"] != component_hashes[index]:
            issues.append(_issue(
                "stage_component_mismatch", f"$/stage_trace/stages/{index}/component_sha256",
                f"expected {component_hashes[index]}",
            ))
        if stage["record_count"] != stage_counts[index]:
            issues.append(_issue(
                "stage_record_count_mismatch", f"$/stage_trace/stages/{index}/record_count",
                f"expected {stage_counts[index]}",
            ))
        previous_output = stage["output_sha256"]
    if stage_trace["ordered_chain_sha256"] != _stage_chain_sha256(stages):
        issues.append(_issue(
            "stage_chain_digest_mismatch", "$/stage_trace/ordered_chain_sha256",
            "ordered stage trace digest does not match its framed records",
        ))
    return issues


def validate_candidate_set_bytes(candidate_raw: bytes, input_raw: bytes) -> dict[str, Any]:
    try:
        candidate = parse_canonical_json(candidate_raw)
        input_value = validated_input(input_raw)
        schema = load_frozen_schema("candidate_set.schema.json", CANDIDATE_SCHEMA_SHA256)
    except DocumentError as error:
        return error_report([error.issue])
    issues = schema_issues(candidate, schema)
    issues.extend(_forbidden_issues(candidate))
    if not issues and candidate["status"] == "candidate_set":
        issues.extend(_candidate_issues(
            candidate,
            input_value.foreground,
            hashlib.sha256(input_value.png_bytes).hexdigest(),
        ))
    if issues:
        return error_report(issues)
    report = success_report(
        "carrier_candidate_set_v0.3",
        candidate_sha256=hashlib.sha256(candidate_raw).hexdigest(),
        candidate_status=candidate["status"],
        graph_class_count=(candidate.get("graph_equivalence") or {}).get("class_count", 0),
        interpretation_count=len(candidate.get("interpretations", [])),
        schema_sha256=CANDIDATE_SCHEMA_SHA256,
        structural_scope="STRUCTURALLY_VALID_NOT_COMPLETE",
    )
    report["status"] = "STRUCTURALLY_VALID_NOT_COMPLETE"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = StructuredArgumentParser()
    parser.add_argument("candidate", help="canonical candidate-set JSON path")
    parser.add_argument("--input", required=True, help="paired canonical anonymous input")
    args = parser.parse_args(argv)
    try:
        candidate_raw = Path(args.candidate).read_bytes()
        input_raw = Path(args.input).read_bytes()
    except OSError as error:
        return emit_report(error_report([
            ValidationIssue("input_read_failed", "$", str(error)),
        ]), error=True)
    report = validate_candidate_set_bytes(candidate_raw, input_raw)
    return emit_report(report, error=report["status"] == "INVALID")


if __name__ == "__main__":
    raise SystemExit(main())
