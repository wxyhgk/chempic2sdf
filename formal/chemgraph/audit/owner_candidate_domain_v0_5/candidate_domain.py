"""Construct the factorized known-intersection R0 candidate domain."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    compile_fragment_root,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    compile_support_partition,
)

from .build_observation import support_input_from_observation
from .common import (
    LIMITS,
    CandidateDomainError,
    canonical_bytes,
    capacity_gate,
    decode_runs,
    fail,
    framed_items_sha256,
    ratio,
    sha256,
)


CONSTRUCTOR_PATH = Path(__file__).resolve()
SCHEMA_VERSION = "chemgraph-owner-candidate-domain-v0.5-r0-research"
PROTOCOL_COMMIT = "5a8bdd9"


def _canonicalize_support_input(support_input: Any) -> dict[str, Any]:
    geometry_root = support_input.get("geometry_root") if isinstance(support_input, dict) else None
    parent_supports = support_input.get("parent_supports") if isinstance(support_input, dict) else None
    if not isinstance(geometry_root, dict) or set(geometry_root) != {
        "base_strokes", "schema_version", "training_authorized"
    }:
        fail("GEOMETRY_ROOT_NOT_CLOSED")
    if geometry_root["training_authorized"] is not False:
        fail("TRAINING_AUTHORIZED")
    strokes = geometry_root["base_strokes"]
    if not isinstance(strokes, list) or not strokes or not isinstance(parent_supports, list):
        fail("SOURCE_FIBERS_INVALID")
    support_by_old_id = {}
    for support in parent_supports:
        if not isinstance(support, dict) or set(support) != {
            "base_stroke_id", "support_pixel_count", "support_rle_sha256", "support_runs"
        }:
            fail("PARENT_SUPPORT_NOT_CLOSED")
        old_id = support["base_stroke_id"]
        if type(old_id) is not int or old_id in support_by_old_id:
            fail("PARENT_SUPPORT_ID_INVALID")
        support_by_old_id[old_id] = support

    normalized_records = []
    seen_old_ids = set()
    seen_boundary_member_ids = set()
    for stroke in strokes:
        if not isinstance(stroke, dict) or set(stroke) != {
            "boundary_member_ids", "id", "segment_q4"
        }:
            fail("BASE_STROKE_NOT_CLOSED")
        old_id = stroke["id"]
        members = stroke["boundary_member_ids"]
        segment = stroke["segment_q4"]
        if type(old_id) is not int or old_id in seen_old_ids:
            fail("BASE_STROKE_ID_INVALID")
        if not isinstance(members, list) or len(members) != 1 or type(members[0]) is not int:
            fail("R0_BOUNDARY_MEMBERS_INVALID")
        if members[0] in seen_boundary_member_ids:
            fail("R0_BOUNDARY_MEMBER_ID_NOT_UNIQUE")
        if not (
            isinstance(segment, list) and len(segment) == 2
            and all(
                isinstance(point, list) and len(point) == 2
                and all(type(value) is int for value in point)
                for point in segment
            )
        ):
            fail("SEGMENT_INVALID")
        if old_id not in support_by_old_id:
            fail("PARENT_SUPPORT_FIBER_INCOMPLETE")
        begin, end = sorted(segment)
        if begin == end:
            fail("ZERO_LENGTH_SEGMENT")
        seen_old_ids.add(old_id)
        seen_boundary_member_ids.add(members[0])
        normalized_records.append((begin, end, old_id))
    if seen_old_ids != set(support_by_old_id):
        fail("PARENT_SUPPORT_FIBER_INCOMPLETE")
    normalized_records.sort()
    if len({(tuple(begin), tuple(end)) for begin, end, _ in normalized_records}) != len(normalized_records):
        fail("DUPLICATE_SEGMENT")

    normalized_strokes = []
    normalized_supports = []
    for new_id, (begin, end, old_id) in enumerate(normalized_records):
        old_support = support_by_old_id[old_id]
        normalized_strokes.append({
            "boundary_member_ids": [new_id],
            "id": new_id,
            "segment_q4": [begin, end],
        })
        normalized_supports.append({
            "base_stroke_id": new_id,
            "support_pixel_count": old_support["support_pixel_count"],
            "support_rle_sha256": old_support["support_rle_sha256"],
            "support_runs": old_support["support_runs"],
        })
    return {
        "geometry_root": {
            "base_strokes": normalized_strokes,
            "schema_version": geometry_root["schema_version"],
            "training_authorized": False,
        },
        "parent_supports": normalized_supports,
        "schema_version": support_input["schema_version"],
        "training_authorized": False,
    }


def _anchor_pixel(point_q4_exact: Any) -> list[int]:
    if not (
        isinstance(point_q4_exact, list)
        and len(point_q4_exact) == 2
        and all(
            isinstance(axis, list)
            and len(axis) == 2
            and all(type(value) is int for value in axis)
            and axis[1] > 0
            for axis in point_q4_exact
        )
    ):
        fail("INTERSECTION_POINT_INVALID")
    x = Fraction(point_q4_exact[0][0], point_q4_exact[0][1])
    y = Fraction(point_q4_exact[1][0], point_q4_exact[1][1])
    row, column = y.numerator // (16 * y.denominator), x.numerator // (16 * x.denominator)
    if not (0 <= row < 768 and 0 <= column < 768):
        fail("INTERSECTION_ANCHOR_OUTSIDE_RASTER")
    return [row, column]


def _section_hashes(sections: list[tuple[str, list[Any]]]) -> list[dict[str, Any]]:
    return [
        {"item_count": len(items), "section": name, "sha256": framed_items_sha256(items)}
        for name, items in sections
    ]


def build_candidate_domain(observation: Any) -> dict[str, Any]:
    raw_support_input = support_input_from_observation(observation)
    support_input = _canonicalize_support_input(raw_support_input)
    observation_sha = sha256(canonical_bytes(observation))
    geometry = compile_fragment_root(support_input["geometry_root"])
    support = compile_support_partition(support_input)

    capacity_gate("max_carriers", geometry["base_stroke_count"])
    capacity_gate("max_intersections", geometry["intersection_count"])
    capacity_gate("max_atomic_fragments", geometry["atomic_fragment_count"])
    capacity_gate("max_total_rle_runs", support["child_support_run_count_total"])
    if geometry["intersection_count"] != 2:
        fail("R0_REQUIRES_TWO_INTERSECTIONS")

    source_binding = {
        "compiled_fragment_root_sha256": sha256(canonical_bytes(geometry)),
        "compiled_support_partition_sha256": sha256(canonical_bytes(support)),
        "constructor_protocol_commit": PROTOCOL_COMMIT,
        "geometry_root_sha256": geometry["source_root_sha256"],
        "observation_sha256": observation_sha,
        "png_sha256": observation["png_sha256"],
        "renderer_sha256": observation["renderer_sha256"],
        "support_input_sha256": observation["support_input_sha256"],
    }

    owner_options = []
    for intersection in geometry["intersections"]:
        owner_options.append({
            "anchor_pixel": _anchor_pixel(intersection["point_q4_exact"]),
            "geometry_intersection_id": intersection["id"],
            "id": len(owner_options),
            "incident_carrier_ids": intersection["base_stroke_ids"],
            "kind": "intersection",
        })
    capacity_gate("max_owner_options", len(owner_options))

    child_support_pixels: dict[int, set[tuple[int, int]]] = {}
    child_carriers: dict[int, int] = {}
    for child in support["child_supports"]:
        fragment_id = child["atomic_fragment_id"]
        child_support_pixels[fragment_id] = decode_runs(child["support_runs"])
        child_carriers[fragment_id] = child["base_stroke_id"]

    overlap_records = support["cross_carrier_overlaps"]
    capacity_gate("max_pixel_variables", len(overlap_records))
    overlap_pixels = [tuple(record["pixel"]) for record in overlap_records]
    pixel_to_variable = {pixel: index for index, pixel in enumerate(overlap_pixels)}

    evidence_memberships: list[dict[str, Any]] = []
    domains: list[tuple[int, ...]] = []
    for variable_id, record in enumerate(overlap_records):
        pixel = tuple(record["pixel"])
        carriers = record["base_stroke_ids"]
        fragment_memberships = []
        for carrier_id in carriers:
            matches = sorted(
                fragment_id for fragment_id, pixels in child_support_pixels.items()
                if child_carriers[fragment_id] == carrier_id and pixel in pixels
            )
            if len(matches) != 1:
                fail("INCIDENCE_FRAGMENT_FIBER_INCOMPLETE")
            fragment_memberships.append({
                "atomic_fragment_id": matches[0],
                "carrier_id": carrier_id,
            })
        evidence_memberships.append({
            "carrier_ids": carriers,
            "fragment_memberships": fragment_memberships,
            "id": variable_id,
            "pixel": list(pixel),
        })
        domain = tuple(
            option["id"] for option in owner_options
            if set(option["incident_carrier_ids"]).issubset(carriers)
        )
        if not domain:
            fail("UNOWNED_OVERLAP")
        domains.append(domain)

    unique_domains = sorted(set(domains))
    capacity_gate("max_domain_classes", len(unique_domains))
    domain_classes = [
        {"id": index, "owner_option_ids": list(domain)}
        for index, domain in enumerate(unique_domains)
    ]
    domain_to_class = {domain: index for index, domain in enumerate(unique_domains)}
    pixel_variables = [
        {
            "domain_class_id": domain_to_class[domains[index]],
            "evidence_membership_id": index,
            "id": index,
            "pixel": list(pixel),
        }
        for index, pixel in enumerate(overlap_pixels)
    ]

    adjacency_edges = []
    for left_id, (left_row, left_column) in enumerate(overlap_pixels):
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                right = left_row + row_delta, left_column + column_delta
                right_id = pixel_to_variable.get(right)
                if right_id is not None and left_id < right_id:
                    adjacency_edges.append({
                        "id": len(adjacency_edges),
                        "left_variable_id": left_id,
                        "right_variable_id": right_id,
                    })
    capacity_gate("max_adjacency_edges", len(adjacency_edges))

    constraints: list[dict[str, Any]] = [{
        "id": 0,
        "kind": "exact_one",
        "variable_ids": list(range(len(pixel_variables))),
    }]
    for option in owner_options:
        anchor = tuple(option["anchor_pixel"])
        seed_variable_id = pixel_to_variable.get(anchor)
        if seed_variable_id is None or option["id"] not in domains[seed_variable_id]:
            fail("SEED_OWNER_UNAVAILABLE")
        eligible_variable_ids = [
            index for index, domain in enumerate(domains) if option["id"] in domain
        ]
        eligible_set = set(eligible_variable_ids)
        eligible_edge_ids = [
            edge["id"] for edge in adjacency_edges
            if edge["left_variable_id"] in eligible_set
            and edge["right_variable_id"] in eligible_set
        ]
        constraints.extend([
            {
                "id": len(constraints),
                "kind": "seed_owner",
                "owner_option_id": option["id"],
                "variable_id": seed_variable_id,
            },
            {
                "adjacency_edge_ids": eligible_edge_ids,
                "eligible_variable_ids": eligible_variable_ids,
                "id": len(constraints) + 1,
                "kind": "connected_to_seed",
                "owner_option_id": option["id"],
                "seed_variable_id": seed_variable_id,
            },
            {
                "eligible_variable_ids": eligible_variable_ids,
                "id": len(constraints) + 2,
                "kind": "option_incidence",
                "owner_option_id": option["id"],
            },
        ])
    capacity_gate("max_constraints", len(constraints))

    ambiguous_count = sum(len(domain) > 1 for domain in domains)
    capacity_gate("max_ambiguous_variables", ambiguous_count)
    raw_assignment_count = 1
    for domain in domains:
        raw_assignment_count *= len(domain)
        capacity_gate("max_raw_assignments", raw_assignment_count)
    universal_memberships = len(pixel_variables) * len(owner_options)
    candidate_memberships = sum(len(domain) for domain in domains)
    eliminated = universal_memberships - candidate_memberships
    eligible_option_count = sum(
        set(option["incident_carrier_ids"]).issubset(evidence["carrier_ids"])
        for evidence in evidence_memberships
        for option in owner_options
    )
    retained_eligible_count = candidate_memberships
    ineligible_option_count = universal_memberships - eligible_option_count
    eliminated_ineligible_count = eliminated
    histogram = [
        {"domain_size": size, "variable_count": sum(len(domain) == size for domain in domains)}
        for size in sorted({len(domain) for domain in domains})
    ]
    metrics = {
        "baseline_id": "all_and_only_known_intersections_per_variable-v0.5-r0",
        "candidate_option_membership_count": candidate_memberships,
        "domain_size_histogram": histogram,
        "eligible_option_count": eligible_option_count,
        "eligible_option_recall": ratio(retained_eligible_count, eligible_option_count),
        "eliminated_option_membership_count": eliminated,
        "eliminated_option_ratio": ratio(eliminated, universal_memberships),
        "eliminated_ineligible_option_count": eliminated_ineligible_count,
        "ineligible_option_count": ineligible_option_count,
        "ineligible_option_elimination_recall": ratio(
            eliminated_ineligible_count, ineligible_option_count
        ),
        "max_domain_size": max(len(domain) for domain in domains),
        "mean_domain_size": ratio(candidate_memberships, len(domains)),
        "multi_intersection_ratio": ratio(ambiguous_count, len(domains)),
        "raw_cartesian_assignment_count": raw_assignment_count,
        "retained_eligible_option_count": retained_eligible_count,
        "residual_containing_ratio": [0, 1],
        "residual_only_ratio": [0, 1],
        "singleton_ratio": ratio(len(domains) - ambiguous_count, len(domains)),
        "universal_option_membership_count": universal_memberships,
    }

    counts = {
        "adjacency_edge_count": len(adjacency_edges),
        "ambiguous_variable_count": ambiguous_count,
        "atomic_fragment_count": geometry["atomic_fragment_count"],
        "carrier_count": geometry["base_stroke_count"],
        "constraint_count": len(constraints),
        "domain_class_count": len(domain_classes),
        "evidence_membership_count": len(evidence_memberships),
        "owner_option_count": len(owner_options),
        "pixel_variable_count": len(pixel_variables),
        "residual_reason_count": 0,
    }
    residual_reasons: list[Any] = []
    sections = [
        ("source_binding", [source_binding]),
        ("limits", [LIMITS]),
        ("counts", [counts]),
        ("owner_options", owner_options),
        ("evidence_memberships", evidence_memberships),
        ("domain_classes", domain_classes),
        ("pixel_variables", pixel_variables),
        ("adjacency_edges", adjacency_edges),
        ("constraints", constraints),
        ("residual_reasons", residual_reasons),
        ("metrics", [metrics]),
    ]
    section_hashes = _section_hashes(sections)
    section_digest = {item["section"]: item["sha256"] for item in section_hashes}
    constructor_sha = sha256(CONSTRUCTOR_PATH.read_bytes())
    stage_names = [
        "source_binding", "owner_options", "evidence_memberships", "domain_classes",
        "pixel_variables", "adjacency_edges", "constraints", "metrics",
    ]
    previous = observation_sha
    stage_trace = []
    section_items = dict(sections)
    for index, name in enumerate(stage_names):
        output = section_digest[name]
        stage_trace.append({
            "component_sha256": constructor_sha,
            "index": index,
            "input_sha256": previous,
            "output_sha256": output,
            "record_count": len(section_items[name]),
            "stage": name,
        })
        previous = output

    artifact = {
        "adjacency_edges": adjacency_edges,
        "constraints": constraints,
        "counts": counts,
        "domain_classes": domain_classes,
        "evidence_memberships": evidence_memberships,
        "limits": LIMITS,
        "metrics": metrics,
        "owner_options": owner_options,
        "pixel_variables": pixel_variables,
        "residual_reasons": residual_reasons,
        "schema_version": SCHEMA_VERSION,
        "section_hashes": section_hashes,
        "source_binding": source_binding,
        "stage_trace": stage_trace,
        "status": "candidate_domain",
        "training_authorized": False,
    }
    encoded = canonical_bytes(artifact)
    capacity_gate("max_output_bytes", len(encoded))
    return artifact


def ensure_constructor_has_no_oracle_literal() -> None:
    source = CONSTRUCTOR_PATH.read_text(encoding="utf-8")
    forbidden = (
        "partition_" + "annotations",
        "locally_admissible_" + "partition_count",
    )
    if any(token in source for token in forbidden):
        raise CandidateDomainError("CONSTRUCTOR_ORACLE_LITERAL_FORBIDDEN")
