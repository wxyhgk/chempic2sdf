"""Independent LegalR0 and factorized-CSP verifier.

This file deliberately does not import the R0 constructor, observation builder,
or their shared helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from formal.chemgraph.audit.fragment_root_constructor_v0_4.fragment_root import (
    compile_fragment_root,
)
from formal.chemgraph.audit.fragment_support_partition_v0_4.support_partition import (
    compile_support_partition,
)
from formal.chemgraph.challenge_v0_4.rasterizer_v0_4 import (
    Q,
    LineStyle,
    Segment,
    rasterize_gray,
    rasterize_support,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OBSERVATION_PATH = ROOT / "r0_observation.json"
CANDIDATE_PATH = ROOT / "r0_candidate_domain.json"
CONFIRMED_WITNESS_PATH = (
    REPOSITORY_ROOT
    / "formal/chemgraph/audit/png_ownership_realization_v0_5/representative_witness.json"
)
PNG_PATH = (
    REPOSITORY_ROOT
    / "formal/chemgraph/audit/png_ownership_realization_v0_5/representative_edge_w5_round_box4.png"
)
RENDERER_PATH = REPOSITORY_ROOT / "formal/chemgraph/challenge_v0_4/rasterizer_v0_4.py"
CONSTRUCTOR_PATH = ROOT / "candidate_domain.py"
PROTOCOL_COMMIT = "5a8bdd9"

LIMITS = {
    "max_adjacency_edges": 32_768,
    "max_ambiguous_variables": 20,
    "max_atomic_fragments": 64,
    "max_carriers": 8,
    "max_constraints": 32_768,
    "max_domain_classes": 64,
    "max_intersections": 8,
    "max_output_bytes": 16_777_216,
    "max_owner_options": 16,
    "max_pixel_variables": 4_096,
    "max_raw_assignments": 1_048_576,
    "max_total_rle_runs": 65_536,
}


class R0VerificationError(ValueError):
    """Closed independent R0 verification failure."""


def _fail(code: str) -> None:
    raise R0VerificationError(code)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encode_runs(pixels: Iterable[tuple[int, int]]) -> list[list[int]]:
    ordered = sorted(set(pixels))
    if not ordered:
        return []
    row, start = ordered[0]
    end = start + 1
    runs: list[list[int]] = []
    for next_row, column in ordered[1:]:
        if next_row == row and column == end:
            end += 1
            continue
        runs.append([row, start, end - start])
        row, start, end = next_row, column, column + 1
    runs.append([row, start, end - start])
    return runs


def _decode_runs(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        _fail("RLE_NOT_ARRAY")
    result: set[tuple[int, int]] = set()
    previous: tuple[int, int, int] | None = None
    for run in value:
        if not (
            isinstance(run, list)
            and len(run) == 3
            and all(type(item) is int for item in run)
        ):
            _fail("RLE_RUN_INVALID")
        row, start, length = run
        if not (0 <= row < 768 and 0 <= start < 768 and 1 <= length <= 768 - start):
            _fail("RLE_RUN_OUT_OF_RANGE")
        if previous is not None:
            old_row, old_start, old_length = previous
            if (row, start) <= (old_row, old_start):
                _fail("RLE_ORDER_NONCANONICAL")
            if row == old_row and start <= old_start + old_length:
                _fail("RLE_NOT_MAXIMAL")
        for column in range(start, start + length):
            pixel = row, column
            if pixel in result:
                _fail("RLE_PIXEL_DUPLICATE")
            result.add(pixel)
        previous = row, start, length
    return result


def _rle_sha256(runs: list[list[int]]) -> str:
    digest = hashlib.sha256()
    digest.update(len(runs).to_bytes(4, "big"))
    for row, start, length in runs:
        digest.update(row.to_bytes(4, "big"))
        digest.update(start.to_bytes(4, "big"))
        digest.update(length.to_bytes(4, "big"))
    return digest.hexdigest()


def _framed_items_sha256(items: list[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(len(items).to_bytes(4, "big"))
    for item in items:
        encoded = canonical_bytes(item)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> list[int]:
    value = Fraction(numerator, denominator)
    return [value.numerator, value.denominator]


def _load_canonical(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise R0VerificationError(f"JSON_INVALID:{path.name}") from error
    if raw != canonical_bytes(value):
        _fail(f"JSON_NONCANONICAL:{path.name}")
    return value


def _fixture() -> tuple[list[Segment], LineStyle, list[list[list[int]]]]:
    return [
        Segment("H", 10 * Q, 20 * Q, 30 * Q, 20 * Q),
        Segment("V1", 18 * Q, 10 * Q, 18 * Q, 20 * Q),
        Segment("V2", 22 * Q, 20 * Q, 22 * Q, 30 * Q),
    ], LineStyle(width_q16=5 * Q, cap="round", antialias_profile="box_4x4"), [
        [[160, 320], [480, 320]],
        [[288, 160], [288, 320]],
        [[352, 320], [352, 480]],
    ]


def _expected_observation() -> tuple[dict[str, Any], list[set[tuple[int, int]]]]:
    segments, style, q4_segments = _fixture()
    support_indexes = [rasterize_support([segment], style) for segment in segments]
    supports = [{divmod(pixel, 768) for pixel in support} for support in support_indexes]
    gray = rasterize_gray(segments, style)
    if set().union(*support_indexes) != {
        index for index, value in enumerate(gray) if value != 255
    }:
        _fail("RENDERER_SUPPORT_UNION_MISMATCH")
    geometry_root = {
        "base_strokes": [
            {"boundary_member_ids": [index], "id": index, "segment_q4": segment}
            for index, segment in enumerate(q4_segments)
        ],
        "schema_version": "chemgraph-fragment-root-v0.4-research",
        "training_authorized": False,
    }
    parent_supports = []
    for index, pixels in enumerate(supports):
        runs = _encode_runs(pixels)
        parent_supports.append({
            "base_stroke_id": index,
            "support_pixel_count": len(pixels),
            "support_rle_sha256": _rle_sha256(runs),
            "support_runs": runs,
        })
    support_input = {
        "geometry_root": geometry_root,
        "parent_supports": parent_supports,
        "schema_version": "chemgraph-fragment-support-partition-v0.4-research",
        "training_authorized": False,
    }
    observation = {
        "geometry_root": geometry_root,
        "parent_supports": parent_supports,
        "png_sha256": _sha256(PNG_PATH.read_bytes()),
        "renderer_sha256": _sha256(RENDERER_PATH.read_bytes()),
        "schema_version": "owner-candidate-r0-observation-v0.5-research",
        "support_input_sha256": _sha256(canonical_bytes(support_input)),
        "training_authorized": False,
    }
    return observation, supports


def _connected(variable_ids: set[int], adjacency: dict[int, set[int]]) -> bool:
    if not variable_ids:
        return False
    unseen = set(variable_ids)
    queue = deque([unseen.pop()])
    while queue:
        variable = queue.popleft()
        for neighbor in adjacency[variable]:
            if neighbor in unseen:
                unseen.remove(neighbor)
                queue.append(neighbor)
    return not unseen


def _legal_r0(
    assignment: tuple[int, ...], domains: list[tuple[int, ...]],
    owner_options: list[dict[str, Any]], evidence: list[dict[str, Any]],
    adjacency: dict[int, set[int]], pixel_to_variable: dict[tuple[int, int], int],
) -> bool:
    if len(assignment) != len(domains):
        return False
    for variable_id, owner in enumerate(assignment):
        if owner not in domains[variable_id]:
            return False
        incident = set(owner_options[owner]["incident_carrier_ids"])
        if not incident.issubset(evidence[variable_id]["carrier_ids"]):
            return False
    for option in owner_options:
        owner = option["id"]
        fiber = {index for index, value in enumerate(assignment) if value == owner}
        seed = pixel_to_variable.get(tuple(option["anchor_pixel"]))
        if seed is None or seed not in fiber or not _connected(fiber, adjacency):
            return False
    return True


def _candidate_feasible(
    assignment: tuple[int, ...], candidate: dict[str, Any],
    domains: list[tuple[int, ...]], adjacency: dict[int, set[int]],
) -> bool:
    if any(owner not in domains[index] for index, owner in enumerate(assignment)):
        return False
    for constraint in candidate["constraints"]:
        kind = constraint["kind"]
        if kind == "exact_one":
            if constraint["variable_ids"] != list(range(len(assignment))):
                return False
        elif kind == "seed_owner":
            if assignment[constraint["variable_id"]] != constraint["owner_option_id"]:
                return False
        elif kind == "option_incidence":
            eligible = set(constraint["eligible_variable_ids"])
            owner = constraint["owner_option_id"]
            if any(value == owner and index not in eligible for index, value in enumerate(assignment)):
                return False
        elif kind == "connected_to_seed":
            owner = constraint["owner_option_id"]
            fiber = {index for index, value in enumerate(assignment) if value == owner}
            if constraint["seed_variable_id"] not in fiber or not _connected(fiber, adjacency):
                return False
        else:
            return False
    return True


def _stored_annotation_assignments(
    pixel_to_variable: dict[tuple[int, int], int], variable_count: int,
) -> list[tuple[int, ...]]:
    witness = _load_canonical(CONFIRMED_WITNESS_PATH)
    annotations = witness["configuration"]["partition_annotations"]
    assignments = []
    for annotation in annotations:
        assignment: list[int | None] = [None] * variable_count
        for owner, key in ((0, "left_region_runs"), (1, "right_region_runs")):
            for pixel in _decode_runs(annotation[key]):
                variable_id = pixel_to_variable.get(pixel)
                if variable_id is None or assignment[variable_id] is not None:
                    _fail("STORED_ANNOTATION_INVALID")
                assignment[variable_id] = owner
        if any(value is None for value in assignment):
            _fail("STORED_ANNOTATION_INCOMPLETE")
        assignments.append(tuple(value for value in assignment if value is not None))
    return assignments


def verify_payload(observation: Any, candidate: Any) -> dict[str, Any]:
    expected_observation, parent_supports = _expected_observation()
    if observation != expected_observation:
        _fail("OBSERVATION_REBUILD_MISMATCH")
    if not isinstance(candidate, dict) or set(candidate) != {
        "adjacency_edges", "constraints", "counts", "domain_classes",
        "evidence_memberships", "limits", "metrics", "owner_options",
        "pixel_variables", "residual_reasons", "schema_version", "section_hashes",
        "source_binding", "stage_trace", "status", "training_authorized"
    }:
        _fail("CANDIDATE_NOT_CLOSED")
    if candidate["schema_version"] != "chemgraph-owner-candidate-domain-v0.5-r0-research":
        _fail("CANDIDATE_SCHEMA_MISMATCH")
    if candidate["status"] != "candidate_domain" or candidate["training_authorized"] is not False:
        _fail("CANDIDATE_STATUS_INVALID")
    if candidate["limits"] != LIMITS:
        _fail("LIMITS_MISMATCH")
    if candidate["residual_reasons"] != []:
        _fail("R0_RESIDUAL_FORBIDDEN")

    support_input = {
        "geometry_root": observation["geometry_root"],
        "parent_supports": observation["parent_supports"],
        "schema_version": "chemgraph-fragment-support-partition-v0.4-research",
        "training_authorized": False,
    }
    geometry = compile_fragment_root(observation["geometry_root"])
    support = compile_support_partition(support_input)
    expected_source_binding = {
        "compiled_fragment_root_sha256": _sha256(canonical_bytes(geometry)),
        "compiled_support_partition_sha256": _sha256(canonical_bytes(support)),
        "constructor_protocol_commit": PROTOCOL_COMMIT,
        "geometry_root_sha256": geometry["source_root_sha256"],
        "observation_sha256": _sha256(canonical_bytes(observation)),
        "png_sha256": observation["png_sha256"],
        "renderer_sha256": observation["renderer_sha256"],
        "support_input_sha256": observation["support_input_sha256"],
    }
    if candidate["source_binding"] != expected_source_binding:
        _fail("SOURCE_BINDING_MISMATCH")

    expected_options = [
        {
            "anchor_pixel": [
                Fraction(item["point_q4_exact"][1][0], item["point_q4_exact"][1][1]).numerator
                // (16 * Fraction(item["point_q4_exact"][1][0], item["point_q4_exact"][1][1]).denominator),
                Fraction(item["point_q4_exact"][0][0], item["point_q4_exact"][0][1]).numerator
                // (16 * Fraction(item["point_q4_exact"][0][0], item["point_q4_exact"][0][1]).denominator),
            ],
            "geometry_intersection_id": item["id"],
            "id": item["id"],
            "incident_carrier_ids": item["base_stroke_ids"],
            "kind": "intersection",
        }
        for item in geometry["intersections"]
    ]
    if candidate["owner_options"] != expected_options:
        _fail("OWNER_OPTIONS_MISMATCH")

    child_pixels = {
        child["atomic_fragment_id"]: _decode_runs(child["support_runs"])
        for child in support["child_supports"]
    }
    child_carrier = {
        child["atomic_fragment_id"]: child["base_stroke_id"]
        for child in support["child_supports"]
    }
    overlaps = support["cross_carrier_overlaps"]
    expected_evidence = []
    domains: list[tuple[int, ...]] = []
    for index, record in enumerate(overlaps):
        pixel = tuple(record["pixel"])
        fragments = []
        for carrier in record["base_stroke_ids"]:
            matches = sorted(
                fragment for fragment, pixels in child_pixels.items()
                if child_carrier[fragment] == carrier and pixel in pixels
            )
            if len(matches) != 1:
                _fail("FRAGMENT_MEMBERSHIP_NOT_UNIQUE")
            fragments.append({"atomic_fragment_id": matches[0], "carrier_id": carrier})
        expected_evidence.append({
            "carrier_ids": record["base_stroke_ids"],
            "fragment_memberships": fragments,
            "id": index,
            "pixel": record["pixel"],
        })
        domains.append(tuple(
            option["id"] for option in expected_options
            if set(option["incident_carrier_ids"]).issubset(record["base_stroke_ids"])
        ))
    if candidate["evidence_memberships"] != expected_evidence:
        _fail("EVIDENCE_MEMBERSHIP_MISMATCH")
    if any(not domain for domain in domains):
        _fail("UNOWNED_OVERLAP")

    unique_domains = sorted(set(domains))
    expected_domain_classes = [
        {"id": index, "owner_option_ids": list(domain)}
        for index, domain in enumerate(unique_domains)
    ]
    if candidate["domain_classes"] != expected_domain_classes:
        _fail("DOMAIN_CLASSES_MISMATCH")
    class_by_domain = {domain: index for index, domain in enumerate(unique_domains)}
    expected_variables = [
        {
            "domain_class_id": class_by_domain[domains[index]],
            "evidence_membership_id": index,
            "id": index,
            "pixel": record["pixel"],
        }
        for index, record in enumerate(overlaps)
    ]
    if candidate["pixel_variables"] != expected_variables:
        _fail("PIXEL_VARIABLES_MISMATCH")

    pixel_to_variable = {
        tuple(record["pixel"]): index for index, record in enumerate(overlaps)
    }
    expected_edges = []
    for left_id, record in enumerate(overlaps):
        row, column = record["pixel"]
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                right_id = pixel_to_variable.get((row + row_delta, column + column_delta))
                if right_id is not None and left_id < right_id:
                    expected_edges.append({
                        "id": len(expected_edges),
                        "left_variable_id": left_id,
                        "right_variable_id": right_id,
                    })
    if candidate["adjacency_edges"] != expected_edges:
        _fail("ADJACENCY_EDGES_MISMATCH")
    adjacency = {index: set() for index in range(len(overlaps))}
    for edge in expected_edges:
        left, right = edge["left_variable_id"], edge["right_variable_id"]
        adjacency[left].add(right)
        adjacency[right].add(left)

    expected_constraints: list[dict[str, Any]] = [{
        "id": 0, "kind": "exact_one", "variable_ids": list(range(len(overlaps)))
    }]
    for option in expected_options:
        seed = pixel_to_variable[tuple(option["anchor_pixel"])]
        eligible = [index for index, domain in enumerate(domains) if option["id"] in domain]
        eligible_set = set(eligible)
        edge_ids = [
            edge["id"] for edge in expected_edges
            if edge["left_variable_id"] in eligible_set
            and edge["right_variable_id"] in eligible_set
        ]
        expected_constraints.extend([
            {"id": len(expected_constraints), "kind": "seed_owner",
             "owner_option_id": option["id"], "variable_id": seed},
            {"adjacency_edge_ids": edge_ids, "eligible_variable_ids": eligible,
             "id": len(expected_constraints) + 1, "kind": "connected_to_seed",
             "owner_option_id": option["id"], "seed_variable_id": seed},
            {"eligible_variable_ids": eligible, "id": len(expected_constraints) + 2,
             "kind": "option_incidence", "owner_option_id": option["id"]},
        ])
    if candidate["constraints"] != expected_constraints:
        _fail("CONSTRAINTS_MISMATCH")

    ambiguous_count = sum(len(domain) > 1 for domain in domains)
    raw_count = 1
    for domain in domains:
        raw_count *= len(domain)
    universal = len(domains) * len(expected_options)
    candidate_memberships = sum(len(domain) for domain in domains)
    eliminated = universal - candidate_memberships
    eligible_option_count = sum(
        set(option["incident_carrier_ids"]).issubset(evidence["carrier_ids"])
        for evidence in expected_evidence
        for option in expected_options
    )
    retained_eligible_count = candidate_memberships
    ineligible_option_count = universal - eligible_option_count
    eliminated_ineligible_count = eliminated
    expected_metrics = {
        "baseline_id": "all_and_only_known_intersections_per_variable-v0.5-r0",
        "candidate_option_membership_count": candidate_memberships,
        "domain_size_histogram": [
            {"domain_size": size, "variable_count": sum(len(domain) == size for domain in domains)}
            for size in sorted({len(domain) for domain in domains})
        ],
        "eligible_option_count": eligible_option_count,
        "eligible_option_recall": _ratio(retained_eligible_count, eligible_option_count),
        "eliminated_option_membership_count": eliminated,
        "eliminated_option_ratio": _ratio(eliminated, universal),
        "eliminated_ineligible_option_count": eliminated_ineligible_count,
        "ineligible_option_count": ineligible_option_count,
        "ineligible_option_elimination_recall": _ratio(
            eliminated_ineligible_count, ineligible_option_count
        ),
        "max_domain_size": max(len(domain) for domain in domains),
        "mean_domain_size": _ratio(candidate_memberships, len(domains)),
        "multi_intersection_ratio": _ratio(ambiguous_count, len(domains)),
        "raw_cartesian_assignment_count": raw_count,
        "retained_eligible_option_count": retained_eligible_count,
        "residual_containing_ratio": [0, 1],
        "residual_only_ratio": [0, 1],
        "singleton_ratio": _ratio(len(domains) - ambiguous_count, len(domains)),
        "universal_option_membership_count": universal,
    }
    if candidate["metrics"] != expected_metrics:
        _fail("METRICS_MISMATCH")
    expected_counts = {
        "adjacency_edge_count": len(expected_edges),
        "ambiguous_variable_count": ambiguous_count,
        "atomic_fragment_count": geometry["atomic_fragment_count"],
        "carrier_count": geometry["base_stroke_count"],
        "constraint_count": len(expected_constraints),
        "domain_class_count": len(unique_domains),
        "evidence_membership_count": len(expected_evidence),
        "owner_option_count": len(expected_options),
        "pixel_variable_count": len(expected_variables),
        "residual_reason_count": 0,
    }
    if candidate["counts"] != expected_counts:
        _fail("COUNTS_MISMATCH")

    section_items = {
        "source_binding": [candidate["source_binding"]],
        "limits": [candidate["limits"]],
        "counts": [candidate["counts"]],
        "owner_options": candidate["owner_options"],
        "evidence_memberships": candidate["evidence_memberships"],
        "domain_classes": candidate["domain_classes"],
        "pixel_variables": candidate["pixel_variables"],
        "adjacency_edges": candidate["adjacency_edges"],
        "constraints": candidate["constraints"],
        "residual_reasons": candidate["residual_reasons"],
        "metrics": [candidate["metrics"]],
    }
    expected_section_hashes = [
        {"item_count": len(items), "section": name, "sha256": _framed_items_sha256(items)}
        for name, items in section_items.items()
    ]
    if candidate["section_hashes"] != expected_section_hashes:
        _fail("SECTION_HASHES_MISMATCH")
    section_hash = {item["section"]: item["sha256"] for item in expected_section_hashes}
    constructor_sha = _sha256(CONSTRUCTOR_PATH.read_bytes())
    stage_names = [
        "source_binding", "owner_options", "evidence_memberships", "domain_classes",
        "pixel_variables", "adjacency_edges", "constraints", "metrics",
    ]
    previous = expected_source_binding["observation_sha256"]
    expected_trace = []
    for index, name in enumerate(stage_names):
        output = section_hash[name]
        expected_trace.append({
            "component_sha256": constructor_sha,
            "index": index,
            "input_sha256": previous,
            "output_sha256": output,
            "record_count": len(section_items[name]),
            "stage": name,
        })
        previous = output
    if candidate["stage_trace"] != expected_trace:
        _fail("STAGE_TRACE_MISMATCH")

    pointwise_assignments = list(itertools.product(*domains))
    if len(pointwise_assignments) != raw_count or len(set(pointwise_assignments)) != raw_count:
        _fail("POINTWISE_ENUMERATION_INVALID")
    legal = {
        assignment for assignment in pointwise_assignments
        if _legal_r0(
            assignment, domains, expected_options, expected_evidence,
            adjacency, pixel_to_variable,
        )
    }
    feasible_list = [
        assignment for assignment in pointwise_assignments
        if _candidate_feasible(assignment, candidate, domains, adjacency)
    ]
    feasible = set(feasible_list)
    if len(feasible) != len(feasible_list):
        _fail("CANDIDATE_EXPANSION_DUPLICATE")
    if feasible != legal:
        _fail("CANDIDATE_LEGAL_SET_MISMATCH")
    if len(legal) != 873:
        _fail("LEGAL_R0_TRUTH_TABLE_MISMATCH")

    stored = _stored_annotation_assignments(pixel_to_variable, len(domains))
    if not all(assignment in feasible for assignment in stored):
        _fail("STORED_ANNOTATION_NOT_COVERED")
    realized_by_variable = [
        {assignment[index] for assignment in legal}
        for index in range(len(domains))
    ]
    false_singletons = sum(
        len(domain) == 1 and realized != set(domain)
        for domain, realized in zip(domains, realized_by_variable, strict=True)
    )
    false_multi = sum(
        len(domain) > 1 and realized != set(domain)
        for domain, realized in zip(domains, realized_by_variable, strict=True)
    )
    if false_singletons or false_multi:
        _fail("DOMAIN_REALIZABILITY_MISMATCH")

    return {
        "artifact_sha256": _sha256(canonical_bytes(candidate)),
        "candidate_assignment_count": len(feasible),
        "candidate_nodup": True,
        "complete_count": len(legal & feasible),
        "false_multi_option_count": false_multi,
        "false_singleton_count": false_singletons,
        "legal_assignment_count": len(legal),
        "observation_sha256": _sha256(canonical_bytes(observation)),
        "pointwise_assignment_count": len(pointwise_assignments),
        "protocol_commit": PROTOCOL_COMMIT,
        "schema_version": "owner-candidate-domain-v0.5-r0-verification",
        "sound_count": len(feasible & legal),
        "status": "passed",
        "stored_annotation_coverage": [sum(item in feasible for item in stored), len(stored)],
        "training_authorized": False,
        "whole_world_exact": True,
    }


def verify_frozen_r0() -> dict[str, Any]:
    observation = _load_canonical(OBSERVATION_PATH)
    candidate = _load_canonical(CANDIDATE_PATH)
    return verify_payload(observation, candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="freeze the canonical result to r0_verification.json",
    )
    args = parser.parse_args()
    result = canonical_bytes(verify_frozen_r0())
    if args.write:
        ROOT.joinpath("r0_verification.json").write_bytes(result)
    print(result.decode("ascii"), end="")


if __name__ == "__main__":
    main()
