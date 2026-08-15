#!/usr/bin/env python3
"""Independently enumerate v0.3 boundary matchings without trusting self-signed flags."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from formal.chemgraph.audit.carrier_candidate_set_v0_3._common import (
    StructuredArgumentParser,
    canonical_json_bytes,
    emit_report,
    framed_sha256,
    parse_canonical_json,
)
from formal.chemgraph.audit.carrier_candidate_set_v0_3.validate_candidate_set import (
    validate_candidate_set_bytes,
)


MATCHING_CAPACITY = 128
MISSING_ROOT_DOMAINS = (
    "stroke_fragment_universe",
    "intersection_hypothesis_universe",
    "ownership_assignment_universe",
    "atom_instance_universe",
    "endpoint_attachment_universe",
    "port_partition_universe",
    "lane_chain_universe",
    "bond_grouping_universe",
)


@dataclass(frozen=True, slots=True)
class MatchingEnumeration:
    records: tuple[dict[str, Any], ...]
    overflow: bool


def _matching_body(pair_ids: tuple[int, ...], used: frozenset[int], boundary_ids: tuple[int, ...]) -> dict[str, Any]:
    return {
        "compatible_pair_ids": list(pair_ids),
        "unmatched_boundary_ids": [value for value in boundary_ids if value not in used],
    }


def enumerate_boundary_matchings(
    boundary_records: list[dict[str, Any]], pair_records: list[dict[str, Any]],
    *, capacity: int = MATCHING_CAPACITY,
) -> MatchingEnumeration:
    boundary_ids = tuple(record["id"] for record in boundary_records)
    pair_members = {
        record["id"]: tuple(record["boundary_ids"])
        for record in pair_records
    }
    pairs_by_boundary: dict[int, list[int]] = {value: [] for value in boundary_ids}
    for pair_id, members in pair_members.items():
        for boundary_id in members:
            pairs_by_boundary.setdefault(boundary_id, []).append(pair_id)
    for values in pairs_by_boundary.values():
        values.sort()

    bodies: list[dict[str, Any]] = []

    def visit(remaining: tuple[int, ...], selected: tuple[int, ...], used: frozenset[int]) -> None:
        if len(bodies) > capacity:
            return
        if not remaining:
            pair_ids = tuple(sorted(selected))
            bodies.append(_matching_body(pair_ids, used, boundary_ids))
            return
        first = remaining[0]
        tail = remaining[1:]
        visit(tail, selected, used)
        if len(bodies) > capacity:
            return
        remaining_set = set(tail)
        for pair_id in pairs_by_boundary.get(first, []):
            left, right = pair_members[pair_id]
            other = right if left == first else left
            if other not in remaining_set:
                continue
            visit(
                tuple(value for value in tail if value != other),
                (*selected, pair_id),
                used | frozenset((first, other)),
            )
            if len(bodies) > capacity:
                return

    visit(boundary_ids, (), frozenset())
    if len(bodies) > capacity:
        return MatchingEnumeration((), True)
    ordered = sorted(bodies, key=canonical_json_bytes)
    records = tuple(
        {"matching_index": index, **body}
        for index, body in enumerate(ordered)
    )
    return MatchingEnumeration(records, False)


def root_projection(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "atom_proposal_universe": document["atom_proposal_universe"],
        "boundary_universe": document["boundary_universe"],
        "compatible_boundary_pairs": document["compatible_boundary_pairs"],
        "direct_pair_universe": document["direct_pair_universe"],
        "image": document["image"],
        "fixed_thresholds": document["provenance"]["fixed_thresholds"],
    }


def _emitted_matchings(document: dict[str, Any]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for interpretation in document["interpretations"]:
        matching = interpretation["boundary_matching"]
        by_index[matching["matching_index"]] = matching
    return [by_index[index] for index in sorted(by_index)]


def compare_boundary_matchings(
    expected: list[dict[str, Any]], emitted: list[dict[str, Any]],
) -> dict[str, Any]:
    def bodies(records: list[dict[str, Any]]) -> set[bytes]:
        return {
            canonical_json_bytes({key: value for key, value in item.items() if key != "matching_index"})
            for item in records
        }

    expected_bodies = bodies(expected)
    emitted_bodies = bodies(emitted)
    return {
        "canonical_order_mismatch": expected_bodies == emitted_bodies and expected != emitted,
        "exact": expected == emitted,
        "extra_matching_count": len(emitted_bodies - expected_bodies),
        "missing_matching_count": len(expected_bodies - emitted_bodies),
    }


def verify_candidate_bytes(candidate_raw: bytes, input_raw: bytes) -> dict[str, Any]:
    structural = validate_candidate_set_bytes(candidate_raw, input_raw)
    if structural["status"] == "INVALID":
        return {
            "errors": structural["errors"],
            "status": "INVALID",
            "training_authorized": False,
        }
    document = parse_canonical_json(candidate_raw)
    if document["status"] == "reject":
        return {
            "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
            "reason": "closed reject does not carry the root universes needed for independent replay",
            "status": "ROOT_UNAVAILABLE_FROM_REJECT",
            "training_authorized": False,
        }

    enumeration = enumerate_boundary_matchings(
        document["boundary_universe"], document["compatible_boundary_pairs"],
    )
    root_value = root_projection(document)
    common = {
        "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "missing_root_domains": list(MISSING_ROOT_DOMAINS),
        "root_projection_sha256": framed_sha256([root_value]),
        "training_authorized": False,
    }
    if enumeration.overflow:
        return {
            **common,
            "boundary_matching_capacity": MATCHING_CAPACITY,
            "status": "BOUNDARY_MATCHINGS_OUTSIDE_CAPACITY",
        }

    expected = list(enumeration.records)
    emitted = _emitted_matchings(document)
    comparison = compare_boundary_matchings(expected, emitted)
    if not comparison["exact"]:
        status = (
            "BOUNDARY_MATCHINGS_NONCANONICAL"
            if comparison["canonical_order_mismatch"]
            else "BOUNDARY_MATCHINGS_INCOMPLETE"
        )
        return {
            **common,
            "emitted_boundary_matching_count": len(emitted),
            "emitted_boundary_matchings_sha256": framed_sha256(emitted),
            "expected_boundary_matching_count": len(expected),
            "expected_boundary_matchings_sha256": framed_sha256(expected),
            "canonical_order_mismatch": comparison["canonical_order_mismatch"],
            "extra_matching_count": comparison["extra_matching_count"],
            "missing_matching_count": comparison["missing_matching_count"],
            "status": status,
        }
    return {
        **common,
        "boundary_matching_count": len(expected),
        "boundary_matchings_sha256": framed_sha256(expected),
        "interpretation_completeness_certified": False,
        "status": "BOUNDARY_MATCHINGS_COMPLETE_INTERPRETATIONS_UNPROVABLE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = StructuredArgumentParser()
    parser.add_argument("candidate", help="canonical v0.3 candidate-set JSON")
    parser.add_argument("--input", required=True, help="paired canonical anonymous input")
    args = parser.parse_args(argv)
    try:
        candidate_raw = Path(args.candidate).read_bytes()
        input_raw = Path(args.input).read_bytes()
    except OSError as error:
        return emit_report({
            "errors": [{"code": "input_read_failed", "message": str(error), "path": "$"}],
            "status": "INVALID",
            "training_authorized": False,
        }, error=True)
    report = verify_candidate_bytes(candidate_raw, input_raw)
    return emit_report(report, error=report["status"] == "INVALID")


if __name__ == "__main__":
    raise SystemExit(main())
