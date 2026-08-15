#!/usr/bin/env python3
"""Build same-root documents showing why v0.3 cannot certify interpretation completeness."""

from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path
from typing import Any

from formal.chemgraph.audit.blind_carrier_v0_2._common import canonical_json_bytes
from formal.chemgraph.audit.carrier_candidate_set_v0_3._common import framed_sha256
from formal.chemgraph.audit.carrier_candidate_set_v0_3.build_fixtures import (
    build_documents,
    _refresh_collision_document,
)
from formal.chemgraph.audit.carrier_candidate_set_v0_3.validate_candidate_set import (
    _stage_chain_sha256,
    validate_candidate_set_bytes,
)

from .verify_boundary_completeness import root_projection, verify_candidate_bytes


OUTPUT_ROOT = Path(__file__).resolve().parent / "counterexamples"


def _clear_singleton_taxonomy(document: dict[str, Any]) -> None:
    taxonomy = document["ambiguity_taxonomy"]
    taxonomy["entries"] = []
    taxonomy["entry_count"] = 0
    final_stage = document["stage_trace"]["stages"][-1]
    final_stage["output_sha256"] = framed_sha256([taxonomy])
    final_stage["record_count"] = 0
    document["stage_trace"]["ordered_chain_sha256"] = _stage_chain_sha256(
        document["stage_trace"]["stages"],
    )


def _strip_one_interpretation(document: dict[str, Any], png: bytes) -> dict[str, Any]:
    stripped = copy.deepcopy(document)
    kept = next(
        interpretation
        for interpretation in stripped["interpretations"]
        if interpretation["bonds"]
    )
    kept["ownership_hypothesis_index"] = 0
    stripped["interpretations"] = [kept]
    _refresh_collision_document(stripped, png)
    _clear_singleton_taxonomy(stripped)
    return stripped


def _paired_only_matching(document: dict[str, Any], png: bytes) -> dict[str, Any]:
    paired = copy.deepcopy(document)
    original_boundary = paired["boundary_universe"][0]

    def boundary(boundary_id: int, y_q4: int) -> dict[str, Any]:
        value = copy.deepcopy(original_boundary)
        value["id"] = boundary_id
        value["segment_q4"] = [[1608, y_q4], [8008, y_q4]]
        value["source_lsd_index"] = boundary_id
        return value

    boundaries = [boundary(0, 6144), boundary(1, 6160)]
    pair = {
        "boundary_ids": [0, 1],
        "canonical_centerline_q4": [[1608, 6152], [8008, 6152]],
        "id": 0,
        "interior_darkness_q24": 16777216,
        "interior_ink_occupancy_ppm": 1000000,
        "observed_angle_mdeg": 0,
        "overlap_q4": 6400,
        "separation_q4": 16,
        "shorter_overlap_ppm": 1000000,
    }
    interpretation = paired["interpretations"][0]
    interpretation["boundaries"] = copy.deepcopy(boundaries)
    interpretation["boundary_matching"] = {
        "compatible_pair_ids": [0],
        "matching_index": 0,
        "unmatched_boundary_ids": [],
    }
    fragment = interpretation["stroke_fragments"][0]
    fragment["boundary_member_ids"] = [0, 1]
    fragment["derivation"] = "paired_boundaries"
    paired["boundary_universe"] = boundaries
    paired["compatible_boundary_pairs"] = [pair]
    paired["enumeration"]["compatible_boundary_pair_count"] = 1
    paired["enumeration"]["raw_boundary_count"] = 2
    _refresh_collision_document(paired, png)
    _clear_singleton_taxonomy(paired)
    return paired


def build_values() -> dict[str, Any]:
    fixtures = build_documents()
    input_value = fixtures["valid_input.json"]
    input_raw = canonical_json_bytes(input_value)
    png = base64.b64decode(input_value["image"]["bytes_base64"], validate=True)
    full = copy.deepcopy(fixtures["valid_glyph_bond_collision.json"])
    omitted = _strip_one_interpretation(full, png)
    paired_only = _paired_only_matching(
        copy.deepcopy(fixtures["valid_single_bond.json"]), png,
    )
    full_raw = canonical_json_bytes(full)
    omitted_raw = canonical_json_bytes(omitted)
    paired_only_raw = canonical_json_bytes(paired_only)
    full_structural = validate_candidate_set_bytes(full_raw, input_raw)
    omitted_structural = validate_candidate_set_bytes(omitted_raw, input_raw)
    paired_only_structural = validate_candidate_set_bytes(paired_only_raw, input_raw)
    if full_structural["status"] != "STRUCTURALLY_VALID_NOT_COMPLETE":
        raise RuntimeError(f"full collision fixture became invalid: {full_structural}")
    if omitted_structural["status"] != "STRUCTURALLY_VALID_NOT_COMPLETE":
        raise RuntimeError(f"omitted-alternative fixture became invalid: {omitted_structural}")
    if paired_only_structural["status"] != "STRUCTURALLY_VALID_NOT_COMPLETE":
        raise RuntimeError(f"paired-only matching fixture became invalid: {paired_only_structural}")
    full_root_sha = framed_sha256([root_projection(full)])
    omitted_root_sha = framed_sha256([root_projection(omitted)])
    if full_root_sha != omitted_root_sha:
        raise RuntimeError("counterexample documents must have byte-identical root projections")
    full_probe = verify_candidate_bytes(full_raw, input_raw)
    omitted_probe = verify_candidate_bytes(omitted_raw, input_raw)
    paired_only_probe = verify_candidate_bytes(paired_only_raw, input_raw)
    manifest = {
        "conclusion": "same root projection admits structurally valid documents with different interpretation sets; root-only exact interpretation completeness is unprovable",
        "full_candidate_sha256": hashlib.sha256(full_raw).hexdigest(),
        "full_interpretation_count": len(full["interpretations"]),
        "full_probe_status": full_probe["status"],
        "omitted_candidate_sha256": hashlib.sha256(omitted_raw).hexdigest(),
        "omitted_interpretation_count": len(omitted["interpretations"]),
        "omitted_probe_status": omitted_probe["status"],
        "paired_only_candidate_sha256": hashlib.sha256(paired_only_raw).hexdigest(),
        "paired_only_missing_matching_count": paired_only_probe.get("missing_matching_count"),
        "paired_only_probe_status": paired_only_probe["status"],
        "root_projection_sha256": full_root_sha,
        "schema_version": "carrier-completeness-counterexample-v0.3",
        "training_authorized": False,
    }
    return {
        "input.json": input_value,
        "full_collision.json": full,
        "omitted_alternative.json": omitted,
        "paired_only_matching.json": paired_only,
        "manifest.json": manifest,
    }


def write_values(root: Path = OUTPUT_ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, value in build_values().items():
        (root / name).write_bytes(canonical_json_bytes(value))


def main() -> None:
    write_values()


if __name__ == "__main__":
    main()
