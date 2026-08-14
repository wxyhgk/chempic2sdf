#!/usr/bin/env python3
"""Independent stdlib-only verifier for Raster Evidence v0.3 artifacts.

This file does not import the extractor, runner, Scene generator, renderer,
candidate enumerator, or graph canonicalizer.  It authenticates and regroups
the frozen v0.1 scene manifest independently.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "raster-evidence-independent-review-v0.3"
FROZEN_SCENE_MANIFEST_SHA256 = (
    "f7519924b13b685f235a90dbd837772ced626cdffc9bbbacd4ff423d22c406f7"
)
DOMAIN_LAYERS = {
    "simple_domain": frozenset({"simple_line_angle_valid"}),
    "crossing_domain": frozenset({"simple_line_angle_valid", "crossing_allowed"}),
}


class IndependentReviewError(RuntimeError):
    """The submitted artifacts do not satisfy the independent contract."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IndependentReviewError(f"invalid_json:{path}:{error}") from error
    if not isinstance(value, dict):
        raise IndependentReviewError(f"json_root_not_object:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise IndependentReviewError(
                        f"jsonl_row_not_object:{path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise IndependentReviewError(f"invalid_jsonl:{path}:{error}") from error
    return rows


def _write_json(path: Path, value: Any) -> str:
    data = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _expected_groups(
    scene_rows: Iterable[dict[str, Any]], domain: str
) -> dict[frozenset[str], dict[str, Any]]:
    layers = DOMAIN_LAYERS[domain]
    by_raster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        if row.get("exclusive_grammar_layer") in layers:
            raw_raster = row.get("raw_raster_sha256")
            if not isinstance(raw_raster, str):
                raise IndependentReviewError("frozen_scene_missing_raw_raster")
            by_raster[raw_raster].append(row)
    groups: dict[frozenset[str], dict[str, Any]] = {}
    for raw_raster, rows in by_raster.items():
        scene_ids = frozenset(row.get("scene_id") for row in rows)
        if None in scene_ids or len(scene_ids) != len(rows):
            raise IndependentReviewError("frozen_scene_ids_missing_or_duplicate")
        if scene_ids in groups:
            raise IndependentReviewError("different_raw_rasters_share_scene_group")
        graphs = frozenset(row.get("canonical_graph_signature_structure") for row in rows)
        if None in graphs:
            raise IndependentReviewError("frozen_scene_missing_graph_signature")
        groups[scene_ids] = {
            "graphs": graphs,
            "raw_raster_sha256": raw_raster,
        }
    return groups


def verify(
    scene_manifest_path: Path,
    submitted_manifest_path: Path,
    summary_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    if _sha256_path(scene_manifest_path) != FROZEN_SCENE_MANIFEST_SHA256:
        raise IndependentReviewError("frozen_scene_manifest_sha256_mismatch")
    scene_rows = _read_jsonl(scene_manifest_path)
    submitted_rows = _read_jsonl(submitted_manifest_path)
    summary = _read_json(summary_path)
    validation = _read_json(validation_path)
    if summary.get("schema_version") != "raster-evidence-audit-v0.3":
        raise IndependentReviewError("summary_schema_mismatch")
    if validation.get("schema_version") != "raster-evidence-audit-v0.3":
        raise IndependentReviewError("validation_schema_mismatch")
    if summary.get("mode") != "full":
        raise IndependentReviewError("independent_review_requires_full_run")
    if summary.get("training_authorized") is not False:
        raise IndependentReviewError("summary_training_authority_must_be_false")
    if validation.get("training_authorized") is not False:
        raise IndependentReviewError("validation_training_authority_must_be_false")
    manifest_sha = _sha256_path(submitted_manifest_path)
    if summary.get("artifacts", {}).get("raster_evidence_manifest_sha256") != manifest_sha:
        raise IndependentReviewError("submitted_manifest_sha256_mismatch")

    rows_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in submitted_rows:
        domain = row.get("domain")
        if domain not in DOMAIN_LAYERS:
            raise IndependentReviewError(f"unsupported_submitted_domain:{domain}")
        rows_by_domain[domain].append(row)

    domain_results: dict[str, Any] = {}
    for domain in DOMAIN_LAYERS:
        expected = _expected_groups(scene_rows, domain)
        submitted = rows_by_domain.get(domain, [])
        if len(submitted) != len(expected):
            raise IndependentReviewError(
                f"observation_count_mismatch:{domain}:{len(submitted)}:{len(expected)}"
            )
        seen_groups: set[frozenset[str]] = set()
        strict_count = 0
        ambiguity_count = 0
        for row in submitted:
            raw_scene_ids = row.get("scene_ids_audit_only")
            if not isinstance(raw_scene_ids, list) or any(
                not isinstance(value, str) for value in raw_scene_ids
            ):
                raise IndependentReviewError("invalid_submitted_scene_ids")
            scene_ids = frozenset(raw_scene_ids)
            if len(scene_ids) != len(raw_scene_ids) or scene_ids in seen_groups:
                raise IndependentReviewError("duplicate_submitted_scene_group")
            seen_groups.add(scene_ids)
            expected_record = expected.get(scene_ids)
            if expected_record is None:
                raise IndependentReviewError("submitted_scene_group_not_in_frozen_domain")
            expected_graphs = expected_record["graphs"]
            oracle_graphs = row.get("oracle_candidate_graphs")
            predicted_graphs = row.get("predicted_candidate_graphs")
            if not isinstance(oracle_graphs, list) or not isinstance(predicted_graphs, list):
                raise IndependentReviewError("candidate_graphs_not_lists")
            if oracle_graphs != sorted(set(oracle_graphs)):
                raise IndependentReviewError("oracle_candidate_graphs_not_canonical")
            if predicted_graphs != sorted(set(predicted_graphs)):
                raise IndependentReviewError("predicted_candidate_graphs_not_canonical")
            if set(oracle_graphs) != expected_graphs:
                raise IndependentReviewError("oracle_candidate_set_mismatch")
            if set(predicted_graphs) != expected_graphs:
                raise IndependentReviewError("predicted_candidate_set_mismatch")
            if row.get("candidate_sound") is not True or row.get("candidate_complete") is not True:
                raise IndependentReviewError("submitted_candidate_gate_not_true")
            if len(expected_graphs) == 1:
                strict_count += 1
                expected_graph = next(iter(expected_graphs))
                if row.get("decision") != "Graph" or row.get("output_graph") != expected_graph:
                    raise IndependentReviewError("strict_decision_mismatch")
            else:
                ambiguity_count += 1
                if row.get("decision") != "Ambiguous" or row.get("output_graph") is not None:
                    raise IndependentReviewError("ambiguity_decision_mismatch")
        if seen_groups != set(expected):
            raise IndependentReviewError(f"submitted_domain_coverage_incomplete:{domain}")
        declared = summary.get("domains", {}).get(domain, {})
        if declared.get("audited_observation_count") != len(expected):
            raise IndependentReviewError("summary_observation_count_mismatch")
        domain_results[domain] = {
            "ambiguous_observation_count": ambiguity_count,
            "scene_count": sum(len(group) for group in expected),
            "strict_observation_count": strict_count,
            "verified_observation_count": len(expected),
        }

    fixtures = summary.get("fixture_factorization")
    if not isinstance(fixtures, dict) or set(fixtures) != {
        "crossing_junction", "parallel_multiplicity"
    }:
        raise IndependentReviewError("fixture_factorization_missing")
    for record in fixtures.values():
        if record.get("png_bytes_equal") is not True or record.get("evidence_bytes_equal") is not True:
            raise IndependentReviewError("fixture_factorization_failed")
    if validation.get("candidate_soundness_and_completeness_100_percent") is not True:
        raise IndependentReviewError("validation_candidate_gate_not_true")
    if validation.get("exact_collision_factorization") is not True:
        raise IndependentReviewError("validation_factorization_gate_not_true")
    if validation.get("hidden_challenge_passed") is not False:
        raise IndependentReviewError("hidden_challenge_must_remain_pending")
    if validation.get("stability_gate_passed") is not False:
        raise IndependentReviewError("stability_must_remain_pending")
    return {
        "candidate_sets_exact": True,
        "domain_results": domain_results,
        "fixture_factorization_verified": True,
        "frozen_scene_manifest_sha256": FROZEN_SCENE_MANIFEST_SHA256,
        "overall_status": "passed_independent_finite_audit_only",
        "schema_version": SCHEMA_VERSION,
        "submitted_manifest_sha256": manifest_sha,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--submitted-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(
        args.scene_manifest,
        args.submitted_manifest,
        args.summary,
        args.validation,
    )
    output_sha = _write_json(args.output, result)
    print(json.dumps({"output_sha256": output_sha, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
