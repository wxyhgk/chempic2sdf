#!/usr/bin/env python3
"""Read-only score-frontier diagnostics for frozen v0.9.11 fresh-dev runs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    build_conditional_ordinary_pair_target,
)
from chemvector.evidence_carrier_fresh_dev_data import (
    FreshEvidenceCarrierPackage,
    load_fresh_evidence_carrier_package,
)
from chemvector.evidence_carrier_pair_coverage import (
    EvidenceCarrierPairCoverageConfig,
    _coverage_masks,
    _coverage_upper_bound,
)
from chemvector.evidence_carrier_pair_evaluation import (
    score_evidence_carrier_pair_record,
)
from chemvector.evaluate_evidence_carrier_p0_pair_scorer import (
    score_frozen_p0_pair_target,
)
from chemvector.evaluate_evidence_carrier_pair_coverage import (
    DEVELOPMENT_THRESHOLD,
    evaluate_pair_coverage_sample,
)
from chemvector.evaluate_evidence_carrier_pair_coverage_fresh import (
    CALIBRATION_REPORT_SHA256,
    CHECKPOINT_SHA256,
    COMMITMENT_REGISTRY,
    SCHEMA_VERSION as PAIRED_REPORT_SCHEMA_VERSION,
    _canonical_sha256,
    _registered_commitment,
    _tensor_mapping_sha256,
    _tensor_sha256,
    load_committed_v0911_pipeline,
)
from chemvector.train_evidence_carrier_extent import file_sha256


SCHEMA_VERSION = "evidence-carrier-fresh-score-frontier-audit-0.1"
DEFAULT_COMMITMENT = str(COMMITMENT_REGISTRY["dev"]["path"])
DEFAULT_PAIRED_REPORT = (
    "outputs/evaluations/"
    "chemvector-v0911-synergistic-pair-coverage-fresh8k/paired-run1.json"
)
PAIRED_REPORT_SHA256 = (
    "033f23b34eb87f0389bde032abf485dbe47c89b9c25e9a5560ed5e5389e28744"
)
PAIRED_REPORT_REGISTRY: dict[str, dict[str, str]] = {
    "fresh_dev_run1": {
        "path": DEFAULT_PAIRED_REPORT,
        "sha256": PAIRED_REPORT_SHA256,
    }
}
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v0914-fresh-score-frontier/audit.json"
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_FORBIDDEN_PATH_MARKERS = ("internal_test", "sealed", "mr_tadf", "mrtadf")


def _strict_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _strict_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _strict_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _same_float(left: Any, right: float, *, label: str) -> None:
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} differs from frozen v0.9.11")


def _reject_forbidden_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    normalized = str(resolved).lower().replace("-", "_")
    if any(marker in normalized for marker in _FORBIDDEN_PATH_MARKERS):
        raise ValueError(f"{label} cannot reference sealed/internal/MR-TADF data")
    return resolved


def _json_no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in paired report: {key}")
        result[key] = value
    return result


def _registered_paired_report(path: str | Path) -> dict[str, str]:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = Path(__file__).resolve().parents[1] / requested
    requested = requested.resolve(strict=False)
    for role, record in PAIRED_REPORT_REGISTRY.items():
        registered = Path(record["path"])
        if not registered.is_absolute():
            registered = Path(__file__).resolve().parents[1] / registered
        registered = registered.resolve(strict=False)
        if requested != registered:
            continue
        return {
            "role": role,
            "path": str(registered),
            "sha256": _strict_sha256(
                record["sha256"], label=f"registered {role} paired report"
            ),
        }
    raise ValueError("Paired report path is not present in the formal registry")


def _load_paired_report(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    if path.name != "paired-run1.json":
        raise ValueError("Score-frontier input must be the paired-run1 JSON")
    digest = file_sha256(path)
    if digest != _strict_sha256(
        expected_sha256, label="expected paired-run1 SHA256"
    ):
        raise ValueError("paired-run1 SHA256 differs from the formal registry")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_no_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid paired-run1 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("paired-run1 JSON must contain an object")
    return payload, digest


def _resolved_equal(left: Any, right: Path, *, label: str) -> None:
    if not isinstance(left, str):
        raise ValueError(f"{label} must be a path")
    if Path(left).expanduser().resolve(strict=False) != right.resolve(strict=False):
        raise ValueError(f"{label} differs from the registered commitment")


def _validate_report_records(
    report: Mapping[str, Any],
    package: FreshEvidenceCarrierPackage,
) -> dict[str, Mapping[str, Any]]:
    records = _strict_list(
        report.get("sample_evaluations"), label="sample_evaluations"
    )
    if len(records) != package.sample_count:
        raise ValueError("paired-run1 sample count differs from the commitment")
    expected_keys = [identity.sample_key for identity in package.identities]
    result: dict[str, Mapping[str, Any]] = {}
    observed_keys: list[str] = []
    observed_fingerprints: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _strict_mapping(raw_record, label=f"sample_evaluations[{index}]")
        identity = _strict_mapping(
            record.get("identity"), label=f"sample_evaluations[{index}].identity"
        )
        sample_key = identity.get("sample_key")
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError("paired-run1 record lacks a sample_key")
        if sample_key in result:
            raise ValueError(f"Duplicate paired-run1 sample_key: {sample_key}")
        expected_identity = package.identities[index].as_dict()
        if dict(identity) != expected_identity:
            raise ValueError(f"paired-run1 identity mismatch for {sample_key}")
        input_fingerprint = _strict_sha256(
            record.get("input_fingerprint"),
            label=f"{sample_key}.input_fingerprint",
        )
        if input_fingerprint in observed_fingerprints:
            raise ValueError("paired-run1 input fingerprints are not unique")
        observed_fingerprints.add(input_fingerprint)
        target_sha256 = _strict_sha256(
            record.get("target_sha256"), label=f"{sample_key}.target_sha256"
        )
        score_sha256 = _strict_sha256(
            record.get("score_sha256"), label=f"{sample_key}.score_sha256"
        )
        expected_route = {
            "input_sha256": input_fingerprint,
            "target_sha256": target_sha256,
            "score_sha256": score_sha256,
        }
        for route_name in ("route_b_fingerprints", "route_c_fingerprints"):
            route = _strict_mapping(
                record.get(route_name), label=f"{sample_key}.{route_name}"
            )
            if dict(route) != expected_route:
                raise ValueError(f"{sample_key} has inconsistent route fingerprints")
        baseline = _strict_mapping(
            record.get("baseline"), label=f"{sample_key}.baseline"
        )
        coverage = _strict_mapping(
            record.get("coverage"), label=f"{sample_key}.coverage"
        )
        _same_float(
            baseline.get("threshold"),
            DEVELOPMENT_THRESHOLD,
            label=f"{sample_key}.baseline.threshold",
        )
        _same_float(
            coverage.get("threshold"),
            DEVELOPMENT_THRESHOLD,
            label=f"{sample_key}.coverage.threshold",
        )
        observed_keys.append(sample_key)
        result[sample_key] = record
    if observed_keys != expected_keys:
        raise ValueError("paired-run1 sample order differs from the commitment")
    return result


def _validate_report_before_pipeline(
    report: Mapping[str, Any],
    registration: Mapping[str, str],
    package: FreshEvidenceCarrierPackage,
) -> dict[str, Mapping[str, Any]]:
    if report.get("schema_version") != PAIRED_REPORT_SCHEMA_VERSION:
        raise ValueError("paired-run1 schema differs from the frozen evaluator")
    if report.get("role") != "dev" or package.role != "dev":
        raise ValueError("Score-frontier audit accepts only registered fresh dev")
    if report.get("training_steps") != 0 or report.get("parameter_updates") != 0:
        raise ValueError("paired-run1 is not a zero-training frozen run")
    if report.get("independent_generalization") is not True:
        raise ValueError("paired-run1 is not an independent fresh-dev run")
    protocol = _strict_mapping(report.get("protocol"), label="protocol")
    required_protocol = {
        "training_steps": 0,
        "parameter_updates": 0,
        "single_dataset_pass": True,
        "shared_pair_scores": True,
        "threshold_recalibrated": False,
        "coverage_recalibrated": False,
        "sample_limit_exposed": False,
        "internal_test_read": False,
        "sealed_audit_read": False,
        "registered_commitment": True,
    }
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise ValueError("paired-run1 protocol is not the frozen fresh-dev protocol")
    _same_float(
        protocol.get("threshold"),
        DEVELOPMENT_THRESHOLD,
        label="protocol.threshold",
    )
    report_input = _strict_mapping(report.get("input"), label="input")
    _resolved_equal(
        report_input.get("commitment_path"),
        Path(registration["path"]),
        label="input.commitment_path",
    )
    if report_input.get("commitment_sha256") != registration["sha256"]:
        raise ValueError("paired-run1 commitment SHA256 differs from registry")
    if report_input.get("commitment_sha256") != package.commitment_sha256:
        raise ValueError("paired-run1 commitment SHA256 differs from loader")
    if report_input.get("package_input_fingerprint") != package.input_fingerprint:
        raise ValueError("paired-run1 package input fingerprint differs")
    if report_input.get("fresh_audit_eligible") is not True:
        raise ValueError("paired-run1 input is not fresh-audit eligible")
    if report_input.get("independence_verified") is not True:
        raise ValueError("paired-run1 input isolation was not verified")
    if report_input.get("samples") != package.sample_count:
        raise ValueError("paired-run1 input sample count differs")
    if report_input.get("molecules") != package.molecule_count:
        raise ValueError("paired-run1 input molecule count differs")
    if report_input.get("manifest_sha256") != package.manifest_sha256:
        raise ValueError("paired-run1 manifest SHA256 differs")
    if report_input.get("freeze_sha256") != package.freeze_sha256:
        raise ValueError("paired-run1 freeze SHA256 differs")
    if report_input.get("isolation_sha256") != package.isolation_sha256:
        raise ValueError("paired-run1 isolation SHA256 differs")

    paired_execution = _strict_mapping(
        report.get("paired_execution"), label="paired_execution"
    )
    if (
        paired_execution.get("sample_count_b") != package.sample_count
        or paired_execution.get("sample_count_c") != package.sample_count
        or paired_execution.get("target_build_count") != package.sample_count
        or paired_execution.get("p0_forward_count") != package.sample_count
        or paired_execution.get("paired_any_mismatch_count") != 0
        or paired_execution.get("paired_input_mismatch_count") != 0
        or paired_execution.get("paired_target_mismatch_count") != 0
        or paired_execution.get("paired_score_mismatch_count") != 0
    ):
        raise ValueError("paired-run1 execution fingerprints are incomplete")

    report_pipeline = _strict_mapping(report.get("pipeline"), label="pipeline")
    if report_pipeline.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("paired-run1 checkpoint SHA256 differs from v0.9.11")
    if (
        report_pipeline.get("calibration_report_sha256")
        != CALIBRATION_REPORT_SHA256
    ):
        raise ValueError("paired-run1 calibration SHA256 differs from v0.9.11")
    _same_float(
        report_pipeline.get("development_threshold"),
        DEVELOPMENT_THRESHOLD,
        label="pipeline.development_threshold",
    )
    for route in ("b", "c"):
        config_name = f"route_{route}_config"
        hash_name = f"route_{route}_config_sha256"
        config = _strict_mapping(
            report_pipeline.get(config_name), label=f"pipeline.{config_name}"
        )
        declared = _strict_sha256(
            report_pipeline.get(hash_name), label=f"pipeline.{hash_name}"
        )
        if _canonical_sha256(config) != declared:
            raise ValueError(f"paired-run1 {config_name} hash is invalid")
    return _validate_report_records(report, package)


def _validate_report_against_pipeline(
    report: Mapping[str, Any], pipeline: Mapping[str, Any]
) -> None:
    report_pipeline = _strict_mapping(report.get("pipeline"), label="pipeline")
    expected_scalars = {
        "checkpoint_sha256": pipeline["pair_checkpoint_sha256"],
        "calibration_report_sha256": pipeline["calibration_report_sha256"],
        "p0_checkpoint_sha256": pipeline["p0_checkpoint_sha256"],
        "route_b_config_sha256": pipeline["route_b_config_sha256"],
        "route_c_config_sha256": pipeline["route_c_config_sha256"],
    }
    for key, value in expected_scalars.items():
        if report_pipeline.get(key) != value:
            raise ValueError(f"paired-run1 pipeline {key} differs from frozen replay")
    if report_pipeline.get("route_b_config") != pipeline["route_b"]:
        raise ValueError("paired-run1 route B config differs from frozen replay")
    if report_pipeline.get("route_c_config") != pipeline["route_c"]:
        raise ValueError("paired-run1 route C config differs from frozen replay")
    if report_pipeline.get("implementation") != pipeline["implementation"]:
        raise ValueError("paired-run1 implementation hashes differ from frozen replay")


def _is_related_record(record: Mapping[str, Any]) -> bool:
    baseline = _strict_mapping(record.get("baseline"), label="baseline")
    coverage = _strict_mapping(record.get("coverage"), label="coverage")
    coverage_branch = _strict_mapping(
        coverage.get("coverage"), label="coverage.coverage"
    )
    return bool(
        int(baseline.get("multiplicity_2_matched_count", 0))
        < int(baseline.get("multiplicity_2_target_count", 0))
        or baseline.get("pairable_sample_exact") is False
        or int(baseline.get("selected_cross_target_pairs", 0)) > 0
        or int(coverage_branch.get("vetoed_pair_count", 0)) > 0
    )


def _iter_relevant_samples(
    package: FreshEvidenceCarrierPackage, sample_keys: Iterable[str]
) -> Iterable[dict[str, Any]]:
    requested = set(sample_keys)
    rows = tuple(row for row in package._rows if row.sample_key in requested)
    if len(rows) != len(requested):
        found = {row.sample_key for row in rows}
        raise ValueError(f"Missing committed replay samples: {sorted(requested - found)}")
    identities = tuple(row.identity for row in rows)
    molecules = {identity.molecule_id for identity in identities}
    selections = tuple(
        item for item in package._selections if item.molecule_id in molecules
    )
    shard_names = {row.shard for row in rows}
    shards = tuple(
        shard for shard in package.shards if shard.relative_path in shard_names
    )
    selected_package = replace(
        package,
        identities=identities,
        shards=shards,
        _rows=rows,
        _selections=selections,
    )
    yield from selected_package.iter_samples()


def _array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _score_ranks(scores: np.ndarray, indices: Iterable[int]) -> dict[int, int]:
    ranked = sorted(indices, key=lambda index: (-float(scores[index]), int(index)))
    return {int(index): rank + 1 for rank, index in enumerate(ranked)}


def _normalized_rank(rank_1based: int, candidate_count: int) -> float:
    if candidate_count < 1 or not 1 <= rank_1based <= candidate_count:
        raise ValueError("Normalized rank inputs are inconsistent")
    if candidate_count == 1:
        return 0.0
    return (rank_1based - 1) / (candidate_count - 1)


def _coverage_branch_reason(
    candidate_index: int,
    *,
    scores: np.ndarray,
    pairs: np.ndarray,
    decoded: np.ndarray,
    coverage_result: Mapping[str, Any],
) -> dict[str, Any]:
    decoded = np.asarray(decoded, dtype=np.float32)
    coverage = _strict_mapping(
        coverage_result.get("coverage"), label="coverage result"
    )
    config = EvidenceCarrierPairCoverageConfig(
        **dict(_strict_mapping(coverage.get("config"), label="coverage config"))
    )
    selected = [int(value) for value in coverage["initial_selected_candidate_indices"]]
    ranked = sorted(
        selected,
        key=lambda index: (
            -float(scores[index]),
            int(pairs[index, 0]),
            int(pairs[index, 1]),
        ),
    )
    if candidate_index not in ranked:
        return {"reason": "not_initially_selected"}
    rank = ranked.index(candidate_index)
    diagnostics = {
        int(item["candidate_index"]): item
        for item in _strict_list(
            coverage.get("veto_diagnostics"), label="coverage.veto_diagnostics"
        )
    }
    if candidate_index in diagnostics:
        return {
            "reason": "vetoed_synergistic_coverage",
            "coverage_rank_zero_based": rank,
            "coverage_rank_1based": rank + 1,
            "diagnostic": dict(diagnostics[candidate_index]),
        }
    if rank < config.minimum_contributor_pairs:
        return {
            "reason": "kept_insufficient_stronger_pairs",
            "coverage_rank_zero_based": rank,
            "coverage_rank_1based": rank + 1,
            "stronger_pair_count": rank,
        }
    ranked_segments = decoded[pairs[np.asarray(ranked, dtype=np.int64)]]
    query_segments = ranked_segments[rank]
    contributor_segments = ranked_segments[:rank]
    upper_bound = _coverage_upper_bound(
        query_segments, contributor_segments, config=config
    )
    base = {
        "coverage_rank_zero_based": rank,
        "coverage_rank_1based": rank + 1,
        "union_coverage_upper_bound": [float(value) for value in upper_bound],
    }
    if float(upper_bound.min()) + 1e-12 < config.minimum_union_common_coverage:
        return {**base, "reason": "kept_union_upper_bound_below_minimum"}
    masks = _coverage_masks(query_segments, contributor_segments, config=config)
    per_contributor = masks.mean(axis=-1)
    union_coverage = masks.any(axis=0).mean(axis=-1)
    union_common = float(union_coverage.min())
    single_common = per_contributor.min(axis=-1)
    best_single_common = float(single_common.max())
    contributor_fraction = per_contributor.mean(axis=-1)
    active = np.where(
        contributor_fraction + 1e-12 >= config.minimum_contributor_fraction
    )[0]
    details = {
        **base,
        "union_coverage": [float(value) for value in union_coverage],
        "union_common_coverage": union_common,
        "best_single_common_coverage": best_single_common,
        "active_contributor_count": int(len(active)),
        "active_contributor_candidate_indices": [int(ranked[index]) for index in active],
    }
    if union_common + 1e-12 < config.minimum_union_common_coverage:
        return {**details, "reason": "kept_union_common_below_minimum"}
    if len(active) < config.minimum_contributor_pairs:
        return {**details, "reason": "kept_insufficient_active_contributors"}
    if best_single_common >= config.maximum_single_common_coverage - 1e-12:
        return {**details, "reason": "kept_single_stronger_pair_explains_candidate"}
    raise RuntimeError("Coverage branch diagnosis disagrees with frozen veto output")


def _candidate_diagnostics(
    target: Mapping[str, Any],
    pair_scores: Any,
    baseline: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    scores = _array(pair_scores, dtype=np.float64)
    pairs = _array(target["pair_indices"], dtype=np.int64)
    labels = _array(target["pair_labels"], dtype=np.int64)
    target_ids = _array(target["anchor_target_line_ids"], dtype=np.int64)
    decoded = _array(target["decoded_segments_pixels"], dtype=np.float32)
    selected = {int(value) for value in baseline["selected_candidate_indices"]}
    kept = {int(value) for value in coverage["selected_candidate_indices"]}
    vetoed = {
        int(value)
        for value in coverage["coverage"]["vetoed_candidate_indices"]
    }
    score_ranks = _score_ranks(scores, range(len(scores)))
    target_counts = Counter(map(int, target_ids.tolist()))
    pair_rows = pairs.tolist()

    candidate_by_pair = {
        tuple(map(int, pair)): index for index, pair in enumerate(pair_rows)
    }
    anchors_by_target: dict[int, list[int]] = {}
    for anchor_id, target_id in enumerate(target_ids.tolist()):
        anchors_by_target.setdefault(int(target_id), []).append(anchor_id)
    expected_positive = []
    for target_id, target_anchors in sorted(anchors_by_target.items()):
        if len(target_anchors) != 2:
            continue
        anchor_ids = tuple(target_anchors)
        index = candidate_by_pair.get(anchor_ids)
        if index is None:
            expected_positive.append(
                {
                    "candidate_index": None,
                    "candidate_available": False,
                    "anchor_ids": list(anchor_ids),
                    "target_line_id": target_id,
                    "target_anchor_multiplicity": 2,
                    "is_multiplicity_2_expected": True,
                    "score": None,
                    "threshold": DEVELOPMENT_THRESHOLD,
                    "margin_to_frozen_threshold": None,
                    "score_rank_1based": None,
                    "incident_rank_1based_by_anchor": None,
                    "incident_negative_count": None,
                    "best_incident_negative_candidate_index_by_anchor": None,
                    "best_incident_negative_score_by_anchor": None,
                    "margin_to_best_incident_negative_by_anchor": None,
                    "best_incident_negative_candidate_index": None,
                    "best_incident_negative_score": None,
                    "margin_to_best_incident_negative": None,
                    "incident_negative_outranks_positive": None,
                    "positive_strictly_outscores_all_incident_negatives": None,
                    "selected_by_baseline": False,
                    "selected_after_coverage": False,
                    "vetoed_by_coverage": False,
                    "conflicts_with_selected_pair_anchor": False,
                    "conflicting_selected_candidate_indices": [],
                    "selection_state": "candidate_absent",
                }
            )
            continue
        if int(labels[index]) != 1:
            raise ValueError("Multiplicity-2 expected pair is not labelled positive")
        incident_ranks: dict[str, int] = {}
        best_negative_indices: dict[str, int | None] = {}
        best_negative_scores: dict[str, float | None] = {}
        negative_margins: dict[str, float | None] = {}
        incident_negative_indices: set[int] = set()
        for anchor_id in anchor_ids:
            incident = [
                other
                for other, pair in enumerate(pair_rows)
                if anchor_id in map(int, pair)
            ]
            key = str(anchor_id)
            incident_ranks[key] = _score_ranks(scores, incident)[index]
            negatives = [other for other in incident if int(labels[other]) == 0]
            incident_negative_indices.update(negatives)
            if not negatives:
                best_negative_indices[key] = None
                best_negative_scores[key] = None
                negative_margins[key] = None
                continue
            best_negative = min(
                negatives,
                key=lambda other: (-float(scores[other]), int(other)),
            )
            best_negative_score = float(scores[best_negative])
            best_negative_indices[key] = best_negative
            best_negative_scores[key] = best_negative_score
            negative_margins[key] = float(scores[index]) - best_negative_score
        best_incident_negative = (
            min(
                incident_negative_indices,
                key=lambda other: (-float(scores[other]), int(other)),
            )
            if incident_negative_indices
            else None
        )
        best_incident_negative_score = (
            float(scores[best_incident_negative])
            if best_incident_negative is not None
            else None
        )
        margin_to_best_incident_negative = (
            float(scores[index]) - best_incident_negative_score
            if best_incident_negative_score is not None
            else None
        )
        conflicts = sorted(
            selected_index
            for selected_index in selected
            if selected_index != index
            and set(anchor_ids).intersection(map(int, pairs[selected_index]))
        )
        margin = float(scores[index]) - DEVELOPMENT_THRESHOLD
        is_selected = index in selected
        if is_selected:
            state = "selected"
        elif conflicts:
            state = "blocked_by_selected_anchor_conflict"
        elif margin <= 0.0:
            state = "at_or_below_frozen_threshold"
        else:
            state = "unselected_without_local_anchor_conflict"
        expected_positive.append(
            {
                "candidate_index": index,
                "candidate_available": True,
                "anchor_ids": list(anchor_ids),
                "target_line_id": target_id,
                "target_anchor_multiplicity": target_counts[target_id],
                "is_multiplicity_2_expected": True,
                "score": float(scores[index]),
                "threshold": DEVELOPMENT_THRESHOLD,
                "margin_to_frozen_threshold": margin,
                "score_rank_1based": score_ranks[index],
                "incident_rank_1based_by_anchor": incident_ranks,
                "incident_negative_count": len(incident_negative_indices),
                "best_incident_negative_candidate_index_by_anchor": (
                    best_negative_indices
                ),
                "best_incident_negative_score_by_anchor": best_negative_scores,
                "margin_to_best_incident_negative_by_anchor": negative_margins,
                "best_incident_negative_candidate_index": best_incident_negative,
                "best_incident_negative_score": best_incident_negative_score,
                "margin_to_best_incident_negative": (
                    margin_to_best_incident_negative
                ),
                "incident_negative_outranks_positive": bool(
                    margin_to_best_incident_negative is not None
                    and margin_to_best_incident_negative < 0.0
                ),
                "positive_strictly_outscores_all_incident_negatives": bool(
                    margin_to_best_incident_negative is None
                    or margin_to_best_incident_negative > 0.0
                ),
                "selected_by_baseline": is_selected,
                "selected_after_coverage": index in kept,
                "vetoed_by_coverage": index in vetoed,
                "conflicts_with_selected_pair_anchor": bool(conflicts),
                "conflicting_selected_candidate_indices": conflicts,
                "selection_state": state,
            }
        )

    cross_candidates = []
    for index in sorted(selected):
        if int(labels[index]) != 0:
            continue
        anchor_ids = tuple(map(int, pairs[index]))
        incident_ranks = {}
        incident_counts = {}
        incident_normalized_ranks = {}
        target_multiplicities = {}
        positive_indices_by_anchor = {}
        positive_counts_by_anchor = {}
        positive_available_by_anchor = {}
        for anchor_id in anchor_ids:
            incident = [
                other
                for other, pair in enumerate(pair_rows)
                if anchor_id in map(int, pair)
            ]
            key = str(anchor_id)
            incident_rank = _score_ranks(scores, incident)[index]
            incident_ranks[key] = incident_rank
            incident_counts[key] = len(incident)
            incident_normalized_ranks[key] = _normalized_rank(
                incident_rank, len(incident)
            )
            target_multiplicities[key] = target_counts[int(target_ids[anchor_id])]
            same_target_positive = [
                other for other in incident if int(labels[other]) == 1
            ]
            positive_indices_by_anchor[key] = same_target_positive
            positive_counts_by_anchor[key] = len(same_target_positive)
            positive_available_by_anchor[key] = bool(same_target_positive)
        top_flags = [rank == 1 for rank in incident_ranks.values()]
        global_rank = score_ranks[index]
        cross_candidates.append(
            {
                "candidate_index": index,
                "anchor_ids": list(anchor_ids),
                "target_line_ids": [int(target_ids[value]) for value in anchor_ids],
                "score": float(scores[index]),
                "threshold": DEVELOPMENT_THRESHOLD,
                "margin_to_frozen_threshold": (
                    float(scores[index]) - DEVELOPMENT_THRESHOLD
                ),
                "candidate_count": len(scores),
                "score_rank_1based": global_rank,
                "score_rank_normalized_0_best": _normalized_rank(
                    global_rank, len(scores)
                ),
                "incident_rank_1based_by_anchor": incident_ranks,
                "incident_candidate_count_by_anchor": incident_counts,
                "incident_rank_normalized_0_best_by_anchor": (
                    incident_normalized_ranks
                ),
                "incident_top_either": any(top_flags),
                "incident_top_both": all(top_flags),
                "target_anchor_multiplicity_by_anchor": target_multiplicities,
                "same_target_positive_candidate_available_by_anchor": (
                    positive_available_by_anchor
                ),
                "same_target_positive_candidate_count_by_anchor": (
                    positive_counts_by_anchor
                ),
                "same_target_positive_candidate_indices_by_anchor": (
                    positive_indices_by_anchor
                ),
                "selected_after_coverage": index in kept,
                "vetoed_by_coverage": index in vetoed,
                "coverage_branch": _coverage_branch_reason(
                    index,
                    scores=scores,
                    pairs=pairs,
                    decoded=decoded,
                    coverage_result=coverage,
                ),
            }
        )
    vetoed_same_target_candidates = []
    for index in sorted(vetoed):
        if int(labels[index]) != 1:
            continue
        anchor_ids = tuple(map(int, pairs[index]))
        target_id = int(target_ids[anchor_ids[0]])
        if target_id != int(target_ids[anchor_ids[1]]):
            raise ValueError("Positive vetoed pair has inconsistent target IDs")
        vetoed_same_target_candidates.append(
            {
                "candidate_index": index,
                "anchor_ids": list(anchor_ids),
                "target_line_id": target_id,
                "target_anchor_multiplicity": target_counts[target_id],
                "score": float(scores[index]),
                "candidate_count": len(scores),
                "score_rank_1based": score_ranks[index],
                "score_rank_normalized_0_best": _normalized_rank(
                    score_ranks[index], len(scores)
                ),
                "coverage_branch": _coverage_branch_reason(
                    index,
                    scores=scores,
                    pairs=pairs,
                    decoded=decoded,
                    coverage_result=coverage,
                ),
            }
        )
    return {
        "expected_positive_candidates": expected_positive,
        "selected_cross_candidates": cross_candidates,
        "vetoed_same_target_candidates": vetoed_same_target_candidates,
    }


def _stable_replay_check(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    fields = (
        "anchor_count",
        "candidate_pair_count",
        "selected_pair_count",
        "selected_same_target_pairs",
        "selected_cross_target_pairs",
        "selected_anchor_pairs",
        "selected_candidate_indices",
        "multiplicity_2_target_count",
        "multiplicity_2_matched_count",
        "pairable_multiplicity_2_target_count",
        "pairable_sample_eligible",
        "pairable_sample_exact",
    )
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{label}.{field} differs from paired-run1")


@torch.inference_mode()
def _replay_sample(
    pipeline: Mapping[str, Any],
    sample: Mapping[str, Any],
    report_record: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    sample_key = sample["identity"].sample_key
    if sample["input_fingerprint"] != report_record["input_fingerprint"]:
        raise ValueError(f"{sample_key} input fingerprint differs from paired-run1")
    target = build_conditional_ordinary_pair_target(
        sample["image"],
        sample["primitives"],
        sample["layers"],
        target_config=pipeline["target_config"],
        pairing_config=pipeline["pairing_config"],
    )
    scored = score_frozen_p0_pair_target(
        pipeline["pair_model"],
        pipeline["p0_model"],
        sample["image"],
        target,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    target_sha256 = _tensor_mapping_sha256(target)
    score_sha256 = _tensor_sha256(scored["pair_scores"])
    if target_sha256 != report_record["target_sha256"]:
        raise ValueError(f"{sample_key} target_sha256 differs from paired-run1")
    if score_sha256 != report_record["score_sha256"]:
        raise ValueError(f"{sample_key} score_sha256 differs from paired-run1")
    record = dict(target)
    record["decoded_segments_pixels"] = scored["decoded_segments_pixels"]
    baseline = score_evidence_carrier_pair_record(
        record,
        scored["pair_scores"],
        DEVELOPMENT_THRESHOLD,
        verify_permutation=True,
    )
    coverage = evaluate_pair_coverage_sample(
        record,
        scored["pair_scores"],
        threshold=DEVELOPMENT_THRESHOLD,
        verify_permutation=True,
    )
    _stable_replay_check(
        baseline, report_record["baseline"], label=f"{sample_key}.baseline"
    )
    _stable_replay_check(
        coverage, report_record["coverage"], label=f"{sample_key}.coverage"
    )
    if (
        coverage["coverage"]["vetoed_candidate_indices"]
        != report_record["coverage"]["coverage"]["vetoed_candidate_indices"]
    ):
        raise ValueError(f"{sample_key} coverage veto differs from paired-run1")
    candidate_diagnostics = _candidate_diagnostics(
        record, scored["pair_scores"], baseline, coverage
    )
    identity = sample["identity"].as_dict()
    source_prefix = str(identity.get("source_record_id", "")).split(":", 1)[0].lower()
    source = {"chembl": "ChEMBL", "coconut": "COCONUT"}.get(source_prefix)
    if source is None:
        raise ValueError(f"Unsupported fresh-dev source for {sample_key}")
    baseline_report = report_record["baseline"]
    coverage_report = report_record["coverage"]
    return {
        "identity": identity,
        "source": source,
        "variant": identity["variant"],
        "anchor_count": int(baseline["anchor_count"]),
        "reasons": {
            "m2_miss": (
                int(baseline_report["multiplicity_2_matched_count"])
                < int(baseline_report["multiplicity_2_target_count"])
            ),
            "pairable_exact_false": (
                baseline_report["pairable_sample_exact"] is False
            ),
            "selected_cross": int(baseline_report["selected_cross_target_pairs"]) > 0,
            "coverage_veto": (
                int(coverage_report["coverage"]["vetoed_pair_count"]) > 0
            ),
        },
        "fingerprints": {
            "input_sha256": sample["input_fingerprint"],
            "target_sha256": target_sha256,
            "score_sha256": score_sha256,
        },
        **candidate_diagnostics,
    }


def _float_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()) if len(array) else None,
        "p50": float(np.percentile(array, 50)) if len(array) else None,
        "mean": float(array.mean()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def _aggregate_bucket(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [
        candidate
        for sample in samples
        for candidate in sample["expected_positive_candidates"]
    ]
    cross = [
        candidate
        for sample in samples
        for candidate in sample["selected_cross_candidates"]
    ]
    vetoed_same_target = [
        candidate
        for sample in samples
        for candidate in sample["vetoed_same_target_candidates"]
    ]
    available_missed = [
        candidate
        for candidate in positives
        if candidate["candidate_available"]
        and not candidate["selected_by_baseline"]
    ]
    at_or_below = [
        candidate
        for candidate in available_missed
        if candidate["margin_to_frozen_threshold"] <= 0.0
    ]
    above = [
        candidate
        for candidate in available_missed
        if candidate["margin_to_frozen_threshold"] > 0.0
    ]
    locally_outscored = [
        candidate
        for candidate in available_missed
        if candidate["incident_negative_outranks_positive"]
    ]
    return {
        "sample_count": len(samples),
        "m2_miss_sample_count": sum(sample["reasons"]["m2_miss"] for sample in samples),
        "pairable_exact_false_sample_count": sum(
            sample["reasons"]["pairable_exact_false"] for sample in samples
        ),
        "selected_cross_sample_count": sum(
            sample["reasons"]["selected_cross"] for sample in samples
        ),
        "coverage_veto_sample_count": sum(
            sample["reasons"]["coverage_veto"] for sample in samples
        ),
        "expected_positive_candidate_count": len(positives),
        "expected_positive_available_count": sum(
            candidate["candidate_available"] for candidate in positives
        ),
        "expected_positive_absent_count": sum(
            not candidate["candidate_available"] for candidate in positives
        ),
        "expected_positive_selected_count": sum(
            candidate["selected_by_baseline"] for candidate in positives
        ),
        "expected_positive_missed_count": sum(
            not candidate["selected_by_baseline"] for candidate in positives
        ),
        "expected_positive_available_missed_count": len(available_missed),
        "expected_positive_anchor_conflict_count": sum(
            candidate["conflicts_with_selected_pair_anchor"] for candidate in positives
        ),
        "expected_positive_margin": _float_summary(
            [
                candidate["margin_to_frozen_threshold"]
                for candidate in positives
                if candidate["margin_to_frozen_threshold"] is not None
            ]
        ),
        "missed_available_at_or_below_threshold_count": len(at_or_below),
        "missed_available_above_threshold_count": len(above),
        "missed_available_incident_negative_outranks_count": len(
            locally_outscored
        ),
        "missed_available_positive_does_not_strictly_outscore_count": sum(
            not candidate[
                "positive_strictly_outscores_all_incident_negatives"
            ]
            for candidate in available_missed
        ),
        "missed_available_no_incident_negative_count": sum(
            candidate["incident_negative_count"] == 0
            for candidate in available_missed
        ),
        "missed_available_margin_to_threshold": _float_summary(
            [candidate["margin_to_frozen_threshold"] for candidate in available_missed]
        ),
        "missed_available_margin_to_best_incident_negative": _float_summary(
            [
                candidate["margin_to_best_incident_negative"]
                for candidate in available_missed
                if candidate["margin_to_best_incident_negative"] is not None
            ]
        ),
        "missed_available_quadrants": {
            "at_or_below_threshold__positive_strictly_outscores": sum(
                candidate["margin_to_frozen_threshold"] <= 0.0
                and candidate[
                    "positive_strictly_outscores_all_incident_negatives"
                ]
                for candidate in available_missed
            ),
            "at_or_below_threshold__positive_does_not_strictly_outscore": sum(
                candidate["margin_to_frozen_threshold"] <= 0.0
                and not candidate[
                    "positive_strictly_outscores_all_incident_negatives"
                ]
                for candidate in available_missed
            ),
            "above_threshold__positive_strictly_outscores": sum(
                candidate["margin_to_frozen_threshold"] > 0.0
                and candidate[
                    "positive_strictly_outscores_all_incident_negatives"
                ]
                for candidate in available_missed
            ),
            "above_threshold__positive_does_not_strictly_outscore": sum(
                candidate["margin_to_frozen_threshold"] > 0.0
                and not candidate[
                    "positive_strictly_outscores_all_incident_negatives"
                ]
                for candidate in available_missed
            ),
        },
        "selected_cross_candidate_count": len(cross),
        "selected_cross_vetoed_count": sum(
            candidate["vetoed_by_coverage"] for candidate in cross
        ),
        "selected_cross_endpoint_count": 2 * len(cross),
        "selected_cross_endpoint_same_target_positive_available_count": sum(
            sum(
                candidate[
                    "same_target_positive_candidate_available_by_anchor"
                ].values()
            )
            for candidate in cross
        ),
        "selected_cross_margin": _float_summary(
            [candidate["margin_to_frozen_threshold"] for candidate in cross]
        ),
        "vetoed_same_target_candidate_count": len(vetoed_same_target),
        "vetoed_same_target_target_multiplicity_counts": dict(
            sorted(
                Counter(
                    str(candidate["target_anchor_multiplicity"])
                    for candidate in vetoed_same_target
                ).items()
            )
        ),
    }


def _aggregate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def grouped(field: str) -> dict[str, Any]:
        values: dict[str, list[Mapping[str, Any]]] = {}
        for sample in samples:
            values.setdefault(str(sample[field]), []).append(sample)
        return {
            key: _aggregate_bucket(bucket)
            for key, bucket in sorted(values.items())
        }

    return {
        "overall": _aggregate_bucket(samples),
        "by_source": grouped("source"),
        "by_variant": grouped("variant"),
        "by_anchor_count": grouped("anchor_count"),
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _implementation_map(
    pipeline: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    frozen = _strict_mapping(
        pipeline.get("implementation"), label="frozen implementation map"
    )
    result = {
        str(name): {
            "path": str(_strict_mapping(record, label=str(name))["path"]),
            "sha256": _strict_sha256(
                _strict_mapping(record, label=str(name))["sha256"],
                label=f"{name} implementation SHA256",
            ),
        }
        for name, record in frozen.items()
    }
    current = Path(__file__).resolve()
    result[current.name] = {
        "path": str(current),
        "sha256": file_sha256(current),
    }
    return result


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    commitment_path = _reject_forbidden_path(args.commitment, label="commitment")
    paired_report_path = _reject_forbidden_path(
        args.paired_report, label="paired report"
    )
    output = _reject_forbidden_path(args.output, label="output")
    if output.exists():
        raise FileExistsError(f"Score-frontier output exists: {output}")
    registration = _registered_commitment(commitment_path)
    if registration["role"] != "dev":
        raise ValueError("Score-frontier audit requires registered fresh dev")
    paired_registration = _registered_paired_report(paired_report_path)
    if paired_registration["role"] != "fresh_dev_run1":
        raise ValueError("Score-frontier audit requires formal fresh paired-run1")
    report, report_sha256 = _load_paired_report(
        Path(paired_registration["path"]),
        expected_sha256=paired_registration["sha256"],
    )
    package = load_fresh_evidence_carrier_package(
        registration["path"],
        expected_role="dev",
        expected_commitment_sha256=registration["sha256"],
    )
    records = _validate_report_before_pipeline(report, registration, package)
    relevant_keys = [
        identity.sample_key
        for identity in package.identities
        if _is_related_record(records[identity.sample_key])
    ]
    if not relevant_keys:
        raise ValueError("paired-run1 contains no score-frontier diagnostic samples")
    device = _device(args.device)
    pipeline = load_committed_v0911_pipeline(device=device)
    _validate_report_against_pipeline(report, pipeline)
    implementation = _implementation_map(pipeline)
    diagnostics = []
    for sample in _iter_relevant_samples(package, relevant_keys):
        sample_key = sample["identity"].sample_key
        diagnostics.append(
            _replay_sample(
                pipeline,
                sample,
                records[sample_key],
                device=device,
            )
        )
    replayed_keys = [item["identity"]["sample_key"] for item in diagnostics]
    if replayed_keys != relevant_keys:
        raise ValueError("Selective replay order differs from paired-run1")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC-COMPLETE",
        "diagnostic_only": True,
        "fresh_exposed": True,
        "not_for_selection": True,
        "training_steps": 0,
        "parameter_updates": 0,
        "sealed_audit_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
        "protocol": {
            "diagnostic_only": True,
            "fresh_exposed": True,
            "not_for_selection": True,
            "training_steps": 0,
            "parameter_updates": 0,
            "threshold_searched": False,
            "threshold_recalibrated": False,
            "coverage_recalibrated": False,
            "new_threshold_provided": False,
            "sealed_audit_read": False,
            "internal_test_read": False,
            "mr_tadf_read": False,
            "replay_scope": (
                "paired-run1 m2 miss, pairable-exact false, selected-cross, "
                "or coverage-veto samples only"
            ),
        },
        "input": {
            "commitment_path": str(package.commitment_path),
            "commitment_sha256": package.commitment_sha256,
            "package_input_fingerprint": package.input_fingerprint,
            "paired_report_path": str(paired_report_path),
            "paired_report_sha256": report_sha256,
            "paired_report_sample_count": package.sample_count,
            "replayed_sample_count": len(diagnostics),
        },
        "frozen_pipeline": {
            "version": "v0.9.11",
            "device": str(device),
            "checkpoint_sha256": pipeline["pair_checkpoint_sha256"],
            "p0_checkpoint_sha256": pipeline["p0_checkpoint_sha256"],
            "calibration_report_sha256": pipeline["calibration_report_sha256"],
            "development_threshold": DEVELOPMENT_THRESHOLD,
            "route_b_config_sha256": pipeline["route_b_config_sha256"],
            "route_c_config_sha256": pipeline["route_c_config_sha256"],
            "audit_script_sha256": implementation[Path(__file__).name]["sha256"],
        },
        "implementation": implementation,
        "aggregation": _aggregate(diagnostics),
        "sample_diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commitment", default=DEFAULT_COMMITMENT)
    parser.add_argument("--paired-report", default=DEFAULT_PAIRED_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "status": report["status"],
                "diagnostic_only": report["diagnostic_only"],
                "replayed_sample_count": report["input"]["replayed_sample_count"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_VERSION",
    "audit",
    "build_parser",
    "main",
]
