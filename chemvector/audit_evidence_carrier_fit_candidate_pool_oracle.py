#!/usr/bin/env python3
"""Audit the fit-only candidate-pool ceiling for Evidence Carrier v0.9.18."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import chemvector.evidence_carrier_pair_evaluation as evaluation_module
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    joined_training_records,
    load_train_only_p0_pair_caches,
)
from chemvector.train_evidence_carrier_pair_conditioned_residual import (
    load_pair_residual_manifest,
)


SCHEMA_VERSION = "evidence-carrier-fit-candidate-pool-oracle-0.1"
STATUS_GO = "FIT-CANDIDATE-POOL-ORACLE-GO"
STATUS_NO_GO = "FIT-CANDIDATE-POOL-ORACLE-NO-GO"
DEFAULT_INDEX = "outputs/cache/chemvector-v0917-pair-residual-fit-dev.pt"
DEFAULT_OUTPUT = (
    "outputs/evaluations/chemvector-v0918-fit-candidate-pool-oracle/audit.json"
)
EXPECTED_INDEX_SHA256 = (
    "e34cab3ce79030c0e4b054f245b0a65c5619e9c5453f4ff0368f90b5b0b73c5f"
)
EXPECTED_FIT_SIDECAR_SHA256 = (
    "e599b59f9b991ae3525ae54f394e2027d303e3415c2f6dae2abefc1356b41834"
)
EXPECTED_SAMPLE_COUNT = 7_200
EXPECTED_PAIR_COUNT = 368_573
EXPECTED_POSITIVE_COUNT = 316_552
EXPECTED_NEGATIVE_COUNT = 52_021
ORACLE_RELATIVE_GAIN = 1.0
_EVALUATION_FIELDS = (
    "anchor_segments_pixels",
    "anchor_target_line_ids",
    "anchor_source_indices",
    "anchor_group_ids",
    "anchor_group_types",
    "pair_indices",
    "pair_labels",
)


class FitCandidatePoolOracleError(RuntimeError):
    """Raised when the frozen fit-only oracle contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def oracle_pair_scores(
    pair_labels: torch.Tensor,
    *,
    tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU,
) -> torch.Tensor:
    labels = torch.as_tensor(pair_labels, dtype=torch.long).detach().cpu()
    if labels.ndim != 1 or (
        len(labels)
        and not bool(((labels == 0) | (labels == 1)).all())
    ):
        raise ValueError("pair_labels must be a binary one-dimensional tensor")
    if not math.isfinite(float(tau0)):
        raise ValueError("tau0 must be finite")
    signs = labels.to(torch.float64).mul(2.0).sub(1.0)
    return signs.mul(ORACLE_RELATIVE_GAIN).add(float(tau0))


def _source_scores(
    fit_record: Mapping[str, Any], canonical_scores: torch.Tensor
) -> torch.Tensor:
    mapping = torch.as_tensor(
        fit_record["target_free"]["source_to_canonical_candidate_indices"],
        dtype=torch.long,
    )
    scores = torch.as_tensor(canonical_scores, dtype=torch.float64).detach().cpu()
    if mapping.shape != scores.shape:
        raise FitCandidatePoolOracleError("canonical/source score mapping differs")
    return scores[mapping].contiguous()


def _evaluation_target(source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("partition") != "fit":
        raise FitCandidatePoolOracleError("oracle received a non-fit source record")
    missing = [name for name in _EVALUATION_FIELDS if name not in source]
    if missing:
        raise FitCandidatePoolOracleError(
            f"source evaluation target is missing {missing[0]}"
        )
    return {name: source[name] for name in _EVALUATION_FIELDS}


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "sample_count_equal_7200": metrics["sample_count"] == EXPECTED_SAMPLE_COUNT,
        "selected_cross_target_equal_0": metrics["selected_cross_target_pairs"] == 0,
        "selected_pair_precision_equal_1": metrics["selected_pair_precision"] == 1.0,
        "multiplicity_2_recall_at_least_0_995": (
            (metrics["multiplicity_2_recall"] or 0.0) >= 0.995
        ),
        "pairable_sample_exact_at_least_0_95": (
            (metrics["pairable_sample_exact"] or 0.0) >= 0.95
        ),
        "parallel_2_groups_present": metrics["parallel_2_group_count"] > 0,
        "parallel_2_preservation_equal_1": (
            metrics["parallel_2_distinct_preservation"] == 1.0
        ),
        "ordinary_support_loss_at_most_0_1pp": (
            metrics["ordinary_support_loss_pp"] is not None
            and metrics["ordinary_support_loss_pp"] <= 0.1
        ),
        "matching_optimal_equal_7200": (
            metrics["matching_optimal_sample_count"] == EXPECTED_SAMPLE_COUNT
        ),
        "permutation_deterministic": metrics["permutation_deterministic"] is True,
    }
    return {
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
    }


def evaluate_fit_candidate_pool_oracle(
    fit_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(fit_records) != EXPECTED_SAMPLE_COUNT:
        raise FitCandidatePoolOracleError("fit record count differs")
    by_key = {
        str(record["identity"]["sample_key"]): record for record in source_records
    }
    if len(by_key) != len(source_records):
        raise FitCandidatePoolOracleError("source sample keys are not unique")

    baseline_rows = []
    oracle_rows = []
    oracle_failures = []
    pair_count = positive_count = negative_count = 0
    for fit_record in fit_records:
        identity = fit_record.get("identity")
        if not isinstance(identity, Mapping):
            raise FitCandidatePoolOracleError("fit identity is missing")
        sample_key = str(identity["sample_key"])
        source = by_key.get(sample_key)
        if source is None:
            raise FitCandidatePoolOracleError("fit source record is missing")
        if dict(source["identity"]) != dict(identity):
            raise FitCandidatePoolOracleError("fit/source identity differs")

        canonical_tokens = torch.as_tensor(
            fit_record["target_free"]["pair_tokens"], dtype=torch.float64
        )
        canonical_labels = torch.as_tensor(
            fit_record["target_aware"]["pair_labels"], dtype=torch.long
        )
        if canonical_tokens.ndim != 2 or canonical_tokens.shape[1] != 21:
            raise FitCandidatePoolOracleError("fit pair token shape differs")
        if canonical_labels.shape != (len(canonical_tokens),):
            raise FitCandidatePoolOracleError("fit pair label shape differs")
        raw_scores = _source_scores(fit_record, canonical_tokens[:, -1])
        source_labels = torch.as_tensor(source["pair_labels"], dtype=torch.long)
        mapped_labels = _source_scores(fit_record, canonical_labels).to(torch.long)
        if not torch.equal(source_labels, mapped_labels):
            raise FitCandidatePoolOracleError("fit/source pair labels differ")

        target = _evaluation_target(source)
        baseline = evaluate_evidence_carrier_pair_sample(
            target,
            raw_scores,
            threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
            verify_permutation=True,
        )
        oracle = evaluate_evidence_carrier_pair_sample(
            target,
            oracle_pair_scores(source_labels),
            threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
            verify_permutation=True,
        )
        baseline_rows.append(baseline)
        oracle_rows.append(oracle)
        pair_count += len(source_labels)
        positive_count += int((source_labels == 1).sum())
        negative_count += int((source_labels == 0).sum())
        if (
            oracle["selected_cross_target_pairs"]
            or oracle["parallel_2_violated_group_count"]
            or oracle["ordinary_support_lost_target_count"]
            or oracle["matching_optimal"] is not True
            or oracle["permutation_deterministic"] is not True
            or (
                oracle["pairable_sample_eligible"]
                and oracle["pairable_sample_exact"] is not True
            )
        ):
            oracle_failures.append(
                {
                    "identity": dict(identity),
                    "oracle": oracle,
                }
            )

    actual_counts = {
        "sample_count": len(fit_records),
        "pair_count": pair_count,
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
    }
    expected_counts = {
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "pair_count": EXPECTED_PAIR_COUNT,
        "positive_pair_count": EXPECTED_POSITIVE_COUNT,
        "negative_pair_count": EXPECTED_NEGATIVE_COUNT,
    }
    if actual_counts != expected_counts:
        raise FitCandidatePoolOracleError(
            f"fit candidate counts differ: {actual_counts}"
        )
    baseline_metrics = aggregate_evidence_carrier_pair_evaluations(baseline_rows)
    oracle_metrics = aggregate_evidence_carrier_pair_evaluations(oracle_rows)
    gate = _gate(oracle_metrics)
    return {
        "status": STATUS_GO if gate["status"] == "GO" else STATUS_NO_GO,
        "counts": actual_counts,
        "baseline_selected_metrics": baseline_metrics,
        "oracle_selected_metrics": oracle_metrics,
        "gate": gate,
        "oracle_failure_sample_count": len(oracle_failures),
        "oracle_failure_samples": oracle_failures,
    }


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _plain(payload),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def implementation_sha256() -> dict[str, str]:
    return {
        "oracle_audit": file_sha256(Path(__file__).resolve()),
        "pair_evaluation": file_sha256(Path(evaluation_module.__file__).resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal artifact: {output}")
    index, fit_records, metadata = load_pair_residual_manifest(args.index)
    if metadata["sha256"] != EXPECTED_INDEX_SHA256:
        raise FitCandidatePoolOracleError("v0.9.17 index SHA differs")
    if metadata["fit_sidecar_sha256"] != EXPECTED_FIT_SIDECAR_SHA256:
        raise FitCandidatePoolOracleError("v0.9.17 fit sidecar SHA differs")

    source_input = index["input"]
    base_cache, augmentation_cache = load_train_only_p0_pair_caches(
        source_input["base_cache"]["path"],
        source_input["augmentation_cache"]["path"],
    )
    source_records = joined_training_records(base_cache, augmentation_cache, "fit")
    result = evaluate_fit_candidate_pool_oracle(fit_records, source_records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
        "oracle_relative_gain": ORACLE_RELATIVE_GAIN,
        "index_path": metadata["path"],
        "index_sha256": metadata["sha256"],
        "fit_sidecar_path": metadata["fit_sidecar_path"],
        "fit_sidecar_sha256": metadata["fit_sidecar_sha256"],
        "source_sha256": dict(index["source_sha256"]),
        "implementation_sha256": implementation_sha256(),
        "v0917_dev_sidecar_read": False,
        "v0917_dev_report_read": False,
        "original_calibration_records_accessed": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    _write_json_no_overwrite(output, payload)
    return {**payload, "artifact_sha256": file_sha256(output)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(_plain(run(args)), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_INDEX",
    "DEFAULT_OUTPUT",
    "FitCandidatePoolOracleError",
    "ORACLE_RELATIVE_GAIN",
    "SCHEMA_VERSION",
    "STATUS_GO",
    "STATUS_NO_GO",
    "evaluate_fit_candidate_pool_oracle",
    "oracle_pair_scores",
]
