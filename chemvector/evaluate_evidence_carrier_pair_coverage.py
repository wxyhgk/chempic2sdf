#!/usr/bin/env python3
"""Post-hoc train/cal audit for the v0.9.11 whole-set coverage veto."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from chemvector.evidence_carrier_pair_coverage import (
    EvidenceCarrierPairCoverageConfig,
    apply_synergistic_pair_coverage_veto,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    score_evidence_carrier_pair_record,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    file_sha256,
    joined_training_records,
    load_p0_pair_scorer_checkpoint,
    load_train_only_p0_pair_caches,
    score_p0_pair_records,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    pair_scorer_train_cal_gate,
)


SCHEMA_VERSION = "evidence-carrier-pair-coverage-train-cal-audit-0.1"
DEVELOPMENT_THRESHOLD = 0.7209501005709171
DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-scorer-pilot9k-train-cal/best.pt"
)
DEFAULT_CALIBRATION_REPORT = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-scorer-pilot9k-train-cal/calibration.json"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v0911-synergistic-pair-coverage-train-cal900/audit.json"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "67ba638de2841089612b428a8cf43b45206cd21f4da9aa7e182fb269160c7234"
)
EXPECTED_CALIBRATION_REPORT_SHA256 = (
    "7a987a423a12b1c8a480be6b90bff7c73b925af31e22a312c8b62e92c134265a"
)


def _contains_internal_test(value: Any) -> bool:
    return "internal_test" in str(value).lower().replace("-", "_")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(name): _plain_json(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _anchor_pair_set(indices: Sequence[int], pairs: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, pairs[index])) for index in indices}


def _coverage_permutation_deterministic(
    coverage: Mapping[str, Any],
    pair_scores: np.ndarray,
    pair_indices: np.ndarray,
    decoded_segments_pixels: np.ndarray,
    *,
    config: EvidenceCarrierPairCoverageConfig,
) -> bool:
    order = np.arange(len(pair_indices) - 1, -1, -1, dtype=np.int64)
    lookup = {
        tuple(map(int, pair)): index
        for index, pair in enumerate(pair_indices[order].tolist())
    }
    permuted_selected = [
        lookup[tuple(map(int, pair_indices[index]))]
        for index in coverage["initial_selected_candidate_indices"]
    ]
    permuted = apply_synergistic_pair_coverage_veto(
        permuted_selected,
        pair_scores[order],
        pair_indices[order],
        decoded_segments_pixels,
        config=config,
    )
    return (
        _anchor_pair_set(
            coverage["kept_candidate_indices"], pair_indices
        )
        == _anchor_pair_set(
            permuted["kept_candidate_indices"], pair_indices[order]
        )
        and _anchor_pair_set(
            coverage["vetoed_candidate_indices"], pair_indices
        )
        == _anchor_pair_set(
            permuted["vetoed_candidate_indices"], pair_indices[order]
        )
    )


def evaluate_pair_coverage_sample(
    record: Mapping[str, Any],
    pair_scores: Any,
    *,
    threshold: float,
    config: EvidenceCarrierPairCoverageConfig = EvidenceCarrierPairCoverageConfig(),
    verify_permutation: bool = True,
) -> dict[str, Any]:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Coverage threshold must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("Coverage threshold must be finite")
    if not isinstance(verify_permutation, bool):
        raise ValueError("verify_permutation must be boolean")
    scores = _array(pair_scores, dtype=np.float64)
    pairs = _array(record["pair_indices"], dtype=np.int64)
    decoded = _array(record["decoded_segments_pixels"], dtype=np.float64)
    initial = score_evidence_carrier_pair_record(
        record,
        scores,
        threshold,
        verify_permutation=verify_permutation,
    )
    coverage = apply_synergistic_pair_coverage_veto(
        initial["selected_candidate_indices"],
        scores,
        pairs,
        decoded,
        config=config,
    )
    coverage_deterministic = (
        _coverage_permutation_deterministic(
            coverage,
            scores,
            pairs,
            decoded,
            config=config,
        )
        if verify_permutation
        else None
    )

    adjusted = np.full_like(scores, threshold - 1.0, dtype=np.float64)
    kept = list(map(int, coverage["kept_candidate_indices"]))
    adjusted[kept] = scores[kept]
    accounting_started = perf_counter()
    final = score_evidence_carrier_pair_record(
        record,
        adjusted,
        threshold,
        verify_permutation=False,
    )
    accounting_latency = 1000.0 * (perf_counter() - accounting_started)
    if _anchor_pair_set(final["selected_candidate_indices"], pairs) != _anchor_pair_set(
        kept, pairs
    ):
        raise RuntimeError("Coverage metric accounting changed the frozen kept pairs")

    labels = _array(record["pair_labels"], dtype=np.int64)
    target_ids = _array(record["anchor_target_line_ids"], dtype=np.int64)
    target_counts = {
        int(target): int((target_ids == target).sum())
        for target in np.unique(target_ids)
    }
    vetoed = list(map(int, coverage["vetoed_candidate_indices"]))
    vetoed_same = sum(int(labels[index] == 1) for index in vetoed)
    vetoed_cross = len(vetoed) - vetoed_same
    vetoed_multiplicity_two = 0
    for index in vetoed:
        first, second = map(int, pairs[index])
        if target_ids[first] == target_ids[second] and target_counts[
            int(target_ids[first])
        ] == 2:
            vetoed_multiplicity_two += 1

    pipeline_latency = float(initial["matching_latency_ms"]) + float(
        coverage["latency_ms"]
    )
    return {
        **final,
        "matching_latency_ms": pipeline_latency,
        "permutation_checked": verify_permutation,
        "permutation_deterministic": (
            bool(initial["permutation_deterministic"])
            and bool(coverage_deterministic)
            if verify_permutation
            else None
        ),
        "initial_matching": initial,
        "coverage": coverage,
        "coverage_permutation_deterministic": coverage_deterministic,
        "coverage_vetoed_same_target_pairs": vetoed_same,
        "coverage_vetoed_cross_target_pairs": vetoed_cross,
        "coverage_vetoed_multiplicity_2_pairs": vetoed_multiplicity_two,
        "coverage_pipeline_latency_ms": pipeline_latency,
        "coverage_metric_accounting_latency_ms": accounting_latency,
    }


def aggregate_pair_coverage_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = list(evaluations)
    metrics = aggregate_evidence_carrier_pair_evaluations(values)
    coverage_latencies = np.asarray(
        [float(item["coverage"]["latency_ms"]) for item in values],
        dtype=np.float64,
    )
    return {
        **metrics,
        "coverage_vetoed_pair_count": sum(
            int(item["coverage"]["vetoed_pair_count"]) for item in values
        ),
        "coverage_vetoed_same_target_pairs": sum(
            int(item["coverage_vetoed_same_target_pairs"]) for item in values
        ),
        "coverage_vetoed_cross_target_pairs": sum(
            int(item["coverage_vetoed_cross_target_pairs"]) for item in values
        ),
        "coverage_vetoed_multiplicity_2_pairs": sum(
            int(item["coverage_vetoed_multiplicity_2_pairs"]) for item in values
        ),
        "coverage_latency_ms": {
            "mean": float(coverage_latencies.mean()) if len(coverage_latencies) else None,
            "p50": float(np.percentile(coverage_latencies, 50))
            if len(coverage_latencies)
            else None,
            "p95": float(np.percentile(coverage_latencies, 95))
            if len(coverage_latencies)
            else None,
            "max": float(coverage_latencies.max()) if len(coverage_latencies) else None,
        },
        "coverage_permutation_deterministic": all(
            bool(item["coverage_permutation_deterministic"]) for item in values
        ),
    }


def _validate_threshold_commitment(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_CALIBRATION_REPORT_SHA256:
        raise ValueError("B calibration report SHA256 differs from the commitment")
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace = payload.get("threshold_trace")
    if not isinstance(trace, list):
        raise ValueError("B calibration report lacks threshold_trace")
    matches = [
        row
        for row in trace
        if math.isclose(
            float(row.get("threshold", math.nan)),
            DEVELOPMENT_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise ValueError("Development threshold is absent or duplicated in B trace")
    row = matches[0]
    expected = {
        "selected_cross_target_pairs": 1,
        "multiplicity_2_matched_count": 36399,
        "pairable_sample_exact_count": 707,
    }
    if any(int(row.get(name, -1)) != value for name, value in expected.items()):
        raise ValueError("Development threshold trace metrics changed")
    return row


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.checkpoint, args.calibration_report, args.output]
    if any(_contains_internal_test(path) for path in paths):
        raise ValueError("Coverage audit paths cannot reference internal_test")
    if any("validation" in str(path).lower() for path in paths):
        raise ValueError("Coverage audit cannot read validation paths")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    calibration_path = Path(args.calibration_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Coverage audit output exists: {output}")
    if file_sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("B checkpoint SHA256 differs from the coverage commitment")
    threshold_trace = _validate_threshold_commitment(calibration_path)

    model, checkpoint = load_p0_pair_scorer_checkpoint(
        checkpoint_path,
        device="cpu",
    )
    base_cache, augmentation_cache = load_train_only_p0_pair_caches(
        checkpoint["base_cache_path"], checkpoint["augmentation_cache_path"]
    )
    records = joined_training_records(
        base_cache, augmentation_cache, "calibration"
    )
    scores = score_p0_pair_records(
        model,
        records,
        anchor_feature_dim=int(checkpoint["anchor_feature_dim"]),
        batch_size=64,
        device=torch.device("cpu"),
    )
    config = EvidenceCarrierPairCoverageConfig()
    evaluator_path = Path(__file__).resolve()
    coverage_module_path = evaluator_path.with_name(
        "evidence_carrier_pair_coverage.py"
    )
    baseline = [
        score_evidence_carrier_pair_record(
            record,
            score,
            DEVELOPMENT_THRESHOLD,
            verify_permutation=True,
        )
        for record, score in zip(records, scores, strict=True)
    ]
    covered = [
        evaluate_pair_coverage_sample(
            record,
            score,
            threshold=DEVELOPMENT_THRESHOLD,
            config=config,
            verify_permutation=True,
        )
        for record, score in zip(records, scores, strict=True)
    ]
    baseline_metrics = aggregate_evidence_carrier_pair_evaluations(baseline)
    coverage_metrics = aggregate_pair_coverage_evaluations(covered)
    component_count = base_cache.get("identity_partition", {}).get(
        "calibration_component_count"
    )
    base_gate = pair_scorer_train_cal_gate(
        {"selected_metrics": coverage_metrics},
        calibration_records=records,
        calibration_component_count=component_count,
    )
    extra_checks = {
        "coverage_vetoed_at_least_one_pair": (
            coverage_metrics["coverage_vetoed_pair_count"] > 0
        ),
        "coverage_vetoed_all_residual_cross_pairs": (
            baseline_metrics["selected_cross_target_pairs"]
            == coverage_metrics["coverage_vetoed_cross_target_pairs"]
            and coverage_metrics["selected_cross_target_pairs"] == 0
        ),
        "coverage_vetoed_no_multiplicity_2_pair": (
            coverage_metrics["coverage_vetoed_multiplicity_2_pairs"] == 0
        ),
        "coverage_permutation_deterministic": coverage_metrics[
            "coverage_permutation_deterministic"
        ],
        "fresh_audit_still_required": True,
    }
    development_go = base_gate["status"] == "GO" and all(extra_checks.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DEV-GO" if development_go else "DEV-NO-GO",
        "classification": "post-hoc train-cal development audit",
        "allows_validation": False,
        "reason_validation_forbidden": (
            "The coverage rule was designed after inspecting calibration residuals, "
            "and the existing validation500 was already observed by upstream P0."
        ),
        "threshold": DEVELOPMENT_THRESHOLD,
        "threshold_source": {
            "path": str(calibration_path),
            "sha256": EXPECTED_CALIBRATION_REPORT_SHA256,
            "trace_row": threshold_trace,
        },
        "coverage_config": asdict(config),
        "implementation": {
            "coverage_module_path": str(coverage_module_path),
            "coverage_module_sha256": file_sha256(coverage_module_path),
            "evaluator_path": str(evaluator_path),
            "evaluator_sha256": file_sha256(evaluator_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": EXPECTED_CHECKPOINT_SHA256,
            "model_type": checkpoint["model_type"],
            "best_epoch": checkpoint["best_epoch"],
        },
        "cache_provenance": {
            "base_cache_path": base_cache["cache_path"],
            "base_cache_sha256": base_cache["cache_sha256"],
            "augmentation_cache_path": augmentation_cache["cache_path"],
            "augmentation_cache_sha256": augmentation_cache["cache_sha256"],
            "uses_validation": False,
            "uses_internal_test": False,
        },
        "baseline_metrics": baseline_metrics,
        "coverage_metrics": coverage_metrics,
        "train_cal_gate": base_gate,
        "coverage_checks": extra_checks,
        "vetoed_samples": [
            {
                "identity": record["identity"],
                "vetoed_candidate_indices": result["coverage"][
                    "vetoed_candidate_indices"
                ],
                "veto_diagnostics": result["coverage"]["veto_diagnostics"],
                "vetoed_same_target_pairs": result[
                    "coverage_vetoed_same_target_pairs"
                ],
                "vetoed_cross_target_pairs": result[
                    "coverage_vetoed_cross_target_pairs"
                ],
            }
            for record, result in zip(records, covered, strict=True)
            if result["coverage"]["vetoed_pair_count"]
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration-report", default=DEFAULT_CALIBRATION_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "status": report["status"],
                "cross": report["coverage_metrics"][
                    "selected_cross_target_pairs"
                ],
                "m2_recall": report["coverage_metrics"][
                    "multiplicity_2_recall"
                ],
                "pairable_exact": report["coverage_metrics"][
                    "pairable_sample_exact"
                ],
                "vetoed": report["coverage_metrics"][
                    "coverage_vetoed_pair_count"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEVELOPMENT_THRESHOLD",
    "SCHEMA_VERSION",
    "aggregate_pair_coverage_evaluations",
    "audit",
    "build_parser",
    "evaluate_pair_coverage_sample",
]
