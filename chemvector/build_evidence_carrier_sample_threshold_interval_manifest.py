#!/usr/bin/env python3
"""Build train-only robust interval labels for the v0.9.16 tau predictor.

This builder consumes only the frozen 8,100 fit / 900 calibration pair caches
and the committed v0.9.10 P0 pair scorer.  It always enumerates the complete
per-sample parametric upper hull; the v0.9.15 early capability certificate is
intentionally unavailable on this path because it does not contain the real
adjacent hull boundary needed as a training label.

The resulting ``.pt`` manifest keeps model inputs and target-aware interval
labels separate.  Predictor inputs are the frozen normalized 20-D pair
descriptor plus the raw pair gain, and four O(P) target-free count features.
Multiplicity-two targets whose required pair is absent are retained for audit
but excluded from predictor loss.  Candidate-reachable samples with no robust
target-consistent scalar interval are separately classified as scalar
unreachable and are also excluded from interval loss.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import chemvector.audit_evidence_carrier_sample_threshold_reachability as reachability
import chemvector.evidence_carrier_sample_threshold_predictor as threshold_predictor
import chemvector.train_evidence_carrier_p0_pair_scorer as p0_pair_training
from chemvector.evidence_carrier_conditional_pair_data import PAIR_FEATURE_NAMES
from chemvector.evidence_carrier_p0_pair_scorer import (
    P0_PAIR_CONSISTENCY_FEATURE_NAMES,
    P0_PAIR_DESCRIPTOR_DIM,
)
from chemvector.evidence_carrier_pair_evaluation import (
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    DEFAULT_AUGMENTATION_CACHE,
    DEFAULT_BASE_CACHE,
    collate_p0_pair_training_records,
    joined_training_records,
    load_p0_pair_scorer_checkpoint,
    load_train_only_p0_pair_caches,
)


SCHEMA_VERSION = "evidence-carrier-sample-threshold-interval-manifest-0.1"
SUMMARY_SCHEMA_VERSION = (
    "evidence-carrier-sample-threshold-interval-manifest-summary-0.1"
)
TAU0 = 0.7209501005709171
LOSS_BOUNDARY_INSET_FRACTION = 0.05
DEFAULT_CHECKPOINT = reachability.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT = (
    "outputs/cache/chemvector-v0916-sample-tau-interval-manifest.pt"
)
DEFAULT_SUMMARY_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v0916-sample-tau-interval-manifest/audit.json"
)
PARTITIONS = reachability.PARTITIONS
EXPECTED_PARTITION_SAMPLE_COUNTS = (
    reachability.EXPECTED_PARTITION_SAMPLE_COUNTS
)
BASE_CACHE_SHA256 = reachability.BASE_CACHE_SHA256
AUGMENTATION_CACHE_SHA256 = reachability.AUGMENTATION_CACHE_SHA256
CHECKPOINT_SHA256 = reachability.CHECKPOINT_SHA256

NORMALIZED_DESCRIPTOR_FEATURE_NAMES = tuple(
    f"normalized_{name}"
    for name in (*PAIR_FEATURE_NAMES, *P0_PAIR_CONSISTENCY_FEATURE_NAMES)
)
PAIR_TOKEN_FEATURE_NAMES = (
    *NORMALIZED_DESCRIPTOR_FEATURE_NAMES,
    "raw_pair_gain",
)
RAW_GLOBAL_COUNT_FEATURE_NAMES = (
    "anchor_count",
    "candidate_pair_count",
    "gain_above_tau0_count",
    "gain_above_tau0_incident_unique_anchor_count",
)
GLOBAL_COUNT_FEATURE_NAMES = tuple(
    f"log1p_{name}" for name in RAW_GLOBAL_COUNT_FEATURE_NAMES
)
PAIR_TOKEN_DIM = len(PAIR_TOKEN_FEATURE_NAMES)
GLOBAL_COUNT_FEATURE_DIM = len(GLOBAL_COUNT_FEATURE_NAMES)
_ORACLE_RECORD_FIELDS = (
    "anchor_segments_pixels",
    "anchor_target_line_ids",
    "anchor_source_indices",
    "anchor_group_ids",
    "anchor_group_types",
    "pair_indices",
)


class IntervalManifestError(RuntimeError):
    """Raised when a frozen input or interval-label contract is violated."""


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _implementation_sha256() -> dict[str, str]:
    return {
        "interval_manifest": _file_sha256(Path(__file__).resolve()),
        "p0_pair_training": _file_sha256(
            Path(p0_pair_training.__file__).resolve()
        ),
        "sample_threshold_predictor": _file_sha256(
            Path(threshold_predictor.__file__).resolve()
        ),
        **{
            f"reachability_{name}": value
            for name, value in reachability._implementation_sha256().items()
        },
    }


def _frozen_input_sha256(
    base_path: Path,
    augmentation_path: Path,
    checkpoint_path: Path,
) -> dict[str, str]:
    return {
        "base_cache": _file_sha256(base_path),
        "augmentation_cache": _file_sha256(augmentation_path),
        "checkpoint": _file_sha256(checkpoint_path),
    }


def _plain_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(name): _plain_json(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _as_float_boundary(value: Any, *, lower: bool) -> float:
    if value is None:
        return -math.inf if lower else math.inf
    result = float(value)
    if math.isnan(result):
        raise IntervalManifestError("Oracle interval boundary cannot be NaN")
    return result


def robust_loss_interval(
    lower_threshold: float | None,
    upper_threshold: float | None,
    *,
    tau0: float = TAU0,
) -> dict[str, float]:
    """Return one open hull interval and its deterministic inset loss bounds."""

    lower = _as_float_boundary(lower_threshold, lower=True)
    upper = _as_float_boundary(upper_threshold, lower=False)
    if not lower < upper:
        raise IntervalManifestError("Oracle interval must have positive width")
    finite_lower = math.isfinite(lower)
    finite_upper = math.isfinite(upper)
    if finite_lower and finite_upper:
        scale = upper - lower
    elif finite_lower:
        scale = max(1.0, abs(lower - float(tau0)))
    elif finite_upper:
        scale = max(1.0, abs(upper - float(tau0)))
    else:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        raise IntervalManifestError(
            "Oracle interval normalization scale must be finite and positive"
        )
    inset = LOSS_BOUNDARY_INSET_FRACTION * scale
    loss_lower = lower + inset if finite_lower else -math.inf
    loss_upper = upper - inset if finite_upper else math.inf
    if not loss_lower < loss_upper:
        raise IntervalManifestError("Inset oracle loss interval is empty")
    fp32_values = torch.tensor(
        (lower, upper, scale, loss_lower, loss_upper),
        dtype=torch.float32,
    )
    contract = threshold_predictor.fp32_open_interval_contract(
        fp32_values[0:1],
        fp32_values[1:2],
        fp32_values[2:3],
        inward_margin_fraction=LOSS_BOUNDARY_INSET_FRACTION,
    )
    if not bool(contract.item()):
        raise IntervalManifestError(
            "Oracle loss interval is not representable after FP32 conversion"
        )
    return {
        "lower_threshold": lower,
        "upper_threshold": upper,
        "normalization_scale": float(scale),
        "loss_lower_threshold": float(loss_lower),
        "loss_upper_threshold": float(loss_upper),
        "loss_boundary_inset_fraction": LOSS_BOUNDARY_INSET_FRACTION,
        "fp32_training_representable": True,
    }


def candidate_reachability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Describe absent multiplicity-two target pairs without changing candidates."""

    target_ids = torch.as_tensor(
        record["anchor_target_line_ids"], dtype=torch.long
    ).detach().cpu()
    pair_indices = torch.as_tensor(
        record["pair_indices"], dtype=torch.long
    ).detach().cpu()
    if pair_indices.numel() == 0:
        pair_indices = torch.empty((0, 2), dtype=torch.long)
    if pair_indices.ndim != 2 or pair_indices.size(-1) != 2:
        raise IntervalManifestError("pair_indices must have shape [pairs,2]")
    anchors_by_target: dict[int, list[int]] = defaultdict(list)
    for anchor_index, target_id in enumerate(target_ids.tolist()):
        anchors_by_target[int(target_id)].append(anchor_index)
    expected = tuple(
        sorted(
            tuple(indices)
            for indices in anchors_by_target.values()
            if len(indices) == 2
        )
    )
    available = {
        tuple(map(int, pair)) for pair in pair_indices.tolist()
    }
    missing = tuple(pair for pair in expected if pair not in available)
    return {
        "multiplicity_2_target_count": len(expected),
        "candidate_reachable_multiplicity_2_target_count": (
            len(expected) - len(missing)
        ),
        "candidate_unreachable_target_count": len(missing),
        "candidate_unreachable_anchor_pairs": [list(pair) for pair in missing],
        "candidate_reachable": not missing,
    }


def _structural_oracle_key(evaluation: Mapping[str, Any]) -> tuple[Any, ...]:
    precision = evaluation.get("selected_pair_precision")
    return (
        -int(evaluation["selected_cross_target_pairs"]),
        -int(evaluation["parallel_2_violated_group_count"]),
        int(evaluation["multiplicity_2_matched_count"]),
        int(evaluation.get("pairable_sample_exact") is True),
        -int(evaluation["ordinary_support_lost_target_count"]),
        float(precision or 0.0),
    )


def _target_consistent(evaluation: Mapping[str, Any]) -> bool:
    precision = evaluation.get("selected_pair_precision")
    selected_pair_count = int(evaluation["selected_pair_count"])
    precision_ok = selected_pair_count == 0 or precision == 1.0
    pairable_ok = (
        not bool(evaluation["pairable_sample_eligible"])
        or evaluation.get("pairable_sample_exact") is True
    )
    return bool(
        int(evaluation["selected_cross_target_pairs"]) == 0
        and int(evaluation["parallel_2_violated_group_count"]) == 0
        and int(evaluation["multiplicity_2_matched_count"])
        == int(evaluation["multiplicity_2_target_count"])
        and int(evaluation["ordinary_support_lost_target_count"]) == 0
        and precision_ok
        and pairable_ok
    )


def _compact_oracle_evaluation(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    names = (
        "anchor_count",
        "candidate_pair_count",
        "selected_pair_count",
        "selected_singleton_count",
        "selected_same_target_pairs",
        "selected_cross_target_pairs",
        "selected_pair_precision",
        "multiplicity_2_target_count",
        "multiplicity_2_matched_count",
        "pairable_multiplicity_2_target_count",
        "parallel_2_group_count",
        "parallel_2_violated_group_count",
        "ordinary_supported_target_count_before",
        "ordinary_supported_target_count_after",
        "ordinary_support_lost_target_count",
        "pairable_sample_eligible",
        "pairable_sample_exact",
    )
    return {name: evaluation.get(name) for name in names}


def build_sample_oracle_label(
    record: Mapping[str, Any],
    pair_scores: Any,
    *,
    tau0: float = TAU0,
) -> dict[str, Any]:
    """Enumerate the full hull and emit every equivalent robust loss interval."""

    reachability_status = candidate_reachability(record)
    hull = reachability.discover_parametric_upper_hull(record, pair_scores)
    replay = reachability.replay_parametric_open_intervals(
        record,
        pair_scores,
        hull,
    )
    states = tuple(replay["states"])
    if not states:
        raise IntervalManifestError(
            "Full hull replay produced no production-certified open state"
        )

    consistent_states = tuple(
        state for state in states if _target_consistent(state["evaluation"])
    )
    if not reachability_status["candidate_reachable"]:
        loss_eligible = False
        exclusion_reason = "candidate_unreachable"
        equivalent_states: tuple[Mapping[str, Any], ...] = ()
    elif not consistent_states:
        loss_eligible = False
        exclusion_reason = "scalar_unreachable"
        equivalent_states = ()
    else:
        loss_eligible = True
        exclusion_reason = None
        best_key = max(
            _structural_oracle_key(state["evaluation"])
            for state in consistent_states
        )
        equivalent_states = tuple(
            state
            for state in consistent_states
            if _structural_oracle_key(state["evaluation"]) == best_key
        )

    intervals = []
    seen = set()
    for state in equivalent_states:
        expected_state = tuple(
            tuple(map(int, pair)) for pair in state["selected_anchor_pairs"]
        )
        for witness in state["witnesses"]:
            if witness.get("robust_open_interval") is not True:
                continue
            thresholds = tuple(map(float, witness["thresholds"]))
            if len(thresholds) != 3:
                raise IntervalManifestError(
                    "Robust hull witness must contain 25/50/75 thresholds"
                )
            verification = evaluate_evidence_carrier_pair_sample(
                record,
                pair_scores,
                threshold=thresholds[1],
                verify_permutation=True,
            )
            verified_state = tuple(
                sorted(
                    tuple(map(int, pair))
                    for pair in verification["selected_anchor_pairs"]
                )
            )
            if verified_state != expected_state:
                raise IntervalManifestError(
                    "Permutation verification changed a robust oracle state"
                )
            if verification.get("permutation_deterministic") is not True:
                raise IntervalManifestError(
                    "Robust oracle state is not permutation deterministic"
                )
            interval = robust_loss_interval(
                witness.get("lower_threshold"),
                witness.get("upper_threshold"),
                tau0=tau0,
            )
            signature = (
                interval["lower_threshold"],
                interval["upper_threshold"],
                tuple(state["selected_candidate_indices"]),
            )
            if signature in seen:
                continue
            seen.add(signature)
            intervals.append(
                {
                    **interval,
                    "production_witness_thresholds": list(thresholds),
                    "minimum_boundary_margin": witness.get(
                        "minimum_boundary_margin"
                    ),
                    "state_cardinality": int(state["cardinality"]),
                    "selected_candidate_indices": list(
                        map(int, state["selected_candidate_indices"])
                    ),
                    "selected_candidate_index_space": "source_pair_indices",
                    "structural_oracle_key": list(
                        _structural_oracle_key(state["evaluation"])
                    ),
                    "production_certified_open_interval": True,
                    "permutation_deterministic": True,
                    "oracle_evaluation": _compact_oracle_evaluation(
                        state["evaluation"]
                    ),
                }
            )
    intervals.sort(
        key=lambda item: (
            item["lower_threshold"],
            item["upper_threshold"],
            item["selected_candidate_indices"],
        )
    )
    if loss_eligible and not intervals:
        raise IntervalManifestError(
            "Loss-eligible sample has no robust oracle interval"
        )
    scalar_reachable = bool(consistent_states)
    return {
        **reachability_status,
        "loss_eligible": loss_eligible,
        "exclusion_reason": exclusion_reason,
        "scalar_reachable_target_consistent": scalar_reachable,
        "early_certificate_used": False,
        "parametric_hull_fully_enumerated": True,
        "parametric_hull_state_count": len(hull),
        "production_reachable_state_count": len(states),
        "target_consistent_state_count": len(consistent_states),
        "structurally_equivalent_oracle_state_count": len(equivalent_states),
        "oracle_equivalent_robust_intervals": intervals,
    }


@torch.inference_mode()
def score_and_tokenize_records(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    anchor_feature_dim: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return frozen raw gains and normalized 20-D descriptors per sample."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    scores: list[torch.Tensor] = []
    descriptors: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        current = records[start : start + batch_size]
        collated = collate_p0_pair_training_records(
            current,
            anchor_feature_dim=anchor_feature_dim,
        )
        counts = tuple(map(int, collated["pair_count"].tolist()))
        output = model(
            collated["anchor_features"].to(device),
            collated["pair_indices"].to(device),
            collated["pair_features"].to(device),
            collated["pair_consistency_features"].to(device),
            collated["pair_mask"].to(device),
        )
        gains = output["pair_gains"].detach().cpu()
        normalized = output["normalized_pair_descriptors"].detach().cpu()
        for gain_row, descriptor_row, count in zip(
            gains,
            normalized,
            counts,
            strict=True,
        ):
            scores.append(gain_row[:count].to(torch.float32).contiguous())
            descriptors.append(
                descriptor_row[:count].to(torch.float32).contiguous()
            )
    if len(scores) != len(records) or len(descriptors) != len(records):
        raise IntervalManifestError("Frozen scorer returned the wrong sample count")
    return scores, descriptors


def build_sample_features(
    record: Mapping[str, Any],
    pair_scores: Any,
    normalized_descriptors: Any,
    *,
    tau0: float = TAU0,
) -> dict[str, Any]:
    """Build the strictly target-free predictor input for one image."""

    scores = torch.as_tensor(pair_scores, dtype=torch.float32).detach().cpu()
    descriptors = torch.as_tensor(
        normalized_descriptors, dtype=torch.float32
    ).detach().cpu()
    pair_indices = torch.as_tensor(
        record["pair_indices"], dtype=torch.long
    ).detach().cpu()
    if pair_indices.numel() == 0:
        pair_indices = torch.empty((0, 2), dtype=torch.long)
    pair_count = len(pair_indices)
    if scores.shape != (pair_count,):
        raise IntervalManifestError("Raw pair gains differ from pair_indices")
    if descriptors.shape != (pair_count, P0_PAIR_DESCRIPTOR_DIM):
        raise IntervalManifestError(
            "Normalized pair descriptors must have shape [pairs,20]"
        )
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(descriptors).all()
    ):
        raise IntervalManifestError("Predictor pair tokens contain non-finite values")
    active = scores > float(tau0)
    active_indices = pair_indices[active]
    incident_unique = (
        int(torch.unique(active_indices).numel()) if len(active_indices) else 0
    )
    raw_counts = torch.tensor(
        (
            len(record["anchor_segments_pixels"]),
            pair_count,
            int(active.sum()),
            incident_unique,
        ),
        dtype=torch.long,
    )
    global_features = torch.log1p(raw_counts.to(torch.float32))
    source_tokens = torch.cat(
        (descriptors, scores[:, None]), dim=-1
    ).contiguous()
    canonical_source_indices = torch.tensor(
        sorted(
            range(pair_count),
            key=lambda index: (
                int(pair_indices[index, 0]),
                int(pair_indices[index, 1]),
            ),
        ),
        dtype=torch.long,
    )
    source_to_canonical = torch.empty(pair_count, dtype=torch.long)
    if pair_count:
        source_to_canonical[canonical_source_indices] = torch.arange(
            pair_count, dtype=torch.long
        )
    canonical_pairs = pair_indices[canonical_source_indices].contiguous()
    tokens = source_tokens[canonical_source_indices].contiguous()
    if tokens.shape != (pair_count, PAIR_TOKEN_DIM):
        raise IntervalManifestError("Predictor pair token width changed")
    return {
        "identity": dict(record.get("identity", {})),
        "target_free": {
            "pair_indices": canonical_pairs,
            "pair_tokens": tokens,
            "canonical_to_source_candidate_indices": (
                canonical_source_indices.contiguous()
            ),
            "source_to_canonical_candidate_indices": (
                source_to_canonical.contiguous()
            ),
            "raw_global_counts": raw_counts,
            "global_count_features": global_features,
        },
    }


def _oracle_process_payload(
    record: Mapping[str, Any],
    score: torch.Tensor,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    compact: dict[str, np.ndarray] = {}
    for name in _ORACLE_RECORD_FIELDS:
        value = record[name]
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        compact[name] = np.ascontiguousarray(array).copy()
    return (
        compact,
        np.ascontiguousarray(score.detach().cpu().numpy()).copy(),
    )


def _oracle_task(task: tuple[Mapping[str, Any], Any]) -> dict[str, Any]:
    record, score = task
    return build_sample_oracle_label(record, score)


def _ordered_oracle_labels(
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[torch.Tensor],
    *,
    workers: int,
):
    tasks = (
        _oracle_process_payload(record, score)
        for record, score in zip(records, scores, strict=True)
    )
    if workers == 1:
        for task in tasks:
            yield _oracle_task(task)
        return
    context = mp.get_context("spawn")
    with reachability._single_thread_worker_environment():
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=reachability._initialise_audit_worker,
        ) as executor:
            indexed_tasks = iter(enumerate(tasks))
            pending = {}
            ready = {}
            next_output_index = 0
            window = max(1, workers * 4)

            def fill_window() -> None:
                while len(pending) + len(ready) < window:
                    try:
                        index, task = next(indexed_tasks)
                    except StopIteration:
                        return
                    pending[executor.submit(_oracle_task, task)] = index

            fill_window()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    ready[pending.pop(future)] = future
                while next_output_index in ready:
                    yield ready.pop(next_output_index).result()
                    next_output_index += 1
                    fill_window()


def _partition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        record for record in records if record["target_aware"]["loss_eligible"]
    ]
    candidate_unreachable = [
        record
        for record in records
        if record["target_aware"]["exclusion_reason"]
        == "candidate_unreachable"
    ]
    scalar_unreachable = [
        record
        for record in records
        if record["target_aware"]["exclusion_reason"] == "scalar_unreachable"
    ]
    intervals = [
        interval
        for record in eligible
        for interval in record["target_aware"][
            "oracle_equivalent_robust_intervals"
        ]
    ]
    return {
        "sample_count": len(records),
        "loss_eligible_sample_count": len(eligible),
        "candidate_unreachable_sample_count": len(candidate_unreachable),
        "candidate_unreachable_target_count": sum(
            int(record["candidate_unreachable_target_count"])
            for record in (
                item["target_aware"] for item in candidate_unreachable
            )
        ),
        "scalar_unreachable_sample_count": len(scalar_unreachable),
        "oracle_interval_count": len(intervals),
        "multi_interval_loss_sample_count": sum(
            len(
                record["target_aware"]["oracle_equivalent_robust_intervals"]
            )
            > 1
            for record in eligible
        ),
        "finite_interval_count": sum(
            math.isfinite(interval["lower_threshold"])
            and math.isfinite(interval["upper_threshold"])
            for interval in intervals
        ),
        "lower_unbounded_interval_count": sum(
            not math.isfinite(interval["lower_threshold"])
            for interval in intervals
        ),
        "upper_unbounded_interval_count": sum(
            not math.isfinite(interval["upper_threshold"])
            for interval in intervals
        ),
        "full_hull_sample_count": sum(
            record["target_aware"]["parametric_hull_fully_enumerated"] is True
            for record in records
        ),
        "early_certificate_sample_count": sum(
            record["target_aware"]["early_certificate_used"] is True
            for record in records
        ),
    }


def _device(value: str) -> torch.device:
    return reachability._device(value)


def _validate_args(args: argparse.Namespace) -> None:
    requested = tuple(args.partitions)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("partitions must be unique and non-empty")
    if any(partition not in PARTITIONS for partition in requested):
        raise ValueError("partitions may contain only fit and calibration")
    if args.limit_per_partition is not None and (
        isinstance(args.limit_per_partition, bool)
        or not isinstance(args.limit_per_partition, int)
        or args.limit_per_partition < 1
    ):
        raise ValueError("limit_per_partition must be positive or None")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (
        isinstance(args.workers, bool)
        or not isinstance(args.workers, int)
        or args.workers < 1
    ):
        raise ValueError("workers must be a positive integer")


def _build_lock_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".lock")


def _acquire_build_lock(
    manifest_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, int]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _build_lock_path(manifest_path)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"Interval manifest build lock exists: {lock_path}"
        ) from error
    payload = {
        "schema_version": "evidence-carrier-interval-manifest-build-lock-0.1",
        "status": "running",
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "expected_source_sha256": {
            "base_cache": BASE_CACHE_SHA256,
            "augmentation_cache": AUGMENTATION_CACHE_SHA256,
            "checkpoint": CHECKPOINT_SHA256,
        },
        "requested_partitions": list(args.partitions),
        "limit_per_partition": args.limit_per_partition,
    }
    os.write(
        descriptor,
        (
            json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    os.fsync(descriptor)
    _fsync_directory(lock_path.parent)
    return lock_path, descriptor


def _discard_build_lock(lock_path: Path, descriptor: int) -> None:
    os.close(descriptor)
    lock_path.unlink(missing_ok=True)
    _fsync_directory(lock_path.parent)


def _commit_build_lock(
    lock_path: Path,
    descriptor: int,
    *,
    manifest_path: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
) -> None:
    os.close(descriptor)
    completed = {
        "schema_version": "evidence-carrier-interval-manifest-build-lock-0.1",
        "status": "completed",
        "pid": os.getpid(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": summary["manifest_sha256"],
        "summary_path": str(summary_path),
        "summary_sha256": _file_sha256(summary_path),
        "formal_eligible": bool(summary["formal_eligible"]),
        "source_sha256": summary["source_sha256"],
    }
    temporary = lock_path.with_suffix(lock_path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    completed,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, lock_path)
        _fsync_directory(lock_path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_outputs(
    manifest_path: Path,
    summary_path: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest_path == summary_path:
        raise ValueError("Manifest and summary outputs must differ")
    for path in (manifest_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    unique = f".tmp.{os.getpid()}"
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + unique)
    summary_tmp = summary_path.with_suffix(summary_path.suffix + unique)
    manifest_committed = False
    try:
        for path in (manifest_path, summary_path):
            if path.exists():
                raise FileExistsError(f"Interval manifest output exists: {path}")
        for path in (manifest_tmp, summary_tmp):
            if path.exists():
                raise FileExistsError(
                    f"Interval manifest temporary output exists: {path}"
                )
        torch.save(dict(manifest), manifest_tmp)
        _fsync_file(manifest_tmp)
        manifest_sha256 = _file_sha256(manifest_tmp)
        final_summary = {
            **dict(summary),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "manifest_size_bytes": manifest_tmp.stat().st_size,
            "commitment": {
                "status": "committed",
                "manifest_committed_before_summary": True,
                "summary_binds_manifest_sha256": True,
                "summary_is_final_commit_marker": True,
                "completed_build_lock_required": True,
                "no_overwrite": True,
            },
        }
        with summary_tmp.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _plain_json(final_summary),
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication gives create-if-absent semantics.  The audit
        # summary is linked last and is therefore the formal commit marker.
        os.link(manifest_tmp, manifest_path)
        manifest_committed = True
        _fsync_directory(manifest_path.parent)
        manifest_tmp.unlink()
        os.link(summary_tmp, summary_path)
        _fsync_directory(summary_path.parent)
        summary_tmp.unlink()
        return final_summary
    finally:
        if manifest_committed and not summary_path.exists():
            manifest_path.unlink(missing_ok=True)
            _fsync_directory(manifest_path.parent)
        manifest_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)


@torch.inference_mode()
def _build_manifest_under_lock(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_args(args)
    base_path = reachability._reject_forbidden_path(
        args.base_cache, label="base cache"
    )
    augmentation_path = reachability._reject_forbidden_path(
        args.augmentation_cache, label="P0 sidecar"
    )
    checkpoint_path = reachability._reject_forbidden_path(
        args.checkpoint, label="B checkpoint"
    )
    output_path = reachability._reject_forbidden_path(
        args.output, label="manifest output"
    )
    summary_path = reachability._reject_forbidden_path(
        args.summary_output, label="summary output"
    )
    expected_hashes = {
        "base_cache": BASE_CACHE_SHA256,
        "augmentation_cache": AUGMENTATION_CACHE_SHA256,
        "checkpoint": CHECKPOINT_SHA256,
    }
    actual_hashes = _frozen_input_sha256(
        base_path,
        augmentation_path,
        checkpoint_path,
    )
    if actual_hashes != expected_hashes:
        raise IntervalManifestError(
            f"Frozen input SHA256 mismatch: {actual_hashes}"
        )
    implementation_sha256 = _implementation_sha256()

    base_cache, augmentation_cache = load_train_only_p0_pair_caches(
        base_path,
        augmentation_path,
    )
    model, checkpoint = load_p0_pair_scorer_checkpoint(
        checkpoint_path,
        device="cpu",
        verify_source_hashes=False,
    )
    if (
        checkpoint.get("base_cache_sha256") != BASE_CACHE_SHA256
        or checkpoint.get("augmentation_cache_sha256")
        != AUGMENTATION_CACHE_SHA256
    ):
        raise IntervalManifestError(
            "Frozen scorer checkpoint is not bound to the train-only caches"
        )
    device = _device(args.device)
    if device.type != "cpu" and args.workers != 1:
        raise ValueError("workers > 1 requires CPU scoring")
    model.to(device)
    anchor_feature_dim = int(augmentation_cache["anchor_feature_dim"])

    partitions: dict[str, Any] = {}
    partition_summaries: dict[str, Any] = {}
    complete_counts = True
    for partition in args.partitions:
        source_records = joined_training_records(
            base_cache,
            augmentation_cache,
            partition,
        )
        if args.limit_per_partition is not None:
            source_records = source_records[: args.limit_per_partition]
        for record in source_records:
            reachability._identity_is_train_only(record)
        expected_count = EXPECTED_PARTITION_SAMPLE_COUNTS[partition]
        complete_partition = len(source_records) == expected_count
        complete_counts = complete_counts and complete_partition
        scores, descriptors = score_and_tokenize_records(
            model,
            source_records,
            anchor_feature_dim=anchor_feature_dim,
            batch_size=args.batch_size,
            device=device,
        )
        labels = list(
            _ordered_oracle_labels(
                source_records,
                scores,
                workers=min(args.workers, max(len(source_records), 1)),
            )
        )
        records = []
        for source, score, descriptor, label in zip(
            source_records,
            scores,
            descriptors,
            labels,
            strict=True,
        ):
            features = build_sample_features(source, score, descriptor)
            records.append(
                {
                    **features,
                    "partition": partition,
                    "target_aware": label,
                }
            )
        loss_eligible_indices = torch.tensor(
            [
                index
                for index, record in enumerate(records)
                if record["target_aware"]["loss_eligible"]
            ],
            dtype=torch.long,
        )
        candidate_unreachable_indices = torch.tensor(
            [
                index
                for index, record in enumerate(records)
                if record["target_aware"]["exclusion_reason"]
                == "candidate_unreachable"
            ],
            dtype=torch.long,
        )
        scalar_unreachable_indices = torch.tensor(
            [
                index
                for index, record in enumerate(records)
                if record["target_aware"]["exclusion_reason"]
                == "scalar_unreachable"
            ],
            dtype=torch.long,
        )
        summary = {
            **_partition_summary(records),
            "expected_samples": expected_count,
            "complete_partition": complete_partition,
        }
        partitions[partition] = {
            **summary,
            "records": records,
            "loss_eligible_indices": loss_eligible_indices,
            "candidate_unreachable_indices": candidate_unreachable_indices,
            "scalar_unreachable_indices": scalar_unreachable_indices,
        }
        partition_summaries[partition] = summary

    formal_eligible = bool(
        tuple(args.partitions) == PARTITIONS
        and args.limit_per_partition is None
        and complete_counts
    )
    status = (
        "TRAIN-CAL-INTERVAL-MANIFEST-COMPLETE"
        if formal_eligible
        else "PARTIAL-INTERVAL-MANIFEST"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "formal_eligible": formal_eligible,
        "tau0": TAU0,
        "source_sha256": {
            **actual_hashes,
            "manifest_implementation": implementation_sha256[
                "interval_manifest"
            ],
        },
        "early_shortcut_enabled": False,
        "commitment_protocol": {
            "manifest_path": str(output_path),
            "summary_path": str(summary_path),
            "completed_build_lock_path": str(_build_lock_path(output_path)),
            "summary_binds_manifest_sha256": True,
            "completed_build_lock_required_for_training": True,
            "no_overwrite": True,
        },
        "target_free_predictor_inputs": True,
        "target_aware_fields_are_loss_labels_only": True,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
        "protocol": {
            "complete_parametric_upper_hull_required": True,
            "early_capability_certificate_allowed": False,
            "production_interval_witnesses": "strict_25_50_75_percent",
            "multiple_structurally_equivalent_intervals_retained": True,
            "candidate_unreachable_excluded_from_loss": True,
            "scalar_unreachable_excluded_from_loss": True,
            "loss_boundary_inset_fraction": LOSS_BOUNDARY_INSET_FRACTION,
            "finite_interval_normalization_scale": "upper-lower",
            "one_sided_interval_normalization_scale": (
                "max(1,abs(finite_boundary-tau0))"
            ),
            "fully_unbounded_interval_normalization_scale": 1.0,
            "fp32_training_interval_representability_required": True,
            "frozen_input_sha256_reverified_before_commit": True,
        },
        "feature_contract": {
            "pair_token_dim": PAIR_TOKEN_DIM,
            "pair_token_feature_names": list(PAIR_TOKEN_FEATURE_NAMES),
            "normalized_descriptor_dim": P0_PAIR_DESCRIPTOR_DIM,
            "normalized_descriptor_feature_names": list(
                NORMALIZED_DESCRIPTOR_FEATURE_NAMES
            ),
            "descriptor_source": (
                "frozen_pair_scorer_transform_plus_checkpoint_mean_scale"
            ),
            "raw_pair_gain_index": PAIR_TOKEN_DIM - 1,
            "pair_order": (
                "lexicographic_first_anchor_id_then_second_anchor_id"
            ),
            "candidate_index_mapping": (
                "target_free stores canonical_to_source and source_to_canonical"
            ),
            "global_count_feature_dim": GLOBAL_COUNT_FEATURE_DIM,
            "raw_global_count_feature_names": list(
                RAW_GLOBAL_COUNT_FEATURE_NAMES
            ),
            "global_count_feature_names": list(GLOBAL_COUNT_FEATURE_NAMES),
            "global_count_transform": "log1p(raw_count)",
            "gain_above_tau0_comparison": "strict_greater_than",
        },
        "frozen_descriptor_normalization": {
            "mean": model.descriptor_mean.detach().cpu().clone(),
            "scale": model.descriptor_scale.detach().cpu().clone(),
            "source": "committed_pair_scorer_checkpoint",
        },
        "input": {
            "base_cache_path": str(base_path),
            "augmentation_cache_path": str(augmentation_path),
            "checkpoint_path": str(checkpoint_path),
            "sha256": actual_hashes,
            "partitions": list(args.partitions),
            "limit_per_partition": args.limit_per_partition,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "worker_start_method": "spawn",
            "worker_numeric_threads": 1,
            "highs_threads": 1,
            "worker_task_window_per_worker": 4,
        },
        "implementation_sha256": implementation_sha256,
        "partitions": partitions,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "manifest_schema_version": SCHEMA_VERSION,
        "status": status,
        "formal_eligible": formal_eligible,
        "tau0": TAU0,
        "source_sha256": manifest["source_sha256"],
        "early_shortcut_enabled": False,
        "completed_build_lock_path": str(_build_lock_path(output_path)),
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
        "input": manifest["input"],
        "feature_contract": manifest["feature_contract"],
        "implementation_sha256": implementation_sha256,
        "partitions": partition_summaries,
    }
    if _implementation_sha256() != implementation_sha256:
        raise IntervalManifestError(
            "Interval manifest implementation changed during execution"
        )
    final_input_hashes = _frozen_input_sha256(
        base_path,
        augmentation_path,
        checkpoint_path,
    )
    if final_input_hashes != actual_hashes:
        raise IntervalManifestError(
            "Frozen input SHA256 changed during interval manifest generation"
        )
    final_summary = _write_outputs(
        output_path,
        summary_path,
        manifest,
        summary,
    )
    return manifest, final_summary


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Acquire the sibling build lock before any cache load or hull solve."""

    _validate_args(args)
    output_path = reachability._reject_forbidden_path(
        args.output, label="manifest output"
    )
    summary_path = reachability._reject_forbidden_path(
        args.summary_output, label="summary output"
    )
    lock_path, lock_descriptor = _acquire_build_lock(
        output_path,
        summary_path,
        args,
    )
    try:
        for path in (output_path, summary_path):
            if path.exists():
                raise FileExistsError(f"Interval manifest output exists: {path}")
        manifest, summary = _build_manifest_under_lock(args)
    except BaseException:
        _discard_build_lock(lock_path, lock_descriptor)
        raise
    _commit_build_lock(
        lock_path,
        lock_descriptor,
        manifest_path=output_path,
        summary_path=summary_path,
        summary=summary,
    )
    return manifest, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", default=DEFAULT_BASE_CACHE)
    parser.add_argument("--augmentation-cache", default=DEFAULT_AUGMENTATION_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=PARTITIONS,
        default=list(PARTITIONS),
    )
    parser.add_argument("--limit-per-partition", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary = build_manifest(args)
    print(
        json.dumps(
            {
                "manifest": str(Path(args.output).expanduser().resolve()),
                "summary": str(Path(args.summary_output).expanduser().resolve()),
                "status": summary["status"],
                "formal_eligible": summary["formal_eligible"],
                "manifest_sha256": summary["manifest_sha256"],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GLOBAL_COUNT_FEATURE_NAMES",
    "PAIR_TOKEN_FEATURE_NAMES",
    "RAW_GLOBAL_COUNT_FEATURE_NAMES",
    "SCHEMA_VERSION",
    "TAU0",
    "IntervalManifestError",
    "build_manifest",
    "build_parser",
    "build_sample_features",
    "build_sample_oracle_label",
    "candidate_reachability",
    "main",
    "robust_loss_interval",
    "score_and_tokenize_records",
]
