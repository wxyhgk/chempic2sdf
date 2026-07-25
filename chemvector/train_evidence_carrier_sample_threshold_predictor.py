#!/usr/bin/env python3
"""Train the pre-registered tiny per-image threshold predictor.

The optimizer is deliberately fit-only.  This command performs exactly 1,500
fixed steps, seals the final checkpoint, then consumes calibration exactly once
through the existing production matcher.  Calibration never affects the
optimizer, checkpoint selection, or any early-stopping rule.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

import chemvector.build_evidence_carrier_sample_threshold_interval_manifest as interval_manifest_module
import chemvector.evidence_carrier_pair_evaluation as pair_evaluation_module
import chemvector.evidence_carrier_pair_matching as pair_matching_module
import chemvector.evidence_carrier_sample_threshold_predictor as predictor_module
import chemvector.train_evidence_carrier_p0_pair_scorer as p0_pair_trainer_module
import chemvector.train_evidence_carrier_pair_scorer as pair_trainer_module
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES,
    SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
    SAMPLE_THRESHOLD_PREDICTOR_PARAMETER_COUNT,
    EvidenceCarrierSampleThresholdPredictor,
    robust_open_interval_loss,
    sample_threshold_global_count_features,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    joined_training_records,
    load_train_only_p0_pair_caches,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    pair_scorer_train_cal_gate,
)


INTERVAL_MANIFEST_SCHEMA_VERSION = interval_manifest_module.SCHEMA_VERSION
CHECKPOINT_SCHEMA_VERSION = (
    "evidence-carrier-sample-threshold-predictor-checkpoint-0.1"
)
SUMMARY_SCHEMA_VERSION = (
    "evidence-carrier-sample-threshold-predictor-train-summary-0.1"
)
CALIBRATION_SCHEMA_VERSION = (
    "evidence-carrier-sample-threshold-predictor-calibration-0.1"
)
EXPECTED_PARTITION_SAMPLE_COUNTS = {"fit": 8_100, "calibration": 900}
TRAINING_STEPS = 1_500
TRAINING_BATCH_SIZE = 64
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0
TRAINING_SEED = 20_260_722
DEFAULT_MANIFEST = interval_manifest_module.DEFAULT_OUTPUT
DEFAULT_MANIFEST_AUDIT = interval_manifest_module.DEFAULT_SUMMARY_OUTPUT
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0916-sample-tau-predictor"
)
DEFAULT_CALIBRATION_OUTPUT = (
    "outputs/evaluations/chemvector-v0916-sample-tau-predictor/calibration.json"
)
DEFAULT_CALIBRATION_CONSUMPTION_MARKER = (
    "outputs/evaluations/chemvector-v0916-sample-tau-predictor/"
    "calibration-consumption.json"
)
_FORBIDDEN_PATH_MARKERS = (
    "fresh",
    "validation",
    "sealed",
    "internal_test",
    "mr_tadf",
    "mrtadf",
)
_RAW_GLOBAL_COUNT_FEATURE_NAMES = (
    "anchor_count",
    "candidate_pair_count",
    "gain_above_tau0_count",
    "gain_above_tau0_incident_unique_anchor_count",
)


class SampleThresholdManifestError(RuntimeError):
    """Raised when the frozen interval-manifest contract is violated."""


@dataclass(frozen=True, slots=True)
class SampleThresholdIntervalRecord:
    identity: Mapping[str, Any]
    partition: str
    anchor_count: int
    pair_tokens: torch.Tensor
    pair_indices: torch.Tensor
    global_count_features: torch.Tensor
    interval_lower: torch.Tensor
    interval_upper: torch.Tensor
    interval_normalization_scale: torch.Tensor
    loss_eligible: bool
    exclusion_reason: str | None

    @property
    def candidate_pair_count(self) -> int:
        return len(self.pair_tokens)

    @property
    def interval_count(self) -> int:
        return len(self.interval_lower)


@dataclass(frozen=True, slots=True)
class SampleThresholdManifestPartition:
    name: str
    records: tuple[SampleThresholdIntervalRecord, ...]

    @property
    def loss_records(self) -> tuple[SampleThresholdIntervalRecord, ...]:
        return tuple(record for record in self.records if record.loss_eligible)

    @property
    def excluded_records(self) -> tuple[SampleThresholdIntervalRecord, ...]:
        return tuple(record for record in self.records if not record.loss_eligible)


@dataclass(frozen=True, slots=True)
class SampleThresholdIntervalManifest:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    partitions: Mapping[str, SampleThresholdManifestPartition]
    audit_path: Path | None = None
    audit_sha256: str | None = None
    audit: Mapping[str, Any] | None = None
    commitment_path: Path | None = None
    commitment_sha256: str | None = None
    commitment: Mapping[str, Any] | None = None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictor_implementation_sha256() -> dict[str, str]:
    return {
        "model": file_sha256(Path(predictor_module.__file__).resolve()),
        "trainer": file_sha256(Path(__file__).resolve()),
        "p0_pair_trainer": file_sha256(
            Path(p0_pair_trainer_module.__file__).resolve()
        ),
        "pair_trainer_and_gate": file_sha256(
            Path(pair_trainer_module.__file__).resolve()
        ),
        "pair_evaluation": file_sha256(
            Path(pair_evaluation_module.__file__).resolve()
        ),
        "pair_matching": file_sha256(
            Path(pair_matching_module.__file__).resolve()
        ),
    }


def _normalised_path_text(value: str | Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _train_only_manifest_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    resolved = raw.resolve(strict=False)
    for candidate in (raw, resolved):
        normalised = _normalised_path_text(candidate)
        if any(marker in normalised for marker in _FORBIDDEN_PATH_MARKERS):
            raise ValueError(
                "threshold manifest cannot reference fresh/validation/sealed/"
                "internal-test/MR-TADF data"
            )
    return resolved


def _tensor(value: Any, *, dtype: torch.dtype, name: str) -> torch.Tensor:
    try:
        result = (
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )
        if dtype == torch.long:
            if result.dtype == torch.bool:
                raise ValueError("boolean values are not integer indices/counts")
            if result.is_floating_point() and (
                not bool(torch.isfinite(result).all())
                or not bool((result == result.round()).all())
            ):
                raise ValueError("floating values are not exact integers")
        result = result.to(dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as error:
        raise SampleThresholdManifestError(f"{name} is not tensor-like") from error
    return result.contiguous()


def _expected_interval_scale(lower: float, upper: float) -> float:
    if math.isfinite(lower) and math.isfinite(upper):
        return upper - lower
    if math.isfinite(lower):
        return max(1.0, abs(lower - SAMPLE_THRESHOLD_BASELINE_TAU))
    if math.isfinite(upper):
        return max(1.0, abs(upper - SAMPLE_THRESHOLD_BASELINE_TAU))
    return 1.0


def _validate_interval(
    interval: Mapping[str, Any],
    *,
    sample_label: str,
) -> tuple[float, float, float]:
    try:
        lower = float(interval["lower_threshold"])
        upper = float(interval["upper_threshold"])
        scale = float(interval["normalization_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise SampleThresholdManifestError(
            f"{sample_label} has a malformed oracle interval"
        ) from error
    if math.isnan(lower) or math.isnan(upper):
        raise SampleThresholdManifestError(f"{sample_label} interval contains NaN")
    if lower == math.inf or upper == -math.inf or not lower < upper:
        raise SampleThresholdManifestError(f"{sample_label} interval is not open")
    expected_scale = _expected_interval_scale(lower, upper)
    if not math.isfinite(scale) or scale <= 0.0 or not math.isclose(
        scale, expected_scale, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} interval normalization scale differs from protocol"
        )
    margin = scale * SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION
    expected_loss_lower = lower + margin if math.isfinite(lower) else lower
    expected_loss_upper = upper - margin if math.isfinite(upper) else upper
    if "loss_lower_threshold" in interval and not math.isclose(
        float(interval["loss_lower_threshold"]),
        expected_loss_lower,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} loss lower bound differs from the 5% inset"
        )
    if "loss_upper_threshold" in interval and not math.isclose(
        float(interval["loss_upper_threshold"]),
        expected_loss_upper,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} loss upper bound differs from the 5% inset"
        )
    if not expected_loss_lower < expected_loss_upper:
        raise SampleThresholdManifestError(
            f"{sample_label} interval collapses after the 5% inset"
        )
    fp32_values = torch.tensor(
        (lower, upper, scale, expected_loss_lower, expected_loss_upper),
        dtype=torch.float32,
    )
    contract = predictor_module.fp32_open_interval_contract(
        fp32_values[0:1],
        fp32_values[1:2],
        fp32_values[2:3],
        inward_margin_fraction=SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION,
    )
    if not bool(contract.item()):
        raise SampleThresholdManifestError(
            f"{sample_label} interval is not representable in FP32"
        )
    if interval.get("fp32_training_representable") is not True:
        raise SampleThresholdManifestError(
            f"{sample_label} interval lacks the FP32 representability certificate"
        )
    if interval.get("production_certified_open_interval", True) is not True:
        raise SampleThresholdManifestError(
            f"{sample_label} includes a non-certified training interval"
        )
    if interval.get("permutation_deterministic", True) is not True:
        raise SampleThresholdManifestError(
            f"{sample_label} includes a permutation-unstable training interval"
        )
    if interval.get(
        "loss_boundary_inset_fraction",
        SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION,
    ) != SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION:
        raise SampleThresholdManifestError(
            f"{sample_label} interval inset fraction differs from protocol"
        )
    return lower, upper, scale


def _parse_record(
    raw: Mapping[str, Any],
    *,
    partition: str,
    index: int,
) -> SampleThresholdIntervalRecord:
    if not isinstance(raw, Mapping):
        raise SampleThresholdManifestError("manifest records must be mappings")
    identity = raw.get("identity")
    if not isinstance(identity, Mapping):
        raise SampleThresholdManifestError("manifest record identity is missing")
    sample_label = str(identity.get("sample_key", f"{partition}:{index}"))
    if raw.get("partition") != partition:
        raise SampleThresholdManifestError(
            f"{sample_label} partition differs from its manifest container"
        )
    target_free = raw.get("target_free")
    target_aware = raw.get("target_aware")
    if not isinstance(target_free, Mapping) or not isinstance(
        target_aware, Mapping
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} target_free/target_aware layers are missing"
        )
    pair_tokens = _tensor(
        target_free.get("pair_tokens"), dtype=torch.float32, name="pair_tokens"
    )
    pair_indices = _tensor(
        target_free.get("pair_indices"), dtype=torch.long, name="pair_indices"
    )
    global_features = _tensor(
        target_free.get("global_count_features"),
        dtype=torch.float32,
        name="global_count_features",
    )
    raw_global_counts = _tensor(
        target_free.get("raw_global_counts"),
        dtype=torch.long,
        name="raw_global_counts",
    )
    if pair_tokens.ndim != 2 or pair_tokens.size(-1) != SAMPLE_THRESHOLD_PAIR_TOKEN_DIM:
        raise SampleThresholdManifestError(
            f"{sample_label} pair_tokens must have shape [pairs,21]"
        )
    if pair_indices.shape != (len(pair_tokens), 2):
        raise SampleThresholdManifestError(
            f"{sample_label} pair_indices must have shape [pairs,2]"
        )
    if len(pair_indices) and (
        bool((pair_indices[:, 0] >= pair_indices[:, 1]).any())
        or len({tuple(map(int, pair)) for pair in pair_indices.tolist()})
        != len(pair_indices)
        or pair_indices.tolist() != sorted(pair_indices.tolist())
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} pair_indices are not unique ascending pairs"
        )
    canonical_to_source = _tensor(
        target_free.get("canonical_to_source_candidate_indices"),
        dtype=torch.long,
        name="canonical_to_source_candidate_indices",
    )
    source_to_canonical = _tensor(
        target_free.get("source_to_canonical_candidate_indices"),
        dtype=torch.long,
        name="source_to_canonical_candidate_indices",
    )
    expected_permutation = list(range(len(pair_indices)))
    if (
        canonical_to_source.shape != (len(pair_indices),)
        or source_to_canonical.shape != (len(pair_indices),)
        or sorted(canonical_to_source.tolist()) != expected_permutation
        or sorted(source_to_canonical.tolist()) != expected_permutation
        or (
            len(pair_indices)
            and not torch.equal(
                source_to_canonical[canonical_to_source],
                torch.arange(len(pair_indices), dtype=torch.long),
            )
        )
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} canonical/source candidate mapping is invalid"
        )
    if global_features.shape != (SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,):
        raise SampleThresholdManifestError(
            f"{sample_label} global count feature width differs"
        )
    if raw_global_counts.shape != (SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,) or bool(
        (raw_global_counts < 0).any()
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} raw global counts are invalid"
        )
    if not bool(torch.isfinite(pair_tokens).all()) or not bool(
        torch.isfinite(global_features).all()
    ):
        raise SampleThresholdManifestError(f"{sample_label} features are non-finite")

    raw_pair_gains = pair_tokens[:, -1]
    anchor_count = int(raw_global_counts[0])
    try:
        expected_global = sample_threshold_global_count_features(
            anchor_count,
            pair_indices,
            raw_pair_gains,
        )
    except ValueError as error:
        raise SampleThresholdManifestError(
            f"{sample_label} pair geometry/count inputs are invalid"
        ) from error
    above = raw_pair_gains > SAMPLE_THRESHOLD_BASELINE_TAU
    incident_count = (
        int(torch.unique(pair_indices[above].reshape(-1)).numel())
        if bool(above.any())
        else 0
    )
    expected_raw_counts = torch.tensor(
        (anchor_count, len(pair_indices), int(above.sum()), incident_count),
        dtype=torch.long,
    )
    if not torch.equal(raw_global_counts, expected_raw_counts):
        raise SampleThresholdManifestError(
            f"{sample_label} raw global counts differ from pair inputs"
        )
    if not torch.allclose(global_features, expected_global, rtol=0.0, atol=1e-7):
        raise SampleThresholdManifestError(
            f"{sample_label} global count features differ from raw target-free counts"
        )

    loss_eligible = target_aware.get("loss_eligible")
    if not isinstance(loss_eligible, bool):
        raise SampleThresholdManifestError(
            f"{sample_label} loss eligibility flag must be boolean"
        )
    exclusion_reason = target_aware.get("exclusion_reason")
    if exclusion_reason is not None and (
        not isinstance(exclusion_reason, str) or not exclusion_reason
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} exclusion_reason must be a non-empty string or null"
        )
    if loss_eligible != (exclusion_reason is None):
        raise SampleThresholdManifestError(
            f"{sample_label} loss eligibility and exclusion reason differ"
        )
    if exclusion_reason not in {None, "candidate_unreachable", "scalar_unreachable"}:
        raise SampleThresholdManifestError(
            f"{sample_label} has an unsupported loss exclusion reason"
        )
    candidate_reachable = target_aware.get("candidate_reachable")
    if not isinstance(candidate_reachable, bool) or candidate_reachable != (
        exclusion_reason != "candidate_unreachable"
    ):
        raise SampleThresholdManifestError(
            f"{sample_label} candidate reachability metadata differs"
        )
    scalar_reachable = target_aware.get("scalar_reachable_target_consistent")
    if not isinstance(scalar_reachable, bool) or scalar_reachable != loss_eligible:
        raise SampleThresholdManifestError(
            f"{sample_label} scalar reachability metadata differs"
        )
    if target_aware.get("early_certificate_used") is not False or target_aware.get(
        "parametric_hull_fully_enumerated"
    ) is not True:
        raise SampleThresholdManifestError(
            f"{sample_label} does not contain a full non-early hull label"
        )

    intervals = target_aware.get("oracle_equivalent_robust_intervals")
    if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes)):
        raise SampleThresholdManifestError(
            f"{sample_label} oracle_intervals must be a sequence"
        )
    parsed_intervals = [
        _validate_interval(interval, sample_label=sample_label)
        for interval in intervals
    ]
    if loss_eligible and not parsed_intervals:
        raise SampleThresholdManifestError(
            f"{sample_label} eligible record needs an oracle interval"
        )
    if not loss_eligible and parsed_intervals:
        raise SampleThresholdManifestError(
            f"{sample_label} excluded record cannot enter interval loss"
        )
    lower = torch.tensor([item[0] for item in parsed_intervals], dtype=torch.float32)
    upper = torch.tensor([item[1] for item in parsed_intervals], dtype=torch.float32)
    scale = torch.tensor([item[2] for item in parsed_intervals], dtype=torch.float32)
    return SampleThresholdIntervalRecord(
        identity=dict(identity),
        partition=partition,
        anchor_count=anchor_count,
        pair_tokens=pair_tokens,
        pair_indices=pair_indices,
        global_count_features=global_features,
        interval_lower=lower,
        interval_upper=upper,
        interval_normalization_scale=scale,
        loss_eligible=loss_eligible,
        exclusion_reason=exclusion_reason,
    )


def load_sample_threshold_interval_manifest(
    path: str | Path,
    *,
    require_formal_counts: bool = True,
    audit_path: str | Path | None = None,
    parse_partitions: Sequence[str] = ("fit", "calibration"),
) -> SampleThresholdIntervalManifest:
    requested_partitions = tuple(parse_partitions)
    if (
        not requested_partitions
        or len(set(requested_partitions)) != len(requested_partitions)
        or any(name not in {"fit", "calibration"} for name in requested_partitions)
    ):
        raise ValueError("parse_partitions must contain unique fit/calibration names")
    resolved = _train_only_manifest_path(path)
    manifest_sha256 = file_sha256(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SampleThresholdManifestError("threshold manifest must be a mapping")
    expected_metadata = {
        "schema_version": INTERVAL_MANIFEST_SCHEMA_VERSION,
        "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
        "fresh_read": False,
        "validation_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    mismatches = {
        name: (payload.get(name), expected)
        for name, expected in expected_metadata.items()
        if payload.get(name) != expected
    }
    if mismatches:
        raise SampleThresholdManifestError(
            f"threshold manifest metadata differs from protocol: {mismatches}"
        )
    if payload.get("early_shortcut_enabled") is not False:
        raise SampleThresholdManifestError(
            "threshold manifest must disable the v0.9.15 early shortcut"
        )
    feature_contract = payload.get("feature_contract")
    expected_feature_contract = {
        "pair_token_dim": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
        "pair_token_feature_names": list(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES),
        "normalized_descriptor_dim": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM - 1,
        "normalized_descriptor_feature_names": list(
            SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES[:-1]
        ),
        "descriptor_source": (
            "frozen_pair_scorer_transform_plus_checkpoint_mean_scale"
        ),
        "raw_pair_gain_index": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM - 1,
        "pair_order": "lexicographic_first_anchor_id_then_second_anchor_id",
        "candidate_index_mapping": (
            "target_free stores canonical_to_source and source_to_canonical"
        ),
        "global_count_feature_dim": SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
        "raw_global_count_feature_names": list(_RAW_GLOBAL_COUNT_FEATURE_NAMES),
        "global_count_feature_names": list(
            SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
        ),
        "global_count_transform": "log1p(raw_count)",
        "gain_above_tau0_comparison": "strict_greater_than",
    }
    if not isinstance(feature_contract, Mapping):
        raise SampleThresholdManifestError(
            "threshold manifest feature_contract is missing"
        )
    feature_mismatches = {
        name: (feature_contract.get(name), expected)
        for name, expected in expected_feature_contract.items()
        if feature_contract.get(name) != expected
    }
    if feature_mismatches:
        raise SampleThresholdManifestError(
            "threshold manifest feature contract differs from predictor: "
            f"{feature_mismatches}"
        )
    normalization = payload.get("frozen_descriptor_normalization")
    if not isinstance(normalization, Mapping):
        raise SampleThresholdManifestError(
            "threshold manifest descriptor normalization is missing"
        )
    descriptor_mean = _tensor(
        normalization.get("mean"), dtype=torch.float32, name="descriptor mean"
    )
    descriptor_scale = _tensor(
        normalization.get("scale"), dtype=torch.float32, name="descriptor scale"
    )
    descriptor_dim = SAMPLE_THRESHOLD_PAIR_TOKEN_DIM - 1
    if (
        descriptor_mean.shape != (descriptor_dim,)
        or descriptor_scale.shape != (descriptor_dim,)
        or not bool(torch.isfinite(descriptor_mean).all())
        or not bool(torch.isfinite(descriptor_scale).all())
        or bool((descriptor_scale <= 0.0).any())
    ):
        raise SampleThresholdManifestError(
            "threshold manifest descriptor normalization is invalid"
        )
    protocol = payload.get("protocol")
    expected_protocol = {
        "early_capability_certificate_allowed": False,
        "multiple_structurally_equivalent_intervals_retained": True,
        "candidate_unreachable_excluded_from_loss": True,
        "scalar_unreachable_excluded_from_loss": True,
        "loss_boundary_inset_fraction": (
            SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION
        ),
        "finite_interval_normalization_scale": "upper-lower",
        "one_sided_interval_normalization_scale": (
            "max(1,abs(finite_boundary-tau0))"
        ),
        "fully_unbounded_interval_normalization_scale": 1.0,
    }
    if not isinstance(protocol, Mapping) or any(
        protocol.get(name) != expected
        for name, expected in expected_protocol.items()
    ):
        raise SampleThresholdManifestError(
            "threshold manifest interval protocol differs from predictor loss"
        )
    if require_formal_counts and payload.get("formal_eligible") is not True:
        raise SampleThresholdManifestError(
            "threshold predictor training requires a formal complete manifest"
        )
    manifest_audit_path: Path | None = None
    manifest_audit_sha256: str | None = None
    manifest_audit: Mapping[str, Any] | None = None
    manifest_commitment_path: Path | None = None
    manifest_commitment_sha256: str | None = None
    manifest_commitment: Mapping[str, Any] | None = None
    if require_formal_counts:
        current_implementation = interval_manifest_module._implementation_sha256()
        expected_source_sha = {
            "base_cache": interval_manifest_module.BASE_CACHE_SHA256,
            "augmentation_cache": interval_manifest_module.AUGMENTATION_CACHE_SHA256,
            "checkpoint": interval_manifest_module.CHECKPOINT_SHA256,
            "manifest_implementation": current_implementation[
                "interval_manifest"
            ],
        }
        if payload.get("source_sha256") != expected_source_sha:
            raise SampleThresholdManifestError(
                "threshold manifest source SHA differs from committed constants"
            )
        if payload.get("implementation_sha256") != current_implementation:
            raise SampleThresholdManifestError(
                "threshold manifest implementation SHA differs from live code"
            )
        manifest_input = payload.get("input")
        if not isinstance(manifest_input, Mapping) or manifest_input.get(
            "sha256"
        ) != {
            "base_cache": interval_manifest_module.BASE_CACHE_SHA256,
            "augmentation_cache": interval_manifest_module.AUGMENTATION_CACHE_SHA256,
            "checkpoint": interval_manifest_module.CHECKPOINT_SHA256,
        }:
            raise SampleThresholdManifestError(
                "threshold manifest input SHA differs from committed constants"
            )
        for name, path_name in (
            ("base_cache", "base_cache_path"),
            ("augmentation_cache", "augmentation_cache_path"),
            ("checkpoint", "checkpoint_path"),
        ):
            source_path = manifest_input.get(path_name)
            if not isinstance(source_path, str) or not source_path:
                raise SampleThresholdManifestError(
                    f"threshold manifest {path_name} provenance is invalid"
                )
            live_path = _train_only_manifest_path(source_path)
            if not live_path.is_file() or file_sha256(live_path) != expected_source_sha[name]:
                raise SampleThresholdManifestError(
                    f"threshold manifest live {name} SHA differs"
                )
        manifest_audit_path = _train_only_manifest_path(
            audit_path or DEFAULT_MANIFEST_AUDIT
        )
        try:
            loaded_audit = json.loads(
                manifest_audit_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise SampleThresholdManifestError(
                "threshold manifest audit is missing or malformed"
            ) from error
        if not isinstance(loaded_audit, Mapping):
            raise SampleThresholdManifestError(
                "threshold manifest audit must be a mapping"
            )
        expected_audit = {
            "manifest_schema_version": INTERVAL_MANIFEST_SCHEMA_VERSION,
            "status": "TRAIN-CAL-INTERVAL-MANIFEST-COMPLETE",
            "formal_eligible": True,
            "manifest_sha256": manifest_sha256,
        }
        audit_mismatches = {
            name: (loaded_audit.get(name), expected)
            for name, expected in expected_audit.items()
            if loaded_audit.get(name) != expected
        }
        commitment = loaded_audit.get("commitment")
        if audit_mismatches or not isinstance(commitment, Mapping) or (
            commitment.get("status") != "committed"
            or commitment.get("summary_binds_manifest_sha256") is not True
            or commitment.get("completed_build_lock_required") is not True
            or commitment.get("no_overwrite") is not True
        ):
            raise SampleThresholdManifestError(
                "threshold manifest audit does not commit the loaded manifest"
            )
        manifest_audit = dict(loaded_audit)
        manifest_audit_sha256 = file_sha256(manifest_audit_path)
        commitment_protocol = payload.get("commitment_protocol")
        if not isinstance(commitment_protocol, Mapping):
            raise SampleThresholdManifestError(
                "threshold manifest commitment protocol is missing"
            )
        expected_commitment_protocol = {
            "manifest_path": str(resolved),
            "summary_path": str(manifest_audit_path),
            "completed_build_lock_path": str(
                resolved.with_suffix(resolved.suffix + ".lock")
            ),
            "summary_binds_manifest_sha256": True,
            "completed_build_lock_required_for_training": True,
            "no_overwrite": True,
        }
        if dict(commitment_protocol) != expected_commitment_protocol:
            raise SampleThresholdManifestError(
                "threshold manifest commitment protocol paths differ"
            )
        manifest_commitment_path = Path(
            commitment_protocol["completed_build_lock_path"]
        ).resolve(strict=False)
        try:
            loaded_commitment = json.loads(
                manifest_commitment_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise SampleThresholdManifestError(
                "threshold manifest completed build lock is missing or malformed"
            ) from error
        expected_commitment = {
            "status": "completed",
            "manifest_path": str(resolved),
            "manifest_sha256": manifest_sha256,
            "summary_path": str(manifest_audit_path),
            "summary_sha256": manifest_audit_sha256,
            "formal_eligible": True,
            "source_sha256": expected_source_sha,
        }
        commitment_mismatches = {
            name: (loaded_commitment.get(name), expected)
            for name, expected in expected_commitment.items()
            if loaded_commitment.get(name) != expected
        }
        if commitment_mismatches:
            raise SampleThresholdManifestError(
                "threshold manifest completed build lock differs from artifacts"
            )
        manifest_commitment = dict(loaded_commitment)
        manifest_commitment_sha256 = file_sha256(manifest_commitment_path)
    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, Mapping) or set(raw_partitions) != {
        "fit",
        "calibration",
    }:
        raise SampleThresholdManifestError(
            "threshold manifest must contain only fit and calibration"
        )
    partitions: dict[str, SampleThresholdManifestPartition] = {}
    for name in ("fit", "calibration"):
        raw_partition = raw_partitions[name]
        if isinstance(raw_partition, Mapping):
            raw_records = raw_partition.get("records")
        else:
            raw_records = raw_partition
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes)
        ):
            raise SampleThresholdManifestError(
                f"threshold manifest {name} records must be a sequence"
            )
        if require_formal_counts and (
            len(raw_records) != EXPECTED_PARTITION_SAMPLE_COUNTS[name]
            or not isinstance(raw_partition, Mapping)
            or raw_partition.get("sample_count")
            != EXPECTED_PARTITION_SAMPLE_COUNTS[name]
            or raw_partition.get("complete_partition") is not True
        ):
            raise SampleThresholdManifestError(
                f"threshold manifest {name} count differs from formal protocol"
            )
        if name not in requested_partitions:
            continue
        records = tuple(
            _parse_record(record, partition=name, index=index)
            for index, record in enumerate(raw_records)
        )
        if isinstance(raw_partition, Mapping):
            expected_indices = {
                "loss_eligible_indices": [
                    index for index, record in enumerate(records) if record.loss_eligible
                ],
                "candidate_unreachable_indices": [
                    index
                    for index, record in enumerate(records)
                    if record.exclusion_reason == "candidate_unreachable"
                ],
                "scalar_unreachable_indices": [
                    index
                    for index, record in enumerate(records)
                    if record.exclusion_reason == "scalar_unreachable"
                ],
            }
            for field, expected in expected_indices.items():
                declared = _tensor(
                    raw_partition.get(field), dtype=torch.long, name=field
                )
                if declared.ndim != 1 or declared.tolist() != expected:
                    raise SampleThresholdManifestError(
                        f"threshold manifest {name} {field} differs from records"
                    )
        partitions[name] = SampleThresholdManifestPartition(name, records)
    return SampleThresholdIntervalManifest(
        path=resolved,
        sha256=manifest_sha256,
        payload=payload,
        partitions=partitions,
        audit_path=manifest_audit_path,
        audit_sha256=manifest_audit_sha256,
        audit=manifest_audit,
        commitment_path=manifest_commitment_path,
        commitment_sha256=manifest_commitment_sha256,
        commitment=manifest_commitment,
    )


def collate_sample_threshold_interval_records(
    records: Sequence[SampleThresholdIntervalRecord],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty threshold-predictor batch")
    if any(not record.loss_eligible for record in records):
        raise ValueError("loss-excluded records cannot enter predictor loss")
    pair_capacity = max(max(record.candidate_pair_count for record in records), 1)
    interval_capacity = max(max(record.interval_count for record in records), 1)
    batch = len(records)
    result: dict[str, Any] = {
        "pair_tokens": torch.zeros(
            (batch, pair_capacity, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM),
            dtype=torch.float32,
        ),
        "pair_mask": torch.zeros((batch, pair_capacity), dtype=torch.bool),
        "global_count_features": torch.stack(
            [record.global_count_features for record in records]
        ),
        "interval_lower": torch.full(
            (batch, interval_capacity), -torch.inf, dtype=torch.float32
        ),
        "interval_upper": torch.full(
            (batch, interval_capacity), torch.inf, dtype=torch.float32
        ),
        "interval_normalization_scale": torch.ones(
            (batch, interval_capacity), dtype=torch.float32
        ),
        "interval_mask": torch.zeros((batch, interval_capacity), dtype=torch.bool),
        "identity": [dict(record.identity) for record in records],
    }
    for index, record in enumerate(records):
        pairs = record.candidate_pair_count
        intervals = record.interval_count
        if pairs:
            order = _canonical_pair_order(record.pair_indices)
            result["pair_tokens"][index, :pairs] = record.pair_tokens[order]
            result["pair_mask"][index, :pairs] = True
        result["interval_lower"][index, :intervals] = record.interval_lower
        result["interval_upper"][index, :intervals] = record.interval_upper
        result["interval_normalization_scale"][index, :intervals] = (
            record.interval_normalization_scale
        )
        result["interval_mask"][index, :intervals] = True
    return result


def _canonical_pair_order(pair_indices: torch.Tensor) -> torch.Tensor:
    if pair_indices.ndim != 2 or pair_indices.size(-1) != 2:
        raise ValueError("pair_indices must have shape [pairs,2]")
    order = sorted(
        range(len(pair_indices)),
        key=lambda index: (
            int(pair_indices[index, 0]),
            int(pair_indices[index, 1]),
        ),
    )
    return torch.tensor(order, dtype=torch.long)


def deterministic_fit_batch_indices(
    record_count: int,
    *,
    steps: int,
    batch_size: int = TRAINING_BATCH_SIZE,
    seed: int = TRAINING_SEED,
) -> Iterator[tuple[int, ...]]:
    if record_count < 1 or steps < 1 or batch_size < 1:
        raise ValueError("record_count, steps, and batch_size must be positive")
    rng = random.Random(seed)
    order: list[int] = []
    cursor = 0
    for _ in range(steps):
        batch: list[int] = []
        while len(batch) < batch_size:
            if cursor == len(order):
                order = list(range(record_count))
                rng.shuffle(order)
                cursor = 0
            take = min(batch_size - len(batch), len(order) - cursor)
            batch.extend(order[cursor : cursor + take])
            cursor += take
        yield tuple(batch)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def train_sample_threshold_predictor_steps(
    model: EvidenceCarrierSampleThresholdPredictor,
    optimizer: torch.optim.Optimizer,
    fit_records: Sequence[SampleThresholdIntervalRecord],
    *,
    steps: int,
    batch_size: int,
    seed: int,
    max_grad_norm: float,
    device: torch.device,
) -> dict[str, Any]:
    """Run deterministic fit-only steps; exposed separately for unit tests."""

    if not fit_records or any(record.partition != "fit" for record in fit_records):
        raise ValueError("threshold predictor optimizer may consume only fit records")
    if any(not record.loss_eligible for record in fit_records):
        raise ValueError("loss-excluded records must be excluded before training")
    model.train()
    losses: list[float] = []
    gradient_norms: list[float] = []
    orders = deterministic_fit_batch_indices(
        len(fit_records), steps=steps, batch_size=batch_size, seed=seed
    )
    for indices in orders:
        batch = _to_device(
            collate_sample_threshold_interval_records(
                [fit_records[index] for index in indices]
            ),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["pair_tokens"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        loss_output = robust_open_interval_loss(
            output["tau"],
            batch["interval_lower"],
            batch["interval_upper"],
            batch["interval_mask"],
            batch["interval_normalization_scale"],
        )
        loss = loss_output["loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("threshold predictor produced a non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(torch.as_tensor(gradient_norm).detach().cpu()))
    return {
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "mean_loss": math.fsum(losses) / len(losses),
        "final_loss": losses[-1],
        "maximum_preclip_gradient_norm": max(gradient_norms),
    }


def _identity_key(record: Mapping[str, Any]) -> tuple[str, str]:
    identity = record.get("identity", {})
    if not isinstance(identity, Mapping):
        raise ValueError("calibration record identity is missing")
    return (
        str(identity.get("sample_key", "")),
        str(identity.get("molecule_id", "")),
    )


def _predictor_batches(
    records: Sequence[SampleThresholdIntervalRecord],
    *,
    batch_size: int,
) -> Iterator[Sequence[SampleThresholdIntervalRecord]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_tau_predictor_capability_gate(
    pair_scorer_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep matching latency diagnostic, not part of the capability decision."""

    excluded_checks = {"matching_p95_below_20ms"}
    checks = dict(pair_scorer_gate["checks"])
    diagnostic_checks = {
        name: checks.pop(name) for name in excluded_checks if name in checks
    }
    thresholds = dict(pair_scorer_gate["thresholds"])
    diagnostic_thresholds = {}
    if "maximum_matching_p95_ms" in thresholds:
        diagnostic_thresholds["maximum_matching_p95_ms"] = thresholds.pop(
            "maximum_matching_p95_ms"
        )
    return {
        **dict(pair_scorer_gate),
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "thresholds": thresholds,
        "diagnostic_checks": diagnostic_checks,
        "diagnostic_thresholds": diagnostic_thresholds,
        "excluded_from_capability_status": sorted(excluded_checks),
        "base_pair_scorer_gate_status": pair_scorer_gate["status"],
    }


@torch.inference_mode()
def evaluate_sample_threshold_calibration_once(
    model: EvidenceCarrierSampleThresholdPredictor,
    manifest_records: Sequence[SampleThresholdIntervalRecord],
    source_records: Sequence[Mapping[str, Any]],
    *,
    calibration_component_count: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one fixed calibration pass with no checkpoint or threshold search."""

    if not manifest_records or len(manifest_records) != len(source_records):
        raise ValueError("calibration manifest/source record counts differ or are empty")
    if any(record.partition != "calibration" for record in manifest_records):
        raise ValueError("one-shot calibration accepts only calibration records")
    for manifest_record, source_record in zip(
        manifest_records, source_records, strict=True
    ):
        if _identity_key({"identity": manifest_record.identity}) != _identity_key(
            source_record
        ):
            raise ValueError("calibration manifest/source identity order differs")
        source_pairs = _tensor(
            source_record.get("pair_indices"),
            dtype=torch.long,
            name="source pair_indices",
        )
        if {
            tuple(map(int, pair)) for pair in manifest_record.pair_indices.tolist()
        } != {tuple(map(int, pair)) for pair in source_pairs.tolist()}:
            raise ValueError("calibration manifest/source candidate pairs differ")

    model.eval()
    predicted_tau_parts: list[torch.Tensor] = []
    predicted_delta_parts: list[torch.Tensor] = []
    latency_per_sample_ms: list[float] = []
    for current in _predictor_batches(manifest_records, batch_size=batch_size):
        batch = _to_device(collate_sample_threshold_interval_records_for_inference(current), device)
        _synchronize(device)
        started = time.perf_counter()
        output = model(
            batch["pair_tokens"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_per_sample_ms.extend([elapsed_ms / len(current)] * len(current))
        predicted_tau_parts.append(output["tau"].detach().cpu())
        predicted_delta_parts.append(output["delta"].detach().cpu())
    predicted_tau = torch.cat(predicted_tau_parts)
    predicted_delta = torch.cat(predicted_delta_parts)
    if not bool(torch.isfinite(predicted_tau).all()):
        raise RuntimeError("threshold predictor produced non-finite calibration tau")

    interval_hits: list[bool | None] = []
    interval_violations: list[float | None] = []
    sample_evaluations: list[dict[str, Any]] = []
    structural_evaluations: list[dict[str, Any]] = []
    for index, (manifest_record, source_record) in enumerate(
        zip(manifest_records, source_records, strict=True)
    ):
        tau = float(predicted_tau[index])
        if manifest_record.loss_eligible:
            tau_tensor = predicted_tau[index : index + 1]
            interval_output = robust_open_interval_loss(
                tau_tensor,
                manifest_record.interval_lower[None],
                manifest_record.interval_upper[None],
                torch.ones(
                    (1, manifest_record.interval_count), dtype=torch.bool
                ),
                manifest_record.interval_normalization_scale[None],
            )
            violation = float(interval_output["per_sample_loss"][0])
            interval_hit: bool | None = violation == 0.0
        else:
            violation = None
            interval_hit = None
        canonical_gain_by_pair = {
            tuple(map(int, pair)): float(gain)
            for pair, gain in zip(
                manifest_record.pair_indices.tolist(),
                manifest_record.pair_tokens[:, -1].tolist(),
                strict=True,
            )
        }
        raw_gains = torch.tensor(
            [
                canonical_gain_by_pair[tuple(map(int, pair))]
                for pair in source_record["pair_indices"].tolist()
            ],
            dtype=torch.float64,
        )
        evaluation = evaluate_evidence_carrier_pair_sample(
            source_record,
            raw_gains,
            threshold=tau,
            verify_permutation=True,
        )
        structural_evaluations.append(evaluation)
        interval_hits.append(interval_hit)
        interval_violations.append(violation)
        sample_evaluations.append(
            {
                "identity": dict(manifest_record.identity),
                "predicted_tau": tau,
                "predicted_delta": float(predicted_delta[index]),
                "loss_eligible": manifest_record.loss_eligible,
                "loss_exclusion_reason": manifest_record.exclusion_reason,
                "robust_interval_hit": interval_hit,
                "normalized_interval_violation": violation,
                "structural_evaluation": evaluation,
            }
        )

    aggregate = aggregate_evidence_carrier_pair_evaluations(
        structural_evaluations
    )
    gate = sample_tau_predictor_capability_gate(
        pair_scorer_train_cal_gate(
            {"selected_metrics": aggregate},
            calibration_records=source_records,
            calibration_component_count=calibration_component_count,
        )
    )
    eligible_violations = [
        float(value) for value in interval_violations if value is not None
    ]
    hit_count = sum(value is True for value in interval_hits)
    eligible_count = len(eligible_violations)
    latency = np.asarray(latency_per_sample_ms, dtype=np.float64)
    tau_values = predicted_tau.numpy().astype(np.float64, copy=False)
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": f"CALIBRATION-TAU-PREDICTOR-{gate['status']}",
        "one_shot": True,
        "threshold_search": False,
        "checkpoint_selection": False,
        "sample_count": len(manifest_records),
        "selected_metrics": aggregate,
        "gate": gate,
        "interval_metrics": {
            "loss_eligible_sample_count": eligible_count,
            "excluded_sample_count": len(manifest_records) - eligible_count,
            "robust_interval_hit_count": hit_count,
            "robust_interval_hit_rate": (
                hit_count / eligible_count if eligible_count else None
            ),
            "mean_normalized_interval_violation": (
                math.fsum(eligible_violations) / eligible_count
                if eligible_count
                else None
            ),
            "maximum_normalized_interval_violation": (
                max(eligible_violations) if eligible_count else None
            ),
        },
        "predicted_tau": {
            "minimum": float(tau_values.min()),
            "mean": float(tau_values.mean()),
            "p50": float(np.percentile(tau_values, 50)),
            "p95": float(np.percentile(tau_values, 95)),
            "maximum": float(tau_values.max()),
        },
        "predictor_latency_ms_per_sample": {
            "mean": float(latency.mean()),
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "maximum": float(latency.max()),
        },
        "sample_evaluations": sample_evaluations,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }


def collate_sample_threshold_interval_records_for_inference(
    records: Sequence[SampleThresholdIntervalRecord],
) -> dict[str, Any]:
    """Collate all records without exposing oracle intervals to the model."""

    if not records:
        raise ValueError("cannot collate an empty threshold-predictor batch")
    pair_capacity = max(max(record.candidate_pair_count for record in records), 1)
    batch = len(records)
    pair_tokens = torch.zeros(
        (batch, pair_capacity, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM), dtype=torch.float32
    )
    pair_mask = torch.zeros((batch, pair_capacity), dtype=torch.bool)
    for index, record in enumerate(records):
        count = record.candidate_pair_count
        if count:
            order = _canonical_pair_order(record.pair_indices)
            pair_tokens[index, :count] = record.pair_tokens[order]
            pair_mask[index, :count] = True
    return {
        "pair_tokens": pair_tokens,
        "pair_mask": pair_mask,
        "global_count_features": torch.stack(
            [record.global_count_features for record in records]
        ),
    }


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _plain_json(payload),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                _plain_json(payload),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _begin_calibration_consumption(
    marker_path: str | Path,
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> Path:
    """Exclusively consume the one-shot calibration protocol."""

    marker = Path(marker_path).expanduser().resolve(strict=False)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "evidence-carrier-tau-calibration-consumption-0.1",
        "status": "started",
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256_before_calibration": checkpoint_sha256,
        "protocol": {
            "optimizer_steps": TRAINING_STEPS,
            "checkpoint_selection": "last_step_only",
            "calibration_evaluation_count": 1,
            "threshold_search": False,
            "matching_latency_capability_gate": False,
        },
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        marker,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("calibration marker write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return marker


def _acquire_training_output_lock(output_dir: str | Path) -> Path:
    root = Path(output_dir).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "run.lock"
    payload = {
        "schema_version": "evidence-carrier-tau-predictor-run-lock-0.1",
        "status": "started",
        "output_dir": str(root),
        "protocol": {
            "optimizer_steps": TRAINING_STEPS,
            "checkpoint_selection": "last_step_only",
            "calibration_evaluation_count": 1,
        },
    }
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        lock_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("training lock write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock_path


def _complete_training_output_lock(
    lock_path: str | Path,
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    calibration_path: str | Path,
    calibration_sha256: str,
    summary_path: str | Path,
    summary_sha256: str,
    calibration_consumption_marker: str | Path,
    calibration_consumption_marker_sha256: str,
) -> str:
    lock = Path(lock_path).expanduser().resolve(strict=True)
    started = json.loads(lock.read_text(encoding="utf-8"))
    if started.get("status") != "started":
        raise RuntimeError("training output lock is not in started state")
    complete = {
        **started,
        "status": "complete",
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_path": str(Path(calibration_path).resolve()),
        "calibration_sha256": calibration_sha256,
        "summary_path": str(Path(summary_path).resolve()),
        "summary_sha256": summary_sha256,
        "calibration_consumption_marker": str(
            Path(calibration_consumption_marker).resolve()
        ),
        "calibration_consumption_marker_sha256": (
            calibration_consumption_marker_sha256
        ),
    }
    _write_json(lock, complete)
    return file_sha256(lock)


def _complete_calibration_consumption(
    marker_path: str | Path,
    *,
    calibration_path: str | Path,
    calibration_sha256: str,
    checkpoint_sha256_after_calibration: str,
    calibration_status: str,
) -> str:
    marker = Path(marker_path).expanduser().resolve(strict=True)
    try:
        started = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("calibration consumption marker is unreadable") from error
    if started.get("status") != "started":
        raise RuntimeError("calibration consumption marker is not in started state")
    expected_checkpoint_sha = started.get("checkpoint_sha256_before_calibration")
    if checkpoint_sha256_after_calibration != expected_checkpoint_sha:
        raise RuntimeError("sealed checkpoint changed during calibration")
    complete = {
        **started,
        "status": "complete",
        "checkpoint_sha256_after_calibration": (
            checkpoint_sha256_after_calibration
        ),
        "checkpoint_sha256_unchanged": True,
        "calibration_path": str(Path(calibration_path).resolve()),
        "calibration_sha256": calibration_sha256,
        "calibration_status": calibration_status,
    }
    _write_json(marker, complete)
    return file_sha256(marker)


def load_sample_threshold_predictor_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_provenance_hashes: bool = True,
) -> tuple[EvidenceCarrierSampleThresholdPredictor, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=False)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("sample-threshold checkpoint must be a mapping")
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierSampleThresholdPredictor.model_type,
        "parameter_count": SAMPLE_THRESHOLD_PREDICTOR_PARAMETER_COUNT,
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "calibration_used_for_training": False,
        "calibration_used_for_checkpoint_selection": False,
        "calibration_evaluated": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"sample-threshold checkpoint contract differs: {mismatches}")
    model = EvidenceCarrierSampleThresholdPredictor()
    if checkpoint.get("model_config") != model.config_dict():
        raise ValueError("sample-threshold checkpoint model config differs")
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("sample-threshold checkpoint weights are invalid") from error
    implementation = checkpoint.get("implementation_sha256")
    if not isinstance(implementation, Mapping) or implementation != (
        predictor_implementation_sha256()
    ):
        raise ValueError("sample-threshold checkpoint implementation SHA differs")
    provenance = (
        ("manifest_path", "manifest_sha256"),
        ("manifest_audit_path", "manifest_audit_sha256"),
        ("manifest_commitment_path", "manifest_commitment_sha256"),
        ("base_cache_path", "base_cache_sha256"),
        ("augmentation_cache_path", "augmentation_cache_sha256"),
        ("frozen_pair_scorer_checkpoint_path", "frozen_pair_scorer_checkpoint_sha256"),
    )
    for path_name, sha_name in provenance:
        source_path = checkpoint.get(path_name)
        source_sha = checkpoint.get(sha_name)
        if (
            not isinstance(source_path, str)
            or not source_path
            or not isinstance(source_sha, str)
            or len(source_sha) != 64
        ):
            raise ValueError(
                f"sample-threshold checkpoint provenance {path_name} is invalid"
            )
        if verify_provenance_hashes:
            live_path = _train_only_manifest_path(source_path)
            if not live_path.is_file() or file_sha256(live_path) != source_sha:
                raise ValueError(
                    f"sample-threshold checkpoint provenance {sha_name} differs"
                )
    expected_frozen_sha = {
        "base_cache_sha256": interval_manifest_module.BASE_CACHE_SHA256,
        "augmentation_cache_sha256": (
            interval_manifest_module.AUGMENTATION_CACHE_SHA256
        ),
        "frozen_pair_scorer_checkpoint_sha256": (
            interval_manifest_module.CHECKPOINT_SHA256
        ),
    }
    frozen_mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected_frozen_sha.items()
        if checkpoint.get(name) != value
    }
    if frozen_mismatches:
        raise ValueError(
            f"sample-threshold checkpoint frozen SHA differs: {frozen_mismatches}"
        )
    if checkpoint.get(
        "manifest_implementation_sha256"
    ) != interval_manifest_module._implementation_sha256():
        raise ValueError(
            "sample-threshold checkpoint manifest implementation SHA differs"
        )
    model.to(torch.device(device)).eval()
    return model, dict(checkpoint)


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"threshold predictor output directory is not empty: {output_dir}"
        )
    consumption_marker = Path(
        DEFAULT_CALIBRATION_CONSUMPTION_MARKER
    ).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_lock_path = _acquire_training_output_lock(output_dir)
    manifest = load_sample_threshold_interval_manifest(
        args.manifest,
        require_formal_counts=True,
        audit_path=DEFAULT_MANIFEST_AUDIT,
        parse_partitions=("fit",),
    )
    fit_partition = manifest.partitions["fit"]
    fit_records = fit_partition.loss_records
    if not fit_records:
        raise RuntimeError("threshold manifest has no loss-eligible fit records")
    manifest_input = manifest.payload["input"]
    source_hashes = manifest_input["sha256"]
    source_paths = {
        "base_cache": _train_only_manifest_path(
            manifest_input["base_cache_path"]
        ),
        "augmentation_cache": _train_only_manifest_path(
            manifest_input["augmentation_cache_path"]
        ),
        "checkpoint": _train_only_manifest_path(
            manifest_input["checkpoint_path"]
        ),
    }
    implementation_sha256 = predictor_implementation_sha256()

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = EvidenceCarrierSampleThresholdPredictor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_LEARNING_RATE,
        weight_decay=TRAINING_WEIGHT_DECAY,
    )
    metrics = train_sample_threshold_predictor_steps(
        model,
        optimizer,
        fit_records,
        steps=TRAINING_STEPS,
        batch_size=TRAINING_BATCH_SIZE,
        seed=TRAINING_SEED,
        max_grad_norm=TRAINING_MAX_GRAD_NORM,
        device=device,
    )
    if predictor_implementation_sha256() != implementation_sha256:
        raise RuntimeError(
            "sample-threshold predictor implementation changed during training"
        )
    raw_fit_partition = manifest.payload["partitions"]["fit"]
    raw_calibration_partition = manifest.payload["partitions"]["calibration"]
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": model.model_type,
        "model": model.state_dict(),
        "model_config": model.config_dict(),
        "parameter_count": model.parameter_count(),
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "training_protocol": {
            "optimizer": "AdamW",
            "learning_rate": TRAINING_LEARNING_RATE,
            "weight_decay": TRAINING_WEIGHT_DECAY,
            "batch_size": TRAINING_BATCH_SIZE,
            "max_grad_norm": TRAINING_MAX_GRAD_NORM,
            "seed": TRAINING_SEED,
            "learning_rate_schedule": "constant",
            "loss": "minimum_multi_interval_normalized_linear_hinge",
            "interval_inward_margin_fraction": (
                SAMPLE_THRESHOLD_INTERVAL_INWARD_MARGIN_FRACTION
            ),
        },
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "manifest_audit_path": str(manifest.audit_path),
        "manifest_audit_sha256": manifest.audit_sha256,
        "manifest_commitment_path": str(manifest.commitment_path),
        "manifest_commitment_sha256": manifest.commitment_sha256,
        "manifest_schema_version": INTERVAL_MANIFEST_SCHEMA_VERSION,
        "manifest_implementation_sha256": manifest.payload[
            "implementation_sha256"
        ],
        "implementation_sha256": implementation_sha256,
        "base_cache_path": str(source_paths["base_cache"]),
        "base_cache_sha256": source_hashes["base_cache"],
        "augmentation_cache_path": str(source_paths["augmentation_cache"]),
        "augmentation_cache_sha256": source_hashes["augmentation_cache"],
        "frozen_pair_scorer_checkpoint_path": str(source_paths["checkpoint"]),
        "frozen_pair_scorer_checkpoint_sha256": source_hashes["checkpoint"],
        "fit_sample_count": len(fit_partition.records),
        "fit_loss_sample_count": len(fit_records),
        "fit_candidate_unreachable_count": int(
            raw_fit_partition["candidate_unreachable_sample_count"]
        ),
        "fit_scalar_unreachable_count": int(
            raw_fit_partition["scalar_unreachable_sample_count"]
        ),
        "calibration_sample_count": int(raw_calibration_partition["sample_count"]),
        "calibration_used_for_training": False,
        "calibration_used_for_checkpoint_selection": False,
        "calibration_evaluated": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
        "training_metrics": metrics,
    }
    checkpoint_path = output_dir / "final.pt"
    temporary = checkpoint_path.with_suffix(f".pt.tmp.{os.getpid()}")
    try:
        torch.save(checkpoint, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)
    checkpoint_sha_before = file_sha256(checkpoint_path)
    marker_path = _begin_calibration_consumption(
        consumption_marker,
        manifest_path=manifest.path,
        manifest_sha256=manifest.sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha_before,
    )
    sealed_model, sealed_checkpoint = load_sample_threshold_predictor_checkpoint(
        checkpoint_path,
        device=device,
        verify_provenance_hashes=True,
    )
    if sealed_checkpoint["implementation_sha256"] != implementation_sha256:
        raise RuntimeError("sealed checkpoint implementation commitment changed")
    calibration_manifest = load_sample_threshold_interval_manifest(
        args.manifest,
        require_formal_counts=True,
        audit_path=DEFAULT_MANIFEST_AUDIT,
        parse_partitions=("calibration",),
    )
    if calibration_manifest.sha256 != manifest.sha256:
        raise RuntimeError("interval manifest changed before calibration consumption")
    base_cache, augmentation_cache = load_train_only_p0_pair_caches(
        source_paths["base_cache"], source_paths["augmentation_cache"]
    )
    calibration_source_records = joined_training_records(
        base_cache, augmentation_cache, "calibration"
    )
    component_count = base_cache.get("identity_partition", {}).get(
        "calibration_component_count"
    )
    calibration = evaluate_sample_threshold_calibration_once(
        sealed_model,
        calibration_manifest.partitions["calibration"].records,
        calibration_source_records,
        calibration_component_count=component_count,
        batch_size=TRAINING_BATCH_SIZE,
        device=device,
    )
    if predictor_implementation_sha256() != implementation_sha256:
        raise RuntimeError(
            "sample-threshold calibration implementation changed during execution"
        )
    checkpoint_sha_after = file_sha256(checkpoint_path)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("sealed checkpoint changed during one-shot calibration")
    calibration_payload = {
        **calibration,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before_calibration": checkpoint_sha_before,
        "checkpoint_sha256_after_calibration": checkpoint_sha_after,
        "checkpoint_sha256_unchanged": True,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "manifest_audit_path": str(manifest.audit_path),
        "manifest_audit_sha256": manifest.audit_sha256,
        "manifest_commitment_path": str(manifest.commitment_path),
        "manifest_commitment_sha256": manifest.commitment_sha256,
        "base_cache_path": str(source_paths["base_cache"]),
        "base_cache_sha256": source_hashes["base_cache"],
        "augmentation_cache_path": str(source_paths["augmentation_cache"]),
        "augmentation_cache_sha256": source_hashes["augmentation_cache"],
        "frozen_pair_scorer_checkpoint_path": str(source_paths["checkpoint"]),
        "frozen_pair_scorer_checkpoint_sha256": source_hashes["checkpoint"],
        "implementation_sha256": implementation_sha256,
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only_before_calibration",
        "calibration_used_for_optimizer": False,
        "calibration_used_for_checkpoint_selection": False,
    }
    calibration_path = Path(DEFAULT_CALIBRATION_OUTPUT).expanduser().resolve(
        strict=False
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_no_overwrite(calibration_path, calibration_payload)
    calibration_sha = file_sha256(calibration_path)
    marker_sha = _complete_calibration_consumption(
        marker_path,
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha,
        checkpoint_sha256_after_calibration=checkpoint_sha_after,
        calibration_status=calibration["status"],
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": calibration["status"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before_calibration": checkpoint_sha_before,
        "checkpoint_sha256_after_calibration": checkpoint_sha_after,
        "checkpoint_sha256_unchanged": True,
        "calibration_consumption_marker": str(marker_path),
        "calibration_consumption_marker_sha256": marker_sha,
        "training_output_lock": str(run_lock_path),
        "training_output_lock_completion_required": True,
        "calibration_path": str(calibration_path),
        "calibration_sha256": calibration_sha,
        "calibration_gate": calibration["gate"],
        "calibration_interval_metrics": calibration["interval_metrics"],
        "calibration_selected_metrics": calibration["selected_metrics"],
        "predictor_latency_ms_per_sample": calibration[
            "predictor_latency_ms_per_sample"
        ],
        "training_metrics": metrics,
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "calibration_evaluated": True,
        "calibration_evaluation_count": 1,
        "calibration_used_for_optimizer": False,
        "calibration_used_for_checkpoint_selection": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json_no_overwrite(summary_path, summary)
    summary_sha = file_sha256(summary_path)
    _complete_training_output_lock(
        run_lock_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha_after,
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha,
        summary_path=summary_path,
        summary_sha256=summary_sha,
        calibration_consumption_marker=marker_path,
        calibration_consumption_marker_sha256=marker_sha,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(_plain_json(train(args)), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_CALIBRATION_CONSUMPTION_MARKER",
    "DEFAULT_CALIBRATION_OUTPUT",
    "DEFAULT_MANIFEST",
    "DEFAULT_MANIFEST_AUDIT",
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_PARTITION_SAMPLE_COUNTS",
    "INTERVAL_MANIFEST_SCHEMA_VERSION",
    "SampleThresholdIntervalManifest",
    "SampleThresholdIntervalRecord",
    "SampleThresholdManifestError",
    "SampleThresholdManifestPartition",
    "TRAINING_BATCH_SIZE",
    "TRAINING_LEARNING_RATE",
    "TRAINING_MAX_GRAD_NORM",
    "TRAINING_SEED",
    "TRAINING_STEPS",
    "TRAINING_WEIGHT_DECAY",
    "collate_sample_threshold_interval_records",
    "deterministic_fit_batch_indices",
    "evaluate_sample_threshold_calibration_once",
    "file_sha256",
    "load_sample_threshold_interval_manifest",
    "load_sample_threshold_predictor_checkpoint",
    "predictor_implementation_sha256",
    "sample_tau_predictor_capability_gate",
    "train_sample_threshold_predictor_steps",
]
