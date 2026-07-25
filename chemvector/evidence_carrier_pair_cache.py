"""Train-only torch cache for Evidence Carrier conditional pair records.

The cache freezes expensive raw-LSD association and target-free ``training_wide``
pair generation before scorer training.  It scans every selected train render,
partitions connected molecule/source identities into fit and calibration, and
stores compact CPU tensors rather than raster images or full sidecars.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from braceexpand import braceexpand
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_DIM,
    PAIR_FEATURE_NAMES,
    PAIRING_PROFILE_TRAINING_WIDE,
    SCHEMA_VERSION as PAIR_DATA_SCHEMA_VERSION,
    ConditionalOrdinaryPairWebDataset,
    ConditionalPairSampleIdentity,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
    iter_conditional_pair_identities,
    partition_fit_calibration_identities,
    validate_molecule_disjoint_splits,
)
from chemvector.evidence_carrier_extent_data import (
    EvidenceCarrierExtentTargetConfig,
    reject_internal_test_shards,
)


CACHE_SCHEMA_VERSION = "evidence-carrier-pair-train-cache-0.1"
CACHE_FORMAT = "torch-dict-of-compact-records"
CACHE_SCOPE = "train-only"
CACHE_PAIRING_PROFILE = PAIRING_PROFILE_TRAINING_WIDE
DEFAULT_PARTITION_SEED = "chemvector-v0.9.10"
DEFAULT_PROGRESS_EVERY = 100

_RECORD_TENSOR_FIELDS = (
    "canvas_size",
    "anchor_segments",
    "anchor_segments_pixels",
    "anchor_source_indices",
    "anchor_target_line_ids",
    "anchor_group_ids",
    "anchor_group_types",
    "pair_indices",
    "pair_features",
    "pair_labels",
    "pair_target_line_ids",
    "pair_overlap_centerline",
    "pair_overlap_centerline_pixels",
)
_INTEGER_RECORD_FIELDS = (
    "canvas_size",
    "anchor_source_indices",
    "anchor_target_line_ids",
    "anchor_group_ids",
    "anchor_group_types",
    "pair_indices",
    "pair_labels",
    "pair_target_line_ids",
)
_FLOAT_RECORD_FIELDS = (
    "anchor_segments",
    "anchor_segments_pixels",
    "pair_features",
    "pair_overlap_centerline",
    "pair_overlap_centerline_pixels",
)


class EvidenceCarrierPairCacheError(RuntimeError):
    """Base error for malformed or damaged pair caches."""


class EvidenceCarrierPairCacheValidationError(EvidenceCarrierPairCacheError):
    """Raised when a cache contract or compact record is invalid."""


class EvidenceCarrierPairCacheIntegrityError(EvidenceCarrierPairCacheError):
    """Raised when a source shard no longer matches its frozen digest."""


def _contains_internal_test(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return "internal_test" in normalized


def _contains_validation_split(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return bool(re.search(r"(?:^|_)validation(?:_|$)", normalized))


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise ValueError("Pair cache limit must be an integer >= 2 or None")
    return limit


def _validate_progress_every(progress_every: int | None) -> int | None:
    if progress_every is None:
        return None
    if (
        isinstance(progress_every, bool)
        or not isinstance(progress_every, int)
        or progress_every < 1
    ):
        raise ValueError("Pair cache progress_every must be a positive integer or None")
    return progress_every


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_path_error(path: str | Path) -> None:
    raw = str(path)
    resolved = str(Path(raw).expanduser().resolve(strict=False))
    if _contains_internal_test(raw) or _contains_internal_test(resolved):
        raise ValueError(
            "Evidence Carrier pair cache must not read sealed internal_test shards"
        )
    if _contains_validation_split(raw) or _contains_validation_split(resolved):
        raise ValueError(
            "Evidence Carrier pair cache accepts train shards only; validation "
            f"source rejected: {raw}"
        )


def expand_evidence_carrier_pair_train_shards(
    shards: str | Sequence[str],
) -> tuple[Path, ...]:
    """Expand local train shard paths and reject validation/internal-test aliases."""

    values = (shards,) if isinstance(shards, str) else tuple(shards)
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("Pair cache shards must be one or more nonempty path strings")
    reject_internal_test_shards(values)
    expanded: list[Path] = []
    for pattern in values:
        if "://" in pattern:
            raise ValueError("Pair cache source shards must be local files for SHA256")
        _source_path_error(pattern)
        matches: list[str] = []
        for braced in braceexpand(pattern):
            current = sorted(glob.glob(braced))
            if current:
                matches.extend(current)
            elif Path(braced).expanduser().is_file():
                matches.append(braced)
        for match in matches:
            _source_path_error(match)
            resolved = Path(match).expanduser().resolve(strict=True)
            _source_path_error(resolved)
            if not resolved.is_file():
                raise ValueError(f"Pair cache source is not a file: {resolved}")
            expanded.append(resolved)
    unique = tuple(sorted(set(expanded), key=lambda path: str(path)))
    if not unique:
        raise ValueError(f"No train shards matched: {list(values)}")
    return unique


def scan_evidence_carrier_pair_train_identities(
    shards: str | Sequence[str],
    *,
    limit: int | None = None,
) -> tuple[ConditionalPairSampleIdentity, ...]:
    """Scan train metadata for all render variants without decoding rasters."""

    paths = expand_evidence_carrier_pair_train_shards(shards)
    identities = tuple(
        iter_conditional_pair_identities(
            [str(path) for path in paths],
            split="train",
            render_variant=None,
            limit=_validate_limit(limit),
        )
    )
    if not identities:
        raise ValueError("Pair cache train identity scan returned no samples")
    if any(identity.source_split != "train" for identity in identities):
        raise ValueError("Pair cache identity scan returned a non-train sample")
    return identities


def _identity_from_payload(value: Any) -> ConditionalPairSampleIdentity:
    if not isinstance(value, dict):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache record identity must be a dictionary"
        )
    required = {
        "sample_key",
        "molecule_id",
        "source_structure_id",
        "source_pilot",
        "source_split",
        "source_record",
        "render_variant",
    }
    if set(value) != required:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache identity fields differ from the frozen schema"
        )
    if not all(isinstance(value[name], str) and value[name] for name in required):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache identity values must be nonempty strings"
        )
    identity = ConditionalPairSampleIdentity(**value)
    if identity.source_split != "train":
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache records must come from train only"
        )
    expected_source = (
        f"{identity.source_pilot}|train|{identity.source_record}"
    )
    if identity.source_structure_id != expected_source:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache source_structure_id is inconsistent"
        )
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if _contains_internal_test(serialized) or _contains_validation_split(
        identity.source_structure_id
    ):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache identity references validation/internal_test"
        )
    return identity


def _cpu_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise EvidenceCarrierPairCacheValidationError(
            f"Pair cache record field {name!r} must be a tensor"
        )
    if value.device.type != "cpu":
        raise EvidenceCarrierPairCacheValidationError(
            f"Pair cache record field {name!r} must be on CPU"
        )
    return value


def validate_evidence_carrier_pair_record(record: Any) -> None:
    """Validate one unpadded compact train record in place."""

    if not isinstance(record, dict):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache records must be dictionaries"
        )
    required = {"identity", "partition", *_RECORD_TENSOR_FIELDS}
    missing = required - set(record)
    if missing:
        raise EvidenceCarrierPairCacheValidationError(
            f"Pair cache record fields are missing: {sorted(missing)}"
        )
    _identity_from_payload(record["identity"])
    if record["partition"] not in {"fit", "calibration"}:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache record partition must be fit or calibration"
        )
    tensors = {
        name: _cpu_tensor(record[name], name=name) for name in _RECORD_TENSOR_FIELDS
    }
    for name in _INTEGER_RECORD_FIELDS:
        if tensors[name].dtype != torch.long:
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache record field {name!r} must use torch.long"
            )
    for name in _FLOAT_RECORD_FIELDS:
        if tensors[name].dtype != torch.float32:
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache record field {name!r} must use torch.float32"
            )
        if not bool(torch.isfinite(tensors[name]).all()):
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache record field {name!r} contains non-finite values"
            )

    canvas = tensors["canvas_size"]
    if canvas.shape != (2,) or bool((canvas < 2).any()):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache canvas_size must contain width,height >= 2"
        )
    anchor_segments = tensors["anchor_segments"]
    anchor_pixels = tensors["anchor_segments_pixels"]
    if anchor_segments.ndim != 3 or anchor_segments.shape[1:] != (2, 2):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache anchor_segments must have shape [anchors,2,2]"
        )
    if anchor_pixels.shape != anchor_segments.shape:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache normalized and pixel anchor shapes differ"
        )
    anchor_count = len(anchor_segments)
    for name in (
        "anchor_source_indices",
        "anchor_target_line_ids",
        "anchor_group_ids",
        "anchor_group_types",
    ):
        if tensors[name].shape != (anchor_count,):
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache record field {name!r} has invalid anchor shape"
            )
        if bool((tensors[name] < 0).any()):
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache record field {name!r} must be nonnegative"
            )
    source_indices = tensors["anchor_source_indices"].tolist()
    if len(set(source_indices)) != len(source_indices):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache anchor_source_indices must be unique"
        )

    pair_indices = tensors["pair_indices"]
    pair_features = tensors["pair_features"]
    pair_labels = tensors["pair_labels"]
    pair_targets = tensors["pair_target_line_ids"]
    centerline = tensors["pair_overlap_centerline"]
    centerline_pixels = tensors["pair_overlap_centerline_pixels"]
    if pair_indices.ndim != 2 or pair_indices.shape[1:] != (2,):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache pair_indices must have shape [pairs,2]"
        )
    pair_count = len(pair_indices)
    expected_shapes = {
        "pair_features": (pair_count, PAIR_FEATURE_DIM),
        "pair_labels": (pair_count,),
        "pair_target_line_ids": (pair_count, 2),
        "pair_overlap_centerline": (pair_count, 2, 2),
        "pair_overlap_centerline_pixels": (pair_count, 2, 2),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache record field {name!r} has shape "
                f"{tuple(tensors[name].shape)}, expected {shape}"
            )
    if pair_count:
        if bool((pair_indices < 0).any()) or bool(
            (pair_indices >= anchor_count).any()
        ):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache pair_indices exceed the anchor array"
            )
        if bool((pair_indices[:, 0] >= pair_indices[:, 1]).any()):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache pair_indices must be strictly ascending"
            )
        pair_keys = [tuple(map(int, pair)) for pair in pair_indices.tolist()]
        if len(set(pair_keys)) != len(pair_keys):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache pair_indices contains duplicate rows"
            )
        if bool((pair_features[:, :3] < 0.0).any()):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache angle, separation, and overlap features "
                "must be nonnegative"
            )
        feature_tolerance = 1e-5
        bounded_features = pair_features[:, 3:]
        if bool(
            (
                (bounded_features < -feature_tolerance)
                | (bounded_features > 1.0 + feature_tolerance)
            ).any()
        ):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache overlap ratios and ink features must lie in [0,1]"
            )
        if bool(((pair_labels != 0) & (pair_labels != 1)).any()):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache pair_labels must be binary"
            )
        expected_targets = tensors["anchor_target_line_ids"][pair_indices]
        if not torch.equal(pair_targets, expected_targets):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache pair target IDs disagree with anchor target IDs"
            )
        expected_labels = (
            pair_targets[:, 0] == pair_targets[:, 1]
        ).to(torch.long)
        if not torch.equal(pair_labels, expected_labels):
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache pair labels disagree with target identity"
            )

    scale = torch.tensor(
        (int(canvas[0]) - 1, int(canvas[1]) - 1),
        dtype=torch.float32,
    )
    if not torch.allclose(anchor_segments, anchor_pixels / scale, atol=1e-5, rtol=0):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache normalized anchor geometry is inconsistent"
        )
    if not torch.allclose(
        centerline,
        centerline_pixels / scale,
        atol=1e-5,
        rtol=0,
    ):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache normalized centerline geometry is inconsistent"
        )
    if pair_features.shape != (pair_count, PAIR_FEATURE_DIM):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache feature width differs from the frozen feature schema"
        )


def _clone_cpu(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("Conditional pair target fields must be tensors")
    return value.detach().to(device="cpu", dtype=dtype).clone().contiguous()


def compact_evidence_carrier_pair_record(
    identity: ConditionalPairSampleIdentity,
    target: dict[str, torch.Tensor],
    *,
    partition: str,
    canvas_size: tuple[int, int],
) -> dict[str, Any]:
    """Reduce one generated target to the scorer/evaluator cache contract."""

    if identity.source_split != "train":
        raise ValueError("Pair cache records may only be built from train identities")
    if partition not in {"fit", "calibration"}:
        raise ValueError("Pair cache partition must be fit or calibration")
    if (
        len(canvas_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 2
            for value in canvas_size
        )
    ):
        raise ValueError(
            "Pair cache canvas_size must contain integer width,height >= 2"
        )
    source_indices = _clone_cpu(
        target["anchor_source_indices"], dtype=torch.long
    )
    raw_group_ids = _clone_cpu(target["raw_group_ids"], dtype=torch.long)
    raw_group_types = _clone_cpu(target["raw_group_types"], dtype=torch.long)
    if len(source_indices) and (
        bool((source_indices < 0).any())
        or bool((source_indices >= len(raw_group_ids)).any())
        or len(raw_group_ids) != len(raw_group_types)
    ):
        raise ValueError("Pair cache anchor source indices exceed raw group targets")
    record = {
        "identity": identity.as_dict(),
        "partition": partition,
        "canvas_size": torch.tensor(canvas_size, dtype=torch.long),
        "anchor_segments": _clone_cpu(target["anchor_segments"], dtype=torch.float32),
        "anchor_segments_pixels": _clone_cpu(
            target["anchor_segments_pixels"], dtype=torch.float32
        ),
        "anchor_source_indices": source_indices,
        "anchor_target_line_ids": _clone_cpu(
            target["anchor_target_line_ids"], dtype=torch.long
        ),
        "anchor_group_ids": raw_group_ids[source_indices].clone(),
        "anchor_group_types": raw_group_types[source_indices].clone(),
        "pair_indices": _clone_cpu(target["pair_indices"], dtype=torch.long),
        "pair_features": _clone_cpu(target["pair_features"], dtype=torch.float32),
        "pair_labels": _clone_cpu(target["pair_labels"], dtype=torch.long),
        "pair_target_line_ids": _clone_cpu(
            target["pair_target_line_ids"], dtype=torch.long
        ),
        "pair_overlap_centerline": _clone_cpu(
            target["pair_overlap_centerline"], dtype=torch.float32
        ),
        "pair_overlap_centerline_pixels": _clone_cpu(
            target["pair_overlap_centerline_pixels"], dtype=torch.float32
        ),
    }
    validate_evidence_carrier_pair_record(record)
    return record


def collate_evidence_carrier_pair_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Pad compact cache records for the pair scorer without raster decoding."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("Pair cache collator requires a sequence of records")
    if not records:
        raise ValueError("Pair cache collator cannot collate an empty batch")
    for record in records:
        validate_evidence_carrier_pair_record(record)
    batch_size = len(records)
    anchor_counts = [len(record["anchor_segments"]) for record in records]
    pair_counts = [len(record["pair_indices"]) for record in records]
    anchor_capacity = max(max(anchor_counts), 1)
    pair_capacity = max(max(pair_counts), 1)
    result: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "identity": [record["identity"] for record in records],
        "partition": [record["partition"] for record in records],
        "canvas_size": torch.stack([record["canvas_size"] for record in records]),
        "anchor_count": torch.tensor(anchor_counts, dtype=torch.long),
        "pair_count": torch.tensor(pair_counts, dtype=torch.long),
        "anchor_segments": torch.zeros(
            (batch_size, anchor_capacity, 2, 2), dtype=torch.float32
        ),
        "anchor_segments_pixels": torch.zeros(
            (batch_size, anchor_capacity, 2, 2), dtype=torch.float32
        ),
        "anchor_source_indices": torch.full(
            (batch_size, anchor_capacity), -1, dtype=torch.long
        ),
        "anchor_target_line_ids": torch.full(
            (batch_size, anchor_capacity), -1, dtype=torch.long
        ),
        "anchor_group_ids": torch.full(
            (batch_size, anchor_capacity), -1, dtype=torch.long
        ),
        "anchor_group_types": torch.full(
            (batch_size, anchor_capacity), -1, dtype=torch.long
        ),
        "anchor_mask": torch.zeros(
            (batch_size, anchor_capacity), dtype=torch.bool
        ),
        "pair_indices": torch.full(
            (batch_size, pair_capacity, 2), -1, dtype=torch.long
        ),
        "pair_features": torch.zeros(
            (batch_size, pair_capacity, PAIR_FEATURE_DIM), dtype=torch.float32
        ),
        "pair_labels": torch.full(
            (batch_size, pair_capacity), -1, dtype=torch.long
        ),
        "pair_target_line_ids": torch.full(
            (batch_size, pair_capacity, 2), -1, dtype=torch.long
        ),
        "pair_overlap_centerline": torch.zeros(
            (batch_size, pair_capacity, 2, 2), dtype=torch.float32
        ),
        "pair_overlap_centerline_pixels": torch.zeros(
            (batch_size, pair_capacity, 2, 2), dtype=torch.float32
        ),
        "pair_mask": torch.zeros((batch_size, pair_capacity), dtype=torch.bool),
    }
    anchor_fields = (
        "anchor_segments",
        "anchor_segments_pixels",
        "anchor_source_indices",
        "anchor_target_line_ids",
        "anchor_group_ids",
        "anchor_group_types",
    )
    pair_fields = (
        "pair_indices",
        "pair_features",
        "pair_labels",
        "pair_target_line_ids",
        "pair_overlap_centerline",
        "pair_overlap_centerline_pixels",
    )
    for batch_index, record in enumerate(records):
        anchor_count = anchor_counts[batch_index]
        pair_count = pair_counts[batch_index]
        for name in anchor_fields:
            result[name][batch_index, :anchor_count] = record[name]
        for name in pair_fields:
            result[name][batch_index, :pair_count] = record[name]
        result["anchor_mask"][batch_index, :anchor_count] = True
        result["pair_mask"][batch_index, :pair_count] = True
    return result


def _partition_metadata(partition: Any, calibration_fraction: float) -> dict[str, Any]:
    fit = tuple(partition.fit)
    calibration = tuple(partition.calibration)
    validate_molecule_disjoint_splits(
        fit,
        calibration,
        left_name="fit",
        right_name="calibration",
    )
    return {
        "seed": partition.seed,
        "calibration_fraction": calibration_fraction,
        "component_count": partition.component_count,
        "calibration_component_count": partition.calibration_component_count,
        "fit_sample_keys": [identity.sample_key for identity in fit],
        "calibration_sample_keys": [
            identity.sample_key for identity in calibration
        ],
        "fit_molecule_ids": sorted({identity.molecule_id for identity in fit}),
        "calibration_molecule_ids": sorted(
            {identity.molecule_id for identity in calibration}
        ),
        "fit_source_structure_ids": sorted(
            {identity.source_structure_id for identity in fit}
        ),
        "calibration_source_structure_ids": sorted(
            {identity.source_structure_id for identity in calibration}
        ),
    }


def _configuration(
    *,
    calibration_fraction: float,
    partition_seed: str,
    limit: int | None,
) -> dict[str, Any]:
    pairing_config = conditional_pair_pairing_config(CACHE_PAIRING_PROFILE)
    target_config = EvidenceCarrierExtentTargetConfig()
    return {
        "pairing_profile": CACHE_PAIRING_PROFILE,
        "pairing_config": asdict(pairing_config),
        "target_config": asdict(target_config),
        "render_variant": None,
        "render_mode": "all_variants",
        "source_split": "train",
        "calibration_fraction": calibration_fraction,
        "partition_seed": partition_seed,
        "limit": limit,
    }


def _source_descriptors(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceCarrierPairCacheValidationError(f"{name} is not SHA256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise EvidenceCarrierPairCacheValidationError(
            f"{name} is not SHA256"
        ) from error
    return value.lower()


def _validate_cache_payload(
    payload: Any,
    *,
    validate_source_hashes: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache payload must be a dictionary"
        )
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "scope": CACHE_SCOPE,
        "pair_data_schema_version": PAIR_DATA_SCHEMA_VERSION,
        "pairing_profile": CACHE_PAIRING_PROFILE,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache {name} differs from the frozen contract"
            )
    if tuple(payload.get("pair_feature_names", ())) != PAIR_FEATURE_NAMES:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache feature names/order differ from the frozen schema"
        )
    if payload.get("uses_validation") is not False:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache must declare uses_validation=False"
        )
    if payload.get("uses_internal_test") is not False:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache must declare uses_internal_test=False"
        )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache config must be a dictionary"
        )
    required_config = {
        "pairing_profile": CACHE_PAIRING_PROFILE,
        "pairing_config": asdict(
            conditional_pair_pairing_config(CACHE_PAIRING_PROFILE)
        ),
        "target_config": asdict(EvidenceCarrierExtentTargetConfig()),
        "render_variant": None,
        "render_mode": "all_variants",
        "source_split": "train",
    }
    for name, value in required_config.items():
        if config.get(name) != value:
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache config field {name!r} is invalid"
            )
    calibration_fraction = config.get("calibration_fraction")
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not math.isfinite(float(calibration_fraction))
        or not 0.0 < float(calibration_fraction) < 1.0
    ):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache calibration_fraction is invalid"
        )
    partition_seed = config.get("partition_seed")
    if not isinstance(partition_seed, str) or not partition_seed:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache partition_seed is invalid"
        )
    try:
        _validate_limit(config.get("limit"))
    except ValueError as error:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache limit is invalid"
        ) from error
    if payload.get("configuration_sha256") != _json_sha256(
        {"config": config, "pair_feature_names": list(PAIR_FEATURE_NAMES)}
    ):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache configuration SHA256 is inconsistent"
        )

    raw_sources = payload.get("source_shards")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache source_shards must be a nonempty list"
        )
    source_hashes: dict[str, str] = {}
    for index, source in enumerate(raw_sources):
        if not isinstance(source, dict):
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache source_shards[{index}] must be a dictionary"
            )
        path_value = source.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache source_shards[{index}].path is invalid"
            )
        _source_path_error(path_value)
        digest = _validate_sha256(
            source.get("sha256"),
            name=f"source_shards[{index}].sha256",
        )
        size = source.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise EvidenceCarrierPairCacheValidationError(
                f"Pair cache source_shards[{index}].size_bytes is invalid"
            )
        if path_value in source_hashes:
            raise EvidenceCarrierPairCacheValidationError(
                "Pair cache source_shards contains duplicate paths"
            )
        source_hashes[path_value] = digest
        if validate_source_hashes:
            path = Path(path_value)
            if not path.is_file():
                raise EvidenceCarrierPairCacheIntegrityError(
                    f"Pair cache source shard is missing: {path}"
                )
            if path.stat().st_size != size:
                raise EvidenceCarrierPairCacheIntegrityError(
                    f"Pair cache source shard size changed: {path}"
                )
            actual = _file_sha256(path)
            if actual != digest:
                raise EvidenceCarrierPairCacheIntegrityError(
                    f"Pair cache source shard SHA256 mismatch: {path}"
                )
    if payload.get("shard_sha256") != source_hashes:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache shard_sha256 map differs from source descriptors"
        )

    fit_records = payload.get("fit_records")
    calibration_records = payload.get("calibration_records")
    if not isinstance(fit_records, list) or not isinstance(
        calibration_records, list
    ):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache fit/calibration records must be lists"
        )
    records_by_side = {
        "fit": fit_records,
        "calibration": calibration_records,
    }
    identities_by_side: dict[str, list[ConditionalPairSampleIdentity]] = {
        "fit": [],
        "calibration": [],
    }
    sample_keys: set[str] = set()
    molecule_side: dict[str, str] = {}
    source_side: dict[str, str] = {}
    for side, records in records_by_side.items():
        for record in records:
            validate_evidence_carrier_pair_record(record)
            if record["partition"] != side:
                raise EvidenceCarrierPairCacheValidationError(
                    "Pair cache record is stored under the wrong partition"
                )
            identity = _identity_from_payload(record["identity"])
            if identity.sample_key in sample_keys:
                raise EvidenceCarrierPairCacheValidationError(
                    "Pair cache contains duplicate sample keys"
                )
            sample_keys.add(identity.sample_key)
            identities_by_side[side].append(identity)
            for key, mapping, label in (
                (identity.molecule_id, molecule_side, "molecule"),
                (identity.source_structure_id, source_side, "source structure"),
            ):
                previous = mapping.setdefault(key, side)
                if previous != side:
                    raise EvidenceCarrierPairCacheValidationError(
                        f"Pair cache {label} variants cross fit/calibration"
                    )
    validate_molecule_disjoint_splits(
        identities_by_side["fit"],
        identities_by_side["calibration"],
        left_name="fit",
        right_name="calibration",
    )

    partition = payload.get("identity_partition")
    if not isinstance(partition, dict):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache identity_partition must be a dictionary"
        )
    actual_fit_keys = sorted(
        identity.sample_key for identity in identities_by_side["fit"]
    )
    actual_calibration_keys = sorted(
        identity.sample_key for identity in identities_by_side["calibration"]
    )
    if sorted(partition.get("fit_sample_keys", ())) != actual_fit_keys or sorted(
        partition.get("calibration_sample_keys", ())
    ) != actual_calibration_keys:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache identity partition keys differ from stored records"
        )
    if payload.get("record_count") != len(sample_keys):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache record_count differs from stored records"
        )
    all_identities = (
        *identities_by_side["fit"],
        *identities_by_side["calibration"],
    )
    try:
        expected_partition = partition_fit_calibration_identities(
            all_identities,
            calibration_fraction=float(calibration_fraction),
            seed=partition_seed,
        )
    except ValueError as error:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache identities cannot reproduce the fit/calibration partition"
        ) from error
    if (
        expected_partition.fit_sample_keys != frozenset(actual_fit_keys)
        or expected_partition.calibration_sample_keys
        != frozenset(actual_calibration_keys)
    ):
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache record sides differ from the deterministic identity partition"
        )
    expected_partition_metadata = _partition_metadata(
        expected_partition,
        float(calibration_fraction),
    )
    if partition != expected_partition_metadata:
        raise EvidenceCarrierPairCacheValidationError(
            "Pair cache identity_partition metadata is inconsistent"
        )
    return payload


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            reloaded = torch.load(
                temporary,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            raise EvidenceCarrierPairCacheIntegrityError(
                "Cannot reload staged pair cache"
            ) from error
        _validate_cache_payload(reloaded, validate_source_hashes=False)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_evidence_carrier_pair_train_cache(
    shards: str | Sequence[str],
    output_path: str | Path,
    calibration_fraction: float = 0.1,
    partition_seed: str = DEFAULT_PARTITION_SEED,
    limit: int | None = None,
    progress_every: int | None = DEFAULT_PROGRESS_EVERY,
) -> dict[str, Any]:
    """Build and atomically publish a train-only all-variant pair cache."""

    if isinstance(calibration_fraction, bool) or not isinstance(
        calibration_fraction, (int, float)
    ):
        raise ValueError("Pair cache calibration_fraction must be numeric")
    calibration_fraction = float(calibration_fraction)
    if not math.isfinite(calibration_fraction) or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("Pair cache calibration_fraction must lie in (0, 1)")
    if not isinstance(partition_seed, str) or not partition_seed:
        raise ValueError("Pair cache partition_seed must be a nonempty string")
    limit = _validate_limit(limit)
    progress_every = _validate_progress_every(progress_every)
    destination = Path(output_path).expanduser()
    if not destination.name:
        raise ValueError("Pair cache output_path must name a file")
    if _contains_internal_test(destination) or _contains_internal_test(
        destination.resolve(strict=False)
    ):
        raise ValueError("Pair cache output must not reference internal_test")

    paths = expand_evidence_carrier_pair_train_shards(shards)
    source_descriptors = _source_descriptors(paths)
    identities = tuple(
        iter_conditional_pair_identities(
            [str(path) for path in paths],
            split="train",
            render_variant=None,
            limit=limit,
        )
    )
    if not identities:
        raise ValueError("Pair cache train identity scan returned no samples")
    partition = partition_fit_calibration_identities(
        identities,
        calibration_fraction=calibration_fraction,
        seed=partition_seed,
    )
    fit_keys = set(partition.fit_sample_keys)
    calibration_keys = set(partition.calibration_sample_keys)
    side_by_key = {key: "fit" for key in fit_keys}
    side_by_key.update({key: "calibration" for key in calibration_keys})
    identity_by_key = {identity.sample_key: identity for identity in identities}
    if len(identity_by_key) != len(identities):
        raise ValueError("Pair cache identity scan returned duplicate sample keys")

    pairing_config = conditional_pair_pairing_config(CACHE_PAIRING_PROFILE)
    target_config = EvidenceCarrierExtentTargetConfig()
    dataset = ConditionalOrdinaryPairWebDataset(
        [str(path) for path in paths],
        split="train",
        render_variant=None,
        allowed_sample_keys=set(identity_by_key),
        validate_hashes=True,
    )
    records_by_side: dict[str, list[dict[str, Any]]] = {
        "fit": [],
        "calibration": [],
    }
    seen: set[str] = set()
    for sample_index, sample in enumerate(dataset, 1):
        identity = sample.get("identity")
        if not isinstance(identity, ConditionalPairSampleIdentity):
            raise ValueError("Pair cache dataset did not return a validated identity")
        expected_identity = identity_by_key.get(identity.sample_key)
        if expected_identity != identity:
            raise ValueError("Pair cache build identity differs from the metadata scan")
        if identity.sample_key in seen:
            raise ValueError("Pair cache build encountered a duplicate sample")
        seen.add(identity.sample_key)
        target = build_conditional_ordinary_pair_target(
            sample["image"],
            sample["primitives"],
            sample["layers"],
            target_config=target_config,
            pairing_config=pairing_config,
        )
        records_by_side[side_by_key[identity.sample_key]].append(
            compact_evidence_carrier_pair_record(
                identity,
                target,
                partition=side_by_key[identity.sample_key],
                canvas_size=sample["image"].size,
            )
        )
        if progress_every is not None and (
            sample_index % progress_every == 0 or sample_index == len(identities)
        ):
            print(
                f"pair-cache: {sample_index}/{len(identities)}",
                flush=True,
            )
    missing = set(identity_by_key) - seen
    if missing:
        raise ValueError(
            "Pair cache build did not materialize requested samples: "
            + ", ".join(sorted(missing)[:5])
        )
    for records in records_by_side.values():
        records.sort(key=lambda record: record["identity"]["sample_key"])

    current_descriptors = _source_descriptors(paths)
    if current_descriptors != source_descriptors:
        raise EvidenceCarrierPairCacheIntegrityError(
            "Pair cache source shards changed while the cache was being built"
        )
    config = _configuration(
        calibration_fraction=calibration_fraction,
        partition_seed=partition_seed,
        limit=limit,
    )
    payload: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "scope": CACHE_SCOPE,
        "pair_data_schema_version": PAIR_DATA_SCHEMA_VERSION,
        "pairing_profile": CACHE_PAIRING_PROFILE,
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "config": config,
        "configuration_sha256": _json_sha256(
            {"config": config, "pair_feature_names": list(PAIR_FEATURE_NAMES)}
        ),
        "source_shards": source_descriptors,
        "shard_sha256": {
            source["path"]: source["sha256"] for source in source_descriptors
        },
        "identity_partition": _partition_metadata(
            partition,
            calibration_fraction,
        ),
        "record_count": len(identities),
        "fit_records": records_by_side["fit"],
        "calibration_records": records_by_side["calibration"],
        "uses_validation": False,
        "uses_internal_test": False,
    }
    _validate_cache_payload(payload, validate_source_hashes=True)
    _atomic_torch_save(payload, destination)
    return payload


def load_evidence_carrier_pair_cache(
    path: str | Path,
    validate_source_hashes: bool = True,
) -> dict[str, Any]:
    """Load a cache on CPU and validate schema, records, partition and provenance."""

    if not isinstance(validate_source_hashes, bool):
        raise TypeError("validate_source_hashes must be a boolean")
    source = Path(path).expanduser()
    if _contains_internal_test(source) or _contains_internal_test(
        source.resolve(strict=False)
    ):
        raise ValueError("Pair cache must not load an internal_test path")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except FileNotFoundError as error:
        raise EvidenceCarrierPairCacheIntegrityError(
            f"Pair cache file is missing: {source}"
        ) from error
    except Exception as error:
        raise EvidenceCarrierPairCacheIntegrityError(
            f"Cannot deserialize pair cache: {source}"
        ) from error
    return _validate_cache_payload(
        payload,
        validate_source_hashes=validate_source_hashes,
    )


__all__ = [
    "CACHE_FORMAT",
    "CACHE_PAIRING_PROFILE",
    "CACHE_SCHEMA_VERSION",
    "CACHE_SCOPE",
    "DEFAULT_PARTITION_SEED",
    "EvidenceCarrierPairCacheError",
    "EvidenceCarrierPairCacheIntegrityError",
    "EvidenceCarrierPairCacheValidationError",
    "build_evidence_carrier_pair_train_cache",
    "collate_evidence_carrier_pair_records",
    "compact_evidence_carrier_pair_record",
    "expand_evidence_carrier_pair_train_shards",
    "load_evidence_carrier_pair_cache",
    "scan_evidence_carrier_pair_train_identities",
    "validate_evidence_carrier_pair_record",
]
