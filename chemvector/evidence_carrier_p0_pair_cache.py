"""Train-only frozen-P0 augmentation for the conditional pair cache.

The base cache already freezes LSD detection, ordinary association, candidate
generation, labels, and fit/calibration identity partitioning.  This sidecar
streams only the matching train PNGs, reuses cached anchors, and stores frozen
P0 embeddings, decoded lines, and explicit pair consistency features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    ConditionalOrdinaryPairWebDataset,
    ConditionalPairSampleIdentity,
)
from chemvector.evidence_carrier_p0_pair_scorer import (
    P0_PAIR_CONSISTENCY_FEATURE_DIM,
    P0_PAIR_CONSISTENCY_FEATURE_NAMES,
    EvidenceCarrierP0PairScorerConfig,
    extract_frozen_p0_anchor_evidence,
    p0_decoded_pair_consistency_features,
)
from chemvector.evidence_carrier_extent import EvidenceCarrierExtentConfig
from chemvector.evidence_carrier_extent_data import EvidenceCarrierExtentTargetConfig
from chemvector.train_evidence_carrier_extent import (
    CHECKPOINT_SCHEMA_VERSION as P0_CHECKPOINT_SCHEMA_VERSION,
    P0_EXTENT_WEIGHT,
    P0_KEEP_WEIGHT,
    P0_NORMAL_OFFSET_WEIGHT,
    VALIDATION_SCHEMA_VERSION as P0_VALIDATION_SCHEMA_VERSION,
    file_sha256,
    load_checkpoint,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    load_train_only_cache,
    pair_cache_records,
)


CACHE_SCHEMA_VERSION = "evidence-carrier-p0-pair-augmentation-cache-0.2"
CACHE_FORMAT = "torch-train-only-p0-evidence-sidecar"
DEFAULT_OUTPUT = (
    "outputs/cache/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-augmentation-all-variants.pt"
)
DEFAULT_P0_CHECKPOINT = (
    "outputs/checkpoints/"
    "chemvector-v099-evidence-carrier-extent-pilot9k-p0-step1000/"
    "step-001000.pt"
)
DEFAULT_BASE_CACHE = (
    "outputs/cache/"
    "chemvector-v0910-evidence-carrier-pair-training-wide-all-variants.pt"
)
_EXPLICIT_STEP_CHECKPOINT = re.compile(r"^step-(?P<step>[0-9]{6})\.pt$")
_IDENTITY_FIELDS = tuple(field.name for field in fields(ConditionalPairSampleIdentity))
_CONSISTENCY_CONFIG_FIELDS = tuple(
    field.name for field in fields(EvidenceCarrierP0PairScorerConfig)
)
_P0_LOSS_PROTOCOL = {
    "keep_weight": P0_KEEP_WEIGHT,
    "normal_offset_weight": P0_NORMAL_OFFSET_WEIGHT,
    "extent_weight": P0_EXTENT_WEIGHT,
    "equivalence_weight": "target_line_id_balanced",
}


class EvidenceCarrierP0PairCacheError(RuntimeError):
    pass


class EvidenceCarrierP0PairCacheValidationError(EvidenceCarrierP0PairCacheError):
    pass


def _contains_internal_test(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return "internal_test" in normalized


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sample_keys_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceCarrierP0PairCacheValidationError(f"{name} is not SHA256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise EvidenceCarrierP0PairCacheValidationError(
            f"{name} is not SHA256"
        ) from error
    return value.lower()


def _explicit_checkpoint_step(path: str | Path) -> int:
    match = _EXPLICIT_STEP_CHECKPOINT.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(
            "P0 augmentation requires an explicit step-NNNNNN.pt checkpoint; "
            "best.pt/last.pt and other aliases are forbidden"
        )
    return int(match.group("step"))


def _bucket_partition_metadata(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    fit_keys = [record["identity"]["sample_key"] for record in records["fit"]]
    calibration_keys = [
        record["identity"]["sample_key"] for record in records["calibration"]
    ]
    return {
        "fit_record_count": len(fit_keys),
        "calibration_record_count": len(calibration_keys),
        "fit_sample_keys_sha256": _sample_keys_sha256(fit_keys),
        "calibration_sample_keys_sha256": _sample_keys_sha256(calibration_keys),
        "all_sample_keys_sha256": _sample_keys_sha256(
            [*fit_keys, *calibration_keys]
        ),
    }


def _validate_complete_mapping(
    value: Any,
    *,
    name: str,
    expected_fields: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected_fields):
        raise EvidenceCarrierP0PairCacheValidationError(
            f"P0 pair augmentation {name} is incomplete"
        )
    return value


def _validated_consistency_config(
    payload: Mapping[str, Any],
    *,
    anchor_feature_dim: int,
) -> EvidenceCarrierP0PairScorerConfig:
    raw = _validate_complete_mapping(
        payload.get("pair_consistency_config"),
        name="pair_consistency_config",
        expected_fields=_CONSISTENCY_CONFIG_FIELDS,
    )
    try:
        config = EvidenceCarrierP0PairScorerConfig(**raw)
    except (TypeError, ValueError) as error:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation pair_consistency_config is invalid"
        ) from error
    if asdict(config) != raw or config.anchor_feature_dim != anchor_feature_dim:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation pair_consistency_config differs from cache dimensions"
        )
    return config


def _validate_p0_provenance(
    payload: Mapping[str, Any],
    *,
    anchor_feature_dim: int,
) -> None:
    if payload.get("p0_checkpoint_schema_version") != P0_CHECKPOINT_SCHEMA_VERSION:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation checkpoint schema differs"
        )
    if payload.get("p0_model_type") != "evidence-carrier-extent-v0.1":
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation model_type differs"
        )
    global_step = payload.get("p0_global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation global_step is invalid"
        )
    try:
        filename_step = _explicit_checkpoint_step(payload["p0_checkpoint_path"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceCarrierP0PairCacheValidationError(str(error)) from error
    if filename_step != global_step:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation checkpoint filename and global_step differ"
        )

    raw_model_config = payload.get("p0_model_config")
    raw_target_config = payload.get("p0_target_config")
    if not isinstance(raw_model_config, dict) or not isinstance(
        raw_target_config, dict
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation lacks complete P0 model/target config provenance"
        )
    try:
        model_config = EvidenceCarrierExtentConfig(**raw_model_config)
        target_config = EvidenceCarrierExtentTargetConfig(**raw_target_config)
    except (TypeError, ValueError) as error:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation P0 model/target config provenance is invalid"
        ) from error
    if (
        asdict(model_config) != raw_model_config
        or asdict(target_config) != raw_target_config
        or model_config.hidden_dim != anchor_feature_dim
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation P0 model config differs from cache dimensions"
        )
    if payload.get("p0_loss_protocol") != _P0_LOSS_PROTOCOL:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation P0 loss protocol differs"
        )
    if payload.get("p0_raster_source") != "full_png" or payload.get(
        "p0_oracle_bond_layer"
    ) is not False:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation P0 raster provenance differs"
        )
    for name in ("p0_parameter_count", "p0_trainable_parameter_count"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EvidenceCarrierP0PairCacheValidationError(
                f"P0 pair augmentation {name} is invalid"
            )
    _validate_sha256(payload.get("p0_trainer_sha256"), name="p0_trainer_sha256")

    contains_metrics = payload.get("p0_checkpoint_contains_validation_metrics")
    if not isinstance(contains_metrics, bool):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 checkpoint validation-metrics provenance is invalid"
        )
    validation_schema = payload.get("p0_validation_metrics_schema_version")
    if contains_metrics:
        if validation_schema != P0_VALIDATION_SCHEMA_VERSION:
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 checkpoint validation-metrics schema differs"
            )
    elif validation_schema is not None:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 checkpoint cannot declare a validation schema without metrics"
        )
    if payload.get("p0_checkpoint_selected_by_validation") is not False:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 sidecar checkpoint must be selected by explicit step, not validation"
        )
    selection = _validate_complete_mapping(
        payload.get("p0_checkpoint_selection"),
        name="p0_checkpoint_selection",
        expected_fields=(
            "method",
            "filename",
            "filename_global_step",
            "uses_validation_metrics",
        ),
    )
    expected_selection = {
        "method": "explicit-step-filename",
        "filename": Path(payload["p0_checkpoint_path"]).name,
        "filename_global_step": global_step,
        "uses_validation_metrics": False,
    }
    if selection != expected_selection:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation checkpoint selection provenance differs"
        )


def _validate_extraction_provenance(payload: Mapping[str, Any]) -> None:
    extraction = _validate_complete_mapping(
        payload.get("p0_extraction_provenance"),
        name="p0_extraction_provenance",
        expected_fields=(
            "device",
            "device_type",
            "compute_dtype",
            "autocast_enabled",
            "inference_mode_enabled",
        ),
    )
    try:
        parsed_device = torch.device(extraction["device"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 extraction device provenance is invalid"
        ) from error
    if (
        extraction["device_type"] != parsed_device.type
        or extraction["compute_dtype"] != "float32"
        or extraction["autocast_enabled"] is not False
        or extraction["inference_mode_enabled"] is not True
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 extraction precision provenance differs from the FP32 contract"
        )


def _base_records_by_key(base_cache: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for partition in ("fit", "calibration"):
        for record in pair_cache_records(base_cache, partition):
            identity = record.get("identity")
            if not isinstance(identity, dict) or not isinstance(
                identity.get("sample_key"), str
            ):
                raise EvidenceCarrierP0PairCacheValidationError(
                    "Base pair record lacks a sample identity"
                )
            key = identity["sample_key"]
            if key in result:
                raise EvidenceCarrierP0PairCacheValidationError(
                    "Base pair cache contains duplicate sample keys"
                )
            result[key] = record
    return result


def _validate_augmentation_record(
    record: Any,
    *,
    base_record: Mapping[str, Any] | None,
    anchor_feature_dim: int,
) -> None:
    if not isinstance(record, dict):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation records must be dictionaries"
        )
    required = {
        "identity",
        "partition",
        "anchor_segments_sha256",
        "pair_indices_sha256",
        "anchor_features",
        "decoded_segments_pixels",
        "pair_consistency_features",
    }
    if set(record) != required:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation record fields differ from the frozen schema"
        )
    identity = record["identity"]
    if not isinstance(identity, dict) or set(identity) != set(_IDENTITY_FIELDS):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation identity is invalid"
        )
    if any(
        not isinstance(identity[name], str) or not identity[name]
        for name in _IDENTITY_FIELDS
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation identity fields must be nonempty strings"
        )
    if identity.get("source_split") != "train" or _contains_internal_test(identity):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation records must come from train only"
        )
    if record["partition"] not in {"fit", "calibration"}:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation partition is invalid"
        )
    for name in ("anchor_segments_sha256", "pair_indices_sha256"):
        value = record[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise EvidenceCarrierP0PairCacheValidationError(
                f"P0 pair augmentation {name} is invalid"
            )
    anchor_features = record["anchor_features"]
    decoded = record["decoded_segments_pixels"]
    consistency = record["pair_consistency_features"]
    if not all(
        isinstance(value, torch.Tensor) and value.device.type == "cpu"
        for value in (anchor_features, decoded, consistency)
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation tensors must reside on CPU"
        )
    if anchor_features.dtype != torch.float32 or anchor_features.ndim != 2 or (
        anchor_features.size(-1) != anchor_feature_dim
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair anchor_features must be float32 [anchors,hidden]"
        )
    if decoded.dtype != torch.float32 or decoded.shape != (
        len(anchor_features),
        2,
        2,
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair decoded segments differ from anchor features"
        )
    if (
        consistency.dtype != torch.float32
        or consistency.ndim != 2
        or consistency.size(-1) != P0_PAIR_CONSISTENCY_FEATURE_DIM
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair consistency features have an invalid shape"
        )
    if not all(
        bool(torch.isfinite(value.float()).all())
        for value in (anchor_features, decoded, consistency)
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation contains non-finite values"
        )
    if len(consistency) and bool((consistency < 0.0).any()):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair consistency features must be nonnegative"
        )
    if base_record is not None:
        if identity != base_record.get("identity") or record["partition"] != base_record.get(
            "partition"
        ):
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 pair augmentation identity/partition differs from base cache"
            )
        if len(anchor_features) != len(base_record["anchor_segments"]):
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 pair augmentation anchor count differs from base cache"
            )
        if len(consistency) != len(base_record["pair_indices"]):
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 pair augmentation pair count differs from base cache"
            )
        if record["anchor_segments_sha256"] != _tensor_sha256(
            base_record["anchor_segments"]
        ) or record["pair_indices_sha256"] != _tensor_sha256(
            base_record["pair_indices"]
        ):
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 pair augmentation geometry hashes differ from base cache"
            )


def validate_evidence_carrier_p0_pair_cache(
    payload: Any,
    *,
    base_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation cache must be a dictionary"
        )
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "scope": "train-only",
        "uses_validation": False,
        "uses_internal_test": False,
        "p0_frozen": True,
        "sidecar_validation_data_read": False,
        "anchor_feature_dtype": "float32",
        "decoded_segment_dtype": "float32",
        "pair_consistency_dtype": "float32",
        "pair_consistency_feature_names": list(P0_PAIR_CONSISTENCY_FEATURE_NAMES),
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise EvidenceCarrierP0PairCacheValidationError(
            f"P0 pair augmentation cache contract differs: {mismatches}"
        )
    anchor_feature_dim = payload.get("anchor_feature_dim")
    if (
        isinstance(anchor_feature_dim, bool)
        or not isinstance(anchor_feature_dim, int)
        or anchor_feature_dim < 1
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation anchor_feature_dim is invalid"
        )
    _validated_consistency_config(payload, anchor_feature_dim=anchor_feature_dim)
    _validate_extraction_provenance(payload)
    for name in ("base_cache_path", "p0_checkpoint_path"):
        value = payload.get(name)
        if not isinstance(value, str) or not value or _contains_internal_test(value):
            raise EvidenceCarrierP0PairCacheValidationError(
                f"P0 pair augmentation {name} is invalid"
            )
    for name in ("base_cache_sha256", "p0_checkpoint_sha256"):
        _validate_sha256(payload.get(name), name=name)
    _validate_p0_provenance(payload, anchor_feature_dim=anchor_feature_dim)
    records = payload.get("records")
    if not isinstance(records, dict) or set(records) != {"fit", "calibration"}:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation records must contain fit/calibration"
        )
    base_by_key = _base_records_by_key(base_cache) if base_cache is not None else {}
    seen = set()
    bucket_keys: dict[str, set[str]] = {"fit": set(), "calibration": set()}
    molecule_bucket: dict[str, str] = {}
    source_bucket: dict[str, str] = {}
    count = 0
    for partition in ("fit", "calibration"):
        current = records[partition]
        if not isinstance(current, list):
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 pair augmentation partition records must be lists"
            )
        for record in current:
            key = record.get("identity", {}).get("sample_key")
            if key in seen:
                raise EvidenceCarrierP0PairCacheValidationError(
                    "P0 pair augmentation contains duplicate sample keys"
                )
            seen.add(key)
            _validate_augmentation_record(
                record,
                base_record=base_by_key.get(key) if base_by_key else None,
                anchor_feature_dim=anchor_feature_dim,
            )
            if record["partition"] != partition:
                raise EvidenceCarrierP0PairCacheValidationError(
                    "P0 pair augmentation record is stored under the wrong partition"
                )
            bucket_keys[partition].add(key)
            identity = record["identity"]
            for identity_value, assignments, label in (
                (identity["molecule_id"], molecule_bucket, "molecule"),
                (
                    identity["source_structure_id"],
                    source_bucket,
                    "source structure",
                ),
            ):
                previous = assignments.setdefault(identity_value, partition)
                if previous != partition:
                    raise EvidenceCarrierP0PairCacheValidationError(
                        "P0 pair augmentation bucket partition is not closed over "
                        f"{label} identities"
                    )
            if base_by_key and key not in base_by_key:
                raise EvidenceCarrierP0PairCacheValidationError(
                    "P0 pair augmentation sample is absent from base cache"
                )
            count += 1
    if payload.get("record_count") != count:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation record_count differs from records"
        )
    complete = payload.get("complete")
    if not isinstance(complete, bool):
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation complete flag must be boolean"
        )
    requested_limit = payload.get("requested_limit")
    if complete:
        if requested_limit is not None:
            raise EvidenceCarrierP0PairCacheValidationError(
                "Complete P0 pair augmentation cannot declare requested_limit"
            )
    elif (
        isinstance(requested_limit, bool)
        or not isinstance(requested_limit, int)
        or requested_limit < 1
    ):
        raise EvidenceCarrierP0PairCacheValidationError(
            "Incomplete P0 pair augmentation requires a positive requested_limit"
        )
    if not complete and count > requested_limit:
        raise EvidenceCarrierP0PairCacheValidationError(
            "Incomplete P0 pair augmentation exceeds requested_limit"
        )

    expected_bucket_metadata = _bucket_partition_metadata(records)
    if payload.get("bucket_partition") != expected_bucket_metadata:
        raise EvidenceCarrierP0PairCacheValidationError(
            "P0 pair augmentation bucket_partition metadata is inconsistent"
        )
    if base_by_key:
        ordered_base_keys = sorted(base_by_key)
        expected_keys = set(
            ordered_base_keys if complete else ordered_base_keys[:requested_limit]
        )
        if seen != expected_keys:
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 pair augmentation selection is not closed over the base-cache "
                "bucket partition"
            )
        for partition in ("fit", "calibration"):
            expected_partition_keys = {
                key
                for key in expected_keys
                if base_by_key[key]["partition"] == partition
            }
            if bucket_keys[partition] != expected_partition_keys:
                raise EvidenceCarrierP0PairCacheValidationError(
                    "P0 pair augmentation bucket does not equal the selected base-cache "
                    "partition"
                )
    if base_cache is not None:
        if payload["base_cache_sha256"] != base_cache.get("cache_sha256"):
            raise EvidenceCarrierP0PairCacheValidationError(
                "P0 pair augmentation base cache SHA differs"
            )
    return payload


def _native_image_tensor(image: Any) -> torch.Tensor:
    if getattr(image, "size", None) != (768, 768):
        raise ValueError("Frozen-P0 augmentation requires native 768x768 train PNGs")
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


def _augment_batch(
    samples: Sequence[Mapping[str, Any]],
    base_records: Sequence[Mapping[str, Any]],
    *,
    p0_model: torch.nn.Module,
    device: torch.device,
    consistency_config: EvidenceCarrierP0PairScorerConfig,
) -> list[dict[str, Any]]:
    if len(samples) != len(base_records) or not samples:
        raise ValueError("P0 augmentation samples and base records differ")
    anchor_counts = [len(record["anchor_segments"]) for record in base_records]
    pair_counts = [len(record["pair_indices"]) for record in base_records]
    anchor_capacity = max(max(anchor_counts), 1)
    pair_capacity = max(max(pair_counts), 1)
    batch = len(samples)
    pixels = torch.stack([_native_image_tensor(sample["image"]) for sample in samples])
    anchors = torch.zeros((batch, anchor_capacity, 2, 2), dtype=torch.float32)
    anchor_mask = torch.zeros((batch, anchor_capacity), dtype=torch.bool)
    pairs = torch.full((batch, pair_capacity, 2), -1, dtype=torch.long)
    pair_mask = torch.zeros((batch, pair_capacity), dtype=torch.bool)
    centerlines = torch.zeros((batch, pair_capacity, 2, 2), dtype=torch.float32)
    canvas = torch.zeros((batch, 2), dtype=torch.float32)
    for index, (sample, record) in enumerate(zip(samples, base_records, strict=True)):
        identity = sample.get("identity")
        if not isinstance(identity, ConditionalPairSampleIdentity) or identity.as_dict() != record[
            "identity"
        ]:
            raise ValueError("P0 augmentation raw sample differs from base cache identity")
        anchor_count = anchor_counts[index]
        pair_count = pair_counts[index]
        anchors[index, :anchor_count] = record["anchor_segments"]
        anchor_mask[index, :anchor_count] = True
        pairs[index, :pair_count] = record["pair_indices"]
        pair_mask[index, :pair_count] = True
        centerlines[index, :pair_count] = record["pair_overlap_centerline_pixels"]
        canvas[index] = record["canvas_size"].float()

    pixels = pixels.to(device=device, dtype=torch.float32)
    anchors = anchors.to(device=device, dtype=torch.float32)
    anchor_mask = anchor_mask.to(device)
    pairs = pairs.to(device)
    pair_mask = pair_mask.to(device)
    centerlines = centerlines.to(device=device, dtype=torch.float32)
    canvas = canvas.to(device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=False,
    ):
        evidence = extract_frozen_p0_anchor_evidence(
            p0_model,
            pixels,
            anchors,
            anchor_mask,
            coordinate_mode="normalized",
        )
        anchor_features = evidence["anchor_features"]
        decoded = evidence["decoded_segments_pixels"]
        if anchor_features.dtype != torch.float32 or decoded.dtype != torch.float32:
            raise RuntimeError("Frozen-P0 extraction must produce float32 tensors")
        consistency = p0_decoded_pair_consistency_features(
            decoded,
            pairs,
            centerlines,
            pair_mask,
            canvas,
            config=consistency_config,
        )
        if consistency.dtype != torch.float32:
            raise RuntimeError(
                "Frozen-P0 pair consistency must be computed in float32"
            )
    output = []
    for index, record in enumerate(base_records):
        anchor_count = anchor_counts[index]
        pair_count = pair_counts[index]
        output.append(
            {
                "identity": dict(record["identity"]),
                "partition": record["partition"],
                "anchor_segments_sha256": _tensor_sha256(record["anchor_segments"]),
                "pair_indices_sha256": _tensor_sha256(record["pair_indices"]),
                "anchor_features": anchor_features[
                    index, :anchor_count
                ].detach().cpu().to(torch.float32).contiguous(),
                "decoded_segments_pixels": decoded[
                    index, :anchor_count
                ].detach().cpu().to(torch.float32).contiguous(),
                "pair_consistency_features": consistency[
                    index, :pair_count
                ].detach().cpu().to(torch.float32).contiguous(),
            }
        )
    return output


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_save(
    payload: dict[str, Any],
    destination: Path,
    *,
    base_cache: Mapping[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        validate_evidence_carrier_p0_pair_cache(reloaded, base_cache=base_cache)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_evidence_carrier_p0_pair_cache(
    base_cache_path: str | Path,
    p0_checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    device: torch.device,
    batch_size: int = 4,
    limit: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    if batch_size < 1 or progress_every < 1:
        raise ValueError("P0 augmentation batch_size/progress_every must be positive")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("P0 augmentation limit must be a positive integer or None")
    destination = Path(output_path).expanduser()
    if destination.exists():
        raise RuntimeError(f"P0 augmentation output already exists: {destination}")
    if _contains_internal_test(destination) or _contains_internal_test(
        destination.resolve(strict=False)
    ):
        raise ValueError("P0 augmentation output cannot reference internal_test")
    raw_p0_path = str(p0_checkpoint_path)
    resolved_p0_path = Path(p0_checkpoint_path).expanduser().resolve(strict=False)
    if _contains_internal_test(raw_p0_path) or _contains_internal_test(
        resolved_p0_path
    ):
        raise ValueError("P0 augmentation checkpoint cannot reference internal_test")
    filename_step = _explicit_checkpoint_step(resolved_p0_path)
    if not resolved_p0_path.is_file():
        raise FileNotFoundError(f"P0 augmentation checkpoint is missing: {resolved_p0_path}")
    frozen_p0_sha256 = file_sha256(resolved_p0_path)

    base_cache = load_train_only_cache(base_cache_path)
    base_by_key = _base_records_by_key(base_cache)
    selected_keys = sorted(base_by_key)
    if limit is not None:
        selected_keys = selected_keys[:limit]
    selected = set(selected_keys)
    source_paths = [record["path"] for record in base_cache["source_shards"]]
    p0_model, _, p0_checkpoint = load_checkpoint(resolved_p0_path, device=device)
    if p0_checkpoint["global_step"] != filename_step:
        raise ValueError(
            "P0 checkpoint filename step differs from checkpoint global_step"
        )
    p0_model.float()
    p0_model.eval()
    for parameter in p0_model.parameters():
        parameter.requires_grad_(False)
    hidden_dim = int(p0_model.config.hidden_dim)
    consistency_config = EvidenceCarrierP0PairScorerConfig(
        anchor_feature_dim=hidden_dim
    )
    dataset = ConditionalOrdinaryPairWebDataset(
        source_paths,
        split="train",
        render_variant=None,
        allowed_sample_keys=selected,
        validate_hashes=True,
    )
    records_by_partition: dict[str, list[dict[str, Any]]] = {
        "fit": [],
        "calibration": [],
    }
    sample_batch = []
    base_batch = []
    processed = 0
    for sample in dataset:
        key = sample["identity"].sample_key
        base_record = base_by_key[key]
        sample_batch.append(sample)
        base_batch.append(base_record)
        if len(sample_batch) < batch_size:
            continue
        augmented = _augment_batch(
            sample_batch,
            base_batch,
            p0_model=p0_model,
            device=device,
            consistency_config=consistency_config,
        )
        for record in augmented:
            records_by_partition[record["partition"]].append(record)
        processed += len(augmented)
        if processed % progress_every == 0 or processed == len(selected_keys):
            print(f"p0-pair-cache: {processed}/{len(selected_keys)}", flush=True)
        sample_batch = []
        base_batch = []
    if sample_batch:
        augmented = _augment_batch(
            sample_batch,
            base_batch,
            p0_model=p0_model,
            device=device,
            consistency_config=consistency_config,
        )
        for record in augmented:
            records_by_partition[record["partition"]].append(record)
        processed += len(augmented)
        print(f"p0-pair-cache: {processed}/{len(selected_keys)}", flush=True)
    if processed != len(selected_keys):
        raise RuntimeError(
            f"P0 augmentation processed {processed} samples, expected {len(selected_keys)}"
        )
    if file_sha256(resolved_p0_path) != frozen_p0_sha256:
        raise RuntimeError("P0 checkpoint changed while augmentation was being built")
    for values in records_by_partition.values():
        values.sort(key=lambda record: record["identity"]["sample_key"])
    validation_metrics = p0_checkpoint.get("validation")
    if validation_metrics is not None and not isinstance(validation_metrics, dict):
        raise ValueError(
            "P0 checkpoint validation provenance must be a dictionary or None"
        )
    contains_validation_metrics = bool(validation_metrics)
    validation_metrics_schema = (
        validation_metrics.get("schema_version")
        if contains_validation_metrics
        else None
    )
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "scope": "train-only",
        "base_cache_path": base_cache["cache_path"],
        "base_cache_sha256": base_cache["cache_sha256"],
        "p0_checkpoint_path": str(resolved_p0_path),
        "p0_checkpoint_sha256": frozen_p0_sha256,
        "p0_checkpoint_schema_version": p0_checkpoint[
            "checkpoint_schema_version"
        ],
        "p0_model_type": p0_checkpoint["model_type"],
        "p0_global_step": p0_checkpoint["global_step"],
        "p0_model_config": dict(p0_checkpoint["evidence_carrier_extent_config"]),
        "p0_target_config": dict(p0_checkpoint["target_config"]),
        "p0_loss_protocol": dict(p0_checkpoint["loss_protocol"]),
        "p0_raster_source": p0_checkpoint["raster_source"],
        "p0_oracle_bond_layer": p0_checkpoint["oracle_bond_layer"],
        "p0_parameter_count": p0_checkpoint["parameter_count"],
        "p0_trainable_parameter_count": p0_checkpoint[
            "trainable_parameter_count"
        ],
        "p0_trainer_sha256": p0_checkpoint["trainer_sha256"],
        "p0_checkpoint_contains_validation_metrics": contains_validation_metrics,
        "p0_validation_metrics_schema_version": validation_metrics_schema,
        "p0_checkpoint_selected_by_validation": False,
        "p0_checkpoint_selection": {
            "method": "explicit-step-filename",
            "filename": resolved_p0_path.name,
            "filename_global_step": filename_step,
            "uses_validation_metrics": False,
        },
        "p0_frozen": True,
        "anchor_feature_dim": hidden_dim,
        "anchor_feature_dtype": "float32",
        "decoded_segment_dtype": "float32",
        "pair_consistency_dtype": "float32",
        "pair_consistency_feature_names": list(P0_PAIR_CONSISTENCY_FEATURE_NAMES),
        "pair_consistency_config": asdict(consistency_config),
        "p0_extraction_provenance": {
            "device": str(device),
            "device_type": device.type,
            "compute_dtype": "float32",
            "autocast_enabled": False,
            "inference_mode_enabled": True,
        },
        "record_count": processed,
        "complete": limit is None,
        "requested_limit": limit,
        "records": records_by_partition,
        "bucket_partition": _bucket_partition_metadata(records_by_partition),
        "uses_validation": False,
        "sidecar_validation_data_read": False,
        "uses_internal_test": False,
    }
    validate_evidence_carrier_p0_pair_cache(payload, base_cache=base_cache)
    _atomic_save(payload, destination, base_cache=base_cache)
    return payload


def load_evidence_carrier_p0_pair_cache(
    path: str | Path,
    *,
    base_cache: Mapping[str, Any],
    require_complete: bool = True,
    verify_p0_checkpoint_hash: bool = True,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if _contains_internal_test(source):
        raise ValueError("P0 pair augmentation cache cannot reference internal_test")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    validate_evidence_carrier_p0_pair_cache(payload, base_cache=base_cache)
    if not isinstance(verify_p0_checkpoint_hash, bool):
        raise ValueError("verify_p0_checkpoint_hash must be boolean")
    if verify_p0_checkpoint_hash:
        checkpoint_path = Path(payload["p0_checkpoint_path"]).expanduser().resolve()
        if _contains_internal_test(checkpoint_path):
            raise ValueError("P0 checkpoint cannot reference internal_test")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Frozen P0 checkpoint is missing: {checkpoint_path}"
            )
        if file_sha256(checkpoint_path) != payload["p0_checkpoint_sha256"]:
            raise EvidenceCarrierP0PairCacheValidationError(
                "Frozen P0 checkpoint SHA differs from sidecar provenance"
            )
    if require_complete and not payload["complete"]:
        raise ValueError("B training requires a complete P0 pair augmentation cache")
    payload["cache_path"] = str(source)
    payload["cache_sha256"] = file_sha256(source)
    return payload


def joined_p0_pair_records(
    base_cache: Mapping[str, Any],
    augmentation_cache: Mapping[str, Any],
    partition: str,
) -> list[dict[str, Any]]:
    if partition not in {"fit", "calibration"}:
        raise ValueError("P0 pair joined partition must be fit or calibration")
    validate_evidence_carrier_p0_pair_cache(
        augmentation_cache, base_cache=base_cache
    )
    base_records = pair_cache_records(base_cache, partition)
    augmentation = {
        record["identity"]["sample_key"]: record
        for record in augmentation_cache["records"][partition]
    }
    result = []
    for base_record in base_records:
        key = base_record["identity"]["sample_key"]
        extra = augmentation.get(key)
        if extra is None:
            raise ValueError(f"P0 pair augmentation lacks base sample {key}")
        _validate_augmentation_record(
            extra,
            base_record=base_record,
            anchor_feature_dim=augmentation_cache["anchor_feature_dim"],
        )
        result.append(
            {
                **base_record,
                "anchor_features": extra["anchor_features"],
                "decoded_segments_pixels": extra["decoded_segments_pixels"],
                "pair_consistency_features": extra["pair_consistency_features"],
            }
        )
    return result


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_BASE_CACHE",
    "DEFAULT_OUTPUT",
    "DEFAULT_P0_CHECKPOINT",
    "EvidenceCarrierP0PairCacheValidationError",
    "build_evidence_carrier_p0_pair_cache",
    "joined_p0_pair_records",
    "load_evidence_carrier_p0_pair_cache",
    "validate_evidence_carrier_p0_pair_cache",
]
