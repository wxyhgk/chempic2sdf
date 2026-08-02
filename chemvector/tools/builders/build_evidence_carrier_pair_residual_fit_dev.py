#!/usr/bin/env python3
"""Build the isolated fit/dev pair-residual records for v0.9.17.

The formal index never embeds records.  Fit and dev records are committed as
separate SHA-bound sidecars so the optimizer can load fit data without opening
the one-shot dev payload.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import chemvector.build_evidence_carrier_sample_threshold_interval_manifest as _v0916_features
import chemvector.evidence_carrier_pair_conditioned_residual as _residual_model
import chemvector.train_evidence_carrier_p0_pair_scorer as _p0_training
from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_NEGATIVE_LABEL,
    PAIR_POSITIVE_LABEL,
)
from chemvector.evidence_carrier_pair_conditioned_residual import (
    EvidenceCarrierPairConditionedResidual,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
    sample_threshold_global_count_features,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    DEFAULT_AUGMENTATION_CACHE,
    DEFAULT_BASE_CACHE,
    joined_training_records,
    load_p0_pair_scorer_checkpoint,
    load_train_only_p0_pair_caches,
)


SCHEMA_VERSION = "evidence-carrier-pair-residual-fit-dev-index-0.1"
SIDECAR_SCHEMA_VERSION = "evidence-carrier-pair-residual-records-0.1"
AUDIT_SCHEMA_VERSION = "evidence-carrier-pair-residual-fit-dev-audit-0.1"
STATUS_COMPLETE = "FIT-DEV-PAIR-RESIDUAL-MANIFEST-COMPLETE"
SIDECAR_STATUS_COMPLETE = "PAIR-RESIDUAL-RECORDS-COMPLETE"
DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-scorer-pilot9k-train-cal/best.pt"
)
DEFAULT_OUTPUT = "outputs/cache/chemvector-v0917-pair-residual-fit-dev.pt"
DEFAULT_FIT_SIDECAR = "outputs/cache/chemvector-v0917-pair-residual-fit.pt"
DEFAULT_DEV_SIDECAR = "outputs/cache/chemvector-v0917-pair-residual-dev.pt"
DEFAULT_AUDIT_OUTPUT = (
    "outputs/evaluations/chemvector-v0917-pair-residual-fit-dev/audit.json"
)
SPLIT_SEED = "chemvector-v0.9.17-pair-residual-dev"
EXPECTED_SOURCE_SAMPLE_COUNT = 8_100
EXPECTED_SOURCE_COMPONENT_COUNT = 2_025
EXPECTED_SOURCE_MOLECULE_COUNT = 2_025
EXPECTED_SOURCE_STRUCTURE_COUNT = 2_025
EXPECTED_RENDER_VARIANTS_PER_COMPONENT = 4
EXPECTED_FIT_COMPONENT_COUNT = 1_800
EXPECTED_DEV_COMPONENT_COUNT = 225
EXPECTED_FIT_SAMPLE_COUNT = 7_200
EXPECTED_DEV_SAMPLE_COUNT = 900
EXPECTED_FIT_PAIR_COUNT = 368_573
EXPECTED_DEV_PAIR_COUNT = 46_215
EXPECTED_FIT_POSITIVE_COUNT = 316_552
EXPECTED_FIT_NEGATIVE_COUNT = 52_021
EXPECTED_DEV_POSITIVE_COUNT = 39_418
EXPECTED_DEV_NEGATIVE_COUNT = 6_797
EXPECTED_DEV_COMPONENT_MANIFEST_SHA256 = (
    "e06c179f790853fe51df17a1e1ef26380686881256347579b67a6b1a03301106"
)
EXPECTED_FIT_COMPONENT_MANIFEST_SHA256 = (
    "7e0872eb7cbbbaf5ba2e510722e1c0f527f5e272bff394d34c800e2a3b457a9f"
)
EXPECTED_DEV_SAMPLE_KEY_SHA256 = (
    "ace90a44ecedac87b0edf64b28370f877fb7e90a27d278707cb226508f0ad036"
)
EXPECTED_FIT_SAMPLE_KEY_SHA256 = (
    "2c31c5cec70f73e2b595c8d6e74e8acbf06b001d56f2690f13a233f8bb3016b0"
)
BASE_CACHE_SHA256 = (
    "c0d82ff0c524431635d1609c33ea3e7340c2dd1b23ee279db43e86f27aec7697"
)
AUGMENTATION_CACHE_SHA256 = (
    "b7b50afcb664a7aeb0485f6a61f4bf8358e4b0544d4eaa8b02e6cf5d64a55eb1"
)
CHECKPOINT_SHA256 = (
    "67ba638de2841089612b428a8cf43b45206cd21f4da9aa7e182fb269160c7234"
)
UPSTREAM_P0_CHECKPOINT_SHA256 = (
    "f623e35d17dcca8e97b118c6b790e5a111e5ce474873d345b3c2dd569e041a77"
)

_FORBIDDEN_FLAGS = (
    "original_calibration_records_accessed",
    "validation_read",
    "fresh_read",
    "sealed_read",
    "internal_test_read",
    "mr_tadf_read",
)
_EVALUATION_TARGET_FIELDS = (
    "anchor_segments_pixels",
    "anchor_target_line_ids",
    "anchor_source_indices",
    "anchor_group_ids",
    "anchor_group_types",
)
_TARGET_FREE_FIELDS = (
    "pair_indices",
    "pair_tokens",
    "canonical_to_source_candidate_indices",
    "source_to_canonical_candidate_indices",
    "raw_global_counts",
    "global_count_features",
)


class PairResidualManifestError(RuntimeError):
    """Raised when the frozen v0.9.17 data contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def _contains_forbidden(value: Any) -> bool:
    normalized = str(value).lower().replace("-", "_")
    return any(
        marker in normalized
        for marker in (
            "internal_test",
            "mr_tadf",
            "mrtadf",
            "fresh_dev",
            "validation500",
            "validation_500",
        )
    )


def _safe_path(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    resolved = raw.resolve(strict=False)
    if _contains_forbidden(raw) or _contains_forbidden(resolved):
        raise PairResidualManifestError(f"{label} references a forbidden dataset")
    return resolved


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


def _forbidden_access_flags() -> dict[str, bool]:
    return {name: False for name in _FORBIDDEN_FLAGS}


def feature_contract() -> dict[str, Any]:
    return {
        "pair_token_names": list(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES),
        "pair_token_dim": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
        "normalized_descriptor_dim": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM - 1,
        "raw_pair_gain_index": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM - 1,
        "canonical_pair_order": (
            "lexicographic_first_anchor_id_then_second_anchor_id"
        ),
        "candidate_index_mapping": (
            "canonical_to_source_and_source_to_canonical"
        ),
        "global_count_feature_names": list(
            SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
        ),
        "global_count_feature_dim": SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
        "global_count_transform": "log1p(raw_count)",
        "global_count_boundary": "raw_gain_strictly_greater_than_tau0",
        "semantic_sample_fields": ["pair_tokens", "global_count_features"],
        "pair_mask": "collator-derived-from-pair-count",
        "pair_labels_location": "target_aware_only",
    }


def split_contract() -> dict[str, Any]:
    return {
        "seed": SPLIT_SEED,
        "source_partition": "fit",
        "component_definition": (
            "transitive_closure_shared_molecule_id_or_source_structure_id"
        ),
        "component_canonical_json": (
            "ASCII sort_keys compact JSON of molecule_ids and source_structure_ids"
        ),
        "rank_definition": "sha256(seed + newline + component)",
        "component_row_fields": ["rank", "component", "sample_keys"],
        "component_manifest_serialization": (
            "ASCII sort_keys compact JSON of rank-sorted component rows"
        ),
        "sample_key_digest_serialization": (
            "UTF-8 newline-joined sorted sample keys with trailing newline"
        ),
        "source_sample_count": EXPECTED_SOURCE_SAMPLE_COUNT,
        "source_component_count": EXPECTED_SOURCE_COMPONENT_COUNT,
        "source_molecule_count": EXPECTED_SOURCE_MOLECULE_COUNT,
        "source_structure_count": EXPECTED_SOURCE_STRUCTURE_COUNT,
        "render_variants_per_component": EXPECTED_RENDER_VARIANTS_PER_COMPONENT,
        "fit_component_count": EXPECTED_FIT_COMPONENT_COUNT,
        "dev_component_count": EXPECTED_DEV_COMPONENT_COUNT,
        "fit_sample_count": EXPECTED_FIT_SAMPLE_COUNT,
        "dev_sample_count": EXPECTED_DEV_SAMPLE_COUNT,
        "fit_component_manifest_sha256": (
            EXPECTED_FIT_COMPONENT_MANIFEST_SHA256
        ),
        "dev_component_manifest_sha256": (
            EXPECTED_DEV_COMPONENT_MANIFEST_SHA256
        ),
        "fit_sorted_sample_key_sha256": EXPECTED_FIT_SAMPLE_KEY_SHA256,
        "dev_sorted_sample_key_sha256": EXPECTED_DEV_SAMPLE_KEY_SHA256,
    }


def _expected_source_sha256() -> dict[str, str]:
    return {
        "base_cache": BASE_CACHE_SHA256,
        "augmentation_cache": AUGMENTATION_CACHE_SHA256,
        "frozen_pair_scorer_checkpoint": CHECKPOINT_SHA256,
        "frozen_p0_checkpoint": UPSTREAM_P0_CHECKPOINT_SHA256,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_implementation_commitment(value: Any) -> None:
    expected_names = {
        "manifest_builder",
        "residual_model",
        "frozen_feature_builder",
        "p0_pair_training",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_names
        or any(not _valid_sha256(item) for item in value.values())
    ):
        raise PairResidualManifestError("implementation commitment differs")


def _implementation_sha256() -> dict[str, str]:
    files = {
        "manifest_builder": Path(__file__).resolve(),
        "residual_model": Path(_residual_model.__file__).resolve(),
        "frozen_feature_builder": Path(_v0916_features.__file__).resolve(),
        "p0_pair_training": Path(_p0_training.__file__).resolve(),
    }
    return {name: file_sha256(path) for name, path in files.items()}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(self, item: tuple[str, str]) -> tuple[str, str]:
        if item not in self.parent:
            self.parent[item] = item
            return item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: tuple[str, str], second: tuple[str, str]) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        lower, upper = sorted((first_root, second_root))
        self.parent[upper] = lower


def _identity(record: Mapping[str, Any]) -> dict[str, str]:
    raw = record.get("identity")
    if not isinstance(raw, Mapping):
        raise PairResidualManifestError("source record lacks identity")
    required = (
        "sample_key",
        "molecule_id",
        "source_structure_id",
        "source_pilot",
        "source_split",
        "source_record",
        "render_variant",
    )
    result = {name: str(raw.get(name, "")) for name in required}
    if any(not value for value in result.values()):
        raise PairResidualManifestError("source identity has empty fields")
    if result["source_split"] != "train":
        raise PairResidualManifestError("fit-only manifest received a non-train sample")
    if record.get("partition") != "fit":
        raise PairResidualManifestError("original calibration records are forbidden")
    if _contains_forbidden(json.dumps(result, ensure_ascii=True, sort_keys=True)):
        raise PairResidualManifestError("source identity references a forbidden dataset")
    return result


def _component_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_json_bytes(list(rows))).hexdigest()


def _sample_key_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = sorted(key for row in rows for key in row["sample_keys"])
    return hashlib.sha256(("\n".join(keys) + "\n").encode("utf-8")).hexdigest()


def build_component_partition(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return and verify the precommitted component-disjoint split."""

    if len(records) != EXPECTED_SOURCE_SAMPLE_COUNT:
        raise PairResidualManifestError(
            f"expected {EXPECTED_SOURCE_SAMPLE_COUNT} source fit records"
        )
    identities = [_identity(record) for record in records]
    sample_keys = [identity["sample_key"] for identity in identities]
    if len(set(sample_keys)) != len(sample_keys):
        raise PairResidualManifestError("source fit sample keys are not unique")

    union = _UnionFind()
    for identity in identities:
        union.union(
            ("molecule", identity["molecule_id"]),
            ("source", identity["source_structure_id"]),
        )
    components: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for identity in identities:
        root = union.find(("molecule", identity["molecule_id"]))
        components[root].append(identity)

    rows = []
    for members in components.values():
        component_value = {
            "molecule_ids": sorted({item["molecule_id"] for item in members}),
            "source_structure_ids": sorted(
                {item["source_structure_id"] for item in members}
            ),
        }
        canonical = _json_bytes(component_value).decode("ascii")
        rank = hashlib.sha256(
            f"{SPLIT_SEED}\n{canonical}".encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "rank": rank,
                "component": canonical,
                "sample_keys": sorted(item["sample_key"] for item in members),
            }
        )
    rows.sort(key=lambda item: item["rank"])
    if len(rows) != EXPECTED_SOURCE_COMPONENT_COUNT:
        raise PairResidualManifestError("source fit component count differs")
    if any(
        len(row["sample_keys"]) != EXPECTED_RENDER_VARIANTS_PER_COMPONENT
        for row in rows
    ):
        raise PairResidualManifestError("source component render count differs")
    molecule_ids = {identity["molecule_id"] for identity in identities}
    source_ids = {identity["source_structure_id"] for identity in identities}
    if (
        len(molecule_ids) != EXPECTED_SOURCE_MOLECULE_COUNT
        or len(source_ids) != EXPECTED_SOURCE_STRUCTURE_COUNT
    ):
        raise PairResidualManifestError("source molecule/structure counts differ")

    dev_rows = rows[:EXPECTED_DEV_COMPONENT_COUNT]
    fit_rows = rows[EXPECTED_DEV_COMPONENT_COUNT:]
    result = {
        "seed": SPLIT_SEED,
        "component_count": len(rows),
        "molecule_count": len(molecule_ids),
        "source_structure_count": len(source_ids),
        "fit_component_count": len(fit_rows),
        "dev_component_count": len(dev_rows),
        "fit_components": fit_rows,
        "dev_components": dev_rows,
        "fit_component_manifest_sha256": _component_manifest_sha256(fit_rows),
        "dev_component_manifest_sha256": _component_manifest_sha256(dev_rows),
        "fit_sample_key_sha256": _sample_key_sha256(fit_rows),
        "dev_sample_key_sha256": _sample_key_sha256(dev_rows),
    }
    expected = {
        "component_count": EXPECTED_SOURCE_COMPONENT_COUNT,
        "molecule_count": EXPECTED_SOURCE_MOLECULE_COUNT,
        "source_structure_count": EXPECTED_SOURCE_STRUCTURE_COUNT,
        "fit_component_count": EXPECTED_FIT_COMPONENT_COUNT,
        "dev_component_count": EXPECTED_DEV_COMPONENT_COUNT,
        "fit_component_manifest_sha256": EXPECTED_FIT_COMPONENT_MANIFEST_SHA256,
        "dev_component_manifest_sha256": EXPECTED_DEV_COMPONENT_MANIFEST_SHA256,
        "fit_sample_key_sha256": EXPECTED_FIT_SAMPLE_KEY_SHA256,
        "dev_sample_key_sha256": EXPECTED_DEV_SAMPLE_KEY_SHA256,
    }
    mismatches = {
        name: (result[name], value)
        for name, value in expected.items()
        if result[name] != value
    }
    if mismatches:
        raise PairResidualManifestError(
            f"fit/dev component commitment differs: {mismatches}"
        )
    return result


def _component_assignments(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str]]:
    assignments: dict[str, tuple[str, str]] = {}
    for row in rows:
        component = str(row["component"])
        rank = str(row["rank"])
        for sample_key in row["sample_keys"]:
            if sample_key in assignments:
                raise PairResidualManifestError("component sample keys overlap")
            assignments[str(sample_key)] = (component, rank)
    return assignments


def _record_class_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    positive = negative = pairs = 0
    for record in records:
        labels = torch.as_tensor(record["pair_labels"], dtype=torch.long)
        positive += int((labels == PAIR_POSITIVE_LABEL).sum())
        negative += int((labels == PAIR_NEGATIVE_LABEL).sum())
        pairs += len(labels)
    return {
        "pair_count": pairs,
        "positive_pair_count": positive,
        "negative_pair_count": negative,
    }


def _partition_source_counts(
    source_records: Sequence[Mapping[str, Any]],
    keys: frozenset[str],
) -> dict[str, int]:
    return _record_class_counts(
        [
            record
            for record in source_records
            if _identity(record)["sample_key"] in keys
        ]
    )


def _partition_records(
    source_records: Sequence[Mapping[str, Any]],
    scores: Sequence[torch.Tensor],
    descriptors: Sequence[torch.Tensor],
    component_rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
) -> list[dict[str, Any]]:
    if partition not in {"fit", "dev"}:
        raise ValueError("pair residual partition must be fit or dev")
    assignments = _component_assignments(component_rows)
    result = []
    for source, score, descriptor in zip(
        source_records, scores, descriptors, strict=True
    ):
        identity = _identity(source)
        assignment = assignments.get(identity["sample_key"])
        if assignment is None:
            continue
        features = _v0916_features.build_sample_features(
            source,
            score,
            descriptor,
            tau0=SAMPLE_THRESHOLD_BASELINE_TAU,
        )
        canonical_to_source = features["target_free"][
            "canonical_to_source_candidate_indices"
        ]
        source_labels = torch.as_tensor(
            source["pair_labels"], dtype=torch.long
        ).detach().cpu()
        labels = source_labels[canonical_to_source].contiguous()
        component, rank = assignment
        target_aware: dict[str, Any] = {"pair_labels": labels}
        if partition == "dev":
            evaluation_target = {}
            for name in _EVALUATION_TARGET_FIELDS:
                if name not in source:
                    raise PairResidualManifestError(
                        f"source record lacks evaluator field {name}"
                    )
                evaluation_target[name] = (
                    torch.as_tensor(source[name])
                    .detach()
                    .cpu()
                    .contiguous()
                    .clone()
                )
            target_aware["evaluation_target"] = evaluation_target
        result.append(
            {
                **features,
                "partition": partition,
                "component": component,
                "rank": rank,
                "target_aware": target_aware,
            }
        )
    result.sort(key=lambda record: record["identity"]["sample_key"])
    if set(assignments) != {
        record["identity"]["sample_key"] for record in result
    }:
        raise PairResidualManifestError(
            f"{partition} component/sample assignments are incomplete"
        )
    return result


def _expected_partition_values(partition: str) -> dict[str, Any]:
    if partition == "fit":
        return {
            "sample_count": EXPECTED_FIT_SAMPLE_COUNT,
            "component_count": EXPECTED_FIT_COMPONENT_COUNT,
            "pair_count": EXPECTED_FIT_PAIR_COUNT,
            "positive_pair_count": EXPECTED_FIT_POSITIVE_COUNT,
            "negative_pair_count": EXPECTED_FIT_NEGATIVE_COUNT,
            "component_manifest_sha256": EXPECTED_FIT_COMPONENT_MANIFEST_SHA256,
            "sorted_sample_key_sha256": EXPECTED_FIT_SAMPLE_KEY_SHA256,
        }
    if partition == "dev":
        return {
            "sample_count": EXPECTED_DEV_SAMPLE_COUNT,
            "component_count": EXPECTED_DEV_COMPONENT_COUNT,
            "pair_count": EXPECTED_DEV_PAIR_COUNT,
            "positive_pair_count": EXPECTED_DEV_POSITIVE_COUNT,
            "negative_pair_count": EXPECTED_DEV_NEGATIVE_COUNT,
            "component_manifest_sha256": EXPECTED_DEV_COMPONENT_MANIFEST_SHA256,
            "sorted_sample_key_sha256": EXPECTED_DEV_SAMPLE_KEY_SHA256,
        }
    raise ValueError("partition must be fit or dev")


def _partition_metadata(
    records: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = [
        {"pair_labels": record["target_aware"]["pair_labels"]}
        for record in records
    ]
    counts = _record_class_counts(labels)
    molecules = {record["identity"]["molecule_id"] for record in records}
    sources = {record["identity"]["source_structure_id"] for record in records}
    return {
        "sample_count": len(records),
        "component_count": len(component_rows),
        "molecule_count": len(molecules),
        "source_structure_count": len(sources),
        **counts,
        "component_manifest_sha256": _component_manifest_sha256(component_rows),
        "sorted_sample_key_sha256": _sample_key_sha256(component_rows),
    }


def _validate_component_rows(
    rows: Any,
    *,
    partition: str,
) -> tuple[dict[str, tuple[str, str]], set[str], set[str]]:
    if not isinstance(rows, list):
        raise PairResidualManifestError(f"{partition} component rows are invalid")
    if rows != sorted(rows, key=lambda item: item.get("rank", "")):
        raise PairResidualManifestError(f"{partition} component rank order differs")
    assignments: dict[str, tuple[str, str]] = {}
    molecules: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "rank",
            "component",
            "sample_keys",
        }:
            raise PairResidualManifestError(f"{partition} component row schema differs")
        rank = str(row["rank"])
        component = str(row["component"])
        keys = row["sample_keys"]
        try:
            parsed = json.loads(component)
        except json.JSONDecodeError as error:
            raise PairResidualManifestError(
                f"{partition} component JSON is invalid"
            ) from error
        if (
            not isinstance(parsed, dict)
            or set(parsed) != {"molecule_ids", "source_structure_ids"}
            or component != _json_bytes(parsed).decode("ascii")
        ):
            raise PairResidualManifestError(
                f"{partition} component canonical JSON differs"
            )
        expected_rank = hashlib.sha256(
            f"{SPLIT_SEED}\n{component}".encode("utf-8")
        ).hexdigest()
        if rank != expected_rank:
            raise PairResidualManifestError(f"{partition} component rank differs")
        if (
            not isinstance(keys, list)
            or keys != sorted(keys)
            or len(keys) != EXPECTED_RENDER_VARIANTS_PER_COMPONENT
            or len(set(keys)) != len(keys)
        ):
            raise PairResidualManifestError(
                f"{partition} component sample keys differ"
            )
        molecule_ids = parsed["molecule_ids"]
        source_ids = parsed["source_structure_ids"]
        if (
            not isinstance(molecule_ids, list)
            or not isinstance(source_ids, list)
            or molecule_ids != sorted(set(molecule_ids))
            or source_ids != sorted(set(source_ids))
            or not molecule_ids
            or not source_ids
        ):
            raise PairResidualManifestError(
                f"{partition} component identity lists differ"
            )
        molecules.update(map(str, molecule_ids))
        sources.update(map(str, source_ids))
        for key in keys:
            key = str(key)
            if key in assignments:
                raise PairResidualManifestError(
                    f"{partition} component sample keys overlap"
                )
            assignments[key] = (component, rank)
    return assignments, molecules, sources


def _validate_record(
    record: Any,
    *,
    partition: str,
    assignment: tuple[str, str],
) -> tuple[str, str, int, int, int]:
    if not isinstance(record, Mapping) or set(record) != {
        "identity",
        "partition",
        "component",
        "rank",
        "target_free",
        "target_aware",
    }:
        raise PairResidualManifestError(f"{partition} record schema differs")
    if record.get("partition") != partition:
        raise PairResidualManifestError(f"{partition} record partition differs")
    if (record.get("component"), record.get("rank")) != assignment:
        raise PairResidualManifestError(f"{partition} record component differs")
    identity = record["identity"]
    if not isinstance(identity, Mapping):
        raise PairResidualManifestError(f"{partition} record identity differs")
    required_identity = {
        "sample_key",
        "molecule_id",
        "source_structure_id",
        "source_pilot",
        "source_split",
        "source_record",
        "render_variant",
    }
    if set(identity) != required_identity or identity.get("source_split") != "train":
        raise PairResidualManifestError(f"{partition} record identity differs")
    component_value = json.loads(str(record["component"]))
    if (
        identity["molecule_id"] not in component_value["molecule_ids"]
        or identity["source_structure_id"]
        not in component_value["source_structure_ids"]
    ):
        raise PairResidualManifestError(f"{partition} record identity leaks component")

    target_free = record["target_free"]
    target_aware = record["target_aware"]
    if not isinstance(target_free, Mapping) or set(target_free) != set(
        _TARGET_FREE_FIELDS
    ):
        raise PairResidualManifestError(f"{partition} target-free schema differs")
    expected_target_aware = (
        {"pair_labels"}
        if partition == "fit"
        else {"pair_labels", "evaluation_target"}
    )
    if not isinstance(target_aware, Mapping) or set(target_aware) != (
        expected_target_aware
    ):
        raise PairResidualManifestError(f"{partition} target-aware schema differs")

    pair_indices = torch.as_tensor(target_free["pair_indices"], dtype=torch.long)
    pair_tokens = torch.as_tensor(target_free["pair_tokens"], dtype=torch.float32)
    labels = torch.as_tensor(target_aware["pair_labels"], dtype=torch.long)
    candidate_count = len(pair_tokens)
    if pair_indices.shape != (candidate_count, 2):
        raise PairResidualManifestError(f"{partition} pair index shape differs")
    if pair_tokens.shape != (candidate_count, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM):
        raise PairResidualManifestError(f"{partition} pair token shape differs")
    if labels.shape != (candidate_count,):
        raise PairResidualManifestError(f"{partition} pair label shape differs")
    if candidate_count and (
        pair_indices.tolist() != sorted(pair_indices.tolist())
        or bool((pair_indices[:, 0] >= pair_indices[:, 1]).any())
        or len({tuple(map(int, pair)) for pair in pair_indices.tolist()})
        != candidate_count
    ):
        raise PairResidualManifestError(f"{partition} pair order differs")
    if not bool(torch.isfinite(pair_tokens).all()):
        raise PairResidualManifestError(f"{partition} pair tokens are non-finite")
    valid_labels = (labels == PAIR_POSITIVE_LABEL) | (
        labels == PAIR_NEGATIVE_LABEL
    )
    if candidate_count and not bool(valid_labels.all()):
        raise PairResidualManifestError(f"{partition} pair labels are invalid")

    canonical_to_source = torch.as_tensor(
        target_free["canonical_to_source_candidate_indices"], dtype=torch.long
    )
    source_to_canonical = torch.as_tensor(
        target_free["source_to_canonical_candidate_indices"], dtype=torch.long
    )
    expected_permutation = list(range(candidate_count))
    if (
        canonical_to_source.shape != (candidate_count,)
        or source_to_canonical.shape != (candidate_count,)
        or sorted(canonical_to_source.tolist()) != expected_permutation
        or sorted(source_to_canonical.tolist()) != expected_permutation
        or (
            candidate_count
            and not torch.equal(
                source_to_canonical[canonical_to_source],
                torch.arange(candidate_count, dtype=torch.long),
            )
        )
    ):
        raise PairResidualManifestError(
            f"{partition} canonical/source mapping differs"
        )

    raw_counts = torch.as_tensor(
        target_free["raw_global_counts"], dtype=torch.long
    )
    global_features = torch.as_tensor(
        target_free["global_count_features"], dtype=torch.float32
    )
    if raw_counts.shape != (SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,) or bool(
        (raw_counts < 0).any()
    ):
        raise PairResidualManifestError(f"{partition} raw global counts differ")
    if global_features.shape != (
        SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    ) or not bool(torch.isfinite(global_features).all()):
        raise PairResidualManifestError(f"{partition} global features differ")
    anchor_count = int(raw_counts[0])
    if partition == "dev":
        evaluation = target_aware["evaluation_target"]
        if not isinstance(evaluation, Mapping) or set(evaluation) != set(
            _EVALUATION_TARGET_FIELDS
        ):
            raise PairResidualManifestError(f"{partition} evaluator schema differs")
        anchor_segments = torch.as_tensor(
            evaluation["anchor_segments_pixels"], dtype=torch.float32
        )
        if anchor_segments.shape != (anchor_count, 2, 2) or not bool(
            torch.isfinite(anchor_segments).all()
        ):
            raise PairResidualManifestError(f"{partition} anchor segments differ")
        for name in _EVALUATION_TARGET_FIELDS[1:]:
            values = torch.as_tensor(evaluation[name])
            if values.shape != (anchor_count,):
                raise PairResidualManifestError(
                    f"{partition} evaluator anchor field {name} differs"
                )
        target_ids = torch.as_tensor(
            evaluation["anchor_target_line_ids"], dtype=torch.long
        )
        source_indices = torch.as_tensor(
            evaluation["anchor_source_indices"], dtype=torch.long
        )
        group_ids = torch.as_tensor(
            evaluation["anchor_group_ids"], dtype=torch.long
        )
        group_types = torch.as_tensor(
            evaluation["anchor_group_types"], dtype=torch.long
        )
        if any(
            bool((values < 0).any())
            for values in (target_ids, source_indices, group_ids, group_types)
        ):
            raise PairResidualManifestError(
                f"{partition} evaluator anchor values must be nonnegative"
            )
        if torch.unique(source_indices).numel() != anchor_count:
            raise PairResidualManifestError(
                f"{partition} evaluator source indices must be unique"
            )
        derived_labels = (
            target_ids[pair_indices[:, 0]] == target_ids[pair_indices[:, 1]]
        ).to(torch.long)
        if not torch.equal(labels, derived_labels):
            raise PairResidualManifestError(
                f"{partition} canonical pair labels differ from anchor targets"
            )
    if candidate_count and (
        int(pair_indices.min()) < 0 or int(pair_indices.max()) >= anchor_count
    ):
        raise PairResidualManifestError(f"{partition} pair anchors are out of range")
    expected_global = sample_threshold_global_count_features(
        anchor_count,
        pair_indices,
        pair_tokens[:, -1],
        tau0=SAMPLE_THRESHOLD_BASELINE_TAU,
    )
    expected_raw = torch.tensor(
        (
            anchor_count,
            candidate_count,
            int((pair_tokens[:, -1] > SAMPLE_THRESHOLD_BASELINE_TAU).sum()),
            int(
                torch.unique(
                    pair_indices[
                        pair_tokens[:, -1] > SAMPLE_THRESHOLD_BASELINE_TAU
                    ].reshape(-1)
                ).numel()
            )
            if candidate_count
            and bool(
                (pair_tokens[:, -1] > SAMPLE_THRESHOLD_BASELINE_TAU).any()
            )
            else 0,
        ),
        dtype=torch.long,
    )
    if not torch.equal(raw_counts, expected_raw) or not torch.allclose(
        global_features, expected_global, rtol=0.0, atol=1e-7
    ):
        raise PairResidualManifestError(
            f"{partition} global count features differ from pair tokens"
        )
    return (
        str(identity["molecule_id"]),
        str(identity["source_structure_id"]),
        candidate_count,
        int((labels == PAIR_POSITIVE_LABEL).sum()),
        int((labels == PAIR_NEGATIVE_LABEL).sum()),
    )


def validate_records_sidecar(
    payload: Mapping[str, Any],
    *,
    expected_partition: str | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise PairResidualManifestError("sidecar schema version differs")
    if (
        payload.get("status") != SIDECAR_STATUS_COMPLETE
        or payload.get("formal_eligible") is not True
        or payload.get("source_partition") != "fit"
    ):
        raise PairResidualManifestError("sidecar is not formally complete")
    if any(payload.get(name) is not False for name in _FORBIDDEN_FLAGS):
        raise PairResidualManifestError("sidecar declares forbidden data access")
    partition = payload.get("partition")
    if partition not in {"fit", "dev"} or (
        expected_partition is not None and partition != expected_partition
    ):
        raise PairResidualManifestError("sidecar partition differs")
    if payload.get("tau0") != SAMPLE_THRESHOLD_BASELINE_TAU:
        raise PairResidualManifestError("sidecar tau0 differs")
    if payload.get("feature_contract") != feature_contract():
        raise PairResidualManifestError("sidecar feature contract differs")
    if payload.get("split_contract") != split_contract():
        raise PairResidualManifestError("sidecar split contract differs")
    if payload.get("source_sha256") != _expected_source_sha256():
        raise PairResidualManifestError("sidecar source commitment differs")
    _validate_implementation_commitment(payload.get("implementation_sha256"))
    rows = payload.get("component_rows")
    assignments, row_molecules, row_sources = _validate_component_rows(
        rows, partition=partition
    )
    records = payload.get("records")
    if not isinstance(records, list):
        raise PairResidualManifestError(f"{partition} records are invalid")
    keys = [str(record.get("identity", {}).get("sample_key", "")) for record in records]
    if keys != sorted(keys) or len(set(keys)) != len(keys) or set(keys) != set(
        assignments
    ):
        raise PairResidualManifestError(f"{partition} record keys differ")
    molecules: set[str] = set()
    sources: set[str] = set()
    pair_count = positive = negative = 0
    for record in records:
        sample_key = str(record["identity"]["sample_key"])
        molecule, source, pairs, positives, negatives = _validate_record(
            record,
            partition=partition,
            assignment=assignments[sample_key],
        )
        molecules.add(molecule)
        sources.add(source)
        pair_count += pairs
        positive += positives
        negative += negatives
    if molecules != row_molecules or sources != row_sources:
        raise PairResidualManifestError(f"{partition} component identities differ")
    actual = {
        "sample_count": len(records),
        "component_count": len(rows),
        "molecule_count": len(molecules),
        "source_structure_count": len(sources),
        "pair_count": pair_count,
        "positive_pair_count": positive,
        "negative_pair_count": negative,
        "component_manifest_sha256": _component_manifest_sha256(rows),
        "sorted_sample_key_sha256": _sample_key_sha256(rows),
    }
    expected = _expected_partition_values(partition)
    for name, value in expected.items():
        if actual[name] != value:
            raise PairResidualManifestError(
                f"{partition} formal {name} differs: {actual[name]} != {value}"
            )
    for name, value in actual.items():
        if payload.get(name) != value:
            raise PairResidualManifestError(f"{partition} sidecar {name} differs")
    return actual


def _sidecar_reference(
    path: Path,
    sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    names = (
        "sample_count",
        "component_count",
        "molecule_count",
        "source_structure_count",
        "pair_count",
        "positive_pair_count",
        "negative_pair_count",
        "component_manifest_sha256",
        "sorted_sample_key_sha256",
        "component_rows",
    )
    return {
        "records_path": str(path),
        "records_sha256": sha256,
        "records_schema_version": SIDECAR_SCHEMA_VERSION,
        **{name: payload[name] for name in names},
    }


def validate_manifest_payload(
    payload: Mapping[str, Any],
    *,
    fit_sidecar: Mapping[str, Any] | None = None,
    dev_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the records-free index and optional isolated sidecars."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PairResidualManifestError("index schema version differs")
    if (
        payload.get("status") != STATUS_COMPLETE
        or payload.get("formal_eligible") is not True
        or payload.get("source_partition") != "fit"
    ):
        raise PairResidualManifestError("index is not formally complete")
    if any(payload.get(name) is not False for name in _FORBIDDEN_FLAGS):
        raise PairResidualManifestError("index declares forbidden data access")
    if payload.get("tau0") != SAMPLE_THRESHOLD_BASELINE_TAU:
        raise PairResidualManifestError("index tau0 differs")
    if payload.get("feature_contract") != feature_contract():
        raise PairResidualManifestError("index feature contract differs")
    if payload.get("split_contract") != split_contract():
        raise PairResidualManifestError("index split contract differs")
    if payload.get("source_sha256") != _expected_source_sha256():
        raise PairResidualManifestError("index source commitment differs")
    _validate_implementation_commitment(payload.get("implementation_sha256"))
    if payload.get("model_contract") != EvidenceCarrierPairConditionedResidual().config_dict():
        raise PairResidualManifestError("index model contract differs")
    if (
        payload.get("target_free_forward_only") is not True
        or payload.get("pair_labels_are_loss_and_evaluation_only") is not True
    ):
        raise PairResidualManifestError("index target isolation contract differs")
    input_contract = payload.get("input")
    expected_input_names = {
        "base_cache",
        "augmentation_cache",
        "frozen_pair_scorer_checkpoint",
        "frozen_p0_checkpoint",
        "original_partition",
        "original_fit_record_count",
    }
    if not isinstance(input_contract, Mapping) or set(input_contract) != (
        expected_input_names
    ):
        raise PairResidualManifestError("index input contract differs")
    if (
        input_contract.get("original_partition") != "fit"
        or input_contract.get("original_fit_record_count")
        != EXPECTED_SOURCE_SAMPLE_COUNT
    ):
        raise PairResidualManifestError("index original partition differs")
    for name in (
        "base_cache",
        "augmentation_cache",
        "frozen_pair_scorer_checkpoint",
        "frozen_p0_checkpoint",
    ):
        artifact = input_contract.get(name)
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or artifact.get("sha256") != payload["source_sha256"][name]
            or _contains_forbidden(artifact.get("path"))
        ):
            raise PairResidualManifestError(f"index input artifact {name} differs")
    partitions = payload.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != {"fit", "dev"}:
        raise PairResidualManifestError("index partitions differ")
    for partition, reference in partitions.items():
        expected_reference_fields = {
            "records_path",
            "records_sha256",
            "records_schema_version",
            "sample_count",
            "component_count",
            "molecule_count",
            "source_structure_count",
            "pair_count",
            "positive_pair_count",
            "negative_pair_count",
            "component_manifest_sha256",
            "sorted_sample_key_sha256",
            "component_rows",
        }
        if (
            not isinstance(reference, Mapping)
            or set(reference) != expected_reference_fields
            or "records" in reference
        ):
            raise PairResidualManifestError("index must not embed records")
        expected = _expected_partition_values(partition)
        for name, value in expected.items():
            if reference.get(name) != value:
                raise PairResidualManifestError(
                    f"index {partition} {name} differs"
                )
        if reference.get("records_schema_version") != SIDECAR_SCHEMA_VERSION:
            raise PairResidualManifestError("index sidecar schema differs")
        rows = reference.get("component_rows")
        assignments, molecules, sources = _validate_component_rows(
            rows, partition=partition
        )
        derived = {
            "sample_count": len(assignments),
            "component_count": len(rows),
            "molecule_count": len(molecules),
            "source_structure_count": len(sources),
        }
        for name, value in derived.items():
            if reference.get(name) != value:
                raise PairResidualManifestError(
                    f"index {partition} derived {name} differs"
                )
        if reference.get("component_manifest_sha256") != _component_manifest_sha256(
            rows
        ) or reference.get("sorted_sample_key_sha256") != _sample_key_sha256(rows):
            raise PairResidualManifestError("index component commitment differs")
        sha = reference.get("records_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise PairResidualManifestError("index sidecar SHA256 differs")
        if not isinstance(reference.get("records_path"), str):
            raise PairResidualManifestError("index sidecar path differs")

    sidecars = {"fit": fit_sidecar, "dev": dev_sidecar}
    summaries = {}
    identity_sets = {}
    for partition, sidecar in sidecars.items():
        if sidecar is None:
            continue
        summary = validate_records_sidecar(
            sidecar, expected_partition=partition
        )
        summaries[partition] = summary
        reference = partitions[partition]
        if sidecar.get("source_sha256") != payload.get("source_sha256") or sidecar.get(
            "implementation_sha256"
        ) != payload.get("implementation_sha256"):
            raise PairResidualManifestError(
                f"index/{partition} source or implementation commitment differs"
            )
        for name, value in summary.items():
            if reference.get(name) != value:
                raise PairResidualManifestError(
                    f"index/{partition} sidecar metadata differs"
                )
        if reference.get("component_rows") != sidecar.get("component_rows"):
            raise PairResidualManifestError(
                f"index/{partition} component rows differ"
            )
        identity_sets[partition] = {
            "sample": {
                record["identity"]["sample_key"] for record in sidecar["records"]
            },
            "molecule": {
                record["identity"]["molecule_id"] for record in sidecar["records"]
            },
            "source": {
                record["identity"]["source_structure_id"]
                for record in sidecar["records"]
            },
        }
    if set(identity_sets) == {"fit", "dev"}:
        for name in ("sample", "molecule", "source"):
            if identity_sets["fit"][name] & identity_sets["dev"][name]:
                raise PairResidualManifestError(f"fit/dev {name} overlap")
    return summaries


def validate_index_payload(
    payload: Mapping[str, Any],
    *,
    fit_sidecar: Mapping[str, Any] | None = None,
    dev_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility name for the records-free index validator."""

    return validate_manifest_payload(
        payload,
        fit_sidecar=fit_sidecar,
        dev_sidecar=dev_sidecar,
    )


def validate_sidecar_payload(
    payload: Mapping[str, Any],
    *,
    expected_partition: str | None = None,
) -> dict[str, Any]:
    """Compatibility name for the isolated records-sidecar validator."""

    return validate_records_sidecar(
        payload,
        expected_partition=expected_partition,
    )


def load_records_sidecar(
    path: str | Path,
    *,
    expected_partition: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = _safe_path(path, label=f"{expected_partition} sidecar")
    if expected_sha256 is not None and file_sha256(resolved) != expected_sha256:
        raise PairResidualManifestError(
            f"{expected_partition} sidecar SHA256 differs"
        )
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise PairResidualManifestError("records sidecar must be a dictionary")
    validate_records_sidecar(payload, expected_partition=expected_partition)
    if payload.get("implementation_sha256") != _implementation_sha256():
        raise PairResidualManifestError(
            f"{expected_partition} sidecar implementation changed"
        )
    return payload


def load_fit_dev_index(
    path: str | Path,
    *,
    load_fit: bool = False,
    load_dev: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    resolved = _safe_path(path, label="fit/dev index")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise PairResidualManifestError("fit/dev index must be a dictionary")
    validate_manifest_payload(payload)
    if payload.get("implementation_sha256") != _implementation_sha256():
        raise PairResidualManifestError("fit/dev index implementation changed")
    loaded = {}
    for partition, requested in (("fit", load_fit), ("dev", load_dev)):
        if not requested:
            continue
        reference = payload["partitions"][partition]
        loaded[partition] = load_records_sidecar(
            reference["records_path"],
            expected_partition=partition,
            expected_sha256=reference["records_sha256"],
        )
    if loaded:
        validate_manifest_payload(
            payload,
            fit_sidecar=loaded.get("fit"),
            dev_sidecar=loaded.get("dev"),
        )
    return payload, loaded


def _build_sidecar(
    *,
    partition: str,
    records: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    source_sha256: Mapping[str, str],
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    metadata = _partition_metadata(records, component_rows)
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "status": SIDECAR_STATUS_COMPLETE,
        "formal_eligible": True,
        "partition": partition,
        "source_partition": "fit",
        "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
        "source_sha256": dict(source_sha256),
        "implementation_sha256": dict(implementation_sha256),
        "feature_contract": feature_contract(),
        "split_contract": split_contract(),
        **_forbidden_access_flags(),
        **metadata,
        "component_rows": component_rows,
        "records": records,
    }
    validate_records_sidecar(sidecar, expected_partition=partition)
    return sidecar


def _input_contract(
    *,
    base_path: Path,
    augmentation_path: Path,
    checkpoint_path: Path,
    p0_path: Path,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "base_cache": {
            "path": str(base_path),
            "sha256": source_sha256["base_cache"],
        },
        "augmentation_cache": {
            "path": str(augmentation_path),
            "sha256": source_sha256["augmentation_cache"],
        },
        "frozen_pair_scorer_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": source_sha256["frozen_pair_scorer_checkpoint"],
        },
        "frozen_p0_checkpoint": {
            "path": str(p0_path),
            "sha256": source_sha256["frozen_p0_checkpoint"],
        },
        "original_partition": "fit",
        "original_fit_record_count": EXPECTED_SOURCE_SAMPLE_COUNT,
    }


def _build_index(
    *,
    fit_path: Path,
    fit_sha256: str,
    fit_sidecar: Mapping[str, Any],
    dev_path: Path,
    dev_sha256: str,
    dev_sidecar: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    source_sha256: Mapping[str, str],
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    index = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_COMPLETE,
        "formal_eligible": True,
        "source_partition": "fit",
        "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
        "source_sha256": dict(source_sha256),
        "implementation_sha256": dict(implementation_sha256),
        "input": dict(input_contract),
        "split_contract": split_contract(),
        "feature_contract": feature_contract(),
        "model_contract": EvidenceCarrierPairConditionedResidual().config_dict(),
        "target_free_forward_only": True,
        "pair_labels_are_loss_and_evaluation_only": True,
        **_forbidden_access_flags(),
        "partitions": {
            "fit": _sidecar_reference(fit_path, fit_sha256, fit_sidecar),
            "dev": _sidecar_reference(dev_path, dev_sha256, dev_sidecar),
        },
    }
    validate_manifest_payload(
        index, fit_sidecar=fit_sidecar, dev_sidecar=dev_sidecar
    )
    return index


def _write_torch_temp(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary artifact exists: {temporary}")
    torch.save(dict(payload), temporary)
    _fsync_file(temporary)
    return temporary


def _write_json_temp(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
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
    return temporary


def _publish_no_overwrite(temporary: Path, final: Path) -> None:
    os.link(temporary, final)
    _fsync_directory(final.parent)
    temporary.unlink()


def _build_lock_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".lock")


def _acquire_build_lock(
    *,
    index_path: Path,
    fit_path: Path,
    dev_path: Path,
    audit_path: Path,
) -> Path:
    for path in (index_path, fit_path, dev_path, audit_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _build_lock_path(index_path)
    for path in (index_path, fit_path, dev_path, audit_path, lock_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite formal artifact: {path}")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        payload = {
            "schema_version": "evidence-carrier-pair-residual-build-lock-0.1",
            "status": "started",
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "index_path": str(index_path),
            "fit_sidecar_path": str(fit_path),
            "dev_sidecar_path": str(dev_path),
            "audit_path": str(audit_path),
            "expected_source_sha256": _expected_source_sha256(),
        }
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(lock_path.parent)
    return lock_path


def _commit_build_lock(
    lock_path: Path,
    *,
    artifacts: Mapping[str, Mapping[str, str]],
    source_sha256: Mapping[str, str],
    implementation_sha256: Mapping[str, str],
) -> None:
    complete = {
        "schema_version": "evidence-carrier-pair-residual-build-lock-0.1",
        "status": "complete",
        "pid": os.getpid(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": dict(artifacts),
        "source_sha256": dict(source_sha256),
        "implementation_sha256": dict(implementation_sha256),
    }
    temporary = _write_json_temp(lock_path, complete)
    try:
        os.replace(temporary, lock_path)
        _fsync_directory(lock_path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _source_sha256_now(
    *,
    base_path: Path,
    augmentation_path: Path,
    checkpoint_path: Path,
    p0_path: Path,
) -> dict[str, str]:
    return {
        "base_cache": file_sha256(base_path),
        "augmentation_cache": file_sha256(augmentation_path),
        "frozen_pair_scorer_checkpoint": file_sha256(checkpoint_path),
        "frozen_p0_checkpoint": file_sha256(p0_path),
    }


@torch.inference_mode()
def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build and atomically publish fit/dev sidecars plus their records-free index."""

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    base_path = _safe_path(args.base_cache, label="base cache")
    augmentation_path = _safe_path(
        args.augmentation_cache, label="augmentation cache"
    )
    checkpoint_path = _safe_path(args.checkpoint, label="pair scorer checkpoint")
    index_path = _safe_path(args.output, label="fit/dev index")
    fit_path = _safe_path(args.fit_sidecar, label="fit records sidecar")
    dev_path = _safe_path(args.dev_sidecar, label="dev records sidecar")
    audit_path = _safe_path(args.audit_output, label="manifest audit")
    if len({index_path, fit_path, dev_path, audit_path}) != 4:
        raise ValueError("index, sidecars and audit paths must differ")
    lock_path = _acquire_build_lock(
        index_path=index_path,
        fit_path=fit_path,
        dev_path=dev_path,
        audit_path=audit_path,
    )
    published: list[Path] = []
    temporaries: list[Path] = []
    try:
        first_three = {
            "base_cache": file_sha256(base_path),
            "augmentation_cache": file_sha256(augmentation_path),
            "frozen_pair_scorer_checkpoint": file_sha256(checkpoint_path),
        }
        expected = _expected_source_sha256()
        if any(first_three[name] != expected[name] for name in first_three):
            raise PairResidualManifestError(
                f"frozen source SHA256 differs: {first_three}"
            )
        implementation = _implementation_sha256()

        base_cache, augmentation_cache = load_train_only_p0_pair_caches(
            base_path, augmentation_path
        )
        source_records = joined_training_records(
            base_cache, augmentation_cache, "fit"
        )
        component_partition = build_component_partition(source_records)
        fit_rows = component_partition["fit_components"]
        dev_rows = component_partition["dev_components"]
        fit_keys = frozenset(
            key for row in fit_rows for key in row["sample_keys"]
        )
        dev_keys = frozenset(
            key for row in dev_rows for key in row["sample_keys"]
        )
        if fit_keys & dev_keys or len(fit_keys) != EXPECTED_FIT_SAMPLE_COUNT or len(
            dev_keys
        ) != EXPECTED_DEV_SAMPLE_COUNT:
            raise PairResidualManifestError("fit/dev split sample counts differ")
        for partition, counts in (
            ("fit", _partition_source_counts(source_records, fit_keys)),
            ("dev", _partition_source_counts(source_records, dev_keys)),
        ):
            expected_counts = _expected_partition_values(partition)
            for name in ("pair_count", "positive_pair_count", "negative_pair_count"):
                if counts[name] != expected_counts[name]:
                    raise PairResidualManifestError(
                        f"source {partition} {name} differs"
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
            or checkpoint.get("frozen_p0_checkpoint_sha256")
            != UPSTREAM_P0_CHECKPOINT_SHA256
        ):
            raise PairResidualManifestError(
                "frozen scorer is not bound to the four committed sources"
            )
        if (
            augmentation_cache.get("p0_checkpoint_sha256")
            != UPSTREAM_P0_CHECKPOINT_SHA256
            or augmentation_cache.get("p0_checkpoint_path")
            != checkpoint.get("frozen_p0_checkpoint_path")
        ):
            raise PairResidualManifestError(
                "augmentation cache and frozen scorer disagree on upstream P0"
            )
        p0_path = _safe_path(
            checkpoint.get("frozen_p0_checkpoint_path", ""),
            label="frozen P0 checkpoint",
        )
        source_sha256 = _source_sha256_now(
            base_path=base_path,
            augmentation_path=augmentation_path,
            checkpoint_path=checkpoint_path,
            p0_path=p0_path,
        )
        if source_sha256 != expected:
            raise PairResidualManifestError(
                f"four-source commitment differs: {source_sha256}"
            )

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model.to(device).eval()
        scores, descriptors = _v0916_features.score_and_tokenize_records(
            model,
            source_records,
            anchor_feature_dim=int(augmentation_cache["anchor_feature_dim"]),
            batch_size=args.batch_size,
            device=device,
        )
        fit_records = _partition_records(
            source_records,
            scores,
            descriptors,
            fit_rows,
            partition="fit",
        )
        dev_records = _partition_records(
            source_records,
            scores,
            descriptors,
            dev_rows,
            partition="dev",
        )
        fit_sidecar = _build_sidecar(
            partition="fit",
            records=fit_records,
            component_rows=fit_rows,
            source_sha256=source_sha256,
            implementation_sha256=implementation,
        )
        dev_sidecar = _build_sidecar(
            partition="dev",
            records=dev_records,
            component_rows=dev_rows,
            source_sha256=source_sha256,
            implementation_sha256=implementation,
        )

        fit_tmp = _write_torch_temp(fit_path, fit_sidecar)
        dev_tmp = _write_torch_temp(dev_path, dev_sidecar)
        temporaries.extend((fit_tmp, dev_tmp))
        fit_sha = file_sha256(fit_tmp)
        dev_sha = file_sha256(dev_tmp)
        input_contract = _input_contract(
            base_path=base_path,
            augmentation_path=augmentation_path,
            checkpoint_path=checkpoint_path,
            p0_path=p0_path,
            source_sha256=source_sha256,
        )
        index = _build_index(
            fit_path=fit_path,
            fit_sha256=fit_sha,
            fit_sidecar=fit_sidecar,
            dev_path=dev_path,
            dev_sha256=dev_sha,
            dev_sidecar=dev_sidecar,
            input_contract=input_contract,
            source_sha256=source_sha256,
            implementation_sha256=implementation,
        )
        index_tmp = _write_torch_temp(index_path, index)
        temporaries.append(index_tmp)
        index_sha = file_sha256(index_tmp)
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "formal_eligible": True,
            "index_path": str(index_path),
            "index_sha256": index_sha,
            "fit_sidecar_path": str(fit_path),
            "fit_sidecar_sha256": fit_sha,
            "dev_sidecar_path": str(dev_path),
            "dev_sidecar_sha256": dev_sha,
            "source_sha256": source_sha256,
            "implementation_sha256": implementation,
            "split_contract": split_contract(),
            "partitions": {
                "fit": {
                    name: fit_sidecar[name]
                    for name in _expected_partition_values("fit")
                },
                "dev": {
                    name: dev_sidecar[name]
                    for name in _expected_partition_values("dev")
                },
            },
            "fit_dev_sample_overlap": 0,
            "fit_dev_molecule_overlap": 0,
            "fit_dev_source_structure_overlap": 0,
            **_forbidden_access_flags(),
        }
        audit_tmp = _write_json_temp(audit_path, audit)
        temporaries.append(audit_tmp)
        audit_sha = file_sha256(audit_tmp)

        if _implementation_sha256() != implementation:
            raise PairResidualManifestError(
                "builder implementation changed during generation"
            )
        final_sources = _source_sha256_now(
            base_path=base_path,
            augmentation_path=augmentation_path,
            checkpoint_path=checkpoint_path,
            p0_path=p0_path,
        )
        if final_sources != source_sha256:
            raise PairResidualManifestError(
                "frozen sources changed during generation"
            )

        for temporary, final in (
            (fit_tmp, fit_path),
            (dev_tmp, dev_path),
            (index_tmp, index_path),
            (audit_tmp, audit_path),
        ):
            _publish_no_overwrite(temporary, final)
            published.append(final)
        artifacts = {
            "index": {"path": str(index_path), "sha256": index_sha},
            "fit_sidecar": {"path": str(fit_path), "sha256": fit_sha},
            "dev_sidecar": {"path": str(dev_path), "sha256": dev_sha},
            "audit": {"path": str(audit_path), "sha256": audit_sha},
        }
        _commit_build_lock(
            lock_path,
            artifacts=artifacts,
            source_sha256=source_sha256,
            implementation_sha256=implementation,
        )
        return {
            **audit,
            "audit_sha256": audit_sha,
            "lock_path": str(lock_path),
        }
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        lock_path.unlink(missing_ok=True)
        _fsync_directory(lock_path.parent)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", default=DEFAULT_BASE_CACHE)
    parser.add_argument("--augmentation-cache", default=DEFAULT_AUGMENTATION_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-sidecar", default=DEFAULT_FIT_SIDECAR)
    parser.add_argument("--dev-sidecar", default=DEFAULT_DEV_SIDECAR)
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(_plain(build_manifest(args)), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DEFAULT_AUDIT_OUTPUT",
    "DEFAULT_DEV_SIDECAR",
    "DEFAULT_FIT_SIDECAR",
    "DEFAULT_OUTPUT",
    "PairResidualManifestError",
    "SCHEMA_VERSION",
    "SIDECAR_SCHEMA_VERSION",
    "SIDECAR_STATUS_COMPLETE",
    "STATUS_COMPLETE",
    "UPSTREAM_P0_CHECKPOINT_SHA256",
    "build_component_partition",
    "build_manifest",
    "feature_contract",
    "file_sha256",
    "load_fit_dev_index",
    "load_records_sidecar",
    "split_contract",
    "validate_manifest_payload",
    "validate_index_payload",
    "validate_records_sidecar",
    "validate_sidecar_payload",
]
