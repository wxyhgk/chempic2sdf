#!/usr/bin/env python3
"""Build the frozen fit-only endpoint-incidence OOF folds for v0.9.18."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import chemvector.build_evidence_carrier_pair_residual_fit_dev as v0917_builder
import chemvector.evidence_carrier_endpoint_incidence_residual as model_module
from chemvector.evidence_carrier_endpoint_incidence_residual import (
    EvidenceCarrierEndpointIncidenceResidual,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    joined_training_records,
    load_train_only_p0_pair_caches,
)
from chemvector.train_evidence_carrier_pair_conditioned_residual import (
    load_pair_residual_manifest,
)


INDEX_SCHEMA_VERSION = "evidence-carrier-endpoint-oof-index-0.1"
SIDECAR_SCHEMA_VERSION = "evidence-carrier-endpoint-oof-fold-records-0.1"
AUDIT_SCHEMA_VERSION = "evidence-carrier-endpoint-oof-data-audit-0.1"
STATUS_COMPLETE = "ENDPOINT-OOF-DATA-COMPLETE"
DEFAULT_SOURCE_INDEX = "outputs/cache/chemvector-v0917-pair-residual-fit-dev.pt"
DEFAULT_OUTPUT_INDEX = "outputs/cache/chemvector-v0918-endpoint-oof-index.pt"
DEFAULT_FOLD_PATHS = tuple(
    f"outputs/cache/chemvector-v0918-endpoint-oof-fold{fold}.pt"
    for fold in range(3)
)
DEFAULT_AUDIT = "outputs/evaluations/chemvector-v0918-endpoint-oof-data/audit.json"
FOLD_SEED = "chemvector-v0.9.18-endpoint-oof"
EXPECTED_SOURCE_INDEX_SHA256 = (
    "e34cab3ce79030c0e4b054f245b0a65c5619e9c5453f4ff0368f90b5b0b73c5f"
)
EXPECTED_SOURCE_FIT_SHA256 = (
    "e599b59f9b991ae3525ae54f394e2027d303e3415c2f6dae2abefc1356b41834"
)
EXPECTED_COMPONENT_COUNT = 1_800
EXPECTED_COMPONENTS_PER_FOLD = 600
EXPECTED_SAMPLES_PER_FOLD = 2_400
EXPECTED_FOLD_PAIR_COUNTS = (124_582, 119_228, 124_763)
EXPECTED_FOLD_POSITIVE_COUNTS = (105_803, 104_009, 106_740)
EXPECTED_FOLD_NEGATIVE_COUNTS = (18_779, 15_219, 18_023)
EXPECTED_ALL_ROWS_SHA256 = (
    "00359bf8fe612f58e3407400eaf6dea300dffbb38301fca0e09eb8478a35a83f"
)
EXPECTED_FOLD_ROWS_SHA256 = (
    "9a0d9b06791c5b1f1773bac87959971bbc3c1205006598d7a98403afb07f08e8",
    "61179a108b7de1fcdc5484d31aeaf0529cae08180e5abfcab030bc841c5e0585",
    "ec3c401304456bbf5ac7f61c74da0908986b9d013f64bd30e5cdb85577c2ea14",
)
EXPECTED_FOLD_KEYS_SHA256 = (
    "6f8deccf713db2a1219347965bafbe53651020460f396ee547e009020ab0cf06",
    "0416d7da6b0f755ef982614d3ce1507b02432b1d52d64925186f925b3ca7f687",
    "d5e6cea875267f2192555889ae1b578287ef4acfc68ff1ea7492187308246e3e",
)
_EVALUATION_FIELDS = (
    "anchor_segments_pixels",
    "anchor_target_line_ids",
    "anchor_source_indices",
    "anchor_group_ids",
    "anchor_group_types",
)
_FORBIDDEN_FLAGS = (
    "v0917_dev_sidecar_read",
    "v0917_dev_report_read",
    "original_calibration_records_accessed",
    "fresh_read",
    "sealed_read",
    "internal_test_read",
    "mr_tadf_read",
)


class EndpointOOFDataError(RuntimeError):
    """Raised when the frozen endpoint OOF data contract is violated."""


def file_sha256(path: str | Path) -> str:
    return v0917_builder.file_sha256(path)


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
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def build_fold_partition(component_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(component_rows) != EXPECTED_COMPONENT_COUNT:
        raise EndpointOOFDataError("source component count differs")
    ranked = []
    seen_keys: set[str] = set()
    for source_row in component_rows:
        component = str(source_row["component"])
        keys = sorted(map(str, source_row["sample_keys"]))
        if len(keys) != 4 or seen_keys.intersection(keys):
            raise EndpointOOFDataError("component sample membership differs")
        seen_keys.update(keys)
        rank = hashlib.sha256(f"{FOLD_SEED}\n{component}".encode()).hexdigest()
        ranked.append({"rank": rank, "component": component, "sample_keys": keys})
    ranked.sort(key=lambda row: row["rank"])
    all_sha = hashlib.sha256(_json_bytes(ranked)).hexdigest()
    folds = []
    for fold in range(3):
        rows = ranked[
            fold * EXPECTED_COMPONENTS_PER_FOLD : (fold + 1)
            * EXPECTED_COMPONENTS_PER_FOLD
        ]
        keys = sorted(key for row in rows for key in row["sample_keys"])
        row_sha = hashlib.sha256(_json_bytes(rows)).hexdigest()
        key_sha = hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest()
        folds.append(
            {
                "fold": fold,
                "component_rows": rows,
                "component_count": len(rows),
                "sample_count": len(keys),
                "component_rows_sha256": row_sha,
                "sorted_sample_keys_sha256": key_sha,
                "sample_keys": keys,
            }
        )
    actual = {
        "all_component_rows_sha256": all_sha,
        "fold_component_rows_sha256": tuple(
            fold["component_rows_sha256"] for fold in folds
        ),
        "fold_sorted_sample_keys_sha256": tuple(
            fold["sorted_sample_keys_sha256"] for fold in folds
        ),
        "fold_component_counts": tuple(fold["component_count"] for fold in folds),
        "fold_sample_counts": tuple(fold["sample_count"] for fold in folds),
    }
    expected = {
        "all_component_rows_sha256": EXPECTED_ALL_ROWS_SHA256,
        "fold_component_rows_sha256": EXPECTED_FOLD_ROWS_SHA256,
        "fold_sorted_sample_keys_sha256": EXPECTED_FOLD_KEYS_SHA256,
        "fold_component_counts": (EXPECTED_COMPONENTS_PER_FOLD,) * 3,
        "fold_sample_counts": (EXPECTED_SAMPLES_PER_FOLD,) * 3,
    }
    if actual != expected:
        raise EndpointOOFDataError(f"fold commitment differs: {actual}")
    return {
        "seed": FOLD_SEED,
        "all_component_rows_sha256": all_sha,
        "folds": folds,
    }


def _enrich_records(
    fit_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    partition: Mapping[str, Any],
) -> list[list[dict[str, Any]]]:
    source_by_key = {
        str(record["identity"]["sample_key"]): record for record in source_records
    }
    if len(source_by_key) != len(source_records):
        raise EndpointOOFDataError("source sample keys are not unique")
    key_to_fold = {
        key: int(fold["fold"])
        for fold in partition["folds"]
        for key in fold["sample_keys"]
    }
    result: list[list[dict[str, Any]]] = [[], [], []]
    for fit_record in fit_records:
        identity = fit_record.get("identity")
        if not isinstance(identity, Mapping):
            raise EndpointOOFDataError("fit identity is missing")
        sample_key = str(identity["sample_key"])
        source = source_by_key.get(sample_key)
        if source is None or dict(source["identity"]) != dict(identity):
            raise EndpointOOFDataError("fit/source identity differs")
        fold = key_to_fold.get(sample_key)
        if fold is None:
            raise EndpointOOFDataError("fit sample lacks fold assignment")
        evaluation_target = {
            name: torch.as_tensor(source[name]).detach().cpu().contiguous()
            for name in _EVALUATION_FIELDS
        }
        pair_indices = torch.as_tensor(
            fit_record["target_free"]["pair_indices"], dtype=torch.long
        )
        labels = torch.as_tensor(
            fit_record["target_aware"]["pair_labels"], dtype=torch.long
        )
        target_ids = evaluation_target["anchor_target_line_ids"].to(torch.long)
        derived = (
            target_ids[pair_indices[:, 0]] == target_ids[pair_indices[:, 1]]
        ).to(torch.long)
        if not torch.equal(labels, derived):
            raise EndpointOOFDataError("canonical labels differ from anchor targets")
        result[fold].append(
            {
                **dict(fit_record),
                "oof_fold": fold,
                "target_aware": {
                    "pair_labels": labels.detach().cpu().contiguous(),
                    "evaluation_target": evaluation_target,
                },
            }
        )
    for records in result:
        records.sort(key=lambda record: record["identity"]["sample_key"])
    return result


def _counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    pairs = positive = negative = 0
    for record in records:
        labels = torch.as_tensor(record["target_aware"]["pair_labels"])
        pairs += len(labels)
        positive += int((labels == 1).sum())
        negative += int((labels == 0).sum())
    return {
        "sample_count": len(records),
        "pair_count": pairs,
        "positive_pair_count": positive,
        "negative_pair_count": negative,
    }


def implementation_sha256() -> dict[str, str]:
    return {
        "oof_builder": file_sha256(Path(__file__).resolve()),
        "endpoint_model": file_sha256(Path(model_module.__file__).resolve()),
        "source_builder": file_sha256(Path(v0917_builder.__file__).resolve()),
    }


def _common(
    source_index: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": STATUS_COMPLETE,
        "formal_eligible": True,
        "source_index_path": source_metadata["path"],
        "source_index_sha256": source_metadata["sha256"],
        "source_fit_sidecar_path": source_metadata["fit_sidecar_path"],
        "source_fit_sidecar_sha256": source_metadata["fit_sidecar_sha256"],
        "source_sha256": dict(source_index["source_sha256"]),
        "implementation_sha256": implementation_sha256(),
        "fold_contract": {
            "seed": partition["seed"],
            "all_component_rows_sha256": partition[
                "all_component_rows_sha256"
            ],
            "fold_component_rows_sha256": list(EXPECTED_FOLD_ROWS_SHA256),
            "fold_sorted_sample_keys_sha256": list(EXPECTED_FOLD_KEYS_SHA256),
            "fold_component_count": EXPECTED_COMPONENTS_PER_FOLD,
            "fold_sample_count": EXPECTED_SAMPLES_PER_FOLD,
        },
        "model_contract": EvidenceCarrierEndpointIncidenceResidual().config_dict(),
        **{name: False for name in _FORBIDDEN_FLAGS},
    }


def validate_fold_sidecar(payload: Mapping[str, Any], *, expected_fold: int) -> None:
    if (
        payload.get("schema_version") != SIDECAR_SCHEMA_VERSION
        or payload.get("status") != STATUS_COMPLETE
        or payload.get("formal_eligible") is not True
        or payload.get("fold") != expected_fold
    ):
        raise EndpointOOFDataError("fold sidecar metadata differs")
    if any(payload.get(name) is not False for name in _FORBIDDEN_FLAGS):
        raise EndpointOOFDataError("fold sidecar declares forbidden data access")
    records = payload.get("records")
    rows = payload.get("component_rows")
    if not isinstance(records, list) or not isinstance(rows, list):
        raise EndpointOOFDataError("fold sidecar records or components are missing")
    expected_keys = sorted(key for row in rows for key in row["sample_keys"])
    actual_keys = [str(record["identity"]["sample_key"]) for record in records]
    if actual_keys != expected_keys or len(records) != EXPECTED_SAMPLES_PER_FOLD:
        raise EndpointOOFDataError("fold record membership differs")
    counts = _counts(records)
    expected_counts = {
        "sample_count": EXPECTED_SAMPLES_PER_FOLD,
        "pair_count": EXPECTED_FOLD_PAIR_COUNTS[expected_fold],
        "positive_pair_count": EXPECTED_FOLD_POSITIVE_COUNTS[expected_fold],
        "negative_pair_count": EXPECTED_FOLD_NEGATIVE_COUNTS[expected_fold],
    }
    if counts != expected_counts or payload.get("counts") != expected_counts:
        raise EndpointOOFDataError("fold sidecar counts differ")
    for record in records:
        if record.get("oof_fold") != expected_fold or record.get("partition") != "fit":
            raise EndpointOOFDataError("fold record partition differs")
        target_free = record.get("target_free")
        target_aware = record.get("target_aware")
        if not isinstance(target_free, Mapping) or not isinstance(
            target_aware, Mapping
        ):
            raise EndpointOOFDataError("fold record layers are missing")
        evaluation = target_aware.get("evaluation_target")
        if not isinstance(evaluation, Mapping) or set(evaluation) != set(
            _EVALUATION_FIELDS
        ):
            raise EndpointOOFDataError("fold evaluation target differs")
        pair_indices = torch.as_tensor(target_free["pair_indices"], dtype=torch.long)
        labels = torch.as_tensor(target_aware["pair_labels"], dtype=torch.long)
        target_ids = torch.as_tensor(
            evaluation["anchor_target_line_ids"], dtype=torch.long
        )
        derived = (
            target_ids[pair_indices[:, 0]] == target_ids[pair_indices[:, 1]]
        ).to(torch.long)
        if not torch.equal(labels, derived):
            raise EndpointOOFDataError("fold labels differ from evaluation target")


def validate_index(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != INDEX_SCHEMA_VERSION
        or payload.get("status") != STATUS_COMPLETE
        or payload.get("formal_eligible") is not True
    ):
        raise EndpointOOFDataError("OOF index metadata differs")
    if any(payload.get(name) is not False for name in _FORBIDDEN_FLAGS):
        raise EndpointOOFDataError("OOF index declares forbidden data access")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise EndpointOOFDataError("OOF index fold references differ")
    for fold, reference in enumerate(folds):
        if not isinstance(reference, Mapping) or "records" in reference:
            raise EndpointOOFDataError("OOF index embeds records")
        if reference.get("fold") != fold:
            raise EndpointOOFDataError("OOF index fold order differs")
        if reference.get("component_rows_sha256") != EXPECTED_FOLD_ROWS_SHA256[fold]:
            raise EndpointOOFDataError("OOF index component commitment differs")
        sha = reference.get("sidecar_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise EndpointOOFDataError("OOF index sidecar SHA differs")


def _write_torch(path: Path, payload: Mapping[str, Any]) -> str:
    temporary = v0917_builder._write_torch_temp(path, payload)
    try:
        v0917_builder._publish_no_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    temporary = v0917_builder._write_json_temp(path, payload)
    try:
        v0917_builder._publish_no_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_index_path = Path(args.source_index).expanduser().resolve(strict=True)
    output_index_path = Path(args.output_index).expanduser().resolve(strict=False)
    fold_paths = [
        Path(path).expanduser().resolve(strict=False) for path in args.fold_paths
    ]
    audit_path = Path(args.audit).expanduser().resolve(strict=False)
    lock_path = output_index_path.with_suffix(output_index_path.suffix + ".lock")
    for path in (output_index_path, *fold_paths, audit_path, lock_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite formal artifact: {path}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "schema_version": "evidence-carrier-endpoint-oof-build-lock-0.1",
                    "status": "started",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())

    source_index, fit_records, source_metadata = load_pair_residual_manifest(
        source_index_path
    )
    if source_metadata["sha256"] != EXPECTED_SOURCE_INDEX_SHA256:
        raise EndpointOOFDataError("source index SHA differs")
    if source_metadata["fit_sidecar_sha256"] != EXPECTED_SOURCE_FIT_SHA256:
        raise EndpointOOFDataError("source fit sidecar SHA differs")
    partition = build_fold_partition(source_index["partitions"]["fit"]["component_rows"])
    source_input = source_index["input"]
    base_cache, augmentation_cache = load_train_only_p0_pair_caches(
        source_input["base_cache"]["path"],
        source_input["augmentation_cache"]["path"],
    )
    source_records = joined_training_records(base_cache, augmentation_cache, "fit")
    fold_records = _enrich_records(fit_records, source_records, partition)
    common = _common(source_index, source_metadata, partition)
    sidecars = []
    sidecar_shas = []
    for fold in range(3):
        counts = _counts(fold_records[fold])
        sidecar = {
            **common,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "fold": fold,
            "component_rows": partition["folds"][fold]["component_rows"],
            "counts": counts,
            "records": fold_records[fold],
        }
        validate_fold_sidecar(sidecar, expected_fold=fold)
        sidecars.append(sidecar)

    published: list[Path] = []
    try:
        for path, sidecar in zip(fold_paths, sidecars, strict=True):
            sidecar_shas.append(_write_torch(path, sidecar))
            published.append(path)
        index = {
            **common,
            "schema_version": INDEX_SCHEMA_VERSION,
            "folds": [
                {
                    "fold": fold,
                    "component_count": EXPECTED_COMPONENTS_PER_FOLD,
                    "sample_count": EXPECTED_SAMPLES_PER_FOLD,
                    "pair_count": EXPECTED_FOLD_PAIR_COUNTS[fold],
                    "positive_pair_count": EXPECTED_FOLD_POSITIVE_COUNTS[fold],
                    "negative_pair_count": EXPECTED_FOLD_NEGATIVE_COUNTS[fold],
                    "component_rows_sha256": EXPECTED_FOLD_ROWS_SHA256[fold],
                    "sorted_sample_keys_sha256": EXPECTED_FOLD_KEYS_SHA256[fold],
                    "sidecar_path": str(fold_paths[fold]),
                    "sidecar_sha256": sidecar_shas[fold],
                }
                for fold in range(3)
            ],
        }
        validate_index(index)
        index_sha = _write_torch(output_index_path, index)
        published.append(output_index_path)
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "formal_eligible": True,
            "index_path": str(output_index_path),
            "index_sha256": index_sha,
            "folds": index["folds"],
            "source_index_path": source_metadata["path"],
            "source_index_sha256": source_metadata["sha256"],
            "source_fit_sidecar_path": source_metadata["fit_sidecar_path"],
            "source_fit_sidecar_sha256": source_metadata["fit_sidecar_sha256"],
            "source_sha256": dict(source_index["source_sha256"]),
            "implementation_sha256": implementation_sha256(),
            "fold_contract": common["fold_contract"],
            **{name: False for name in _FORBIDDEN_FLAGS},
        }
        audit_sha = _write_json(audit_path, audit)
        published.append(audit_path)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise

    complete = {
        "schema_version": "evidence-carrier-endpoint-oof-build-lock-0.1",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "index_path": str(output_index_path),
        "index_sha256": index_sha,
        "fold_sidecar_sha256": sidecar_shas,
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "implementation_sha256": implementation_sha256(),
    }
    temporary_lock = v0917_builder._write_json_temp(lock_path, complete)
    temporary_lock.replace(lock_path)
    v0917_builder._fsync_directory(lock_path.parent)
    return {**audit, "audit_sha256": audit_sha, "lock_path": str(lock_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-index", default=DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--fold-paths", nargs=3, default=DEFAULT_FOLD_PATHS)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(_plain(run(args)), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_AUDIT",
    "DEFAULT_FOLD_PATHS",
    "DEFAULT_OUTPUT_INDEX",
    "EndpointOOFDataError",
    "build_fold_partition",
    "validate_fold_sidecar",
    "validate_index",
]
