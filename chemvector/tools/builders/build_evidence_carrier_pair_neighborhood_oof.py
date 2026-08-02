#!/usr/bin/env python3
"""Build target-free carrier-neighborhood sidecars for v0.9.24 OOF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import chemvector.train_evidence_carrier_endpoint_oof as base
from chemvector.build_evidence_carrier_pair_local_oof import (
    EXPECTED_PAIR_COUNTS,
    EXPECTED_SAMPLE_COUNTS,
    file_sha256,
)
from chemvector.evidence_carrier_pair_neighborhood import (
    PAIR_NEIGHBORHOOD_SCHEMA_VERSION,
    PairNeighborhoodConfig,
    build_pair_neighborhood,
)


SCHEMA_VERSION = "evidence-carrier-pair-neighborhood-oof-sidecar-0.1"
INDEX_SCHEMA_VERSION = "evidence-carrier-pair-neighborhood-oof-index-0.1"
STATUS_COMPLETE = "PAIR-NEIGHBORHOOD-OOF-CACHE-COMPLETE"
DEFAULT_OUTPUT_FOLDS = [
    f"outputs/cache/chemvector-v0924-pair-neighborhood-oof-fold{fold}.pt"
    for fold in range(3)
]
DEFAULT_INDEX = "outputs/cache/chemvector-v0924-pair-neighborhood-oof-index.pt"


class PairNeighborhoodOOFBuildError(RuntimeError):
    """Raised when neighborhood cache provenance or shape differs."""


def tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _write_torch_no_overwrite(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def validate_neighborhood_sidecar(
    payload: Mapping[str, Any], *, expected_fold: int | None = None
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != STATUS_COMPLETE:
        raise PairNeighborhoodOOFBuildError("neighborhood sidecar contract differs")
    fold = int(payload.get("fold", -1))
    if fold not in (0, 1, 2) or (expected_fold is not None and fold != expected_fold):
        raise PairNeighborhoodOOFBuildError("neighborhood sidecar fold differs")
    config = payload.get("config")
    if not isinstance(config, Mapping) or config.get("schema_version") != (
        PAIR_NEIGHBORHOOD_SCHEMA_VERSION
    ):
        raise PairNeighborhoodOOFBuildError("neighborhood config differs")
    records = payload.get("records")
    if not isinstance(records, list):
        raise PairNeighborhoodOOFBuildError("neighborhood records differ")
    pair_count = 0
    seen = set()
    for record in records:
        key = str(record.get("sample_key", ""))
        features = record.get("neighbor_features")
        mask = record.get("neighbor_mask")
        if not key or key in seen:
            raise PairNeighborhoodOOFBuildError("neighborhood sample keys differ")
        seen.add(key)
        if not isinstance(features, torch.Tensor) or features.dtype != torch.float16:
            raise PairNeighborhoodOOFBuildError("neighbor features must be float16")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise PairNeighborhoodOOFBuildError("neighbor mask must be boolean")
        if features.ndim != 3 or tuple(features.shape[1:]) != (
            int(config["maximum_neighbors"]),
            8,
        ) or mask.shape != features.shape[:2]:
            raise PairNeighborhoodOOFBuildError("neighborhood tensor shapes differ")
        if record.get("tensor_sha256") != tensor_sha256(features, mask):
            raise PairNeighborhoodOOFBuildError("neighborhood tensor SHA differs")
        pair_count += len(features)
    if payload.get("counts") != {"sample_count": len(records), "pair_count": pair_count}:
        raise PairNeighborhoodOOFBuildError("neighborhood counts differ")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.output_folds) != 3:
        raise ValueError("exactly three output folds are required")
    output_paths = [Path(path).expanduser().resolve() for path in args.output_folds]
    index_path = Path(args.index).expanduser().resolve()
    for path in [*output_paths, index_path]:
        if path.exists():
            raise FileExistsError(path)
    base_index, base_metadata = base.load_oof_index(args.base_index, audit_path=args.base_audit)
    config = PairNeighborhoodConfig()
    fold_references = []
    published = []
    try:
        for fold, output_path in enumerate(output_paths):
            records, source_meta = base.load_oof_fold(base_index, fold)
            output_records = []
            for record in records:
                target = record["target_aware"]["evaluation_target"]
                neighborhood = build_pair_neighborhood(
                    target["anchor_segments_pixels"],
                    record["target_free"]["pair_indices"],
                    config=config,
                )
                features = neighborhood["neighbor_features"].to(torch.float16).contiguous()
                mask = neighborhood["neighbor_mask"].contiguous()
                output_records.append(
                    {
                        "sample_key": str(record["identity"]["sample_key"]),
                        "neighbor_features": features,
                        "neighbor_mask": mask,
                        "tensor_sha256": tensor_sha256(features, mask),
                    }
                )
            output_records.sort(key=lambda row: row["sample_key"])
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": STATUS_COMPLETE,
                "fold": fold,
                "config": config.as_dict(),
                "source_fold": source_meta,
                "counts": {
                    "sample_count": len(output_records),
                    "pair_count": sum(len(row["neighbor_features"]) for row in output_records),
                },
                "records": output_records,
                "target_aware_fields_in_cache": [],
            }
            validate_neighborhood_sidecar(payload, expected_fold=fold)
            if payload["counts"] != {
                "sample_count": EXPECTED_SAMPLE_COUNTS[fold],
                "pair_count": EXPECTED_PAIR_COUNTS[fold],
            }:
                raise PairNeighborhoodOOFBuildError("built neighborhood counts differ")
            sha = _write_torch_no_overwrite(output_path, payload)
            published.append(output_path)
            fold_references.append(
                {"fold": fold, "path": str(output_path), "sha256": sha, **payload["counts"]}
            )
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config": config.as_dict(),
            "base_index": base_metadata,
            "folds": fold_references,
            "counts": {
                "sample_count": sum(EXPECTED_SAMPLE_COUNTS),
                "pair_count": sum(EXPECTED_PAIR_COUNTS),
            },
            "target_aware_fields_in_cache": [],
            "forbidden_reads": {
                "validation": False,
                "fresh": False,
                "sealed": False,
                "internal_test": False,
                "mr_tadf": False,
            },
        }
        index_sha = _write_torch_no_overwrite(index_path, index)
        published.append(index_path)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    return {**index, "index_path": str(index_path), "index_sha256": index_sha}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", default=base.DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--base-audit", default=base.DEFAULT_AUDIT)
    parser.add_argument("--output-folds", nargs=3, default=DEFAULT_OUTPUT_FOLDS)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    return parser


def main() -> None:
    print(json.dumps(run(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["tensor_sha256", "validate_neighborhood_sidecar"]
