#!/usr/bin/env python3
"""Build fit-only v0.9.19 pair-local raster sidecars for three OOF folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    ConditionalOrdinaryPairWebDataset,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_pair_local_raster import (
    PAIR_LOCAL_RASTER_SCHEMA_VERSION,
    PairLocalRasterConfig,
    extract_pair_local_rasters,
    quantize_pair_local_rasters,
)


SCHEMA_VERSION = "evidence-carrier-pair-local-oof-sidecar-0.1"
INDEX_SCHEMA_VERSION = "evidence-carrier-pair-local-oof-index-0.1"
STATUS_COMPLETE = "PAIR-LOCAL-RASTER-OOF-CACHE-COMPLETE"
DEFAULT_SHARDS = [
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/train-000000.tar"
]
DEFAULT_SOURCE_FOLDS = [
    f"outputs/cache/chemvector-v0918-endpoint-oof-fold{fold}.pt"
    for fold in range(3)
]
DEFAULT_OUTPUT_FOLDS = [
    f"outputs/cache/chemvector-v0919-pair-local-oof-fold{fold}.pt"
    for fold in range(3)
]
DEFAULT_INDEX = "outputs/cache/chemvector-v0919-pair-local-oof-index.pt"
EXPECTED_SAMPLE_COUNTS = (2400, 2400, 2400)
EXPECTED_PAIR_COUNTS = (124_582, 119_228, 124_763)


class PairLocalOOFBuildError(RuntimeError):
    """Raised when the frozen pair-local cache contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_torch_no_overwrite(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def _validate_source_record(record: Mapping[str, Any], *, fold: int) -> tuple[str, int]:
    if int(record.get("oof_fold", -1)) != fold:
        raise PairLocalOOFBuildError("source OOF fold differs")
    identity = record.get("identity")
    target_free = record.get("target_free")
    if not isinstance(identity, Mapping) or not isinstance(target_free, Mapping):
        raise PairLocalOOFBuildError("source OOF record schema differs")
    sample_key = str(identity.get("sample_key", ""))
    if not sample_key:
        raise PairLocalOOFBuildError("source OOF sample key is empty")
    pairs = torch.as_tensor(target_free.get("pair_indices"), dtype=torch.long)
    if pairs.ndim != 2 or pairs.size(1) != 2:
        raise PairLocalOOFBuildError("source OOF pair indices differ")
    return sample_key, len(pairs)


def validate_pair_local_sidecar(
    payload: Mapping[str, Any], *, expected_fold: int | None = None
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PairLocalOOFBuildError("pair-local sidecar schema differs")
    if payload.get("status") != STATUS_COMPLETE:
        raise PairLocalOOFBuildError("pair-local sidecar status differs")
    fold = int(payload.get("fold", -1))
    if fold not in (0, 1, 2) or (expected_fold is not None and fold != expected_fold):
        raise PairLocalOOFBuildError("pair-local sidecar fold differs")
    config = payload.get("config")
    if not isinstance(config, Mapping) or config.get("schema_version") != (
        PAIR_LOCAL_RASTER_SCHEMA_VERSION
    ):
        raise PairLocalOOFBuildError("pair-local sidecar config differs")
    records = payload.get("records")
    if not isinstance(records, list):
        raise PairLocalOOFBuildError("pair-local sidecar records differ")
    seen = set()
    pair_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise PairLocalOOFBuildError("pair-local record must be a mapping")
        key = str(record.get("sample_key", ""))
        crops = record.get("pair_local_rasters")
        if not key or key in seen:
            raise PairLocalOOFBuildError("pair-local sample keys differ")
        seen.add(key)
        if not isinstance(crops, torch.Tensor) or crops.dtype != torch.uint8:
            raise PairLocalOOFBuildError("pair-local crops must be uint8 tensors")
        if crops.ndim != 4 or tuple(crops.shape[1:]) != (
            2,
            int(config["height"]),
            int(config["width"]),
        ):
            raise PairLocalOOFBuildError("pair-local crop shape differs")
        if record.get("crop_sha256") != tensor_sha256(crops):
            raise PairLocalOOFBuildError("pair-local crop SHA differs")
        pair_count += len(crops)
    counts = payload.get("counts")
    if counts != {"sample_count": len(records), "pair_count": pair_count}:
        raise PairLocalOOFBuildError("pair-local sidecar counts differ")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.source_folds) != 3 or len(args.output_folds) != 3:
        raise ValueError("exactly three source and output folds are required")
    output_paths = [Path(path).expanduser().resolve() for path in args.output_folds]
    index_path = Path(args.index).expanduser().resolve()
    for path in [*output_paths, index_path]:
        if path.exists():
            raise FileExistsError(path)

    config = PairLocalRasterConfig()
    source_payloads = []
    source_records: dict[str, tuple[int, Mapping[str, Any]]] = {}
    source_references = []
    for fold, path_value in enumerate(args.source_folds):
        path = Path(path_value).expanduser().resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records = payload.get("records")
        if not isinstance(records, list):
            raise PairLocalOOFBuildError("source fold records differ")
        if len(records) != EXPECTED_SAMPLE_COUNTS[fold]:
            raise PairLocalOOFBuildError("source fold sample count differs")
        fold_pairs = 0
        for record in records:
            key, pair_count = _validate_source_record(record, fold=fold)
            if key in source_records:
                raise PairLocalOOFBuildError("source folds overlap by sample key")
            source_records[key] = (fold, record)
            fold_pairs += pair_count
        if fold_pairs != EXPECTED_PAIR_COUNTS[fold]:
            raise PairLocalOOFBuildError("source fold pair count differs")
        source_payloads.append(payload)
        source_references.append({"path": str(path), "sha256": file_sha256(path)})

    buckets: list[list[dict[str, Any]]] = [[], [], []]
    dataset = ConditionalOrdinaryPairWebDataset(
        args.shards,
        split="train",
        render_variant=None,
        allowed_sample_keys=source_records,
        validate_hashes=True,
    )
    for sample in dataset:
        key = str(sample["key"])
        fold, source_record = source_records[key]
        target = build_conditional_ordinary_pair_target(
            sample["image"],
            sample["primitives"],
            sample["layers"],
            pairing_config=conditional_pair_pairing_config("training_wide"),
        )
        frozen_pairs = torch.as_tensor(
            source_record["target_free"]["pair_indices"], dtype=torch.long
        )
        if not torch.equal(target["pair_indices"], frozen_pairs):
            raise PairLocalOOFBuildError(
                f"sample {key} candidates differ from frozen OOF sidecar"
            )
        crops = quantize_pair_local_rasters(
            extract_pair_local_rasters(
                sample["image"],
                target["anchor_segments_pixels"],
                target["pair_indices"],
                pair_centerlines_pixels=target["pair_overlap_centerline_pixels"],
                config=config,
            )
        ).contiguous()
        buckets[fold].append(
            {
                "sample_key": key,
                "pair_local_rasters": crops,
                "crop_sha256": tensor_sha256(crops),
            }
        )

    published = []
    fold_references = []
    try:
        for fold, (records, source, output_path) in enumerate(
            zip(buckets, source_payloads, output_paths, strict=True)
        ):
            records.sort(key=lambda record: record["sample_key"])
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": STATUS_COMPLETE,
                "fold": fold,
                "config": config.as_dict(),
                "source_fold_path": source_references[fold]["path"],
                "source_fold_sha256": source_references[fold]["sha256"],
                "source_fold_contract": source["fold_contract"],
                "counts": {
                    "sample_count": len(records),
                    "pair_count": sum(len(record["pair_local_rasters"]) for record in records),
                },
                "records": records,
            }
            validate_pair_local_sidecar(payload, expected_fold=fold)
            if payload["counts"] != {
                "sample_count": EXPECTED_SAMPLE_COUNTS[fold],
                "pair_count": EXPECTED_PAIR_COUNTS[fold],
            }:
                raise PairLocalOOFBuildError("built fold counts differ")
            sha = _write_torch_no_overwrite(output_path, payload)
            published.append(output_path)
            fold_references.append(
                {
                    "fold": fold,
                    "path": str(output_path),
                    "sha256": sha,
                    **payload["counts"],
                }
            )
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config": config.as_dict(),
            "source_folds": source_references,
            "folds": fold_references,
            "shards": [str(Path(path).expanduser().resolve()) for path in args.shards],
            "counts": {
                "sample_count": sum(EXPECTED_SAMPLE_COUNTS),
                "pair_count": sum(EXPECTED_PAIR_COUNTS),
            },
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
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--source-folds", nargs=3, default=DEFAULT_SOURCE_FOLDS)
    parser.add_argument("--output-folds", nargs=3, default=DEFAULT_OUTPUT_FOLDS)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    return parser


def main() -> None:
    print(json.dumps(run(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PairLocalOOFBuildError",
    "tensor_sha256",
    "validate_pair_local_sidecar",
]
