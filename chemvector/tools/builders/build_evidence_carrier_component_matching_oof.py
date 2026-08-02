#!/usr/bin/env python3
"""Freeze v0.9.23 OOF scores and target-free component matchings for v0.9.26."""

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
import chemvector.train_evidence_carrier_pair_junction_veto_oof as consensus_helper
import chemvector.train_evidence_carrier_pair_local_oof as helper
import chemvector.train_evidence_carrier_pair_local_veto_oof as veto_helper
from chemvector.build_evidence_carrier_pair_local_oof import (
    EXPECTED_PAIR_COUNTS,
    EXPECTED_SAMPLE_COUNTS,
    file_sha256,
)
from chemvector.evidence_carrier_component_matching import (
    enumerate_component_matchings,
)
from chemvector.evidence_carrier_pair_components import pair_component_metadata
from chemvector.evidence_carrier_pair_local_residual import (
    EvidenceCarrierPairLocalResidual,
)
from chemvector.evidence_carrier_pair_local_spatial_residual import (
    EvidenceCarrierPairLocalSpatialResidual,
)
from chemvector.evidence_carrier_pair_profile_veto import (
    EvidenceCarrierPairProfileVeto,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


SCHEMA_VERSION = "evidence-carrier-component-matching-oof-sidecar-0.1"
INDEX_SCHEMA_VERSION = "evidence-carrier-component-matching-oof-index-0.1"
STATUS_COMPLETE = "COMPONENT-MATCHING-OOF-CACHE-COMPLETE"
DEFAULT_PROFILE_CHECKPOINT_DIR = (
    "outputs/checkpoints/chemvector-v0923-pair-profile-veto-oof"
)
DEFAULT_OUTPUT_FOLDS = [
    f"outputs/cache/chemvector-v0926-component-matching-oof-fold{fold}.pt"
    for fold in range(3)
]
DEFAULT_INDEX = "outputs/cache/chemvector-v0926-component-matching-oof-index.pt"


class ComponentMatchingOOFBuildError(RuntimeError):
    """Raised when the component-matching cache contract differs."""


def tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def component_sha256(components: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for component in components:
        digest.update(str(int(component["component_id"])).encode("ascii"))
        for name in ("active_pair_indices", "matching_masks"):
            value = torch.as_tensor(component[name]).detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes(order="C"))
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


def _load_profile(path: Path, device: torch.device) -> EvidenceCarrierPairProfileVeto:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = EvidenceCarrierPairProfileVeto()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def validate_component_matching_sidecar(
    payload: Mapping[str, Any], *, expected_fold: int | None = None
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != STATUS_COMPLETE:
        raise ComponentMatchingOOFBuildError("component sidecar contract differs")
    fold = int(payload.get("fold", -1))
    if fold not in (0, 1, 2) or (expected_fold is not None and fold != expected_fold):
        raise ComponentMatchingOOFBuildError("component sidecar fold differs")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ComponentMatchingOOFBuildError("component records differ")
    seen = set()
    pair_count = active_pair_count = matching_count = component_count = 0
    for record in records:
        key = str(record.get("sample_key", ""))
        scores = record.get("baseline_scores")
        anchor_counts = record.get("component_anchor_counts")
        edge_counts = record.get("component_edge_counts")
        components = record.get("components")
        if not key or key in seen:
            raise ComponentMatchingOOFBuildError("component sample keys differ")
        seen.add(key)
        if not isinstance(scores, torch.Tensor) or scores.dtype != torch.float32:
            raise ComponentMatchingOOFBuildError("baseline scores must be float32")
        if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
            raise ComponentMatchingOOFBuildError("baseline score values differ")
        if not isinstance(anchor_counts, torch.Tensor) or anchor_counts.dtype != torch.uint8:
            raise ComponentMatchingOOFBuildError("component anchor counts must be uint8")
        if not isinstance(edge_counts, torch.Tensor) or edge_counts.dtype != torch.uint8:
            raise ComponentMatchingOOFBuildError("component edge counts must be uint8")
        if anchor_counts.shape != scores.shape or edge_counts.shape != scores.shape:
            raise ComponentMatchingOOFBuildError("component count shapes differ")
        if not isinstance(components, list):
            raise ComponentMatchingOOFBuildError("component configuration list differs")
        if record.get("tensor_sha256") != tensor_sha256(scores, anchor_counts, edge_counts):
            raise ComponentMatchingOOFBuildError("component tensor SHA differs")
        if record.get("component_sha256") != component_sha256(components):
            raise ComponentMatchingOOFBuildError("component configuration SHA differs")
        used = set()
        for component in components:
            indices = component.get("active_pair_indices")
            masks = component.get("matching_masks")
            if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int32:
                raise ComponentMatchingOOFBuildError("active pair indices must be int32")
            if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool:
                raise ComponentMatchingOOFBuildError("matching masks must be boolean")
            if indices.ndim != 1 or masks.ndim != 2 or masks.size(1) != len(indices):
                raise ComponentMatchingOOFBuildError("component matching shapes differ")
            if not len(indices) or not len(masks) or bool((indices < 0).any()) or bool(
                (indices >= len(scores)).any()
            ):
                raise ComponentMatchingOOFBuildError("component active indices differ")
            if used.intersection(indices.tolist()):
                raise ComponentMatchingOOFBuildError("component active pairs overlap")
            used.update(indices.tolist())
            if bool((scores[indices.long()] <= float(SAMPLE_THRESHOLD_BASELINE_TAU)).any()):
                raise ComponentMatchingOOFBuildError("component contains inactive scores")
            active_pair_count += len(indices)
            matching_count += len(masks)
            component_count += 1
        pair_count += len(scores)
    counts = payload.get("counts")
    expected = {
        "sample_count": len(records),
        "pair_count": pair_count,
        "active_component_count": component_count,
        "active_pair_count": active_pair_count,
        "matching_count": matching_count,
    }
    if counts != expected:
        raise ComponentMatchingOOFBuildError("component sidecar counts differ")


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.output_folds) != 3:
        raise ValueError("exactly three output folds are required")
    output_paths = [Path(path).expanduser().resolve() for path in args.output_folds]
    index_path = Path(args.index).expanduser().resolve()
    for path in [*output_paths, index_path]:
        if path.exists():
            raise FileExistsError(path)
    base_index, base_metadata = base.load_oof_index(
        args.base_index, audit_path=args.base_audit
    )
    local_index, local_metadata = helper.load_local_index(args.local_index)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ComponentMatchingOOFBuildError("CUDA was requested but is unavailable")
    tau = float(SAMPLE_THRESHOLD_BASELINE_TAU)
    published = []
    fold_references = []
    try:
        for fold, output_path in enumerate(output_paths):
            records, base_fold_meta = base.load_oof_fold(base_index, fold)
            crops, local_fold_meta = helper.load_local_fold(local_index, fold)
            records = helper.merge_pair_local_records(records, crops)
            avg_path = (
                Path(args.avg_checkpoint_dir).expanduser().resolve()
                / f"fold{fold}"
                / "final.pt"
            )
            spatial_path = (
                Path(args.spatial_checkpoint_dir).expanduser().resolve()
                / f"fold{fold}"
                / "final.pt"
            )
            profile_path = (
                Path(args.profile_checkpoint_dir).expanduser().resolve()
                / f"fold{fold}"
                / "final.pt"
            )
            avg_model, avg_sha = consensus_helper._load_teacher(
                EvidenceCarrierPairLocalResidual, avg_path, device
            )
            spatial_model, spatial_sha = consensus_helper._load_teacher(
                EvidenceCarrierPairLocalSpatialResidual, spatial_path, device
            )
            profile_model = _load_profile(profile_path, device)
            output_records = []
            for start in range(0, len(records), 32):
                current = records[start : start + 32]
                batch = helper._to_device(helper.collate_pair_local_records(current), device)
                avg = helper._forward(avg_model, batch)
                spatial = helper._forward(spatial_model, batch)
                profile = veto_helper._forward(profile_model, batch)
                for image_index, record in enumerate(current):
                    count = int(batch["pair_count"][image_index])
                    consensus = torch.minimum(
                        avg["corrected_gain"][image_index, :count],
                        spatial["corrected_gain"][image_index, :count],
                    ).detach().cpu()
                    logits = profile["veto_logits"][image_index, :count].detach().cpu()
                    scores = torch.where(
                        (consensus > tau) & (logits > 0.0),
                        torch.full_like(consensus, tau - 1.0),
                        consensus,
                    ).to(torch.float32).contiguous()
                    pairs = torch.as_tensor(
                        record["target_free"]["pair_indices"], dtype=torch.long
                    )
                    metadata = pair_component_metadata(pairs)
                    components = []
                    multi_ids = metadata["component_ids"][
                        metadata["multi_anchor_component"]
                    ].unique(sorted=True)
                    for component_id in multi_ids.tolist():
                        component_pairs = torch.where(
                            metadata["component_ids"] == int(component_id)
                        )[0]
                        active = component_pairs[scores[component_pairs] > tau]
                        if not len(active):
                            continue
                        masks = enumerate_component_matchings(pairs[active])
                        components.append(
                            {
                                "component_id": int(component_id),
                                "active_pair_indices": active.to(torch.int32).contiguous(),
                                "matching_masks": masks.contiguous(),
                            }
                        )
                    anchor_counts = metadata["component_anchor_counts"].clamp_max(255).to(
                        torch.uint8
                    ).contiguous()
                    edge_counts = metadata["component_edge_counts"].clamp_max(255).to(
                        torch.uint8
                    ).contiguous()
                    output_records.append(
                        {
                            "sample_key": str(record["identity"]["sample_key"]),
                            "baseline_scores": scores,
                            "component_anchor_counts": anchor_counts,
                            "component_edge_counts": edge_counts,
                            "components": components,
                            "tensor_sha256": tensor_sha256(
                                scores, anchor_counts, edge_counts
                            ),
                            "component_sha256": component_sha256(components),
                        }
                    )
            output_records.sort(key=lambda record: record["sample_key"])
            counts = {
                "sample_count": len(output_records),
                "pair_count": sum(len(record["baseline_scores"]) for record in output_records),
                "active_component_count": sum(
                    len(record["components"]) for record in output_records
                ),
                "active_pair_count": sum(
                    len(component["active_pair_indices"])
                    for record in output_records
                    for component in record["components"]
                ),
                "matching_count": sum(
                    len(component["matching_masks"])
                    for record in output_records
                    for component in record["components"]
                ),
            }
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": STATUS_COMPLETE,
                "fold": fold,
                "threshold": tau,
                "source": {
                    "base": base_fold_meta,
                    "pair_local": local_fold_meta,
                    "avg_checkpoint": {"path": str(avg_path), "sha256": avg_sha},
                    "spatial_checkpoint": {
                        "path": str(spatial_path),
                        "sha256": spatial_sha,
                    },
                    "profile_checkpoint": {
                        "path": str(profile_path),
                        "sha256": file_sha256(profile_path),
                    },
                },
                "counts": counts,
                "records": output_records,
                "target_aware_fields_in_cache": [],
            }
            validate_component_matching_sidecar(payload, expected_fold=fold)
            if counts["sample_count"] != EXPECTED_SAMPLE_COUNTS[fold] or counts[
                "pair_count"
            ] != EXPECTED_PAIR_COUNTS[fold]:
                raise ComponentMatchingOOFBuildError("built fold counts differ")
            sha = _write_torch_no_overwrite(output_path, payload)
            published.append(output_path)
            fold_references.append(
                {"fold": fold, "path": str(output_path), "sha256": sha, **counts}
            )
            del avg_model, spatial_model, profile_model, records
            if device.type == "cuda":
                torch.cuda.empty_cache()
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "status": STATUS_COMPLETE,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "threshold": tau,
            "folds": fold_references,
            "counts": {
                name: sum(int(fold[name]) for fold in fold_references)
                for name in (
                    "sample_count",
                    "pair_count",
                    "active_component_count",
                    "active_pair_count",
                    "matching_count",
                )
            },
            "base_index": base_metadata,
            "local_index": local_metadata,
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
    parser.add_argument("--local-index", default=helper.DEFAULT_LOCAL_INDEX)
    parser.add_argument(
        "--avg-checkpoint-dir", default=consensus_helper.AVG_CHECKPOINT_DIR
    )
    parser.add_argument(
        "--spatial-checkpoint-dir", default=consensus_helper.SPATIAL_CHECKPOINT_DIR
    )
    parser.add_argument(
        "--profile-checkpoint-dir", default=DEFAULT_PROFILE_CHECKPOINT_DIR
    )
    parser.add_argument("--output-folds", nargs=3, default=DEFAULT_OUTPUT_FOLDS)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(run(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "component_sha256",
    "tensor_sha256",
    "validate_component_matching_sidecar",
]
