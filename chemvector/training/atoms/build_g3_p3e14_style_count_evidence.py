#!/usr/bin/env python3
"""Build resumable frozen Count evidence shards for P3-E14."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2b_train5k_multiscale_reachability as p2b,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2d_stride1_top128_cache as p2d,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e14_style_count_manifest as manifest_protocol,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p1x_multiscale_peak_selector_capacity as p1x,
)
from chemvector.training.atoms import train_g3_p1z_continuous_center_field_train5k as p1z
from chemvector.training.atoms import train_g3_p3e9_deepsets_count as p3e9
from chemvector.training.graph.scenes.g3_structural_data import (
    G3SampleManifestRecord,
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e14-style90932-count-evidence-b24"
OUTPUT = Path(f"outputs/cache/{MODEL_TYPE}")
MANIFEST_SHA256 = "5a234b4984d045db635c0196ce481653574ff39897892db0330ab021a063dda1"
SEED = 20260713


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--maximum-shards", type=int)
    parser.add_argument("--candidate-batch-size", type=int, default=24)
    return parser


def load_manifest() -> tuple[list[G3SampleManifestRecord], dict]:
    return load_g3_sample_manifest(
        manifest_protocol.MANIFEST,
        expected_sha256=MANIFEST_SHA256,
        expected_samples=manifest_protocol.EXPECTED_IMAGES,
        expected_molecules=manifest_protocol.EXPECTED_MOLECULES,
        expected_variants=frozenset(manifest_protocol.EXPECTED_VARIANTS),
    )


def group_manifest_by_shard(
    records: list[G3SampleManifestRecord],
) -> list[tuple[str, list[G3SampleManifestRecord]]]:
    groups = defaultdict(list)
    order = []
    for record in records:
        if record.shard not in groups:
            order.append(record.shard)
        groups[record.shard].append(record)
    return [(shard, groups[shard]) for shard in order]


def sample_keys_sha256(records: list[dict] | list[G3SampleManifestRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        key = record["sample_key"] if isinstance(record, dict) else record.sample_key
        digest.update(f"{key}\n".encode())
    return digest.hexdigest()


def sample_key_set_sha256(records: list[dict] | list[G3SampleManifestRecord]) -> str:
    keys = [
        record["sample_key"] if isinstance(record, dict) else record.sample_key
        for record in records
    ]
    return hashlib.sha256("".join(f"{key}\n" for key in sorted(keys)).encode()).hexdigest()


def validate_existing_shard(
    path: Path,
    *,
    source_shard: str,
    expected_records: list[G3SampleManifestRecord],
    candidate_batch_size: int,
) -> dict | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if (
        payload.get("schema_version") != "g3-p3e14-count-evidence-shard-0.1"
        or payload.get("source_shard") != str(Path(source_shard).resolve())
        or payload.get("manifest_sha256") != MANIFEST_SHA256
        or payload.get("candidate_batch_size") != candidate_batch_size
        or not isinstance(records, list)
        or len(records) != len(expected_records)
        or sample_key_set_sha256(records) != sample_key_set_sha256(expected_records)
    ):
        raise ValueError(f"P3-E14 existing evidence shard is invalid: {path}")
    return payload


def build_shard(
    manifest: list[G3SampleManifestRecord],
    *,
    base_model,
    expert_model,
    device: torch.device,
    candidate_batch_size: int,
) -> tuple[list[dict], dict]:
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    records, candidate_lineage = p1x._collect_scale(
        checkpoint=p2b.STRIDE1_CHECKPOINT,
        checkpoint_sha256=p2b.STRIDE1_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=p2d.CANDIDATES,
        domains={"style": manifest},
        collator=collator,
        device=device,
        batch_size=candidate_batch_size,
    )
    cache, candidate_audit = p2d.format_cache(records)
    evidence = p3e9.collect_evidence(
        (base_model, expert_model), {"style": cache["style"]}, device=device
    )["style"]
    return evidence, {
        "candidate_lineage": candidate_lineage,
        "candidate_audit": candidate_audit,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.maximum_shards is not None and args.maximum_shards < 1:
        raise ValueError("P3-E14 maximum shards must be positive")
    if args.candidate_batch_size < 1:
        raise ValueError("P3-E14 candidate batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E14 evidence generation requires CUDA BF16")
    if file_sha256(manifest_protocol.MANIFEST) != MANIFEST_SHA256:
        raise ValueError("P3-E14 manifest SHA256 differs")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
        (p2b.STRIDE1_CHECKPOINT, p2b.STRIDE1_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E14 checkpoint SHA differs: {path}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    manifest, manifest_audit = load_manifest()
    groups = group_manifest_by_shard(manifest)
    selected_groups = groups[: args.maximum_shards] if args.maximum_shards else groups
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    started = time.perf_counter()
    completed = []
    for index, (source_shard, records) in enumerate(selected_groups):
        output_path = args.output_dir / f"evidence-{index:05d}.pt"
        existing = validate_existing_shard(
            output_path,
            source_shard=source_shard,
            expected_records=records,
            candidate_batch_size=args.candidate_batch_size,
        )
        shard_started = time.perf_counter()
        if existing is None:
            evidence, lineage = build_shard(
                records,
                base_model=base_model,
                expert_model=expert_model,
                device=device,
                candidate_batch_size=args.candidate_batch_size,
            )
            payload = {
                "schema_version": "g3-p3e14-count-evidence-shard-0.1",
                "model_type": MODEL_TYPE,
                "manifest_sha256": MANIFEST_SHA256,
                "source_shard": str(Path(source_shard).resolve()),
                "candidate_batch_size": args.candidate_batch_size,
                "sample_keys_sha256": sample_keys_sha256(evidence),
                "sample_key_set_sha256": sample_key_set_sha256(evidence),
                "records": evidence,
                "lineage": lineage,
            }
            _atomic_torch_save(output_path, payload)
            resumed = False
        else:
            payload = existing
            resumed = True
        current = {
            "index": index,
            "path": str(output_path.resolve()),
            "sha256": file_sha256(output_path),
            "source_shard": str(Path(source_shard).resolve()),
            "samples": len(payload["records"]),
            "sample_keys_sha256": payload["sample_keys_sha256"],
            "resumed": resumed,
            "elapsed_seconds": time.perf_counter() - shard_started,
        }
        completed.append(current)
        print(json.dumps({
            "completed_shards": len(completed),
            "total_shards": len(groups),
            **current,
        }, separators=(",", ":")), flush=True)

    complete = len(selected_groups) == len(groups)
    summary = {
        "schema_version": "g3-p3e14-count-evidence-index-0.1",
        "status": "passed" if complete else "partial",
        "model_type": MODEL_TYPE,
        "samples": sum(value["samples"] for value in completed),
        "shards": completed,
        "manifest": manifest_audit,
        "lineage": {
            "candidate_checkpoint": str(p2b.STRIDE1_CHECKPOINT.resolve()),
            "candidate_checkpoint_sha256": p2b.STRIDE1_SHA256,
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "index.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "samples": summary["samples"],
        "shards": len(completed),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
