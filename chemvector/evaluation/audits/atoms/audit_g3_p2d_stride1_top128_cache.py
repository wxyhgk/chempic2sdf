#!/usr/bin/env python3
"""Build the train5k/route cache for the selected stride-1 top128 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
)
from chemvector.evaluation.audits.atoms import audit_g3_p2b_train5k_multiscale_reachability as p2b
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1x_multiscale_peak_selector_capacity as p1x
from chemvector.training.atoms import train_g3_p1z_continuous_center_field_train5k as p1z
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


DEFAULT_OUTPUT = Path("outputs/audits/chemvector-g3-p2d-stride1-top128-cache")
CANDIDATES = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def representable_target_mask(
    candidates: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    valid_indices = torch.where(target_mask)[0]
    expected = targets[valid_indices].double()
    distances = torch.cdist(candidates.double(), expected)
    compatible = distances <= 6.0 / 768.0
    penalty = float(max(len(candidates), len(expected)) + 1)
    cost = torch.where(compatible, distances, torch.full_like(distances, penalty))
    rows, columns = linear_sum_assignment(cost.numpy())
    output = torch.zeros_like(target_mask)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if bool(compatible[row, column]):
            output[valid_indices[column]] = True
    return output


def format_cache(records: dict[str, dict[str, dict]]) -> tuple[dict[str, list[dict]], dict]:
    cache = {}
    digest = hashlib.sha256()
    reachability = {}
    for domain in sorted(records):
        current = []
        matched_atoms = target_atoms = exact_images = 0
        for sample_key in sorted(records[domain]):
            source = records[domain][sample_key]
            full_mask = source["atom_mask"].clone()
            representable = representable_target_mask(
                source["candidate_anchors"], source["atom_anchors"], full_mask
            )
            target_count = int(full_mask.sum())
            matched = int(representable.sum())
            matched_atoms += matched
            target_atoms += target_count
            exact_images += int(matched == target_count)
            record = {
                "sample_key": sample_key,
                "candidate_ids": source["source_ids"].clone(),
                "candidate_features": source["candidate_features"].clone(),
                "candidate_anchors": source["candidate_anchors"].clone(),
                "raw_scores": source["raw_scores"].clone(),
                "normalized_scores": source["normalized_scores"].clone(),
                "scale_ids": torch.ones(CANDIDATES, dtype=torch.long),
                "scale_ranks": torch.arange(CANDIDATES).float() / (CANDIDATES - 1),
                "atom_anchors": source["atom_anchors"].clone(),
                "atom_mask": representable,
                "full_atom_mask": full_mask,
            }
            for name, value in record.items():
                if isinstance(value, torch.Tensor):
                    digest.update(name.encode())
                    digest.update(value.numpy().tobytes())
            current.append(record)
        cache[domain] = current
        reachability[domain] = {
            "matched_atoms": matched_atoms,
            "target_atoms": target_atoms,
            "atom_recall": matched_atoms / target_atoms,
            "exact_images": exact_images,
            "exact_rate": exact_images / len(current),
        }
    return cache, {
        "candidate_protocol": "train5k-stride1-regional-top128",
        "candidate_count": CANDIDATES,
        "candidate_cache_sha256": digest.hexdigest(),
        "reachability": reachability,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D output is not empty: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(20260713)
    np.random.seed(20260713)
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    train, train_audit = p1a.load_train_manifest()
    route, route_audit = p1a.load_route_manifest()
    domains = {"train5k": train, "route100": route}
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    started = time.perf_counter()
    records, lineage = p1x._collect_scale(
        checkpoint=p2b.STRIDE1_CHECKPOINT,
        checkpoint_sha256=p2b.STRIDE1_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=CANDIDATES,
        domains=domains,
        collator=collator,
        device=device,
    )
    cache, cache_audit = format_cache(records)
    del records
    route_reach = cache_audit["reachability"]["route100"]
    checks = {
        "route_matches_p2c_atoms": route_reach["matched_atoms"] == 3_073,
        "route_matches_p2c_exact": route_reach["exact_images"] == 98,
        "fixed_candidate_count128": cache_audit["candidate_count"] == 128,
    }
    decision = (
        "stride1_top128_cache_passed" if all(checks.values()) else "stride1_top128_cache_failed"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = args.output_dir / "candidate-cache.pt"
    _atomic_torch_save(cache_path, cache)
    audit = {
        "schema_version": "g3-p2d-stride1-top128-cache-0.1",
        "decision": decision,
        "checks": checks,
        "candidate_protocol": cache_audit,
        "candidate_cache": str(cache_path.resolve()),
        "candidate_cache_file_sha256": file_sha256(cache_path),
        "manifests": {"train5k": train_audit, "route100": route_audit},
        "lineage": lineage,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({"decision": decision, **cache_audit}, indent=2))


if __name__ == "__main__":
    main()
