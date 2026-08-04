#!/usr/bin/env python3
"""Build the P2-B 96+32 cache from train5k-trained center fields."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    ContinuousStride2AtomCenterField,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1x_multiscale_peak_selector_capacity as p1x
from chemvector.training.atoms import train_g3_p1z_continuous_center_field_train5k as p1z
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


DEFAULT_OUTPUT = Path("outputs/audits/chemvector-g3-p2b-train5k-multiscale-reachability")
STRIDE2_CHECKPOINT = p1z.SCALE_PROTOCOLS["stride2"]["output"] / "step-006250.pt"
STRIDE2_SHA256 = "fb0398d2d5e4c8d5b0d83405892df3469054051aa3627db554df80e35e782753"
STRIDE1_CHECKPOINT = p1z.SCALE_PROTOCOLS["stride1"]["output"] / "step-006250.pt"
STRIDE1_SHA256 = "2869451245f50f98f5e3f5259179ca338e82ad81a8fd59de2d2037a10056404b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-B output is not empty: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("P2-B requires CUDA BF16")
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
    stride2, stride2_lineage = p1x._collect_scale(
        checkpoint=STRIDE2_CHECKPOINT,
        checkpoint_sha256=STRIDE2_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride2"]["model_type"],
        field_factory=ContinuousStride2AtomCenterField,
        feature_name="high_resolution_features",
        maximum_objects=p1x.STRIDE2_K,
        domains=domains,
        collator=collator,
        device=device,
    )
    stride1, stride1_lineage = p1x._collect_scale(
        checkpoint=STRIDE1_CHECKPOINT,
        checkpoint_sha256=STRIDE1_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=p1x.STRIDE1_K,
        domains=domains,
        collator=collator,
        device=device,
    )
    cache, cache_audit = p1x.merge_scale_records(stride2, stride1)
    del stride2, stride1
    reachability = cache_audit["reachability"]
    checks = {
        "train_all_atoms": reachability["train5k"]["matched_atoms"]
        == reachability["train5k"]["target_atoms"],
        "train_all_images": reachability["train5k"]["exact_images"] == len(train),
        "route_all_atoms": reachability["route100"]["matched_atoms"]
        == reachability["route100"]["target_atoms"],
        "route_all_images": reachability["route100"]["exact_images"] == len(route),
        "route_recall_improves_old_96_32": reachability["route100"]["matched_atoms"]
        > 3_032,
        "route_exact_improves_old_96_32": reachability["route100"]["exact_images"] > 69,
    }
    decision = (
        "train5k_fields_fixed128_reachability_go"
        if all(checks.values())
        else "train5k_fields_fixed128_reachability_improved_not_complete"
        if checks["route_recall_improves_old_96_32"]
        and checks["route_exact_improves_old_96_32"]
        else "train5k_fields_fixed128_reachability_no_go"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = args.output_dir / "candidate-cache.pt"
    _atomic_torch_save(cache_path, cache)
    audit = {
        "schema_version": "g3-p2b-train5k-multiscale-reachability-0.1",
        "decision": decision,
        "checks": checks,
        "candidate_protocol": cache_audit,
        "old_capacity64_route96_32": {
            "matched_atoms": 3_032,
            "target_atoms": 3_080,
            "exact_images": 69,
        },
        "candidate_cache": str(cache_path.resolve()),
        "candidate_cache_file_sha256": file_sha256(cache_path),
        "manifests": {"train5k": train_audit, "route100": route_audit},
        "lineage": {"stride2": stride2_lineage, "stride1": stride1_lineage},
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({"decision": decision, "reachability": reachability}, indent=2))


if __name__ == "__main__":
    main()
