#!/usr/bin/env python3
"""Select a fixed total-128 scale allocation on route-select100."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    ContinuousStride2AtomCenterField,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1w_multiscale_peak_rank as p1w
from chemvector.evaluation.audits.atoms import audit_g3_p2b_train5k_multiscale_reachability as p2b
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1z_continuous_center_field_train5k as p1z
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


DEFAULT_OUTPUT = Path("outputs/audits/chemvector-g3-p2c-route128-scale-allocation")
ALLOCATIONS = tuple((128 - stride1_k, stride1_k) for stride1_k in range(0, 129, 16))
EXPANDED_BUDGETS = (
    (32, 128), (16, 144), (0, 160),
    (64, 128), (32, 160), (16, 176), (0, 192),
    (128, 128), (64, 192), (32, 224), (0, 256),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def choose_allocation(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda value: (
            value["metrics"]["exact_images"],
            value["metrics"]["matched_atoms"],
            -value["stride1_k"],
        ),
    )


def failure_diagnostics(stride2: dict, stride1: dict, *, left_k: int, right_k: int) -> list[dict]:
    failures = []
    for sample_key in sorted(stride2):
        expected = stride2[sample_key]["expected"]
        candidates = torch.cat(
            (stride2[sample_key]["anchors"][:left_k], stride1[sample_key]["anchors"][:right_k])
        )
        matched = p1w.match_count(candidates, expected)
        if matched == len(expected):
            continue
        compatible = torch.cdist(expected.float(), candidates.float()) * 768.0 <= 6.0
        failures.append({
            "sample_key": sample_key,
            "target_atoms": len(expected),
            "matched_atoms": matched,
            "targets_without_compatible_peak": torch.where(compatible.sum(dim=1) == 0)[0].tolist(),
            "target_anchors_without_compatible_peak": expected[
                compatible.sum(dim=1) == 0
            ].tolist(),
        })
    return failures


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-C output is not empty: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("P2-C requires CUDA BF16")
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

    route, route_audit = p1a.load_route_manifest()
    domains = {"route100": route}
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    stride2, stride2_lineage = p1w.collect_candidates(
        checkpoint=p2b.STRIDE2_CHECKPOINT,
        checkpoint_sha256=p2b.STRIDE2_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride2"]["model_type"],
        field_factory=ContinuousStride2AtomCenterField,
        domains=domains,
        collator=collator,
        device=device,
    )
    stride1, stride1_lineage = p1w.collect_candidates(
        checkpoint=p2b.STRIDE1_CHECKPOINT,
        checkpoint_sha256=p2b.STRIDE1_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        field_factory=ContinuousStride1AtomCenterField,
        domains=domains,
        collator=collator,
        device=device,
    )
    results = []
    for stride2_k, stride1_k in ALLOCATIONS + EXPANDED_BUDGETS:
        metrics = p1w.evaluate_budget(
            stride2["route100"], stride1["route100"],
            stride2_k=stride2_k, stride1_k=stride1_k,
        )
        results.append({
            "stride2_k": stride2_k,
            "stride1_k": stride1_k,
            "metrics": metrics,
        })
    selected = choose_allocation(results)
    selected_failures = failure_diagnostics(
        stride2["route100"], stride1["route100"],
        left_k=selected["stride2_k"], right_k=selected["stride1_k"],
    )
    maximum_failures = failure_diagnostics(
        stride2["route100"], stride1["route100"], left_k=1024, right_k=1024
    )
    decision = (
        "route128_allocation_full_reachability_go"
        if selected["metrics"]["matched_atoms"] == selected["metrics"]["target_atoms"]
        and selected["metrics"]["exact_images"] == len(route)
        else "route128_allocation_improved_not_complete"
    )
    audit = {
        "schema_version": "g3-p2c-route128-scale-allocation-0.1",
        "decision": decision,
        "selection_rule": "max exact, then matched atoms, then fewer stride1 candidates",
        "allocations": [
            {"stride2_k": left, "stride1_k": right}
            for left, right in ALLOCATIONS + EXPANDED_BUDGETS
        ],
        "results": results,
        "selected_allocation": selected,
        "selected_failures": selected_failures,
        "top1024_per_scale_failures": maximum_failures,
        "manifest": route_audit,
        "lineage": {"stride2": stride2_lineage, "stride1": stride1_lineage},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({
        "decision": decision,
        "selected_allocation": selected,
        "selected_failures": selected_failures,
        "top1024_per_scale_failures": maximum_failures,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
