#!/usr/bin/env python3
"""Independently replay the final P1-X dual-domain capacity gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from chemvector.components.atoms.g3_multiscale_peak_set_ranker import (
    G3MultiscalePeakSetRanker,
)
from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    ContinuousStride2AtomCenterField,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1w_multiscale_peak_rank as p1w
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1u_adjacent_atom_capacity as p1u
from chemvector.training.atoms import train_g3_p1v_continuous_stride1_capacity as p1v
from chemvector.training.atoms import train_g3_p1x_multiscale_peak_selector_capacity as p1x
from chemvector.training.graph.scenes.g3_structural_data import (
    OVERFIT32_MANIFEST,
    load_g3_overfit32_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


CHECKPOINT = p1x.OUTPUT / "step-004800.pt"
CHECKPOINT_SHA256 = "630431b4d9aa0cc0620f9ae0e3e1a0b12587bafda8c00247b9b1b227b290052b"
DEFAULT_OUTPUT = Path(f"outputs/audits/{p1x.MODEL_TYPE}-final")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P1-X final audit output is not empty: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-X final audit requires CUDA BF16")
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("P1-X final checkpoint SHA256 differs")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(20260713)
    np.random.seed(20260713)
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint_checks = {
        "model_type": payload.get("model_type") == p1x.MODEL_TYPE,
        "global_step_4800": payload.get("global_step") == p1x.STEPS,
        "samples_seen_19200": payload.get("samples_seen") == p1x.EXPECTED_EXPOSURE,
        "candidate_protocol": payload.get("candidate_protocol")
        == {
            "stride2_k": p1x.STRIDE2_K,
            "stride1_k": p1x.STRIDE1_K,
            "total": 128,
            "target_dependent_merging": False,
            "pixel_radius_clustering": False,
        },
        "dynamic_matching": payload.get("matching")
        == "maximum-cardinality-score-ranked-one-to-one",
        "known_count_not_model_input": payload.get("known_count_is_model_input") is False,
    }
    if not all(checkpoint_checks.values()):
        raise ValueError("P1-X final checkpoint contract differs")

    device = torch.device("cuda")
    overfit_manifest, overfit_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    adjacent_manifest, adjacent_manifest_audit, _ = p1u.load_adjacent_manifest()
    domains = {"overfit32": overfit_manifest, "adjacent32": adjacent_manifest}
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    stride2, stride2_lineage = p1x._collect_scale(
        checkpoint=p1w.P1U_CHECKPOINT,
        checkpoint_sha256=p1w.P1U_CHECKPOINT_SHA256,
        expected_model_type=p1u.MODEL_TYPE,
        field_factory=ContinuousStride2AtomCenterField,
        feature_name="high_resolution_features",
        maximum_objects=p1x.STRIDE2_K,
        domains=domains,
        collator=collator,
        device=device,
    )
    stride1, stride1_lineage = p1x._collect_scale(
        checkpoint=p1w.P1V_CHECKPOINT,
        checkpoint_sha256=p1w.P1V_CHECKPOINT_SHA256,
        expected_model_type=p1v.MODEL_TYPE,
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=p1x.STRIDE1_K,
        domains=domains,
        collator=collator,
        device=device,
    )
    cache, cache_audit = p1x.merge_scale_records(stride2, stride1)
    ranker = G3MultiscalePeakSetRanker().to(device)
    ranker.load_state_dict(payload["multiscale_peak_set_ranker"], strict=True)
    ranker.eval()
    metrics = {
        domain: p1x.evaluate(ranker, records, device=device)
        for domain, records in cache.items()
    }
    repeated = {
        domain: p1x.evaluate(ranker, records, device=device)
        for domain, records in cache.items()
    }
    decision, capacity_checks = p1x.capacity_gate(metrics)
    saved_metrics = payload["metrics"]["metrics"]
    replay_checks = {
        "candidate_cache_sha256": cache_audit["candidate_cache_sha256"]
        == payload["lineage"]["cache"]["candidate_cache_sha256"],
        "stride2_cache_sha256": stride2_lineage["candidate_tensors_sha256"]
        == payload["lineage"]["p1u"]["candidate_tensors_sha256"],
        "stride1_cache_sha256": stride1_lineage["candidate_tensors_sha256"]
        == payload["lineage"]["p1v"]["candidate_tensors_sha256"],
        "metrics_exactly_match_checkpoint": metrics == saved_metrics,
        "repeat_metrics": repeated == metrics,
        "capacity_go": decision == "dual_domain_selector_capacity_go",
        "all_capacity_checks": all(capacity_checks.values()),
    }
    audit_passed = all(checkpoint_checks.values()) and all(replay_checks.values())
    audit = {
        "schema_version": "g3-p1x-final-audit-0.1",
        "decision": (
            "audit_passed_dual_domain_selector_capacity_go"
            if audit_passed
            else "audit_failed"
        ),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_checks": checkpoint_checks,
        "replay_checks": replay_checks,
        "capacity_checks": capacity_checks,
        "metrics": metrics,
        "cache_audit": cache_audit,
        "lineage": {
            "overfit32_manifest": overfit_audit,
            "adjacent32_manifest": adjacent_manifest_audit,
            "stride2": stride2_lineage,
            "stride1": stride1_lineage,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({"decision": audit["decision"], "metrics": metrics}, indent=2))
    if not audit_passed:
        raise RuntimeError("P1-X independent final audit failed")


if __name__ == "__main__":
    main()
