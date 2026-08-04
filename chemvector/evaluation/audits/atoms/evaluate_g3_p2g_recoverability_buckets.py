#!/usr/bin/env python3
"""Replay frozen P2-F Atom peaks by P2-G render-recoverability bucket."""

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
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p1w_multiscale_peak_rank as p1w,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT = Path("outputs/audits/chemvector-g3-p2g-render-recoverability/audit.json")
OUTPUT = Path(
    "outputs/audits/chemvector-g3-p2g-render-recoverability/"
    "route_checkpoint_buckets.json"
)
CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-g3-p2f-stride1-center-field-train25k/"
    "step-006250.pt"
)
CHECKPOINT_SHA256 = (
    "9e9d7725e9e9ffecde7bcbe9fbbb095f09204a480b18acbb014d88ef5ac83668"
)
MODEL_TYPE = "chemvector-g3-p2f-stride1-center-field-train25k"
BUDGETS = (128, 256, 1_024)
STATUSES = ("eligible", "ambiguous", "severe")


def route_statuses(audit: dict, sample_keys: set[str]) -> dict[str, str]:
    result = {key: "eligible" for key in sample_keys}
    route = audit["results"]["route100"]
    for item in route["flagged"]:
        key = str(item["sample_key"])
        if key not in result:
            raise ValueError(f"P2-G flagged unknown route sample: {key}")
        result[key] = str(item["recoverability_status"])
    if set(result.values()) - set(STATUSES):
        raise ValueError("P2-G route contains an unknown recoverability status")
    return result


def summarize_buckets(
    records: dict[str, dict], statuses: dict[str, str], budgets=BUDGETS
) -> dict:
    if set(records) != set(statuses):
        raise ValueError("P2-G candidate records and statuses differ")
    output = {}
    for budget in budgets:
        by_status = {}
        for status in ("all",) + STATUSES:
            keys = sorted(
                records if status == "all"
                else (key for key, value in statuses.items() if value == status)
            )
            target_atoms = matched_atoms = exact_images = 0
            failures = []
            for key in keys:
                record = records[key]
                expected = record["expected"]
                matched = p1w.match_count(record["anchors"][:budget], expected)
                target_atoms += len(expected)
                matched_atoms += matched
                exact_images += int(matched == len(expected))
                if matched != len(expected):
                    failures.append({
                        "sample_key": key,
                        "target_atoms": len(expected),
                        "matched_atoms": matched,
                    })
            by_status[status] = {
                "images": len(keys),
                "target_atoms": target_atoms,
                "matched_atoms": matched_atoms,
                "atom_recall": matched_atoms / max(target_atoms, 1),
                "exact_images": exact_images,
                "exact_rate": exact_images / max(len(keys), 1),
                "failures": failures,
            }
        output[str(budget)] = by_status
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise RuntimeError(f"P2-G bucket output already exists: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("P2-G checkpoint replay requires CUDA BF16")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "g3-p2g-render-recoverability-0.2":
        raise ValueError("P2-G recoverability audit schema differs")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(20260713)
    np.random.seed(20260713)
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    route, route_audit = p1a.load_route_manifest()
    statuses = route_statuses(audit, {record.sample_key for record in route})
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    candidates, lineage = p1w.collect_candidates(
        checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA256,
        expected_model_type=MODEL_TYPE,
        field_factory=ContinuousStride1AtomCenterField,
        domains={"route100": route},
        collator=collator,
        device=torch.device("cuda"),
    )
    metrics = summarize_buckets(candidates["route100"], statuses)
    top128 = metrics["128"]
    checks = {
        "full_top128_matches_frozen_p2f": (
            top128["all"]["matched_atoms"] == 3_074
            and top128["all"]["target_atoms"] == 3_080
            and top128["all"]["exact_images"] == 99
        ),
        "eligible_top128_complete": (
            top128["eligible"]["matched_atoms"]
            == top128["eligible"]["target_atoms"]
            and top128["eligible"]["exact_images"] == top128["eligible"]["images"]
        ),
        "ambiguous_top128_complete": (
            top128["ambiguous"]["matched_atoms"]
            == top128["ambiguous"]["target_atoms"]
            and top128["ambiguous"]["exact_images"]
            == top128["ambiguous"]["images"]
        ),
        "only_severe_sample_fails_top128": (
            [item["sample_key"] for item in top128["all"]["failures"]]
            == ["train_001276_stress"]
            and top128["severe"]["images"] == 1
        ),
    }
    result = {
        "schema_version": "g3-p2g-checkpoint-recoverability-buckets-0.1",
        "decision": (
            "residual_isolated_to_severe_render"
            if all(checks.values()) else "recoverability_attribution_failed"
        ),
        "metric_policy": {
            "formal_p2f_decision_remains": "route_causal_no_go",
            "full_route100_is_primary": True,
            "bucket_metrics_are_parallel_diagnostics_only": True,
        },
        "checks": checks,
        "budgets": list(BUDGETS),
        "metrics": metrics,
        "recoverability_audit": str(args.audit.resolve()),
        "route_manifest": route_audit,
        "lineage": lineage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(json.dumps({
        "decision": result["decision"],
        "checks": checks,
        "metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
