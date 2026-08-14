#!/usr/bin/env python3
"""One-shot internal-test confirmation for the frozen v1 P1-A model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.vision.audit_v1p1r_semantic_ink_threshold_confirm import (
    _audit_loader,
    _gate_metrics,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    MANIFEST_SHA256,
    evaluate,
    load_safe_manifest,
)


AUDIT_NAME = "chemvector-v1p1ac-binary-ink-internal-confirm"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DEFAULT_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v1p1a-binary-ink-finetune/best.pt"
)
CHECKPOINT_SHA256 = (
    "490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74"
)
P1A_RESULT = Path(
    "outputs/checkpoints/chemvector-v1p1a-binary-ink-finetune/result.json"
)
P1A_RESULT_SHA256 = (
    "ce17cfe8fb0e41b877d3a0ece8d0f6f1de3f082404135b49f67394c21a647612"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v1 P1-A-C confirmation requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P1-A-C output is not empty: {args.output_dir}")
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("v1 P1-A-C checkpoint SHA256 differs")
    if file_sha256(P1A_RESULT) != P1A_RESULT_SHA256:
        raise ValueError("v1 P1-A result SHA256 differs")
    p1a_result = json.loads(P1A_RESULT.read_text(encoding="utf-8"))
    if p1a_result["decision"] != "binary_ink_finetune_go":
        raise ValueError("v1 P1-A validation decision is not GO")

    rows = load_safe_manifest()
    internal_rows = [row for row in rows if row["split"] == "internal_test"]
    if len(internal_rows) != 496:
        raise ValueError("v1 P1-A-C internal-test count differs")
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location=device)
    config = SemanticInkSeparatorConfig(**payload["config"])
    model = SemanticInkSeparator(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    started = time.perf_counter()
    metrics = evaluate(
        model,
        _audit_loader(internal_rows, args.batch_size, args.workers),
        device,
        bond_threshold=0.5,
        text_threshold=0.5,
    )
    if metrics["samples"] != 496:
        raise ValueError("v1 P1-A-C did not read exactly 496 samples")
    checks = _gate_metrics(metrics) | {
        "internal_test_samples_exact496": metrics["samples"] == 496,
        "checkpoint_frozen": True,
        "prediction_thresholds_fixed_0_5": True,
        "validation_not_reselected": True,
    }
    result = {
        "schema_version": "chemvector-v1p1ac-binary-ink-internal-confirm-0.1",
        "audit_name": AUDIT_NAME,
        "decision": "binary_ink_internal_confirm_go" if all(checks.values()) else "binary_ink_internal_confirm_no_go",
        "checks": checks,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "p1a_result_sha256": P1A_RESULT_SHA256,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "prediction_thresholds": {"bond": 0.5, "text": 0.5},
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
