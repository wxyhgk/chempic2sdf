#!/usr/bin/env python3
"""Fine-tune v1 P1 against the frozen binary foreground definition."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
    semantic_ink_loss,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    GO_GATES,
    MANIFEST_SHA256,
    _loader,
    _split_rows,
    evaluate,
    load_safe_manifest,
)


MODEL_NAME = "chemvector-v1p1a-binary-ink-finetune"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
DEFAULT_INIT = Path(
    "outputs/checkpoints/chemvector-v1p1-semantic-ink-separation-pilot/best.pt"
)
INIT_SHA256 = (
    "e4f726b80f73e46a71619f8f4a42ff3b0042a3156cc86c0a4f584154b574263c"
)
ALPHA_FOREGROUND_THRESHOLD = 8.0 / 255.0


def binary_ink_targets(targets: torch.Tensor) -> torch.Tensor:
    return (targets >= ALPHA_FOREGROUND_THRESHOLD).to(dtype=torch.float32)


def _save_checkpoint(
    path: Path,
    model: SemanticInkSeparator,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    train_samples_seen: int,
) -> None:
    payload = {
        "schema_version": "chemvector-v1p1a-binary-ink-checkpoint-0.1",
        "model_name": MODEL_NAME,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 1,
        "metrics": metrics,
        "train_samples_seen": train_samples_seen,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "init_checkpoint_sha256": INIT_SHA256,
        "target_policy": "alpha_gte_8_over_255",
        "internal_test_read": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checks(metrics: dict) -> dict[str, bool]:
    return {
        "validation_samples_exact496": metrics["samples"] == 496,
        "bond_iou_min90pct": metrics["bond"]["iou"] >= GO_GATES["bond_iou"],
        "text_iou_min75pct": metrics["text"]["iou"] >= GO_GATES["text_iou"],
        "foreground_iou_min90pct": metrics["foreground"]["iou"]
        >= GO_GATES["foreground_iou"],
        "stress_bond_iou_min88pct": metrics["bond_by_variant"]["stress"]["iou"]
        >= GO_GATES["stress_bond_iou"],
        "bond_on_text_only_max10pct": metrics["bond_on_text_only_rate"]
        <= GO_GATES["bond_on_text_only_rate_max"],
        "fixed_prediction_threshold_0_5": True,
        "internal_test_unread": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v1 P1-A binary ink fine-tuning requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P1-A output is not empty: {args.output_dir}")
    if file_sha256(args.init_checkpoint) != INIT_SHA256:
        raise ValueError("v1 P1-A init checkpoint SHA256 differs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_safe_manifest()
    train_rows = _split_rows(rows, "train")
    validation_rows = _split_rows(rows, "validation")
    device = torch.device("cuda")
    init = torch.load(args.init_checkpoint, map_location=device)
    config = SemanticInkSeparatorConfig(**init["config"])
    model = SemanticInkSeparator(config).to(device)
    model.load_state_dict(init["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    seen = 0
    running = 0.0
    model.train()
    loader = _loader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    for step, batch in enumerate(loader, 1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = binary_ink_targets(
            batch["targets"].to(device, non_blocking=True)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixels)
            losses = semantic_ink_loss(output, targets, model.config)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"v1 P1-A non-finite loss at step {step}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_samples = pixels.size(0)
        seen += batch_samples
        running += float(losses["loss"].detach()) * batch_samples
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {"step": step, "samples": seen, "loss": running / seen}
                ),
                flush=True,
            )
    if seen != 8_856:
        raise ValueError(f"v1 P1-A read {seen} train samples, expected 8856")

    metrics = evaluate(
        model,
        _loader(
            validation_rows,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
        bond_threshold=0.5,
        text_threshold=0.5,
    )
    checks = _checks(metrics)
    checkpoint = args.output_dir / "best.pt"
    _save_checkpoint(checkpoint, model, optimizer, metrics, seen)
    result = {
        "schema_version": "chemvector-v1p1a-binary-ink-result-0.1",
        "model_name": MODEL_NAME,
        "decision": "binary_ink_finetune_go" if all(checks.values()) else "binary_ink_finetune_no_go",
        "checks": checks,
        "gates": GO_GATES,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": asdict(config),
        "init_checkpoint_sha256": INIT_SHA256,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "train_samples": 8_856,
        "train_samples_seen": seen,
        "validation_samples": 496,
        "internal_test_samples_read": 0,
        "target_policy": "alpha_gte_8_over_255",
        "prediction_thresholds": {"bond": 0.5, "text": 0.5},
        "train_loss": running / seen,
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
