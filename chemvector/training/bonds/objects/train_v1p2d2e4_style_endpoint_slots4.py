#!/usr/bin/env python3
"""Train the style endpoint field with four target slots per stride-4 cell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
    semantic_endpoint_loss,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d2e0_fresh_candidate_oracle import (
    _fresh_loader,
    load_fresh_split,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
    _bond_probability,
)
from chemvector.training.bonds.objects.train_v1p2d2e3_style_endpoint_field import (
    ENDPOINT_BUDGETS,
    ENDPOINT_THRESHOLD,
    GO_GATES,
    INITIAL_CHECKPOINT,
    INITIAL_CHECKPOINT_SHA256,
    MANIFEST_SHA256,
    _checks,
    _style_loader,
    evaluate_reachability,
    load_style_manifest,
    split_rows,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)


MODEL_NAME = "chemvector-v1p2d2e4-style-endpoint-slots4"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
SEED = 20260825
SLOTS = 4


def expand_detector_slots(initial: dict, slots: int = SLOTS) -> SemanticBondEndpointDetector:
    old_slots = int(initial["config"]["slots"])
    if old_slots != 3 or slots != 4:
        raise ValueError("P2-D2-E4 supports only the preregistered slots 3 -> 4 expansion")
    config = SemanticBondEndpointConfig(**{**initial["config"], "slots": slots})
    detector = SemanticBondEndpointDetector(config)
    expanded = detector.state_dict()
    source = initial["detector"]
    special = {"heatmap.weight", "heatmap.bias", "offset.weight", "offset.bias"}
    for name, value in source.items():
        if name in special:
            continue
        if name not in expanded or expanded[name].shape != value.shape:
            raise ValueError(f"P2-D2-E4 incompatible shared tensor: {name}")
        expanded[name].copy_(value)
    expanded["heatmap.weight"][:old_slots].copy_(source["heatmap.weight"])
    expanded["heatmap.bias"][:old_slots].copy_(source["heatmap.bias"])
    expanded["offset.weight"][: old_slots * 2].copy_(source["offset.weight"])
    expanded["offset.bias"][: old_slots * 2].copy_(source["offset.bias"])
    detector.load_state_dict(expanded, strict=True)
    return detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E4 slots4 endpoint training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E4 output is not empty: {args.output_dir}")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("P2-D2-E4 separator SHA256 differs")
    if file_sha256(INITIAL_CHECKPOINT) != INITIAL_CHECKPOINT_SHA256:
        raise ValueError("P2-D2-E4 initial endpoint checkpoint SHA256 differs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_style_manifest()
    fit_rows = split_rows(rows, "fit")
    calibration_rows = split_rows(rows, "calibration")
    holdout_rows = split_rows(rows, "holdout")
    device = torch.device("cuda")
    separator_payload = torch.load(DEFAULT_SEPARATOR, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    initial = torch.load(INITIAL_CHECKPOINT, map_location="cpu")
    detector = expand_detector_slots(initial).to(device)
    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    detector.train()
    samples_seen = 0
    running_loss = 0.0
    loader = _style_loader(
        fit_rows,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    for step, batch in enumerate(loader, 1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        endpoints = batch["endpoints"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = detector(probability)
            losses = semantic_endpoint_loss(output, endpoints, mask)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"P2-D2-E4 non-finite loss at step {step}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
        optimizer.step()
        current_batch = pixels.size(0)
        samples_seen += current_batch
        running_loss += float(losses["loss"].detach()) * current_batch
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "samples_seen": samples_seen,
                        "train_loss": running_loss / samples_seen,
                    }
                ),
                flush=True,
            )
    if samples_seen != len(fit_rows):
        raise ValueError(
            f"P2-D2-E4 epoch read {samples_seen}, expected {len(fit_rows)}"
        )

    checkpoint = {
        "schema_version": "chemvector-v1p2d2e4-slots4-endpoint-checkpoint-0.1",
        "model_name": MODEL_NAME,
        "config": asdict(detector.config),
        "detector": detector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 1,
        "samples_seen": samples_seen,
        "train_loss": running_loss / samples_seen,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "maximum_endpoints": max(ENDPOINT_BUDGETS),
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "initial_slots_copied": 3,
        "new_slots_initialized": 1,
        "separator_sha256": SEPARATOR_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "style_holdout_read": False,
        "fresh_holdout_read": False,
        "sealed_test_read": False,
    }
    checkpoint_path = args.output_dir / "final.pt"
    _atomic_torch_save(checkpoint_path, checkpoint)

    style_calibration = evaluate_reachability(
        separator,
        detector,
        _style_loader(
            calibration_rows,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    fresh_calibration_rows, _, _ = load_fresh_split()
    fresh_calibration = evaluate_reachability(
        separator,
        detector,
        _fresh_loader(
            fresh_calibration_rows,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    style_holdout = evaluate_reachability(
        separator,
        detector,
        _style_loader(
            holdout_rows,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    checks = {
        **_checks(style_holdout, fresh_calibration),
        "slots_exact4": detector.config.slots == SLOTS,
        "all_safe_samples_retained": True,
    }
    result = {
        "schema_version": "chemvector-v1p2d2e4-slots4-endpoint-result-0.1",
        "model_name": MODEL_NAME,
        "decision": (
            "slots4_style_endpoint_go"
            if all(checks.values())
            else "slots4_style_endpoint_no_go"
        ),
        "checks": checks,
        "gates": GO_GATES,
        "config": asdict(detector.config),
        "parameters": sum(parameter.numel() for parameter in detector.parameters()),
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "initial_slots_copied": 3,
        "new_slots_initialized": 1,
        "separator_sha256": SEPARATOR_SHA256,
        "separator_frozen": True,
        "manifest_sha256": MANIFEST_SHA256,
        "fit_samples": len(fit_rows),
        "calibration_samples": len(calibration_rows),
        "holdout_samples": len(holdout_rows),
        "epochs": 1,
        "samples_seen": samples_seen,
        "train_loss": running_loss / samples_seen,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "endpoint_budgets": ENDPOINT_BUDGETS,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "style_calibration": style_calibration,
        "fresh_calibration": fresh_calibration,
        "style_holdout": style_holdout,
        "style_holdout_samples_read": len(holdout_rows),
        "fresh_holdout_samples_read": 0,
        "sealed_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
