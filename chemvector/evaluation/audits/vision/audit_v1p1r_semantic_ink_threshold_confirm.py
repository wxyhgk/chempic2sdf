#!/usr/bin/env python3
"""Calibrate v1 P1 alpha thresholds, then confirm once on internal test."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.core.data import ChemVectorIterableDataset
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    DATA_ROOT,
    GO_GATES,
    MANIFEST_SHA256,
    SemanticInkCollator,
    _binary_metrics,
    _new_binary_counts,
    _update_binary_counts,
    evaluate,
    load_safe_manifest,
)


AUDIT_NAME = "chemvector-v1p1r-semantic-ink-threshold-confirm"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DEFAULT_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v1p1-semantic-ink-separation-pilot/best.pt"
)
CHECKPOINT_SHA256 = (
    "e4f726b80f73e46a71619f8f4a42ff3b0042a3156cc86c0a4f584154b574263c"
)
THRESHOLDS = tuple(
    [round(index / 100.0, 2) for index in range(1, 21)]
    + [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
)


def select_iou_threshold(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Threshold sweep rows must be nonempty")
    return max(rows, key=lambda row: (row["iou"], row["precision"], row["threshold"]))


def _audit_loader(
    rows: list[dict[str, Any]], batch_size: int, workers: int
) -> DataLoader:
    split = str(rows[0]["split"])
    if split not in {"validation", "internal_test"}:
        raise ValueError(f"P1-R cannot audit split {split!r}")
    expected_marker = "/validation-" if split == "validation" else "/internal_test-"
    shards = sorted({str(DATA_ROOT / row["source_shard"]) for row in rows})
    if any(expected_marker not in shard for shard in shards):
        raise ValueError(f"P1-R {split} shard path differs")
    keys = {str(row["sample_key"]) for row in rows}
    dataset = ChemVectorIterableDataset(
        shards,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        include_keys=keys,
        serialize_target=False,
        load_raster_layers=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=SemanticInkCollator(),
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def threshold_sweep(
    model: SemanticInkSeparator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    model.eval()
    counts = {
        channel: [_new_binary_counts() for _ in THRESHOLDS]
        for channel in ("bond", "text")
    }
    samples = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            probabilities = model(pixels)["logits"].float().sigmoid()
        expected = targets >= (8.0 / 255.0)
        samples += pixels.size(0)
        for threshold_index, threshold in enumerate(THRESHOLDS):
            predicted = probabilities >= threshold
            for channel_index, channel in enumerate(("bond", "text")):
                _update_binary_counts(
                    counts[channel][threshold_index],
                    predicted[:, channel_index],
                    expected[:, channel_index],
                )
    return {
        channel: [
            {"threshold": threshold, **_binary_metrics(channel_counts)}
            for threshold, channel_counts in zip(THRESHOLDS, counts[channel])
        ]
        for channel in ("bond", "text")
    } | {"samples": samples}


def _gate_metrics(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "bond_iou_min90pct": metrics["bond"]["iou"] >= GO_GATES["bond_iou"],
        "text_iou_min75pct": metrics["text"]["iou"] >= GO_GATES["text_iou"],
        "foreground_iou_min90pct": metrics["foreground"]["iou"]
        >= GO_GATES["foreground_iou"],
        "stress_bond_iou_min88pct": metrics["bond_by_variant"]["stress"]["iou"]
        >= GO_GATES["stress_bond_iou"],
        "bond_on_text_only_max10pct": metrics["bond_on_text_only_rate"]
        <= GO_GATES["bond_on_text_only_rate_max"],
    }


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
        raise RuntimeError("v1 P1-R threshold audit requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P1-R output is not empty: {args.output_dir}")
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("v1 P1 checkpoint SHA256 differs")
    rows = load_safe_manifest()
    validation_rows = [row for row in rows if row["split"] == "validation"]
    internal_rows = [row for row in rows if row["split"] == "internal_test"]
    if len(validation_rows) != 496 or len(internal_rows) != 496:
        raise ValueError("v1 P1-R split counts differ")

    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location=device)
    config = SemanticInkSeparatorConfig(**payload["config"])
    model = SemanticInkSeparator(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    started = time.perf_counter()
    sweep = threshold_sweep(
        model,
        _audit_loader(validation_rows, args.batch_size, args.workers),
        device,
    )
    if sweep["samples"] != 496:
        raise ValueError("v1 P1-R validation sweep count differs")
    selected = {
        channel: select_iou_threshold(sweep[channel])
        for channel in ("bond", "text")
    }
    bond_threshold = selected["bond"]["threshold"]
    text_threshold = selected["text"]["threshold"]
    validation = evaluate(
        model,
        _audit_loader(validation_rows, args.batch_size, args.workers),
        device,
        bond_threshold=bond_threshold,
        text_threshold=text_threshold,
    )
    validation_checks = _gate_metrics(validation)
    calibration_go = all(validation_checks.values())

    internal = None
    internal_checks = None
    internal_test_read = False
    if calibration_go:
        internal = evaluate(
            model,
            _audit_loader(internal_rows, args.batch_size, args.workers),
            device,
            bond_threshold=bond_threshold,
            text_threshold=text_threshold,
        )
        internal_test_read = True
        if internal["samples"] != 496:
            raise ValueError("v1 P1-R internal-test count differs")
        internal_checks = _gate_metrics(internal)
    decision = (
        "semantic_ink_threshold_confirm_go"
        if calibration_go and internal_checks is not None and all(internal_checks.values())
        else "semantic_ink_threshold_confirm_no_go"
    )
    result = {
        "schema_version": "chemvector-v1p1r-semantic-ink-threshold-confirm-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "threshold_grid": THRESHOLDS,
        "selection_policy": "maximum validation IoU; tie by precision then higher threshold",
        "selected": selected,
        "validation": validation,
        "validation_checks": validation_checks,
        "calibration_go": calibration_go,
        "internal_test_read": internal_test_read,
        "internal_test": internal,
        "internal_test_checks": internal_checks,
        "gates": GO_GATES,
        "sweep": sweep,
        "model_config": asdict(config),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
