#!/usr/bin/env python3
"""Train dense visual vertex detection on Render JSON atom centers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator, decode_vertices, vertex_detection_loss


def load_vertex_model(checkpoint_path: Path, device: torch.device) -> ChemVectorModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    values = {key: value for key, value in checkpoint["config"].items() if key in valid_keys}
    values["vertex_enabled"] = True
    config = ChemVectorConfig(**values)
    model = ChemVectorModel(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key for key in model.state_dict() if key.startswith("vertex_head.")
    }
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "Vertex checkpoint architecture mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.to(device)


def set_vertex_trainable(model: ChemVectorModel) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.vertex_head.parameters():
        parameter.requires_grad_(True)
    return list(model.vertex_head.parameters())


def forward_vertex_batch(
    model: ChemVectorModel,
    batch: dict,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    anchors = batch["anchor_points"].to(device, non_blocking=True)
    mask = batch["anchor_mask"].to(device, non_blocking=True)
    aromatic = batch["aromatic_targets"].to(device, non_blocking=True)
    with torch.no_grad():
        features = model.vision_encoder.forward_stroke_resolution(pixels)
    output = model.vertex_head(features)
    losses = vertex_detection_loss(output, anchors, mask, aromatic)
    return output, losses


def update_detection_metrics(
    totals: dict[str, float],
    decoded: list[list[dict[str, float | bool]]],
    anchors: torch.Tensor,
    anchor_mask: torch.Tensor,
    aromatic_targets: torch.Tensor,
    image_size: int,
    tolerance_pixels: float,
) -> None:
    for batch_index, predictions in enumerate(decoded):
        target_points = anchors[batch_index][anchor_mask[batch_index]].tolist()
        target_aromatic = aromatic_targets[batch_index][anchor_mask[batch_index]].tolist()
        candidates = []
        for prediction_index, prediction in enumerate(predictions):
            for target_index, target in enumerate(target_points):
                distance = math.hypot(
                    (float(prediction["x"]) - target[0]) * image_size,
                    (float(prediction["y"]) - target[1]) * image_size,
                )
                if distance <= tolerance_pixels:
                    candidates.append((distance, prediction_index, target_index))
        matched_predictions = set()
        matched_targets = set()
        for _, prediction_index, target_index in sorted(candidates):
            if prediction_index in matched_predictions or target_index in matched_targets:
                continue
            matched_predictions.add(prediction_index)
            matched_targets.add(target_index)
            totals["aromatic_correct"] += int(
                bool(predictions[prediction_index]["aromatic"])
                == bool(target_aromatic[target_index])
            )
        totals["predictions"] += len(predictions)
        totals["targets"] += len(target_points)
        totals["matches"] += len(matched_targets)
        totals["count_error"] += abs(len(predictions) - len(target_points))
        totals["images"] += 1


def summarize_detection_metrics(totals: dict[str, float]) -> dict[str, float]:
    precision = totals["matches"] / max(totals["predictions"], 1)
    recall = totals["matches"] / max(totals["targets"], 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "count_mae": totals["count_error"] / max(totals["images"], 1),
        "aromatic_accuracy": totals["aromatic_correct"] / max(totals["matches"], 1),
    }


def new_detection_totals() -> dict[str, float]:
    return {
        "predictions": 0,
        "targets": 0,
        "matches": 0,
        "count_error": 0,
        "images": 0,
        "aromatic_correct": 0,
    }


@torch.no_grad()
def validate(
    model: ChemVectorModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    threshold: float,
    tolerance_pixels: float,
) -> dict[str, float]:
    model.eval()
    totals = new_detection_totals()
    loss_totals = {name: 0.0 for name in ("loss", "heatmap_loss", "offset_loss", "aromatic_loss")}
    batches = 0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, losses = forward_vertex_batch(model, batch, device)
        decoded = decode_vertices(output, threshold=threshold)
        update_detection_metrics(
            totals,
            decoded,
            batch["anchor_points"],
            batch["anchor_mask"],
            batch["aromatic_targets"],
            batch["pixel_values"].shape[-1],
            tolerance_pixels,
        )
        for name in loss_totals:
            loss_totals[name] += float(losses[name])
        batches += 1
        if batches >= max_batches:
            break
    result = summarize_detection_metrics(totals)
    result.update({name: value / max(batches, 1) for name, value in loss_totals.items()})
    result["batches"] = batches
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", default=["Data/sdf/webdataset_full_v0/shards/train-*.tar"])
    parser.add_argument("--validation-shards", nargs="+", default=["Data/sdf/webdataset_full_v0/shards/validation-000000.tar"])
    parser.add_argument("--output-dir", default="outputs/chemvector-v5-vertex-10k")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vertex training")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_path = Path(args.init_checkpoint)
    model = load_vertex_model(checkpoint_path, device)
    trainable = set_vertex_trainable(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        ChemVectorIterableDataset(expand_shards(args.train_shards), sequence_version="0.4"),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    validation_loader = DataLoader(
        ChemVectorIterableDataset(expand_shards(args.validation_shards), sequence_version="0.4"),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(model.config), indent=2) + "\n", encoding="utf-8")
    (output_dir / "training_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if tokenizer_path.is_file():
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"trainable_parameters={sum(parameter.numel() for parameter in trainable):,}")
    model.eval()
    model.vertex_head.train()
    samples = 0
    running = {name: 0.0 for name in ("loss", "heatmap_loss", "offset_loss", "aromatic_loss")}
    interval_batches = 0
    started = time.perf_counter()
    for step, batch in enumerate(train_loader, 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, losses = forward_vertex_batch(model, batch, device)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        samples += batch["pixel_values"].size(0)
        interval_batches += 1
        for name in running:
            running[name] += float(losses[name].detach())
        if step % args.log_every == 0:
            print(
                f"step={step} samples={samples} "
                + " ".join(
                    f"{name}={running[name] / interval_batches:.6f}" for name in running
                )
                + f" samples_per_second={samples / (time.perf_counter() - started):.2f}"
            )
            running = {name: 0.0 for name in running}
            interval_batches = 0
        if samples >= args.train_samples:
            break

    metrics = validate(
        model,
        validation_loader,
        device,
        args.validation_batches,
        args.threshold,
        args.tolerance_pixels,
    )
    print("validation=" + json.dumps(metrics, ensure_ascii=False))
    (output_dir / "validation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "global_step": step,
            "training_kind": "vertex_detection_pretraining",
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
