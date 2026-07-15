#!/usr/bin/env python3
"""Train dense visual vertex detection on Render JSON atom centers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.train import expand_shards, iter_mixed_batches
from chemvector.vertex import (
    VertexCollator,
    decode_vertices,
    vertex_detection_loss,
    vertex_teacher_consistency_loss,
)


def load_vertex_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    residual_adapter: bool = False,
    adapter_dim: int = 32,
) -> ChemVectorModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    values = {key: value for key, value in checkpoint["config"].items() if key in valid_keys}
    values["vertex_enabled"] = True
    if residual_adapter:
        values["structure_residual_adapter"] = True
        values["structure_adapter_dim"] = adapter_dim
    config = ChemVectorConfig(**values)
    model = ChemVectorModel(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key
        for key in model.state_dict()
        if key.startswith("vertex_head.")
        or ".pair_feature_adapter." in key
        or ".logit_adapter." in key
    }
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "Vertex checkpoint architecture mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.to(device)


def set_vertex_trainable(
    model: ChemVectorModel,
    *,
    adapter_only: bool = False,
    unfreeze_stroke_blocks: int = 0,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module = model.vertex_head.output_adapter if adapter_only else model.vertex_head
    if module is None:
        raise ValueError("Adapter-only vertex training requires a structure adapter")
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    stroke_stage = model.vision_encoder.stages[0]
    if not 0 <= unfreeze_stroke_blocks <= len(stroke_stage):
        raise ValueError(
            f"unfreeze_stroke_blocks must be in [0, {len(stroke_stage)}]"
        )
    for block in stroke_stage[len(stroke_stage) - unfreeze_stroke_blocks :]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def forward_vertex_batch(
    model: ChemVectorModel,
    batch: dict,
    device: torch.device,
    *,
    teacher_head: torch.nn.Module | None = None,
    teacher_consistency_weight: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    anchors = batch["anchor_points"].to(device, non_blocking=True)
    mask = batch["anchor_mask"].to(device, non_blocking=True)
    aromatic = batch["aromatic_targets"].to(device, non_blocking=True)
    vision_trainable = any(
        parameter.requires_grad for parameter in model.vision_encoder.parameters()
    )
    with torch.set_grad_enabled(vision_trainable):
        features = model.vision_encoder.forward_stroke_resolution(pixels)
    output = model.vertex_head(features)
    losses = vertex_detection_loss(output, anchors, mask, aromatic)
    if teacher_head is not None:
        with torch.no_grad():
            teacher_output = teacher_head(features)
        teacher_losses = vertex_teacher_consistency_loss(output, teacher_output)
        losses.update(teacher_losses)
        losses["loss"] = (
            losses["loss"]
            + teacher_consistency_weight * teacher_losses["teacher_loss"]
        )
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
        count_exact = len(predictions) == len(target_points)
        vertex_exact = count_exact and len(matched_targets) == len(target_points)
        totals["count_exact"] += int(count_exact)
        totals["vertex_exact"] += int(vertex_exact)
        totals["under_detected"] += int(len(predictions) < len(target_points))
        totals["over_detected"] += int(len(predictions) > len(target_points))
        totals["images"] += 1


def summarize_detection_metrics(totals: dict[str, float]) -> dict[str, float]:
    precision = totals["matches"] / max(totals["predictions"], 1)
    recall = totals["matches"] / max(totals["targets"], 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "count_mae": totals["count_error"] / max(totals["images"], 1),
        "count_exact_rate": totals["count_exact"] / max(totals["images"], 1),
        "vertex_exact_rate": totals["vertex_exact"] / max(totals["images"], 1),
        "under_detected_images": int(totals["under_detected"]),
        "over_detected_images": int(totals["over_detected"]),
        "aromatic_accuracy": totals["aromatic_correct"] / max(totals["matches"], 1),
    }


def new_detection_totals() -> dict[str, float]:
    return {
        "predictions": 0,
        "targets": 0,
        "matches": 0,
        "count_error": 0,
        "count_exact": 0,
        "vertex_exact": 0,
        "under_detected": 0,
        "over_detected": 0,
        "images": 0,
        "aromatic_correct": 0,
    }


def save_vertex_checkpoint(
    path: Path,
    model: ChemVectorModel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    samples: int,
    metrics: dict[str, float] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "global_step": step,
            "samples_seen": samples,
            "validation": metrics,
            "training_kind": "vertex_detection_pretraining",
        },
        temporary,
    )
    temporary.replace(path)


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
    parser.add_argument("--replay-shards", nargs="+")
    parser.add_argument("--replay-probability", type=float, default=0.0)
    parser.add_argument("--output-dir", default="outputs/checkpoints/chemvector-v5-vertex-10k")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--residual-adapter", action="store_true")
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--freeze-base-head", action="store_true")
    parser.add_argument("--teacher-consistency-weight", type=float, default=0.0)
    parser.add_argument("--unfreeze-stroke-blocks", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vertex training")
    if not 0.0 <= args.replay_probability < 1.0:
        raise ValueError("replay-probability must be in [0, 1)")
    if args.replay_probability and not args.replay_shards:
        raise ValueError("replay-shards are required when replay is enabled")
    if args.adapter_dim < 1:
        raise ValueError("adapter-dim must be positive")
    if args.freeze_base_head and not args.residual_adapter:
        raise ValueError("Frozen base vertex training requires a residual adapter")
    if args.teacher_consistency_weight < 0:
        raise ValueError("teacher-consistency-weight must be nonnegative")
    if args.teacher_consistency_weight and not args.replay_shards:
        raise ValueError("Teacher consistency requires replay shards")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_path = Path(args.init_checkpoint)
    model = load_vertex_model(
        checkpoint_path,
        device,
        residual_adapter=args.residual_adapter,
        adapter_dim=args.adapter_dim,
    )
    teacher_head = None
    if args.teacher_consistency_weight:
        teacher_head = deepcopy(model.vertex_head).eval().requires_grad_(False)
    trainable = set_vertex_trainable(
        model,
        adapter_only=args.freeze_base_head,
        unfreeze_stroke_blocks=args.unfreeze_stroke_blocks,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.train_shards),
            sample_shuffle_buffer=args.sample_shuffle_buffer,
            sequence_version="0.4",
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    replay_loader = None
    if args.replay_shards:
        replay_loader = DataLoader(
            ChemVectorIterableDataset(
                expand_shards(args.replay_shards),
                sample_shuffle_buffer=args.sample_shuffle_buffer,
                sequence_version="0.4",
            ),
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
    for block in model.vision_encoder.stages[0][
        len(model.vision_encoder.stages[0]) - args.unfreeze_stroke_blocks :
    ]:
        block.train()
    samples = 0
    running_names = ["loss", "heatmap_loss", "offset_loss", "aromatic_loss"]
    if teacher_head is not None:
        running_names.extend(
            (
                "teacher_loss",
                "teacher_heatmap_loss",
                "teacher_offset_loss",
                "teacher_aromatic_loss",
            )
        )
    running = {name: 0.0 for name in running_names}
    interval_batches = 0
    best_f1 = -1.0
    replay_batches = 0
    started = time.perf_counter()
    def training_batches():
        rng = random.Random(args.seed)
        for _ in range(args.epochs):
            yield from iter_mixed_batches(
                train_loader,
                replay_loader,
                args.replay_probability,
                rng=rng,
            )

    for step, (batch, is_replay) in enumerate(training_batches(), 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, losses = forward_vertex_batch(
                model,
                batch,
                device,
                teacher_head=teacher_head if is_replay else None,
                teacher_consistency_weight=args.teacher_consistency_weight,
            )
        if not torch.isfinite(losses["loss"]):
            failure = {
                "step": step,
                "samples": samples,
                "keys": batch["keys"],
                "losses": {
                    name: float(value.detach()) for name, value in losses.items()
                },
            }
            (output_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"Non-finite vertex loss at step {step}: {failure}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        samples += batch["pixel_values"].size(0)
        replay_batches += int(is_replay)
        interval_batches += 1
        for name in running:
            if name in losses:
                running[name] += float(losses[name].detach())
        if step % args.log_every == 0:
            print(
                f"step={step} samples={samples} "
                + " ".join(
                    f"{name}={running[name] / interval_batches:.6f}" for name in running
                )
                + f" replay_batches={replay_batches}"
                + f" samples_per_second={samples / (time.perf_counter() - started):.2f}"
            )
            running = {name: 0.0 for name in running_names}
            interval_batches = 0
        if args.validate_every > 0 and step % args.validate_every == 0:
            metrics = validate(
                model,
                validation_loader,
                device,
                args.validation_batches,
                args.threshold,
                args.tolerance_pixels,
            )
            print(
                f"validation_step={step} "
                + json.dumps(metrics, ensure_ascii=False),
                flush=True,
            )
            checkpoint_path = output_dir / f"checkpoint-{step:07d}.pt"
            save_vertex_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step=step,
                samples=samples,
                metrics=metrics,
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                shutil.copy2(checkpoint_path, output_dir / "best.pt")
            model.eval()
            model.vertex_head.train()
            for block in model.vision_encoder.stages[0][
                len(model.vision_encoder.stages[0]) - args.unfreeze_stroke_blocks :
            ]:
                block.train()
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
    save_vertex_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        step=step,
        samples=samples,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
