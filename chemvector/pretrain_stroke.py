#!/usr/bin/env python3
"""Train a visual pair scorer for ChemSVG stroke primitives."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.stroke import STROKE_CLASSES, StrokePairCollator
from chemvector.train import expand_shards


def load_stroke_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    stroke_band: int | None = None,
    stroke_band_radius: float | None = None,
) -> ChemVectorModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    values = {key: value for key, value in checkpoint["config"].items() if key in valid_keys}
    values["stroke_enabled"] = True
    if stroke_band is not None:
        values["stroke_band"] = stroke_band
    if stroke_band_radius is not None:
        values["stroke_band_radius"] = stroke_band_radius
    config = ChemVectorConfig(**values)
    model = ChemVectorModel(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key for key in model.state_dict() if key.startswith("stroke_head.")
    }
    if unexpected or missing - allowed_missing:
        raise ValueError(
            "Stroke checkpoint architecture mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.to(device)


def set_stroke_trainable(model: ChemVectorModel) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.stroke_head.parameters():
        parameter.requires_grad_(True)
    return list(model.stroke_head.parameters())


def forward_stroke_batch(
    model: ChemVectorModel,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    anchors = batch["anchor_points"].to(device, non_blocking=True)
    pairs = batch["pair_indices"].to(device, non_blocking=True)
    pair_mask = batch["pair_mask"].to(device, non_blocking=True)
    labels = batch["pair_labels"].to(device, non_blocking=True)[pair_mask]
    with torch.no_grad():
        features = model.vision_encoder.forward_stroke_resolution(pixels)
    logits = model.stroke_head(features, anchors, pairs, pair_mask, pixels.shape[-2:])
    return logits, labels


def update_metrics(totals: dict[str, int], logits: torch.Tensor, labels: torch.Tensor) -> None:
    predictions = logits.argmax(dim=-1)
    predicted_edges = predictions != 0
    actual_edges = labels != 0
    totals["pairs"] += int(labels.numel())
    totals["correct"] += int((predictions == labels).sum())
    totals["true_positive"] += int((predicted_edges & actual_edges).sum())
    totals["false_positive"] += int((predicted_edges & ~actual_edges).sum())
    totals["false_negative"] += int((~predicted_edges & actual_edges).sum())
    totals["edge_types"] += int(actual_edges.sum())
    totals["edge_type_correct"] += int(
        ((predictions == labels) & actual_edges).sum()
    )
    for class_id, name in enumerate(STROKE_CLASSES):
        mask = labels == class_id
        totals[f"{name}_count"] += int(mask.sum())
        totals[f"{name}_correct"] += int(((predictions == class_id) & mask).sum())


def new_metric_totals() -> dict[str, int]:
    totals = {
        "pairs": 0,
        "correct": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "edge_types": 0,
        "edge_type_correct": 0,
    }
    for name in STROKE_CLASSES:
        totals[f"{name}_count"] = 0
        totals[f"{name}_correct"] = 0
    return totals


def summarize_metrics(totals: dict[str, int]) -> dict[str, float]:
    precision = totals["true_positive"] / max(
        totals["true_positive"] + totals["false_positive"], 1
    )
    recall = totals["true_positive"] / max(
        totals["true_positive"] + totals["false_negative"], 1
    )
    result = {
        "pair_accuracy": totals["correct"] / max(totals["pairs"], 1),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "edge_type_accuracy": totals["edge_type_correct"]
        / max(totals["edge_types"], 1),
    }
    for name in STROKE_CLASSES:
        result[f"{name.lower()}_recall"] = totals[f"{name}_correct"] / max(
            totals[f"{name}_count"], 1
        )
    return result


@torch.no_grad()
def validate(
    model: ChemVectorModel,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals = new_metric_totals()
    loss_total = 0.0
    batches = 0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, labels = forward_stroke_batch(model, batch, device)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss_total += float(loss)
        update_metrics(totals, logits, labels)
        batches += 1
        if batches >= max_batches:
            break
    result = summarize_metrics(totals)
    result["loss"] = loss_total / max(batches, 1)
    result["batches"] = batches
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/webdataset_full_v0/shards/train-*.tar"],
    )
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=["Data/sdf/webdataset_full_v0/shards/validation-000000.tar"],
    )
    parser.add_argument("--output-dir", default="outputs/checkpoints/chemvector-v5-stroke-10k")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--anchor-jitter-pixels", type=float, default=3.0)
    parser.add_argument("--stroke-band", type=int)
    parser.add_argument("--stroke-band-radius", type=float)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stroke scorer training")
    if args.train_samples < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Stroke training sizes and learning rate must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_path = Path(args.init_checkpoint)
    model = load_stroke_model(
        checkpoint_path,
        device,
        stroke_band=args.stroke_band,
        stroke_band_radius=args.stroke_band_radius,
    )
    trainable = set_stroke_trainable(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    class_weights = torch.tensor(
        (0.25, 1.0, 2.0, 3.0, 1.0),
        device=device,
    )

    train_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.train_shards),
            sequence_version="0.4",
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=StrokePairCollator(
            negative_ratio=args.negative_ratio,
            anchor_jitter_pixels=args.anchor_jitter_pixels,
            seed=args.seed,
        ),
    )
    validation_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.validation_shards),
            sequence_version="0.4",
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=StrokePairCollator(
            negative_ratio=args.negative_ratio,
            anchor_jitter_pixels=0.0,
            seed=args.seed + 1,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(model.config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if tokenizer_path.is_file():
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"trainable_parameters={sum(parameter.numel() for parameter in trainable):,}")
    model.eval()
    model.stroke_head.train()
    totals = new_metric_totals()
    running_loss = 0.0
    interval_batches = 0
    samples = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader, 1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, labels = forward_stroke_batch(model, batch, device)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.detach())
        interval_batches += 1
        samples += batch["pixel_values"].size(0)
        update_metrics(totals, logits.detach(), labels)
        if step % args.log_every == 0:
            metrics = summarize_metrics(totals)
            print(
                f"step={step} samples={samples} "
                f"loss={running_loss / interval_batches:.6f} "
                f"edge_precision={metrics['edge_precision']:.6f} "
                f"edge_recall={metrics['edge_recall']:.6f} "
                f"edge_f1={metrics['edge_f1']:.6f} "
                f"type_accuracy={metrics['edge_type_accuracy']:.6f} "
                f"samples_per_second={samples / (time.perf_counter() - started):.2f}"
            )
            totals = new_metric_totals()
            running_loss = 0.0
            interval_batches = 0
        if samples >= args.train_samples:
            break

    validation = validate(
        model,
        validation_loader,
        device,
        class_weights,
        args.validation_batches,
    )
    print("validation=" + json.dumps(validation, ensure_ascii=False))
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "global_step": step,
            "training_kind": "stroke_pair_pretraining",
        },
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
