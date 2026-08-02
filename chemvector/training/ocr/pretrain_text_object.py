#!/usr/bin/env python3
"""Train an independent visible-text object center and extent detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.text_object import (
    TextObjectBBoxHead,
    TextObjectCenterHead,
    decode_text_objects,
    text_object_loss,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator
from chemvector.pretrain_vertex import new_detection_totals, summarize_detection_metrics, update_detection_metrics


class TextObjectCollator(VertexCollator):
    def __call__(self, samples: list[dict]) -> dict:
        base = super().__call__(samples)
        maximum = max(sum(bool(atom["visible_text"]) for atom in sample["layers"]["atoms"]) for sample in samples)
        maximum = max(maximum, 1)
        centers = torch.zeros((len(samples), maximum, 2))
        bbox_centers = torch.zeros((len(samples), maximum, 2))
        extents = torch.zeros((len(samples), maximum, 2))
        mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
        for sample_index, sample in enumerate(samples):
            visible = [atom for atom in sample["layers"]["atoms"] if atom["visible_text"]]
            for index, atom in enumerate(visible):
                x0, y0, x1, y1 = atom["bbox"]
                centers[sample_index, index] = torch.tensor(atom["center"]) / self.image_size
                bbox_centers[sample_index, index] = torch.tensor(
                    ((x0 + x1) / 2, (y0 + y1) / 2)
                ) / self.image_size
                extents[sample_index, index] = torch.tensor(((x1 - x0) / 2, (y1 - y0) / 2)) / self.image_size
                mask[sample_index, index] = True
        base.update({
            "text_centers": centers,
            "text_bbox_centers": bbox_centers,
            "text_extents": extents,
            "text_mask": mask,
        })
        return base


@torch.no_grad()
def validate(model, head, loader, device, threshold: float) -> dict:
    head.eval()
    totals = new_detection_totals()
    losses = {
        name: 0.0
        for name in (
            "loss",
            "heatmap_loss",
            "offset_loss",
            "extent_loss",
            "bbox_offset_loss",
        )
    }
    batches = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = model.vision_encoder.forward_stroke_resolution(pixels)
            output = head(features)
            current = text_object_loss(
                output,
                batch["text_centers"].to(device),
                batch["text_mask"].to(device),
                batch["text_extents"].to(device),
                batch["text_bbox_centers"].to(device),
            )
        decoded = decode_text_objects(output, threshold=threshold)
        update_detection_metrics(totals, decoded, batch["text_centers"], batch["text_mask"], torch.zeros_like(batch["text_mask"], dtype=torch.float32), 768, 6.0)
        for name in losses:
            losses[name] += float(current[name])
        batches += 1
    result = summarize_detection_metrics(totals)
    result.update({name: value / max(batches, 1) for name, value in losses.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=45000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--predict-bbox-offset", action="store_true")
    parser.add_argument("--train-bbox-offset-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    _, payload = load_grounded_graph_checkpoint(args.grounded_graph_checkpoint, device)
    model, _, _, _ = load_frozen_grounded_pipeline(Path(payload["grounded_selector_checkpoint"]), device)
    head_class = TextObjectBBoxHead if args.predict_bbox_offset else TextObjectCenterHead
    head = head_class(model.config.vision_dims[0]).to(device)
    initialization = None
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        incompatible = head.load_state_dict(initial["text_object_head"], strict=False)
        expected_missing = (
            {"bbox_offset.weight", "bbox_offset.bias"}
            if args.predict_bbox_offset
            and initial.get("head_type", "center") == "center"
            else set()
        )
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError(
                "TextObject initialization is incompatible: "
                f"missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}"
            )
        initialization = args.initial_checkpoint
    if args.train_bbox_offset_only:
        if not args.predict_bbox_offset:
            raise ValueError("bbox-offset-only training requires --predict-bbox-offset")
        head.requires_grad_(False)
        head.bbox_offset.requires_grad_(True)
    trainable = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    collator = TextObjectCollator()
    def loader(shards, shuffle=0):
        return DataLoader(ChemVectorIterableDataset(expand_shards(shards), sequence_version="0.4", sample_shuffle_buffer=shuffle), batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collator)
    train_loader = loader(args.train_shards, 1000)
    validation_loader = loader(args.validation_shards)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")
    best = None
    samples = 0; step = 0
    for _ in range(args.epochs):
        for batch in train_loader:
            step += 1
            pixels = batch["pixel_values"].to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = model.vision_encoder.forward_stroke_resolution(pixels)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = head(features)
                losses = text_object_loss(
                    prediction,
                    batch["text_centers"].to(device),
                    batch["text_mask"].to(device),
                    batch["text_extents"].to(device),
                    batch["text_bbox_centers"].to(device),
                )
            losses["loss"].backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            samples += len(pixels)
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples} "
                    f"loss={float(losses['loss'].detach()):.6f}",
                    flush=True,
                )
            if step % args.validate_every == 0:
                metrics = validate(model, head, validation_loader, device, args.threshold)
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{step:07d}.pt"
                torch.save({
                    "text_object_head": head.state_dict(),
                    "head_type": "bbox" if args.predict_bbox_offset else "center",
                    "input_channels": model.config.vision_dims[0],
                    "samples_seen": samples,
                    "validation": metrics,
                    "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
                    "initial_checkpoint": initialization,
                }, checkpoint)
                score = (
                    -metrics["bbox_offset_loss"],
                    metrics["vertex_exact_rate"],
                    metrics["f1"],
                ) if args.predict_bbox_offset else (
                    metrics["vertex_exact_rate"],
                    metrics["count_exact_rate"],
                    metrics["f1"],
                )
                if best is None or score > best:
                    best = score; shutil.copy2(checkpoint, output_dir / "best.pt")
                head.train()
            if samples >= args.train_samples:
                break
        if samples >= args.train_samples:
            break
    metrics = validate(model, head, validation_loader, device, args.threshold)
    (output_dir / "validation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    torch.save({
        "text_object_head": head.state_dict(),
        "head_type": "bbox" if args.predict_bbox_offset else "center",
        "input_channels": model.config.vision_dims[0],
        "samples_seen": samples,
        "validation": metrics,
        "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
        "initial_checkpoint": initialization,
    }, output_dir / "last.pt")
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
