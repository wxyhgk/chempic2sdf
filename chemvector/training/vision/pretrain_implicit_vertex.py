#!/usr/bin/env python3
"""Train a dense vertex head only on implicit molecular vertices."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.pretrain_vertex import (
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator, decode_vertices, vertex_detection_loss


class ImplicitVertexCollator(VertexCollator):
    def __call__(self, samples: list[dict]) -> dict:
        batch = super().__call__(samples)
        batch["anchor_mask"] &= ~batch["role_targets"].bool()
        return batch


def forward(head, model, batch, device):
    pixels = batch["pixel_values"].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = model.vision_encoder.forward_stroke_resolution(pixels)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = head(features)
        losses = vertex_detection_loss(
            output,
            batch["anchor_points"].to(device),
            batch["anchor_mask"].to(device),
            batch["aromatic_targets"].to(device),
        )
    return output, losses


@torch.no_grad()
def validate(head, model, loader, device, threshold: float) -> dict:
    head.eval()
    totals = new_detection_totals()
    losses = {
        name: 0.0
        for name in ("loss", "heatmap_loss", "offset_loss", "aromatic_loss")
    }
    batches = 0
    for batch in loader:
        output, current = forward(head, model, batch, device)
        decoded = decode_vertices(
            output,
            threshold=threshold,
            maximum_vertices=256,
            nms_kernel_size=1,
        )
        update_detection_metrics(
            totals,
            decoded,
            batch["anchor_points"],
            batch["anchor_mask"],
            batch["aromatic_targets"],
            768,
            6.0,
        )
        for name in losses:
            losses[name] += float(current[name])
        batches += 1
    result = summarize_detection_metrics(totals)
    result.update({name: value / max(batches, 1) for name, value in losses.items()})
    return result


def save(path, head, optimizer, args, samples, metrics):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "implicit_vertex_head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "samples_seen": samples,
            "validation": metrics,
            "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=45000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    device = torch.device("cuda")
    _, payload = load_grounded_graph_checkpoint(args.grounded_graph_checkpoint, device)
    model, _, _, _ = load_frozen_grounded_pipeline(
        Path(payload["grounded_selector_checkpoint"]), device
    )
    head = deepcopy(model.vertex_head).to(device)
    head.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    collator = ImplicitVertexCollator()

    def loader(shards, shuffle=0):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                sequence_version="0.4",
                sample_shuffle_buffer=shuffle,
            ),
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=collator,
        )

    train_loader = loader(args.train_shards, 1000)
    validation_loader = loader(args.validation_shards)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    best = (-1.0, -1.0, -1.0)
    samples = 0
    step = 0
    for _ in range(args.epochs):
        for batch in train_loader:
            step += 1
            _, losses = forward(head, model, batch, device)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples += len(batch["pixel_values"])
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples} "
                    f"loss={float(losses['loss'].detach()):.6f}",
                    flush=True,
                )
            if step % args.validate_every == 0:
                metrics = validate(
                    head, model, validation_loader, device, args.threshold
                )
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{step:07d}.pt"
                save(checkpoint, head, optimizer, args, samples, metrics)
                score = (
                    metrics["vertex_exact_rate"],
                    metrics["count_exact_rate"],
                    metrics["f1"],
                )
                if score > best:
                    best = score
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                head.train()
            if samples >= args.train_samples:
                break
        if samples >= args.train_samples:
            break
    metrics = validate(head, model, validation_loader, device, args.threshold)
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save(output_dir / "last.pt", head, optimizer, args, samples, metrics)
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
