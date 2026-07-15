#!/usr/bin/env python3
"""Train a frozen-vision gate between the base and adapted structure experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.evaluate import load_model
from chemvector.structure_gate import StructureStyleGate, summarize_gate_scores
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator


def make_loader(shards: list[str], batch_size: int, workers: int, *, shuffle: int) -> DataLoader:
    return DataLoader(
        ChemVectorIterableDataset(
            expand_shards(shards),
            sample_shuffle_buffer=shuffle,
            sequence_version="0.4",
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=VertexCollator(),
    )


@torch.no_grad()
def collect_scores(
    vision,
    gate: StructureStyleGate,
    loader: DataLoader,
    device: torch.device,
    samples: int,
) -> torch.Tensor:
    values = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            values.append(gate(features).float().sigmoid().cpu())
        if sum(value.numel() for value in values) >= samples:
            break
    return torch.cat(values)[:samples]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expert-checkpoint", required=True)
    parser.add_argument("--expert-train-shards", nargs="+", required=True)
    parser.add_argument("--base-train-shards", nargs="+", required=True)
    parser.add_argument("--expert-validation-shards", nargs="+", required=True)
    parser.add_argument("--base-validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gate training")
    if min(args.train_samples, args.validation_samples, args.batch_size) < 1:
        raise ValueError("Gate sample and batch sizes must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    base_path = Path(args.base_checkpoint)
    model, _ = load_model(base_path, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vision = model.vision_encoder.eval()
    gate = StructureStyleGate(
        model.config.vision_dims[0],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    expert_loader = make_loader(
        args.expert_train_shards,
        args.batch_size,
        args.workers,
        shuffle=args.shuffle_buffer,
    )
    base_loader = make_loader(
        args.base_train_shards,
        args.batch_size,
        args.workers,
        shuffle=args.shuffle_buffer,
    )
    expert_iterator = iter(expert_loader)
    base_iterator = iter(base_loader)
    seen = 0
    started = time.perf_counter()
    running_loss = 0.0
    batches = 0
    while seen < args.train_samples:
        try:
            expert_batch = next(expert_iterator)
        except StopIteration:
            expert_iterator = iter(expert_loader)
            expert_batch = next(expert_iterator)
        try:
            base_batch = next(base_iterator)
        except StopIteration:
            base_iterator = iter(base_loader)
            base_batch = next(base_iterator)
        expert_pixels = expert_batch["pixel_values"].to(device, non_blocking=True)
        base_pixels = base_batch["pixel_values"].to(device, non_blocking=True)
        pixels = torch.cat((expert_pixels, base_pixels), dim=0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = gate(features)
            targets = torch.cat(
                (
                    torch.ones(expert_pixels.size(0), device=device),
                    torch.zeros(base_pixels.size(0), device=device),
                )
            )
            loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen += pixels.size(0)
        batches += 1
        running_loss += float(loss.detach())
        if batches % args.log_every == 0:
            print(
                f"batches={batches} samples={seen} "
                f"loss={running_loss / args.log_every:.6f} "
                f"samples_per_second={seen / (time.perf_counter() - started):.2f}",
                flush=True,
            )
            running_loss = 0.0

    gate.eval()
    expert_validation = make_loader(
        args.expert_validation_shards,
        args.batch_size,
        args.workers,
        shuffle=0,
    )
    base_validation = make_loader(
        args.base_validation_shards,
        args.batch_size,
        args.workers,
        shuffle=0,
    )
    expert_scores = collect_scores(
        vision, gate, expert_validation, device, args.validation_samples
    )
    base_scores = collect_scores(
        vision, gate, base_validation, device, args.validation_samples
    )
    validation = summarize_gate_scores(expert_scores, base_scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": gate.state_dict(),
        "input_channels": gate.input_channels,
        "hidden_dim": gate.hidden_dim,
        "base_checkpoint": args.base_checkpoint,
        "expert_checkpoint": args.expert_checkpoint,
        "training_args": vars(args),
        "validation": validation,
    }
    torch.save(payload, output_dir / "last.pt")
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer = base_path.parent / "tokenizer.json"
    if tokenizer.is_file():
        shutil.copy2(tokenizer, output_dir / "tokenizer.json")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
