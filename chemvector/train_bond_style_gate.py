#!/usr/bin/env python3
"""Train a frozen-vision style gate between two BondObject experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from chemvector.audit_bond_first_decomposition import _load_models
from chemvector.data import ChemVectorIterableDataset
from chemvector.experiment_loop.artifacts import sha256_file
from chemvector.structure_gate import StructureStyleGate, summarize_gate_scores
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator


def make_loader(
    shards: list[str], batch_size: int, workers: int, *, shuffle: int
) -> DataLoader:
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
def collect_scores(vision, gate, loader, device, samples: int) -> torch.Tensor:
    values = []
    seen = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            values.append(gate(features).float().sigmoid().cpu())
        seen += pixels.size(0)
        if seen >= samples:
            break
    return torch.cat(values)[:samples] if values else torch.empty(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bond-checkpoint", required=True)
    parser.add_argument("--expert-bond-checkpoint", required=True)
    parser.add_argument("--expert-train-shards", nargs="+", required=True)
    parser.add_argument("--base-train-shards", nargs="+", required=True)
    parser.add_argument("--expert-validation-shards", nargs="+", required=True)
    parser.add_argument("--base-validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BondObject style-gate training")
    if min(args.train_samples, args.validation_samples, args.batch_size) < 1:
        raise ValueError("Style-gate sample and batch sizes must be positive")

    output_dir = Path(args.output_dir)
    if (output_dir / "last.pt").exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir / 'last.pt'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    base_path = Path(args.base_bond_checkpoint)
    expert_path = Path(args.expert_bond_checkpoint)
    model, _, graph_path = _load_models(base_path, device, None)
    expert_payload = torch.load(expert_path, map_location="cpu", weights_only=False)
    if Path(expert_payload["grounded_graph_checkpoint"]).resolve() != graph_path.resolve():
        raise ValueError("Base and expert BondObject checkpoints use different vision models")
    vision = model.vision_encoder.eval().requires_grad_(False)
    gate = StructureStyleGate(
        model.config.vision_dims[0], hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
    batches = 0
    running_loss = 0.0
    started = time.perf_counter()
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
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            features = vision.forward_stroke_resolution(pixels)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
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
    expert_scores = collect_scores(
        vision,
        gate,
        make_loader(
            args.expert_validation_shards,
            args.batch_size,
            args.workers,
            shuffle=0,
        ),
        device,
        args.validation_samples,
    )
    base_scores = collect_scores(
        vision,
        gate,
        make_loader(
            args.base_validation_shards,
            args.batch_size,
            args.workers,
            shuffle=0,
        ),
        device,
        args.validation_samples,
    )
    thresholds = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999)
    validation = summarize_gate_scores(expert_scores, base_scores, thresholds)
    payload = {
        "format": "chemvector-bond-style-gate-0.1",
        "gate": gate.state_dict(),
        "input_channels": gate.input_channels,
        "hidden_dim": gate.hidden_dim,
        "base_bond_checkpoint": str(base_path),
        "base_bond_checkpoint_sha256": sha256_file(base_path),
        "expert_bond_checkpoint": str(expert_path),
        "expert_bond_checkpoint_sha256": sha256_file(expert_path),
        "grounded_graph_checkpoint": str(graph_path),
        "training_args": vars(args),
        "validation": validation,
        "uses_uspto_for_training_or_threshold_selection": False,
        "uses_mr_tadf_for_training_or_threshold_selection": False,
    }
    torch.save(payload, output_dir / "last.pt")
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
