#!/usr/bin/env python3
"""Train and evaluate the shadow-mode Atom Query vertex detector."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    atom_query_loss,
    decode_atom_queries,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator


def query_features(
    base_model,
    pixels: torch.Tensor,
    *,
    feature_level: str,
) -> torch.Tensor:
    with torch.no_grad():
        stroke = base_model.vision_encoder.forward_stroke_resolution(pixels)
        if feature_level == "stroke":
            return stroke
        if feature_level == "high":
            return base_model.vision_encoder.forward_high_resolution_from_stroke(stroke)
        raise ValueError(f"Unsupported Atom Query feature level: {feature_level}")


def forward_batch(
    base_model,
    detector,
    batch: dict,
    device: torch.device,
    *,
    feature_level: str,
):
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    anchors = batch["anchor_points"].to(device, non_blocking=True)
    mask = batch["anchor_mask"].to(device, non_blocking=True)
    aromatic = batch["aromatic_targets"].to(device, non_blocking=True)
    features = query_features(base_model, pixels, feature_level=feature_level)
    output = detector(features)
    return output, atom_query_loss(output, anchors, mask, aromatic)


@torch.no_grad()
def evaluate(
    base_model,
    detector,
    loader,
    device: torch.device,
    *,
    threshold: float,
    tolerance_pixels: float,
    maximum_batches: int,
    feature_level: str,
) -> dict:
    base_model.eval()
    detector.eval()
    totals = new_detection_totals()
    loss_totals = {
        name: 0.0
        for name in (
            "loss",
            "presence_loss",
            "center_loss",
            "coarse_center_loss",
            "aromatic_loss",
            "cardinality_loss",
        )
    }
    batches = 0
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output, losses = forward_batch(
                base_model,
                detector,
                batch,
                device,
                feature_level=feature_level,
            )
        decoded = decode_atom_queries(output, threshold=threshold)
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
        if maximum_batches > 0 and batches >= maximum_batches:
            break
    result = summarize_detection_metrics(totals)
    result.update({name: value / max(batches, 1) for name, value in loss_totals.items()})
    result["batches"] = batches
    return result


def save_checkpoint(path: Path, detector, optimizer, config, *, step: int, samples: int, metrics):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "atom_query": detector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "atom_query_config": asdict(config),
            "global_step": step,
            "samples_seen": samples,
            "validation": metrics,
            "training_kind": "atom_query_shadow_pretraining",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--query-init-checkpoint")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--memory-stride", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--feature-level", choices=("stroke", "high"), default="stroke")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Atom Query training")
    if min(args.train_samples, args.epochs, args.batch_size) < 1:
        raise ValueError("training counts must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    base_model = load_vertex_model(Path(args.init_checkpoint), device).eval()
    base_model.requires_grad_(False)
    config = AtomQueryConfig(
        input_channels=base_model.config.vision_dims[
            0 if args.feature_level == "stroke" else 1
        ],
        d_model=args.d_model,
        queries=args.queries,
        layers=args.layers,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        memory_stride=args.memory_stride,
    )
    detector = AtomQueryDetector(config).to(device)
    if args.query_init_checkpoint:
        query_payload = torch.load(
            args.query_init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        saved_config = AtomQueryConfig(**query_payload["atom_query_config"])
        if saved_config != config:
            raise ValueError(
                "Atom Query init checkpoint configuration does not match current arguments"
            )
        detector.load_state_dict(query_payload["atom_query"], strict=True)
    optimizer = torch.optim.AdamW(
        detector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
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
    validation_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.validation_shards),
            sequence_version="0.4",
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"trainable_parameters={sum(p.numel() for p in detector.parameters()):,}")
    samples = 0
    step = 0
    best_exact = -1.0
    started = time.perf_counter()
    running = {
        name: 0.0
        for name in (
            "loss",
            "presence_loss",
            "center_loss",
            "coarse_center_loss",
            "aromatic_loss",
            "cardinality_loss",
        )
    }
    interval_batches = 0
    for _ in range(args.epochs):
        for batch in train_loader:
            step += 1
            detector.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, losses = forward_batch(
                    base_model,
                    detector,
                    batch,
                    device,
                    feature_level=args.feature_level,
                )
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(f"Non-finite Atom Query loss at step {step}")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples += int(batch["pixel_values"].size(0))
            interval_batches += 1
            for name in running:
                running[name] += float(losses[name].detach())
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples} "
                    + " ".join(
                        f"{name}={running[name] / interval_batches:.6f}"
                        for name in running
                    )
                    + f" samples_per_second={samples / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                interval_batches = 0
            if args.validate_every > 0 and step % args.validate_every == 0:
                metrics = evaluate(
                    base_model,
                    detector,
                    validation_loader,
                    device,
                    threshold=args.threshold,
                    tolerance_pixels=args.tolerance_pixels,
                    maximum_batches=args.validation_batches,
                    feature_level=args.feature_level,
                )
                print("validation=" + json.dumps(metrics, ensure_ascii=False), flush=True)
                save_checkpoint(
                    output_dir / f"checkpoint-{step:07d}.pt",
                    detector,
                    optimizer,
                    config,
                    step=step,
                    samples=samples,
                    metrics=metrics,
                )
                if metrics["vertex_exact_rate"] > best_exact:
                    best_exact = metrics["vertex_exact_rate"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        detector,
                        optimizer,
                        config,
                        step=step,
                        samples=samples,
                        metrics=metrics,
                    )
            if samples >= args.train_samples:
                break
        if samples >= args.train_samples:
            break
    metrics = evaluate(
        base_model,
        detector,
        validation_loader,
        device,
        threshold=args.threshold,
        tolerance_pixels=args.tolerance_pixels,
        maximum_batches=args.validation_batches,
        feature_level=args.feature_level,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        output_dir / "last.pt",
        detector,
        optimizer,
        config,
        step=step,
        samples=samples,
        metrics=metrics,
    )
    print("validation=" + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
