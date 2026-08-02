#!/usr/bin/env python3
"""Train one-to-one filtering over frozen dense vertex proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
    vertex_metric_key,
)
from chemvector.proposal_keep import (
    ProposalKeepConfig,
    ProposalKeepHead,
    decode_vertices_with_proposal_keep,
    proposal_keep_loss,
    scale_vertex_batch,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not thresholds or any(not 0.0 < item < 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in (0, 1)")
    if len(set(thresholds)) != len(thresholds):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return thresholds


def forward_dense(model, pixels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        features = model.vision_encoder.forward_stroke_resolution(pixels)
        output = model.vertex_head(features)
    return features, output


@torch.no_grad()
def evaluate(
    model,
    keep_head,
    loader,
    device,
    *,
    thresholds: tuple[float, ...],
    maximum_batches: int,
    cardinality_loss_weight: float = 0.0,
) -> dict:
    model.eval()
    keep_head.eval()
    totals = {
        (threshold, tolerance): new_detection_totals()
        for threshold in thresholds
        for tolerance in (6.0, 16.0)
    }
    loss_total = 0.0
    proposal_recall_total = 0.0
    batches = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        anchors = batch["anchor_points"].to(device, non_blocking=True)
        mask = batch["anchor_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features, dense_output = forward_dense(model, pixels)
            keep_logits = keep_head(features, dense_output)
            losses = proposal_keep_loss(
                keep_logits,
                dense_output,
                anchors,
                mask,
                keep_head.config,
                batch["pixel_values"].shape[-1],
                cardinality_loss_weight=cardinality_loss_weight,
            )
        for threshold in thresholds:
            decoded = decode_vertices_with_proposal_keep(
                dense_output,
                keep_logits,
                keep_head.config,
                keep_threshold=threshold,
            )
            for tolerance in (6.0, 16.0):
                update_detection_metrics(
                    totals[(threshold, tolerance)],
                    decoded,
                    batch["anchor_points"],
                    batch["anchor_mask"],
                    batch["aromatic_targets"],
                    batch["pixel_values"].shape[-1],
                    tolerance,
                )
        loss_total += float(losses["loss"])
        proposal_recall_total += float(losses["proposal_recall"])
        batches += 1
        if maximum_batches > 0 and batches >= maximum_batches:
            break
    grid = {
        f"{threshold:.4f}": {
            f"{tolerance:.4f}": summarize_detection_metrics(
                totals[(threshold, tolerance)]
            )
            for tolerance in (6.0, 16.0)
        }
        for threshold in thresholds
    }
    best_threshold = max(
        thresholds,
        key=lambda item: vertex_metric_key(grid[f"{item:.4f}"]["6.0000"]),
    )
    return {
        "loss": loss_total / max(batches, 1),
        "proposal_recall": proposal_recall_total / max(batches, 1),
        "batches": batches,
        "thresholds": grid,
        "best_threshold": best_threshold,
        "best_6px": grid[f"{best_threshold:.4f}"]["6.0000"],
        "best_16px": grid[f"{best_threshold:.4f}"]["16.0000"],
    }


def save_checkpoint(path, keep_head, optimizer, args, *, step, samples, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "proposal_keep": keep_head.state_dict(),
            "proposal_keep_config": asdict(keep_head.config),
            "optimizer": optimizer.state_dict(),
            "dense_checkpoint": args.dense_checkpoint,
            "global_step": step,
            "samples_seen": samples,
            "validation": validation,
            "training_kind": "one_to_one_proposal_keep",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--train-reject-keys-file")
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-proposal-checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--maximum-proposals", type=int, default=256)
    parser.add_argument("--proposal-threshold", type=float, default=0.05)
    parser.add_argument("--proposal-nms-kernel-size", type=int, default=3)
    parser.add_argument("--matching-radius-pixels", type=float, default=16.0)
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.0)
    parser.add_argument("--minimum-scale", type=float, default=1.0)
    parser.add_argument("--maximum-scale", type=float, default=1.0)
    parser.add_argument("--scale-probability", type=float, default=0.0)
    parser.add_argument(
        "--keep-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.2,0.3,0.4,0.5,0.6,0.7,0.8"),
    )
    parser.add_argument("--validation-batches", type=int, default=125)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=125)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for proposal keep training")
    if min(args.train_samples, args.epochs, args.batch_size) < 1:
        raise ValueError("Training counts must be positive")
    if not 0.0 < args.minimum_scale <= args.maximum_scale <= 1.0:
        raise ValueError("Scale range must lie in (0, 1]")
    if not 0.0 <= args.scale_probability <= 1.0:
        raise ValueError("scale-probability must be in [0, 1]")
    if args.cardinality_loss_weight < 0:
        raise ValueError("cardinality-loss-weight must be nonnegative")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    model = load_vertex_model(Path(args.dense_checkpoint), device).eval()
    model.requires_grad_(False)
    config = ProposalKeepConfig(
        input_channels=model.config.vision_dims[0],
        hidden_dim=args.hidden_dim,
        maximum_proposals=args.maximum_proposals,
        proposal_threshold=args.proposal_threshold,
        proposal_nms_kernel_size=args.proposal_nms_kernel_size,
        matching_radius_pixels=args.matching_radius_pixels,
    )
    keep_head = ProposalKeepHead(config).to(device)
    if args.init_proposal_checkpoint:
        initial = torch.load(
            args.init_proposal_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if ProposalKeepConfig(**initial["proposal_keep_config"]) != config:
            raise ValueError("Initial proposal keep config does not match training config")
        if Path(initial["dense_checkpoint"]).resolve() != Path(
            args.dense_checkpoint
        ).resolve():
            raise ValueError("Initial proposal keep dense checkpoint does not match")
        keep_head.load_state_dict(initial["proposal_keep"])
    optimizer = torch.optim.AdamW(
        keep_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.train_shards),
            sample_shuffle_buffer=args.sample_shuffle_buffer,
            sequence_version="0.4",
            reject_keys=load_reject_keys(args.train_reject_keys_file),
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
            reject_keys=load_reject_keys(args.validation_reject_keys_file),
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"trainable_parameters={sum(p.numel() for p in keep_head.parameters()):,}")
    samples = 0
    step = 0
    best_score = (-1.0, -1.0, -1.0, float("-inf"))
    started = time.perf_counter()
    running_loss = 0.0
    running_recall = 0.0
    interval = 0
    stop = False
    for _ in range(args.epochs):
        for batch in train_loader:
            step += 1
            interval += 1
            keep_head.train()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            anchors = batch["anchor_points"].to(device, non_blocking=True)
            mask = batch["anchor_mask"].to(device, non_blocking=True)
            if args.scale_probability:
                pixels, anchors, _ = scale_vertex_batch(
                    pixels,
                    anchors,
                    minimum_scale=args.minimum_scale,
                    maximum_scale=args.maximum_scale,
                    probability=args.scale_probability,
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features, dense_output = forward_dense(model, pixels)
                keep_logits = keep_head(features, dense_output)
                losses = proposal_keep_loss(
                    keep_logits,
                    dense_output,
                    anchors,
                    mask,
                    config,
                    batch["pixel_values"].shape[-1],
                    cardinality_loss_weight=args.cardinality_loss_weight,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(keep_head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples += int(pixels.size(0))
            running_loss += float(losses["loss"].detach())
            running_recall += float(losses["proposal_recall"])
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples} "
                    f"loss={running_loss / interval:.6f} "
                    f"proposal_recall={running_recall / interval:.4f} "
                    f"samples_per_second={samples / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running_loss = 0.0
                running_recall = 0.0
                interval = 0
            if args.validate_every > 0 and step % args.validate_every == 0:
                validation = evaluate(
                    model,
                    keep_head,
                    validation_loader,
                    device,
                    thresholds=args.keep_thresholds,
                    maximum_batches=args.validation_batches,
                    cardinality_loss_weight=args.cardinality_loss_weight,
                )
                print("validation_step=" + str(step) + " " + json.dumps(validation))
                checkpoint_path = output_dir / f"checkpoint-{step:07d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    keep_head,
                    optimizer,
                    args,
                    step=step,
                    samples=samples,
                    validation=validation,
                )
                score = vertex_metric_key(validation["best_6px"])
                if score > best_score:
                    best_score = score
                    shutil.copy2(checkpoint_path, output_dir / "best.pt")
            if samples >= args.train_samples:
                stop = True
                break
        if stop:
            break

    validation = evaluate(
        model,
        keep_head,
        validation_loader,
        device,
        thresholds=args.keep_thresholds,
        maximum_batches=args.validation_batches,
        cardinality_loss_weight=args.cardinality_loss_weight,
    )
    print("validation=" + json.dumps(validation))
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        output_dir / "last.pt",
        keep_head,
        optimizer,
        args,
        step=step,
        samples=samples,
        validation=validation,
    )


if __name__ == "__main__":
    main()
