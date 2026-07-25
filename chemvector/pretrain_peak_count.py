#!/usr/bin/env python3
"""Train a lightweight per-image count calibrator over frozen dense peaks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.peak_count import (
    PeakCountCalibrator,
    PeakCountConfig,
    decode_vertices_by_count,
    peak_count_features,
    peak_count_loss,
    predict_peak_counts,
)
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator


def forward_dense(model, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        features = model.vision_encoder.forward_stroke_resolution(pixels)
        return model.vertex_head(features)


@torch.no_grad()
def evaluate(model, calibrator, loader, device, *, maximum_batches: int) -> dict:
    model.eval()
    calibrator.eval()
    totals = {
        tolerance: new_detection_totals() for tolerance in (4.0, 6.0, 10.0, 16.0)
    }
    count_loss_total = 0.0
    batches = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = forward_dense(model, pixels)
            features = peak_count_features(output, calibrator.config)
            count_output = calibrator(features)
            loss = peak_count_loss(count_output, batch["anchor_mask"].to(device))
        decoded = decode_vertices_by_count(
            output,
            predict_peak_counts(count_output, calibrator.config.maximum_vertices),
            maximum_vertices=calibrator.config.maximum_vertices,
        )
        for tolerance, values in totals.items():
            update_detection_metrics(
                values,
                decoded,
                batch["anchor_points"],
                batch["anchor_mask"],
                batch["aromatic_targets"],
                batch["pixel_values"].shape[-1],
                tolerance,
            )
        count_loss_total += float(loss)
        batches += 1
        if maximum_batches > 0 and batches >= maximum_batches:
            break
    return {
        "count_loss": count_loss_total / max(batches, 1),
        "batches": batches,
        "tolerances": {
            f"{tolerance:.4f}": summarize_detection_metrics(values)
            for tolerance, values in totals.items()
        },
    }


def save_checkpoint(path, calibrator, optimizer, config, args, *, step, samples, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "peak_count": calibrator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "peak_count_config": asdict(config),
            "dense_checkpoint": args.dense_checkpoint,
            "global_step": step,
            "samples_seen": samples,
            "validation": validation,
            "training_kind": "dense_peak_count_calibration",
        },
        temporary,
    )
    temporary.replace(path)


def metric_key(validation: dict) -> tuple[float, float, float, float]:
    metrics = validation["tolerances"]["6.0000"]
    return (
        metrics["vertex_exact_rate"],
        metrics["count_exact_rate"],
        metrics["f1"],
        -metrics["count_mae"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--train-reject-keys-file")
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=45000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--maximum-vertices", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--soft-count-temperature", type=float, default=0.03)
    parser.add_argument(
        "--objective",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--validation-batches", type=int, default=125)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=125)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for peak count calibration")
    if min(args.train_samples, args.epochs, args.batch_size) < 1:
        raise ValueError("Training counts must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    model = load_vertex_model(Path(args.dense_checkpoint), device).eval()
    model.requires_grad_(False)
    config = PeakCountConfig(
        maximum_vertices=args.maximum_vertices,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        soft_count_temperature=args.soft_count_temperature,
        objective=args.objective,
    )
    calibrator = PeakCountCalibrator(config).to(device)
    optimizer = torch.optim.AdamW(
        calibrator.parameters(),
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
    print(f"trainable_parameters={sum(p.numel() for p in calibrator.parameters()):,}")
    samples = 0
    step = 0
    best_score = (-1.0, -1.0, -1.0, float("-inf"))
    started = time.perf_counter()
    running_loss = 0.0
    interval = 0
    stop = False
    for _ in range(args.epochs):
        for batch in train_loader:
            step += 1
            interval += 1
            calibrator.train()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = forward_dense(model, pixels)
                features = peak_count_features(output, config)
                count_output = calibrator(features)
                loss = peak_count_loss(count_output, batch["anchor_mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples += int(pixels.size(0))
            running_loss += float(loss.detach())
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples} loss={running_loss / interval:.6f} "
                    f"samples_per_second={samples / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running_loss = 0.0
                interval = 0
            if args.validate_every > 0 and step % args.validate_every == 0:
                validation = evaluate(
                    model,
                    calibrator,
                    validation_loader,
                    device,
                    maximum_batches=args.validation_batches,
                )
                print("validation=" + json.dumps(validation, ensure_ascii=False), flush=True)
                save_checkpoint(
                    output_dir / f"checkpoint-{step:07d}.pt",
                    calibrator,
                    optimizer,
                    config,
                    args,
                    step=step,
                    samples=samples,
                    validation=validation,
                )
                current_score = metric_key(validation)
                if current_score > best_score:
                    best_score = current_score
                    save_checkpoint(
                        output_dir / "best.pt",
                        calibrator,
                        optimizer,
                        config,
                        args,
                        step=step,
                        samples=samples,
                        validation=validation,
                    )
            if samples >= args.train_samples:
                stop = True
                break
        if stop:
            break

    validation = evaluate(
        model,
        calibrator,
        validation_loader,
        device,
        maximum_batches=args.validation_batches,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        output_dir / "last.pt",
        calibrator,
        optimizer,
        config,
        args,
        step=step,
        samples=samples,
        validation=validation,
    )
    print("validation=" + json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
