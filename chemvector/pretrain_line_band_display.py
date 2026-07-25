#!/usr/bin/env python3
"""Train a display-only classifier on confirmed two-dimensional bond bands."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import shutil

import torch
from torch.nn import functional as F

from chemvector.data import ChemVectorIterableDataset
from chemvector.line_band_display import (
    LineBandDisplayConfig,
    LineBandDisplayRefiner,
)
from chemvector.pretrain_dense_bond_type_refiner import (
    PositiveBondTypeCollator,
    class_balancing_weights,
)
from chemvector.train import expand_shards


def _forward(
    model: LineBandDisplayRefiner,
    batch: dict,
    device: torch.device,
    *,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = batch["pixel_values"].to(device)
    points = batch["points"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(pixels, points, pair_indices, pair_mask)["display_logits"]
        if symmetric:
            reversed_points = (
                points.reshape(points.size(0), -1, 2, 2).flip(2).reshape_as(points)
            )
            reversed_logits = model(
                pixels, reversed_points, pair_indices, pair_mask
            )["display_logits"]
            logits = (logits + reversed_logits) / 2.0
    return logits, batch["display_targets"].to(device)[pair_mask]


@torch.no_grad()
def evaluate(
    model: LineBandDisplayRefiner,
    records: list[dict],
    collator: PositiveBondTypeCollator,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    counts = torch.zeros(4, dtype=torch.long)
    correct = torch.zeros(4, dtype=torch.long)
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        logits, targets = _forward(model, batch, device, symmetric=True)
        predictions = logits.argmax(-1)
        for class_id in range(4):
            selected = targets == class_id
            counts[class_id] += int(selected.sum())
            correct[class_id] += int(((predictions == targets) & selected).sum())
    total = max(int(counts.sum()), 1)
    from chemvector.bond_object import BOND_TYPES

    return {
        "samples": len(records),
        "bonds": int(counts.sum()),
        "display_accuracy": int(correct.sum()) / total,
        "by_gt_display_type": {
            name: {
                "count": int(counts[class_id]),
                "display_accuracy": int(correct[class_id])
                / max(int(counts[class_id]), 1),
            }
            for class_id, name in enumerate(BOND_TYPES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=8974)
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--train-samples", type=int, default=17948)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--endpoint-jitter-sigma-pixels", type=float, default=1.5)
    parser.add_argument("--endpoint-jitter-max-pixels", type=float, default=6.0)
    parser.add_argument("--endpoint-jitter-probability", type=float, default=1.0)
    parser.add_argument("--endpoint-swap-probability", type=float, default=0.5)
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument("--maximum-class-weight", type=float, default=4.0)
    parser.add_argument("--validate-every", type=int, default=4500)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    device = torch.device("cuda")
    config = LineBandDisplayConfig()
    model = LineBandDisplayRefiner(config).to(device)

    def cache(shards: list[str], count: int) -> list[dict]:
        return list(
            islice(
                iter(
                    ChemVectorIterableDataset(
                        expand_shards(shards), sequence_version="0.4"
                    )
                ),
                count,
            )
        )

    train_records = cache(args.train_shards, args.train_cache_samples)
    validation_records = cache(args.validation_shards, args.validation_samples)
    if len(train_records) != args.train_cache_samples:
        raise ValueError("Training cache is incomplete")
    if len(validation_records) != args.validation_samples:
        raise ValueError("Validation cache is incomplete")

    train_collator = PositiveBondTypeCollator(
        endpoint_jitter_sigma_pixels=args.endpoint_jitter_sigma_pixels,
        endpoint_jitter_max_pixels=args.endpoint_jitter_max_pixels,
        endpoint_jitter_probability=args.endpoint_jitter_probability,
        endpoint_swap_probability=args.endpoint_swap_probability,
        seed=args.seed,
    )
    validation_collator = PositiveBondTypeCollator(seed=args.seed + 1)
    weights = class_balancing_weights(
        train_records,
        "display_type",
        args.class_balance_power,
        args.maximum_class_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "architecture": "line_band_display_refiner",
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "class_weights": weights.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best = -1.0
    running_loss = 0.0
    running_samples = 0
    stop = False
    for _ in range(args.epochs):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            indices = order[start : start + min(args.batch_size, remaining)]
            batch = train_collator([train_records[index] for index in indices])
            model.train()
            logits, targets = _forward(model, batch, device, symmetric=False)
            loss = F.cross_entropy(
                logits, targets, weight=weights.to(logits.dtype)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_loss += float(loss.detach()) * len(indices)
            running_samples += len(indices)
            if samples_seen // args.log_every != (
                samples_seen - len(indices)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} loss={running_loss / running_samples:.6f}",
                    flush=True,
                )
                running_loss = 0.0
                running_samples = 0
            if samples_seen >= next_validation:
                metrics = evaluate(
                    model,
                    validation_records,
                    validation_collator,
                    device,
                    args.batch_size,
                )
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                torch.save(
                    {
                        "line_band_display": model.state_dict(),
                        "config": asdict(config),
                        "samples_seen": samples_seen,
                        "validation": metrics,
                    },
                    checkpoint,
                )
                if metrics["display_accuracy"] > best:
                    best = metrics["display_accuracy"]
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break

    metrics = evaluate(
        model,
        validation_records,
        validation_collator,
        device,
        args.batch_size,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "line_band_display": model.state_dict(),
            "config": asdict(config),
            "samples_seen": samples_seen,
            "validation": metrics,
        },
        output_dir / "last.pt",
    )
    if metrics["display_accuracy"] > best or not (output_dir / "best.pt").exists():
        shutil.copy2(output_dir / "last.pt", output_dir / "best.pt")
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
