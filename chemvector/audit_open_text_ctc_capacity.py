#!/usr/bin/env python3
"""Overfit a fixed balanced Open Text subset to audit OCR model capacity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset

from chemvector.ctc_ocr import ctc_loss
from chemvector.pretrain_open_text_ctc import (
    DEFAULT_DATA_ROOT,
    DEFAULT_INITIAL_CHECKPOINT,
    OpenTextDataset,
    collate_open_text,
    evaluate_open_text,
    load_initial_model,
    optimizer_for_arm,
    source_bucket,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/checkpoints/chemvector-v0811-open-text-capacity-250"
)


def balanced_subset_indices(
    dataset: OpenTextDataset,
    *,
    samples: int,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        by_bucket[source_bucket(entry)].append(index)
    rng = random.Random(seed)
    for values in by_bucket.values():
        rng.shuffle(values)
    buckets = sorted(by_bucket)
    quota = samples // len(buckets)
    selected = []
    counts = {}
    for bucket in buckets:
        take = min(quota, len(by_bucket[bucket]))
        selected.extend(by_bucket[bucket][:take])
        counts[bucket] = take
    remaining = samples - len(selected)
    pool = [
        index
        for bucket in buckets
        for index in by_bucket[bucket][counts[bucket] :]
    ]
    rng.shuffle(pool)
    selected.extend(pool[:remaining])
    for index in pool[:remaining]:
        bucket = source_bucket(dataset.entries[index])
        counts[bucket] = counts.get(bucket, 0) + 1
    if len(selected) != samples:
        raise RuntimeError(f"Could only select {len(selected)} of {samples} samples")
    rng.shuffle(selected)
    return selected, dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pixel-learning-rate", type=float, default=1e-4)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if min(args.samples, args.epochs, args.batch_size, args.evaluate_every) < 1:
        parser.error("counts must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OpenTextDataset(args.data_root, split="train")
    indices, bucket_counts = balanced_subset_indices(
        dataset, samples=args.samples, seed=args.seed
    )
    subset = Subset(dataset, indices)
    model, _ = load_initial_model(args.initial_checkpoint, arm="full")
    model.to(device)
    optimizer = optimizer_for_arm(
        model,
        learning_rate=args.learning_rate,
        pixel_learning_rate=args.pixel_learning_rate,
    )
    collate = lambda samples: collate_open_text(
        samples, model.config.maximum_bytes
    )
    evaluation_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run.json",
        {
            "protocol": "v0.8.11-open-text-capacity-probe",
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "bucket_counts": bucket_counts,
            "selected_indices": indices,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    )
    history = []
    best = -math.inf
    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate,
        )
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in loader:
            pixels = batch["pixels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(pixels)
            loss = ctc_loss(
                logits.float(),
                batch["targets"].to(device),
                batch["target_lengths"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            batches += 1
        if epoch % args.evaluate_every and epoch != args.epochs:
            continue
        metrics = evaluate_open_text(model, evaluation_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(batches, 1),
            }
        )
        history.append(metrics)
        print(
            f"epoch={epoch} "
            + json.dumps(
                {
                    "exact_rate": metrics["exact_rate"],
                    "edit_similarity": metrics["edit_similarity"],
                    "curated_macro_exact_rate": metrics[
                        "curated_macro_exact_rate"
                    ],
                    "train_loss": metrics["train_loss"],
                }
            ),
            flush=True,
        )
        payload = {
            "pretrained_ctc_ocr": model.state_dict(),
            "config": asdict(model.config),
            "arm": "full_capacity",
            "epoch": epoch,
            "validation": metrics,
            "training_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        torch.save(payload, args.output_dir / "last.pt")
        if metrics["exact_rate"] > best:
            best = metrics["exact_rate"]
            torch.save(payload, args.output_dir / "best.pt")
            write_json(args.output_dir / "best-metrics.json", metrics)
    summary = {
        "status": "passed" if best >= 0.95 else "capacity_not_proven",
        "best_exact_rate": best,
        "target_exact_rate": 0.95,
        "bucket_counts": bucket_counts,
        "history": history,
    }
    write_json(args.output_dir / "summary.json", summary)
    print("summary=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
