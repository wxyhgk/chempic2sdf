#!/usr/bin/env python3
"""Train the v1 P1 bond/text semantic ink separator."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
    semantic_ink_loss,
)
from chemvector.core.data import ChemVectorIterableDataset
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


MODEL_NAME = "chemvector-v1p1-semantic-ink-separation-pilot"
DEFAULT_OUTPUT = Path(f"outputs/checkpoints/{MODEL_NAME}")
DEFAULT_MANIFEST = Path(
    "outputs/audits/chemvector-v1p0r-edge-centric-safe-manifest/manifest.jsonl"
)
MANIFEST_SHA256 = (
    "eb9da30c85aaede01b635294f53dca4454825f13db55002cb2df843edb49f11b"
)
DATA_ROOT = Path("Data/sdf/chemvision_layers_v0_1/pilot_10k")
EXPECTED_SPLITS = {"train": 8_856, "validation": 496, "internal_test": 496}
GO_GATES = {
    "bond_iou": 0.90,
    "text_iou": 0.75,
    "foreground_iou": 0.90,
    "stress_bond_iou": 0.88,
    "bond_on_text_only_rate_max": 0.10,
}


def load_safe_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    if file_sha256(path) != MANIFEST_SHA256:
        raise ValueError("v1 P1 safe manifest SHA256 differs")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    split_counts = Counter(str(row["split"]) for row in rows)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise ValueError(f"v1 P1 safe split counts differ: {dict(split_counts)}")
    if len({str(row["sample_key"]) for row in rows}) != len(rows):
        raise ValueError("v1 P1 safe manifest has duplicate sample keys")
    return rows


class SemanticInkCollator:
    @staticmethod
    def _image(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    @staticmethod
    def _layer(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(array)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if any(sample["image"].size != (768, 768) for sample in samples):
            raise ValueError("v1 P1 requires native 768x768 images")
        return {
            "pixel_values": torch.stack([self._image(sample["image"]) for sample in samples]),
            "targets": torch.stack(
                [
                    torch.stack(
                        (
                            self._layer(sample["bond_layer"]),
                            self._layer(sample["text_layer"]),
                        )
                    )
                    for sample in samples
                ]
            ),
            "keys": [sample["key"] for sample in samples],
            "variants": [str(sample["meta"]["variant"]) for sample in samples],
        }


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split]
    if len(selected) != EXPECTED_SPLITS[split]:
        raise ValueError(f"v1 P1 {split} count differs")
    return selected


def _loader(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    keys = {str(row["sample_key"]) for row in rows}
    shards = sorted({str(DATA_ROOT / row["source_shard"]) for row in rows})
    split = str(rows[0]["split"])
    if split == "train":
        if any("/train-" not in shard for shard in shards):
            raise ValueError("v1 P1 train loader received a non-train shard")
    elif split == "validation":
        if any("/validation-" not in shard for shard in shards):
            raise ValueError("v1 P1 validation loader received a non-validation shard")
    else:
        raise ValueError(f"v1 P1 cannot construct loader for split {split!r}")
    dataset = ChemVectorIterableDataset(
        shards,
        shardshuffle=False,
        sample_shuffle_buffer=512 if shuffle else 0,
        sample_shuffle_seed=seed,
        include_keys=keys,
        serialize_target=False,
        load_raster_layers=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=SemanticInkCollator(),
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _new_binary_counts() -> dict[str, int]:
    return {"intersection": 0, "union": 0, "predicted": 0, "target": 0}


def _update_binary_counts(
    counts: dict[str, int], predicted: torch.Tensor, target: torch.Tensor
) -> None:
    counts["intersection"] += int((predicted & target).sum())
    counts["union"] += int((predicted | target).sum())
    counts["predicted"] += int(predicted.sum())
    counts["target"] += int(target.sum())


def _binary_metrics(counts: dict[str, int]) -> dict[str, float]:
    intersection = counts["intersection"]
    return {
        "iou": intersection / max(counts["union"], 1),
        "precision": intersection / max(counts["predicted"], 1),
        "recall": intersection / max(counts["target"], 1),
    }


@torch.no_grad()
def evaluate(
    model: SemanticInkSeparator,
    loader: DataLoader,
    device: torch.device,
    *,
    bond_threshold: float = 0.5,
    text_threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    channel_counts = [_new_binary_counts(), _new_binary_counts()]
    foreground_counts = _new_binary_counts()
    variant_bond_counts: dict[str, dict[str, int]] = defaultdict(_new_binary_counts)
    text_only_pixels = 0
    bond_on_text_only = 0
    samples = 0
    image_both_iou75 = 0
    loss_totals: Counter[str] = Counter()
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixels)
            losses = semantic_ink_loss(output, targets, model.config)
        probabilities = output["logits"].float().sigmoid()
        thresholds = probabilities.new_tensor(
            (bond_threshold, text_threshold)
        ).view(1, 2, 1, 1)
        predicted = probabilities >= thresholds
        expected = targets >= (8.0 / 255.0)
        batch_size = predicted.size(0)
        samples += batch_size
        for name, value in losses.items():
            loss_totals[name] += float(value) * batch_size
        for channel in range(2):
            _update_binary_counts(
                channel_counts[channel], predicted[:, channel], expected[:, channel]
            )
        predicted_foreground = predicted.any(dim=1)
        expected_foreground = expected.any(dim=1)
        _update_binary_counts(
            foreground_counts, predicted_foreground, expected_foreground
        )
        text_only = expected[:, 1] & ~expected[:, 0]
        text_only_pixels += int(text_only.sum())
        bond_on_text_only += int((predicted[:, 0] & text_only).sum())
        for index, variant in enumerate(batch["variants"]):
            _update_binary_counts(
                variant_bond_counts[variant],
                predicted[index, 0],
                expected[index, 0],
            )
            ious = []
            for channel in range(2):
                intersection = int(
                    (predicted[index, channel] & expected[index, channel]).sum()
                )
                union = int(
                    (predicted[index, channel] | expected[index, channel]).sum()
                )
                ious.append(intersection / max(union, 1))
            image_both_iou75 += int(min(ious) >= 0.75)
    return {
        "samples": samples,
        "losses": {name: value / max(samples, 1) for name, value in loss_totals.items()},
        "bond": _binary_metrics(channel_counts[0]),
        "text": _binary_metrics(channel_counts[1]),
        "foreground": _binary_metrics(foreground_counts),
        "bond_by_variant": {
            name: _binary_metrics(counts)
            for name, counts in sorted(variant_bond_counts.items())
        },
        "bond_on_text_only_rate": bond_on_text_only / max(text_only_pixels, 1),
        "image_both_iou75_rate": image_both_iou75 / max(samples, 1),
    }


def _checkpoint(
    path: Path,
    model: SemanticInkSeparator,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    metrics: dict[str, Any],
    train_samples_seen: int,
) -> None:
    payload = {
        "schema_version": "chemvector-v1p1-semantic-ink-checkpoint-0.1",
        "model_name": MODEL_NAME,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "train_samples_seen": train_samples_seen,
        "safe_manifest_sha256": MANIFEST_SHA256,
        "internal_test_read": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v1 P1 semantic ink training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"v1 P1 output is not empty: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_safe_manifest(args.manifest)
    train_rows = _split_rows(rows, "train")
    validation_rows = _split_rows(rows, "validation")
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_validation_samples is not None:
        validation_rows = validation_rows[: args.max_validation_samples]
    if not train_rows or not validation_rows:
        raise ValueError("v1 P1 train and validation rows must be nonempty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = SemanticInkSeparator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    started = time.perf_counter()
    train_samples_seen = 0
    history = []
    best_score = float("-inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = _loader(
            train_rows,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            workers=args.workers,
        )
        epoch_seen = 0
        running = 0.0
        for step, batch in enumerate(loader, 1):
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(pixels)
                losses = semantic_ink_loss(output, targets, model.config)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(
                    f"v1 P1 non-finite loss at epoch={epoch}, step={step}"
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_samples = pixels.size(0)
            epoch_seen += batch_samples
            train_samples_seen += batch_samples
            running += float(losses["loss"].detach()) * batch_samples
            if step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "samples": train_samples_seen,
                            "loss": running / max(epoch_seen, 1),
                        }
                    ),
                    flush=True,
                )
        if epoch_seen != len(train_rows):
            raise ValueError(
                f"v1 P1 epoch read {epoch_seen} samples, expected {len(train_rows)}"
            )
        validation = evaluate(
            model,
            _loader(
                validation_rows,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                workers=args.workers,
            ),
            device,
        )
        if validation["samples"] != len(validation_rows):
            raise ValueError("v1 P1 validation sample count differs")
        score = min(validation["bond"]["iou"], validation["text"]["iou"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / epoch_seen,
                "train_samples_seen": train_samples_seen,
                "validation": validation,
                "selection_score": score,
            }
        )
        _checkpoint(
            args.output_dir / f"epoch-{epoch:02d}.pt",
            model,
            optimizer,
            epoch=epoch,
            metrics=validation,
            train_samples_seen=train_samples_seen,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            _checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                metrics=validation,
                train_samples_seen=train_samples_seen,
            )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    best_payload = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_payload["model"], strict=True)
    final_metrics = evaluate(
        model,
        _loader(
            validation_rows,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
            workers=args.workers,
        ),
        device,
    )
    checks = {
        "validation_samples_exact": final_metrics["samples"] == len(validation_rows),
        "bond_iou_min90pct": final_metrics["bond"]["iou"] >= GO_GATES["bond_iou"],
        "text_iou_min75pct": final_metrics["text"]["iou"] >= GO_GATES["text_iou"],
        "foreground_iou_min90pct": final_metrics["foreground"]["iou"]
        >= GO_GATES["foreground_iou"],
        "stress_bond_iou_min88pct": final_metrics["bond_by_variant"]["stress"]["iou"]
        >= GO_GATES["stress_bond_iou"],
        "bond_on_text_only_max10pct": final_metrics["bond_on_text_only_rate"]
        <= GO_GATES["bond_on_text_only_rate_max"],
        "internal_test_unread": True,
        "native_resolution_768": True,
    }
    result = {
        "schema_version": "chemvector-v1p1-semantic-ink-result-0.1",
        "model_name": MODEL_NAME,
        "decision": "semantic_ink_separation_go" if all(checks.values()) else "semantic_ink_separation_no_go",
        "checks": checks,
        "gates": GO_GATES,
        "config": asdict(model.config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "safe_manifest_sha256": MANIFEST_SHA256,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "internal_test_samples_read": 0,
        "train_samples_seen": train_samples_seen,
        "best_epoch": best_epoch,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "best_checkpoint_sha256": file_sha256(args.output_dir / "best.pt"),
        "metrics": final_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
