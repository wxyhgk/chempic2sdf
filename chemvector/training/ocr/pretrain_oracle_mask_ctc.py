#!/usr/bin/env python3
"""Run balanced Oracle text-layer A/B probes for the v0.9 OCR direction."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import glob
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, IterableDataset

from Data.sdf.chemmol_webdataset import ChemMolWebDataset
from chemvector.ctc_ocr import (
    ConditionedPixelCTCConfig,
    ConditionedPixelCTCOCR,
    ctc_loss,
)
from chemvector.layer_ocr import colorize_visible_text
from chemvector.pretrain_ctc_ocr import (
    collate,
    evaluate,
    is_modifier_label,
    stable_seed,
)


ARM_CONFIGS = {
    "rgb": {"condition_channels": 0, "masked_rgb": False, "height_attention": False},
    "masked_rgb": {
        "condition_channels": 0,
        "masked_rgb": True,
        "height_attention": False,
    },
    "rgb_text_mask": {
        "condition_channels": 1,
        "masked_rgb": False,
        "height_attention": False,
    },
    "rgb_text_mask_anchor": {
        "condition_channels": 2,
        "masked_rgb": False,
        "height_attention": True,
    },
}


def expand_shards(patterns: list[str]) -> list[str]:
    shards = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No Oracle OCR shards matched: {pattern}")
        shards.extend(matches)
    return shards


def count_visible_labels(shards: list[str]) -> Counter:
    counts = Counter()
    dataset = ChemMolWebDataset(
        shards,
        validate_legacy_hash=False,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        sequence_version="0.4",
    )
    for sample in dataset:
        for atom in sample["layers"]["atoms"]:
            if atom.get("visible_text") and atom.get("text"):
                counts[str(atom["text"])] += 1
    return counts


def balanced_label_repeats(counts: Counter, maximum_repeat: int = 8) -> dict[str, int]:
    if not counts:
        raise ValueError("Cannot balance an empty OCR label distribution")
    reference = max(counts.values())
    return {
        label: min(maximum_repeat, max(1, math.ceil(math.sqrt(reference / count))))
        for label, count in counts.items()
    }


def crop_layer(
    image: Image.Image,
    bbox: list[float],
    *,
    context: float,
    jitter_x: float,
    jitter_y: float,
    output_size: tuple[int, int] = (192, 48),
    fill: int | str = 0,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    crop = image.crop(
        (
            x0 - context + jitter_x,
            y0 - context + jitter_y,
            x1 + context + jitter_x,
            y1 + context + jitter_y,
        )
    )
    maximum_width, maximum_height = output_size
    scale = min(maximum_width / max(crop.width, 1), maximum_height / max(crop.height, 1))
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        resample,
    )
    canvas = Image.new(image.mode, output_size, fill)
    canvas.paste(
        resized,
        ((maximum_width - resized.width) // 2, (maximum_height - resized.height) // 2),
    )
    return canvas


def anchor_layer(size: tuple[int, int], center: list[float]) -> Image.Image:
    layer = Image.new("L", size, 0)
    draw = ImageDraw.Draw(layer)
    x, y = (float(value) for value in center)
    draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), fill=255)
    return layer


class OracleMaskCropDataset(IterableDataset):
    def __init__(
        self,
        shards: list[str],
        *,
        arm: str,
        seed: int,
        crop_limit: int,
        color_probability: float,
        training: bool,
        repeats: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if arm not in ARM_CONFIGS:
            raise ValueError(f"Unknown Oracle OCR arm: {arm}")
        self.shards = shards
        self.arm = arm
        self.seed = seed
        self.crop_limit = crop_limit
        self.color_probability = color_probability
        self.training = training
        self.repeats = repeats or {}

    def __iter__(self):
        yielded = 0
        dataset = ChemMolWebDataset(
            self.shards,
            validate_legacy_hash=False,
            shardshuffle=self.training,
            sample_shuffle_buffer=256 if self.training else 0,
            sequence_version="0.4",
        )
        config = ARM_CONFIGS[self.arm]
        for sample in dataset:
            if yielded >= self.crop_limit:
                break
            if "text_layer" not in sample:
                raise ValueError(f"Sample {sample['key']} has no text layer")
            rng = random.Random(stable_seed(self.seed, sample["key"]))
            atoms = [
                atom
                for atom in sample["layers"]["atoms"]
                if atom.get("visible_text") and atom.get("bbox") and atom.get("text")
            ]
            rng.shuffle(atoms)
            colored = rng.random() < self.color_probability
            image = (
                colorize_visible_text(sample["image"], sample["layers"]["atoms"])
                if colored
                else sample["image"]
            )
            for atom in atoms:
                text = str(atom["text"])
                copies = self.repeats.get(text, 1) if self.training else 1
                for copy_index in range(copies):
                    context = rng.uniform(2.0, 10.0) if self.training else 4.0
                    jitter = 1.5 if self.training else 0.0
                    jitter_x = rng.uniform(-jitter, jitter)
                    jitter_y = rng.uniform(-jitter, jitter)
                    rgb = crop_layer(
                        image.convert("RGB"),
                        atom["bbox"],
                        context=context,
                        jitter_x=jitter_x,
                        jitter_y=jitter_y,
                        fill="white",
                    )
                    text_mask = crop_layer(
                        sample["text_layer"],
                        atom["bbox"],
                        context=context,
                        jitter_x=jitter_x,
                        jitter_y=jitter_y,
                        fill=0,
                    )
                    rgb_array = np.asarray(rgb, dtype=np.float32) / 255.0
                    mask_array = np.asarray(text_mask, dtype=np.float32) / 255.0
                    if config["masked_rgb"]:
                        rgb_array = (
                            rgb_array * mask_array[..., None]
                            + (1.0 - mask_array[..., None])
                        )
                    channels = [
                        (torch.from_numpy(rgb_array).permute(2, 0, 1) - 0.5) / 0.5
                    ]
                    if config["condition_channels"]:
                        channels.append(torch.from_numpy(mask_array)[None])
                    if config["condition_channels"] == 2:
                        anchor = crop_layer(
                            anchor_layer(image.size, atom["center"]),
                            atom["bbox"],
                            context=context,
                            jitter_x=jitter_x,
                            jitter_y=jitter_y,
                            fill=0,
                        )
                        channels.append(
                            torch.from_numpy(
                                np.asarray(anchor, dtype=np.float32) / 255.0
                            )[None]
                        )
                    yield {
                        "pixels": torch.cat(channels),
                        "text": text,
                        "modifier": is_modifier_label(text),
                        "colored": colored,
                        "key": f"{sample['key']}-a{atom['atom_id']}-c{copy_index}",
                    }
                    yielded += 1
                    if yielded >= self.crop_limit:
                        break
                if yielded >= self.crop_limit:
                    break


def add_macro_metrics(metrics: dict) -> dict:
    rates = [item["exact_rate"] for item in metrics["labels"].values()]
    supported = [
        item["exact_rate"]
        for item in metrics["labels"].values()
        if item["samples"] >= 20
    ]
    metrics["macro_exact_rate"] = sum(rates) / max(len(rates), 1)
    metrics["supported_macro_exact_rate"] = sum(supported) / max(len(supported), 1)
    metrics["label_count"] = len(rates)
    metrics["supported_label_count"] = len(supported)
    return metrics


def train_arm(
    arm: str,
    args,
    train_shards: list[str],
    validation_shards: list[str],
    repeats: dict[str, int],
    output_root: Path,
    device: torch.device,
) -> dict:
    arm_spec = ARM_CONFIGS[arm]
    config = ConditionedPixelCTCConfig(
        condition_channels=arm_spec["condition_channels"],
        height_attention=arm_spec["height_attention"],
    )
    model = ConditionedPixelCTCOCR(config)
    model.load_pretrained_ctc_checkpoint(args.initial_ctc_checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    make_collate = partial(collate, maximum_bytes=config.maximum_bytes)
    validation_loader = DataLoader(
        OracleMaskCropDataset(
            validation_shards,
            arm=arm,
            seed=args.seed + 1,
            crop_limit=args.validation_crops,
            color_probability=args.color_probability,
            training=False,
        ),
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=make_collate,
    )
    output = output_root / arm
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(
            OracleMaskCropDataset(
                train_shards,
                arm=arm,
                seed=args.seed + epoch,
                crop_limit=args.train_crops,
                color_probability=args.color_probability,
                training=True,
                repeats=repeats,
            ),
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=make_collate,
        )
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in loader:
            logits = model(batch["pixels"].to(device))
            loss = ctc_loss(
                logits,
                batch["targets"].to(device),
                batch["target_lengths"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            batches += 1
        metrics = add_macro_metrics(evaluate(model, validation_loader, device))
        metrics["train_loss"] = loss_sum / max(batches, 1)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"arm={arm} epoch={epoch} " + json.dumps(metrics, ensure_ascii=False),
            flush=True,
        )
        payload = {
            "conditioned_ctc_ocr": model.state_dict(),
            "config": asdict(config),
            "arm": arm,
            "epoch": epoch,
            "validation": metrics,
            "training_args": vars(args),
        }
        torch.save(payload, output / "last.pt")
        score = metrics["supported_macro_exact_rate"]
        if score > best:
            best = score
            torch.save(payload, output / "best.pt")
            (output / "best-validation.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return {
        "arm": arm,
        "best": json.loads((output / "best-validation.json").read_text()),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--initial-ctc-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARM_CONFIGS), default=list(ARM_CONFIGS))
    parser.add_argument("--train-crops", type=int, default=10000)
    parser.add_argument("--validation-crops", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--color-probability", type=float, default=0.5)
    parser.add_argument("--maximum-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards)
    counts = count_visible_labels(train_shards)
    repeats = balanced_label_repeats(counts, args.maximum_repeat)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    distribution = {
        label: {"count": counts[label], "repeat": repeats[label]}
        for label in sorted(counts)
    }
    (output / "label-distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    results = []
    for arm in args.arms:
        results.append(
            train_arm(
                arm,
                args,
                train_shards,
                validation_shards,
                repeats,
                output,
                device,
            )
        )
    summary = {
        "protocol": "v0.9-oracle-text-layer-ab",
        "arms": results,
        "label_distribution": distribution,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
