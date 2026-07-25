#!/usr/bin/env python3
"""Train byte-level CTC OCR on TextObject bounding-box crops."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import glob
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from Data.sdf.chemmol_webdataset import ChemMolWebDataset
from chemvector.ctc_ocr import (
    ChemicalCTCConfig,
    ChemicalCTCOCR,
    bbox_crop,
    ctc_loss,
    decode_ctc,
    encode_text,
)
from chemvector.layer_ocr import colorize_visible_text


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def is_modifier_label(text: str) -> bool:
    return bool(re.search(r"H|[+-]|\d", text))


class ChemicalTextCropDataset(IterableDataset):
    def __init__(
        self,
        shards: list[str],
        *,
        seed: int,
        crop_limit: int,
        color_probability: float,
        training: bool,
        simple_keep_probability: float = 0.5,
    ) -> None:
        super().__init__()
        expanded = []
        for pattern in shards:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No CTC OCR shards matched: {pattern}")
            expanded.extend(matches)
        self.shards = expanded
        self.seed = seed
        self.crop_limit = crop_limit
        self.color_probability = color_probability
        self.training = training
        self.simple_keep_probability = simple_keep_probability

    def __iter__(self):
        yielded = 0
        dataset = ChemMolWebDataset(
            self.shards,
            validate_legacy_hash=False,
            shardshuffle=self.training,
            sample_shuffle_buffer=128 if self.training else 0,
            sequence_version="0.4",
        )
        for sample in dataset:
            if yielded >= self.crop_limit:
                break
            rng = random.Random(stable_seed(self.seed, sample["key"]))
            atoms = [
                atom
                for atom in sample["layers"]["atoms"]
                if atom["visible_text"] and atom.get("bbox") and atom.get("text")
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
                modifier = is_modifier_label(text)
                if not modifier and rng.random() > self.simple_keep_probability:
                    continue
                context = rng.uniform(2.0, 10.0) if self.training else 4.0
                jitter = 1.5 if self.training else 0.0
                crop = bbox_crop(
                    image,
                    atom["bbox"],
                    context=context,
                    jitter_x=rng.uniform(-jitter, jitter),
                    jitter_y=rng.uniform(-jitter, jitter),
                )
                array = np.asarray(crop, dtype=np.float32) / 255.0
                yield {
                    "pixels": (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5,
                    "text": text,
                    "modifier": modifier,
                    "colored": colored,
                    "key": f"{sample['key']}-a{atom['atom_id']}",
                }
                yielded += 1
                if yielded >= self.crop_limit:
                    break


def collate(samples: list[dict], maximum_bytes: int) -> dict:
    encoded = [encode_text(sample["text"], maximum_bytes) for sample in samples]
    return {
        "pixels": torch.stack([sample["pixels"] for sample in samples]),
        "targets": torch.cat(encoded),
        "target_lengths": torch.tensor([len(item) for item in encoded], dtype=torch.long),
        "texts": [sample["text"] for sample in samples],
        "modifiers": [sample["modifier"] for sample in samples],
        "colored": [sample["colored"] for sample in samples],
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    totals = {"samples": 0, "exact": 0, "modifier_samples": 0, "modifier_exact": 0,
              "simple_samples": 0, "simple_exact": 0, "color_samples": 0,
              "color_exact": 0, "bw_samples": 0, "bw_exact": 0,
              "false_modifier": 0}
    labels: dict[str, list[int]] = {}
    for batch in loader:
        logits = model(batch["pixels"].to(device))
        predictions = [decode_ctc(row) for row in logits.argmax(-1).cpu()]
        for expected, predicted, modifier, colored in zip(
            batch["texts"], predictions, batch["modifiers"], batch["colored"]
        ):
            exact = expected == predicted
            totals["samples"] += 1
            totals["exact"] += int(exact)
            bucket = "modifier" if modifier else "simple"
            totals[f"{bucket}_samples"] += 1
            totals[f"{bucket}_exact"] += int(exact)
            color_bucket = "color" if colored else "bw"
            totals[f"{color_bucket}_samples"] += 1
            totals[f"{color_bucket}_exact"] += int(exact)
            totals["false_modifier"] += int(
                not modifier and is_modifier_label(predicted)
            )
            values = labels.setdefault(expected, [0, 0])
            values[0] += int(exact)
            values[1] += 1
    model.train()
    result = {**totals}
    for bucket in ("", "modifier_", "simple_", "color_", "bw_"):
        exact_key = f"{bucket}exact" if bucket else "exact"
        sample_key = f"{bucket}samples" if bucket else "samples"
        result[f"{bucket}exact_rate"] = totals[exact_key] / max(totals[sample_key], 1)
    result["false_modifier_rate"] = totals["false_modifier"] / max(totals["simple_samples"], 1)
    result["labels"] = {
        label: {"exact_rate": correct / count, "samples": count}
        for label, (correct, count) in sorted(labels.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-crops", type=int, default=10000)
    parser.add_argument("--validation-crops", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--color-probability", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    config = ChemicalCTCConfig()
    model = ChemicalCTCOCR(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    make_collate = partial(collate, maximum_bytes=config.maximum_bytes)
    validation_loader = DataLoader(
        ChemicalTextCropDataset(
            args.validation_shards,
            seed=args.seed + 1,
            crop_limit=args.validation_crops,
            color_probability=args.color_probability,
            training=False,
        ),
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=make_collate,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(
            ChemicalTextCropDataset(
                args.train_shards,
                seed=args.seed + epoch,
                crop_limit=args.train_crops,
                color_probability=args.color_probability,
                training=True,
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            batches += 1
        metrics = evaluate(model, validation_loader, device)
        metrics["train_loss"] = loss_sum / max(batches, 1)
        print(f"epoch={epoch} " + json.dumps(metrics, ensure_ascii=False), flush=True)
        payload = {
            "ctc_ocr": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "validation": metrics,
            "training_args": vars(args),
        }
        torch.save(payload, output / "last.pt")
        if metrics["exact_rate"] > best:
            best = metrics["exact_rate"]
            torch.save(payload, output / "best.pt")
            (output / "best-validation.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
