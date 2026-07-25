#!/usr/bin/env python3
"""Train Semantic Bond Relation v0.1 on positive GT endpoint pairs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageFilter
import torch

from chemvector.bond_object import BOND_TYPE_TO_ID, BOND_TYPES
from chemvector.data import ChemVectorIterableDataset
from chemvector.semantic_bond_relation import (
    SemanticBondRelation,
    SemanticBondRelationConfig,
    semantic_bond_relation_loss,
)
from chemvector.train import expand_shards


class PositiveBondRelationCollator:
    def __init__(
        self,
        image_size: int = 768,
        *,
        degradation_probability: float = 0.0,
        seed: int = 20260723,
    ) -> None:
        self.image_size = image_size
        if not 0.0 <= degradation_probability <= 1.0:
            raise ValueError("degradation probability must lie in [0, 1]")
        self.degradation_probability = degradation_probability
        self.rng = random.Random(seed)

    def _pixels(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.rng.random() < self.degradation_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.1))
            reduced = (
                max(1, round(image.width * 0.72)),
                max(1, round(image.height * 0.72)),
            )
            image = image.resize(reduced, Image.Resampling.BILINEAR).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        elif image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5

    def __call__(self, samples: list[dict]) -> dict:
        maximum_atoms = max(len(sample["layers"]["atoms"]) for sample in samples)
        maximum_bonds = max(len(sample["layers"]["bonds"]) for sample in samples)
        points = torch.zeros((len(samples), maximum_atoms, 2))
        pair_indices = torch.zeros(
            (len(samples), maximum_bonds, 2), dtype=torch.long
        )
        pair_mask = torch.zeros((len(samples), maximum_bonds), dtype=torch.bool)
        display = torch.zeros((len(samples), maximum_bonds), dtype=torch.long)
        chemical = torch.zeros_like(display)
        for batch_index, sample in enumerate(samples):
            width = float(sample["layers"]["width"])
            height = float(sample["layers"]["height"])
            for atom in sample["layers"]["atoms"]:
                points[batch_index, atom["atom_id"]] = torch.tensor(
                    (atom["center"][0] / width, atom["center"][1] / height)
                )
            for offset, bond in enumerate(sample["layers"]["bonds"]):
                pair_indices[batch_index, offset] = torch.tensor(
                    (bond["begin_atom"], bond["end_atom"])
                )
                pair_mask[batch_index, offset] = True
                display[batch_index, offset] = BOND_TYPE_TO_ID[bond["display_type"]]
                chemical[batch_index, offset] = BOND_TYPE_TO_ID[bond["chemical_type"]]
        return {
            "pixel_values": torch.stack(
                [self._pixels(sample["image"]) for sample in samples]
            ),
            "points": points,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "display_targets": display[pair_mask],
            "chemical_targets": chemical[pair_mask],
            "bond_counts": pair_mask.sum(-1),
            "keys": [sample["key"] for sample in samples],
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def class_weights(records: list[dict], field: str, device: torch.device) -> torch.Tensor:
    counts = Counter(
        bond[field]
        for record in records
        for bond in record["layers"]["bonds"]
    )
    total = sum(counts.values())
    values = []
    for name in BOND_TYPES:
        count = max(counts.get(name, 0), 1)
        values.append((total / (len(BOND_TYPES) * count)) ** 0.5)
    weights = torch.tensor(values, device=device, dtype=torch.float32)
    return weights / weights.mean()


def forward(
    model: SemanticBondRelation,
    batch: dict,
    device: torch.device,
    display_weights: torch.Tensor | None = None,
    chemical_weights: torch.Tensor | None = None,
):
    current = move_batch(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            current["pixel_values"],
            current["points"],
            current["pair_indices"],
            current["pair_mask"],
        )
        losses = semantic_bond_relation_loss(
            output,
            current["display_targets"],
            current["chemical_targets"],
            display_weights=display_weights,
            chemical_weights=chemical_weights,
        )
    return output, losses, current


@torch.no_grad()
def evaluate(
    model: SemanticBondRelation,
    records: list[dict],
    collator: PositiveBondRelationCollator,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    totals = Counter()
    display_classes = Counter()
    display_hits = Counter()
    chemical_classes = Counter()
    chemical_hits = Counter()
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        output, _, current = forward(model, batch, device)
        display_prediction = output["display_logits"].argmax(-1)
        chemical_prediction = output["chemical_logits"].argmax(-1)
        display_correct = display_prediction == current["display_targets"]
        chemical_correct = chemical_prediction == current["chemical_targets"]
        joint_correct = display_correct & chemical_correct
        totals["bonds"] += len(display_correct)
        totals["display_correct"] += int(display_correct.sum())
        totals["chemical_correct"] += int(chemical_correct.sum())
        totals["joint_correct"] += int(joint_correct.sum())
        offset = 0
        for count in current["bond_counts"].tolist():
            totals["samples"] += 1
            totals["sample_exact"] += int(bool(joint_correct[offset : offset + count].all()))
            offset += count
        for class_id, name in enumerate(BOND_TYPES):
            selected = current["display_targets"] == class_id
            display_classes[name] += int(selected.sum())
            display_hits[name] += int((display_correct & selected).sum())
            selected = current["chemical_targets"] == class_id
            chemical_classes[name] += int(selected.sum())
            chemical_hits[name] += int((chemical_correct & selected).sum())
    return {
        "samples": totals["samples"],
        "bonds": totals["bonds"],
        "display_accuracy": totals["display_correct"] / max(totals["bonds"], 1),
        "chemical_accuracy": totals["chemical_correct"] / max(totals["bonds"], 1),
        "joint_accuracy": totals["joint_correct"] / max(totals["bonds"], 1),
        "sample_exact": totals["sample_exact"] / max(totals["samples"], 1),
        "display_by_type": {
            name: display_hits[name] / max(display_classes[name], 1)
            for name in BOND_TYPES
        },
        "chemical_by_type": {
            name: chemical_hits[name] / max(chemical_classes[name], 1)
            for name in BOND_TYPES
        },
        "display_counts": dict(display_classes),
        "chemical_counts": dict(chemical_classes),
    }


def save(path: Path, model, optimizer, config, args, samples_seen, metrics) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "semantic_bond_relation": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "training_args": vars(args),
            "samples_seen": samples_seen,
            "validation": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--degradation-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    def cache(shards, count):
        dataset = ChemVectorIterableDataset(
            expand_shards(shards), sequence_version="0.4"
        )
        return list(islice(iter(dataset), count))

    train_records = cache(args.train_shards, args.train_cache_samples)
    validation_records = cache(args.validation_shards, args.validation_samples)
    if len(train_records) != args.train_cache_samples:
        raise ValueError("Semantic Bond train cache is incomplete")
    if len(validation_records) != args.validation_samples:
        raise ValueError("Semantic Bond validation cache is incomplete")

    config = SemanticBondRelationConfig()
    model = SemanticBondRelation(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    display_weights = class_weights(train_records, "display_type", device)
    chemical_weights = class_weights(train_records, "chemical_type", device)
    train_collator = PositiveBondRelationCollator(
        degradation_probability=args.degradation_probability,
        seed=args.seed,
    )
    clean_collator = PositiveBondRelationCollator(seed=args.seed + 1)
    degraded_collator = PositiveBondRelationCollator(
        degradation_probability=1.0, seed=args.seed + 2
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best_score = (-1.0, -1.0, -1.0)
    running_loss = 0.0
    running_samples = 0
    while samples_seen < args.train_samples:
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                break
            indices = order[start : start + min(args.batch_size, remaining)]
            batch = train_collator([train_records[index] for index in indices])
            model.train()
            _, losses, _ = forward(
                model,
                batch,
                device,
                display_weights,
                chemical_weights,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_loss += float(losses["loss"].detach()) * len(indices)
            running_samples += len(indices)
            if samples_seen // args.log_every != (samples_seen - len(indices)) // args.log_every:
                print(
                    f"samples={samples_seen} loss={running_loss / max(running_samples, 1):.6f}",
                    flush=True,
                )
                running_loss = 0.0
                running_samples = 0
            if samples_seen >= next_validation:
                clean = evaluate(
                    model, validation_records, clean_collator, device, args.batch_size
                )
                degraded = evaluate(
                    model,
                    validation_records,
                    degraded_collator,
                    device,
                    args.batch_size,
                )
                metrics = {"clean": clean, "degraded": degraded}
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save(checkpoint, model, optimizer, config, args, samples_seen, metrics)
                score = (
                    degraded["joint_accuracy"],
                    clean["joint_accuracy"],
                    degraded["sample_exact"],
                )
                if score > best_score:
                    best_score = score
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every

    metrics = {
        "clean": evaluate(
            model, validation_records, clean_collator, device, args.batch_size
        ),
        "degraded": evaluate(
            model, validation_records, degraded_collator, device, args.batch_size
        ),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save(output_dir / "last.pt", model, optimizer, config, args, samples_seen, metrics)
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
