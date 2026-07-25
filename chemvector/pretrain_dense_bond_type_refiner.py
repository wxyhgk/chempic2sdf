#!/usr/bin/env python3
"""Fine-tune the line-band reader only on confirmed bond objects."""

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

from chemvector.bond_object import BOND_TYPE_TO_ID, BOND_TYPES
from chemvector.data import ChemVectorIterableDataset
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
)
from chemvector.pretrain_node_conditioned_bond import KnownEndpointBondCollator
from chemvector.train import expand_shards


class PositiveBondTypeCollator(KnownEndpointBondCollator):
    """Pack only true bonds, with no all-pairs edge-classification examples."""

    def __init__(
        self,
        image_size: int = 768,
        *,
        endpoint_jitter_sigma_pixels: float = 0.0,
        endpoint_jitter_max_pixels: float = 0.0,
        endpoint_jitter_probability: float = 0.0,
        endpoint_swap_probability: float = 0.0,
        seed: int = 20260718,
    ) -> None:
        super().__init__(
            image_size,
            endpoint_jitter_sigma_pixels=endpoint_jitter_sigma_pixels,
            endpoint_jitter_max_pixels=endpoint_jitter_max_pixels,
            endpoint_jitter_probability=endpoint_jitter_probability,
            seed=seed,
        )
        if not 0.0 <= endpoint_swap_probability <= 1.0:
            raise ValueError("Endpoint swap probability must lie in [0, 1]")
        self.endpoint_swap_probability = endpoint_swap_probability

    def __call__(self, samples: list[dict]) -> dict:
        maximum_bonds = max(len(sample["layers"]["bonds"]) for sample in samples)
        points = torch.zeros((len(samples), maximum_bonds * 2, 2))
        point_mask = torch.zeros(
            (len(samples), maximum_bonds * 2), dtype=torch.bool
        )
        pair_indices = torch.zeros(
            (len(samples), maximum_bonds, 2), dtype=torch.long
        )
        pair_mask = torch.zeros(
            (len(samples), maximum_bonds), dtype=torch.bool
        )
        display = torch.zeros(
            (len(samples), maximum_bonds), dtype=torch.long
        )
        chemical = torch.zeros_like(display)
        for batch_index, sample in enumerate(samples):
            width = float(sample["layers"]["width"])
            height = float(sample["layers"]["height"])
            for bond_index, bond in enumerate(sample["layers"]["bonds"]):
                first = bond_index * 2
                points[batch_index, first] = torch.tensor(
                    (bond["begin"][0] / width, bond["begin"][1] / height)
                )
                points[batch_index, first + 1] = torch.tensor(
                    (bond["end"][0] / width, bond["end"][1] / height)
                )
                pair_indices[batch_index, bond_index] = torch.tensor(
                    (first, first + 1)
                )
                point_mask[batch_index, first : first + 2] = True
                pair_mask[batch_index, bond_index] = True
                display[batch_index, bond_index] = BOND_TYPE_TO_ID[
                    bond["display_type"]
                ]
                chemical[batch_index, bond_index] = BOND_TYPE_TO_ID[
                    bond["chemical_type"]
                ]
        points = self._jitter_points(points, point_mask)
        if self.endpoint_swap_probability:
            swaps = (
                torch.rand(pair_mask.shape, generator=self.generator)
                < self.endpoint_swap_probability
            ) & pair_mask
            shaped = points.reshape(len(samples), maximum_bonds, 2, 2)
            points = torch.where(
                swaps[:, :, None, None], shaped.flip(2), shaped
            ).reshape(len(samples), maximum_bonds * 2, 2)
        return {
            "pixel_values": torch.stack(
                [self._pixels(sample["image"]) for sample in samples]
            ),
            "points": points,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "display_targets": display,
            "chemical_targets": chemical,
            "keys": [sample["key"] for sample in samples],
        }


def class_balancing_weights(
    records: list[dict], field: str, power: float, maximum: float
) -> torch.Tensor:
    counts = torch.zeros(len(BOND_TYPES), dtype=torch.float64)
    for sample in records:
        for bond in sample["layers"]["bonds"]:
            counts[BOND_TYPE_TO_ID[bond[field]]] += 1
    if (counts == 0).any():
        counts = counts.clamp_min(1.0)
    weights = (counts.sum() / counts).pow(power)
    weights /= weights.mean()
    return weights.clamp_max(maximum).float()


def _forward(
    model: NodeConditionedBondObject,
    batch: dict,
    device: torch.device,
    *,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = batch["pixel_values"].to(device)
    points = batch["points"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(pixels, points, pair_indices, pair_mask)
        display_logits = output["display_logits"]
        chemical_logits = output["chemical_logits"]
        if symmetric:
            reversed_points = points.reshape(points.size(0), -1, 2, 2).flip(2).reshape_as(points)
            reversed_output = model(
                pixels, reversed_points, pair_indices, pair_mask
            )
            display_logits = (
                display_logits + reversed_output["display_logits"]
            ) / 2.0
            chemical_logits = (
                chemical_logits + reversed_output["chemical_logits"]
            ) / 2.0
    return (
        display_logits,
        chemical_logits,
        batch["display_targets"].to(device)[pair_mask],
        batch["chemical_targets"].to(device)[pair_mask],
    )


@torch.no_grad()
def evaluate(
    model: NodeConditionedBondObject,
    records: list[dict],
    collator: PositiveBondTypeCollator,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    totals = {
        "bonds": 0,
        "display_correct": 0,
        "chemical_correct": 0,
        "object_correct": 0,
        "by_display": {
            name: {"count": 0, "display_correct": 0, "object_correct": 0}
            for name in BOND_TYPES
        },
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        display_logits, chemical_logits, display_targets, chemical_targets = _forward(
            model, batch, device, symmetric=True
        )
        display_correct = display_logits.argmax(-1) == display_targets
        chemical_correct = chemical_logits.argmax(-1) == chemical_targets
        object_correct = display_correct & chemical_correct
        totals["bonds"] += len(display_targets)
        totals["display_correct"] += int(display_correct.sum())
        totals["chemical_correct"] += int(chemical_correct.sum())
        totals["object_correct"] += int(object_correct.sum())
        for class_id, name in enumerate(BOND_TYPES):
            selected = display_targets == class_id
            bucket = totals["by_display"][name]
            bucket["count"] += int(selected.sum())
            bucket["display_correct"] += int((display_correct & selected).sum())
            bucket["object_correct"] += int((object_correct & selected).sum())
    bonds = max(totals["bonds"], 1)
    return {
        "samples": len(records),
        "bonds": totals["bonds"],
        "display_accuracy": totals["display_correct"] / bonds,
        "chemical_accuracy": totals["chemical_correct"] / bonds,
        "object_accuracy": totals["object_correct"] / bonds,
        "by_gt_display_type": {
            name: {
                "count": values["count"],
                "display_accuracy": values["display_correct"]
                / max(values["count"], 1),
                "object_accuracy": values["object_correct"]
                / max(values["count"], 1),
            }
            for name, values in totals["by_display"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=8974)
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--train-samples", type=int, default=8974)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
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
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = NodeConditionedBondConfig(**initial["config"])
    model = NodeConditionedBondObject(config).to(device)
    model.load_state_dict(initial["node_conditioned_bond"])
    model.edge.requires_grad_(False)
    model.direction.requires_grad_(False)

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
    display_weights = class_balancing_weights(
        train_records,
        "display_type",
        args.class_balance_power,
        args.maximum_class_weight,
    ).to(device)
    chemical_weights = class_balancing_weights(
        train_records,
        "chemical_type",
        args.class_balance_power,
        args.maximum_class_weight,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "architecture": "positive_bond_line_band_type_refiner",
                "trainable_parameters": sum(p.numel() for p in trainable),
                "display_class_weights": display_weights.tolist(),
                "chemical_class_weights": chemical_weights.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best = (-1.0, -1.0, -1.0)
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
            display_logits, chemical_logits, display_targets, chemical_targets = _forward(
                model, batch, device, symmetric=False
            )
            display_loss = F.cross_entropy(
                display_logits,
                display_targets,
                weight=display_weights.to(display_logits.dtype),
            )
            chemical_loss = F.cross_entropy(
                chemical_logits,
                chemical_targets,
                weight=chemical_weights.to(chemical_logits.dtype),
            )
            loss = display_loss + chemical_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                        "node_conditioned_bond": model.state_dict(),
                        "config": asdict(config),
                        "samples_seen": samples_seen,
                        "validation": metrics,
                    },
                    checkpoint,
                )
                score = (
                    metrics["object_accuracy"],
                    metrics["display_accuracy"],
                    metrics["chemical_accuracy"],
                )
                if score > best:
                    best = score
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
            "node_conditioned_bond": model.state_dict(),
            "config": asdict(config),
            "samples_seen": samples_seen,
            "validation": metrics,
        },
        output_dir / "last.pt",
    )
    final_score = (
        metrics["object_accuracy"],
        metrics["display_accuracy"],
        metrics["chemical_accuracy"],
    )
    if final_score > best or not (output_dir / "best.pt").exists():
        shutil.copy2(output_dir / "last.pt", output_dir / "best.pt")
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
