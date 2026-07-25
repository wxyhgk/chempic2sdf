#!/usr/bin/env python3
"""Train an all-pairs semantic bond reader on known atom endpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from chemvector.bond_object import BOND_TYPE_TO_ID, DIRECTION_TO_ID, BOND_TYPES
from chemvector.data import ChemVectorIterableDataset
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
    enumerate_all_pairs,
    node_conditioned_bond_loss,
)
from chemvector.train import expand_shards


class KnownEndpointBondCollator:
    def __init__(
        self,
        image_size: int = 768,
        *,
        endpoint_jitter_sigma_pixels: float = 0.0,
        endpoint_jitter_max_pixels: float = 0.0,
        endpoint_jitter_probability: float = 0.0,
        seed: int = 20260717,
    ) -> None:
        self.image_size = image_size
        if endpoint_jitter_sigma_pixels < 0.0 or endpoint_jitter_max_pixels < 0.0:
            raise ValueError("Endpoint jitter distances must be nonnegative")
        if not 0.0 <= endpoint_jitter_probability <= 1.0:
            raise ValueError("Endpoint jitter probability must lie in [0, 1]")
        if endpoint_jitter_sigma_pixels and not endpoint_jitter_max_pixels:
            raise ValueError("Positive jitter sigma requires a positive maximum")
        self.endpoint_jitter_sigma_pixels = endpoint_jitter_sigma_pixels
        self.endpoint_jitter_max_pixels = endpoint_jitter_max_pixels
        self.endpoint_jitter_probability = endpoint_jitter_probability
        self.generator = torch.Generator().manual_seed(seed)

    def _jitter_points(
        self, points: torch.Tensor, point_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.endpoint_jitter_sigma_pixels or not self.endpoint_jitter_probability:
            return points
        noise = torch.randn(points.shape, generator=self.generator)
        noise *= self.endpoint_jitter_sigma_pixels
        lengths = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
        noise *= (self.endpoint_jitter_max_pixels / lengths.clamp_min(1e-8)).clamp_max(1.0)
        selected = (
            torch.rand(point_mask.shape, generator=self.generator)
            < self.endpoint_jitter_probability
        ) & point_mask
        normalized = noise / float(self.image_size)
        return (points + normalized * selected.unsqueeze(-1)).clamp(0.0, 1.0)

    def _pixels(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5

    def __call__(self, samples: list[dict]) -> dict:
        maximum_atoms = max(len(sample["layers"]["atoms"]) for sample in samples)
        points = torch.zeros((len(samples), maximum_atoms, 2))
        point_mask = torch.zeros((len(samples), maximum_atoms), dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            width = sample["layers"]["width"]
            height = sample["layers"]["height"]
            for atom in sample["layers"]["atoms"]:
                atom_id = atom["atom_id"]
                points[batch_index, atom_id] = torch.tensor(atom["center"]) / torch.tensor(
                    (width, height)
                )
                point_mask[batch_index, atom_id] = True
        points = self._jitter_points(points, point_mask)
        pair_indices, pair_mask = enumerate_all_pairs(point_mask)
        edge = torch.zeros(pair_mask.shape)
        display = torch.zeros(pair_mask.shape, dtype=torch.long)
        chemical = torch.zeros(pair_mask.shape, dtype=torch.long)
        direction = torch.zeros(pair_mask.shape, dtype=torch.long)
        for batch_index, sample in enumerate(samples):
            pair_lookup = {
                tuple(pair): offset
                for offset, pair in enumerate(
                    pair_indices[batch_index, pair_mask[batch_index]].tolist()
                )
            }
            for bond in sample["layers"]["bonds"]:
                atom_pair = tuple(sorted((bond["begin_atom"], bond["end_atom"])))
                offset = pair_lookup[atom_pair]
                edge[batch_index, offset] = 1.0
                display[batch_index, offset] = BOND_TYPE_TO_ID[bond["display_type"]]
                chemical[batch_index, offset] = BOND_TYPE_TO_ID[bond["chemical_type"]]
                direction[batch_index, offset] = DIRECTION_TO_ID[bond["direction"]]
        return {
            "pixel_values": torch.stack([self._pixels(sample["image"]) for sample in samples]),
            "points": points,
            "point_mask": point_mask,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "edge_targets": edge,
            "display_targets": display,
            "chemical_targets": chemical,
            "direction_targets": direction,
            "keys": [sample["key"] for sample in samples],
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def forward(model: NodeConditionedBondObject, batch: dict, device: torch.device):
    current = move_batch(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            current["pixel_values"],
            current["points"],
            current["pair_indices"],
            current["pair_mask"],
        )
        valid = current["pair_mask"]
        losses = node_conditioned_bond_loss(
            output,
            current["edge_targets"][valid],
            current["display_targets"][valid],
            current["chemical_targets"][valid],
            current["direction_targets"][valid],
        )
    return output, losses, current


@torch.no_grad()
def evaluate(
    model: NodeConditionedBondObject,
    records: list[dict],
    collator: KnownEndpointBondCollator,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> dict:
    model.eval()
    totals = {
        "samples": 0,
        "pairs": 0,
        "true_edges": 0,
        "predicted_edges": 0,
        "true_positive_edges": 0,
        "skeleton_exact": 0,
        "bond_object_exact": 0,
        "display_correct": 0,
        "chemical_correct": 0,
        "direction_correct": 0,
    }
    class_totals = {name: 0 for name in BOND_TYPES}
    class_edge_hits = {name: 0 for name in BOND_TYPES}
    class_object_hits = {name: 0 for name in BOND_TYPES}
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        output, _, current = forward(model, batch, device)
        offset = 0
        for batch_index in range(len(batch["keys"])):
            count = int(current["pair_mask"][batch_index].sum())
            edge_targets = current["edge_targets"][batch_index, :count].bool()
            display_targets = current["display_targets"][batch_index, :count]
            chemical_targets = current["chemical_targets"][batch_index, :count]
            direction_targets = current["direction_targets"][batch_index, :count]
            edge_predictions = output["edge_logits"][offset : offset + count].sigmoid() >= threshold
            display_predictions = output["display_logits"][offset : offset + count].argmax(-1)
            chemical_predictions = output["chemical_logits"][offset : offset + count].argmax(-1)
            direction_predictions = output["direction_logits"][offset : offset + count].argmax(-1)
            true_positive = edge_predictions & edge_targets
            display_correct = display_predictions == display_targets
            chemical_correct = chemical_predictions == chemical_targets
            direction_correct = direction_predictions == direction_targets
            object_correct = (
                true_positive & display_correct & chemical_correct & direction_correct
            )
            totals["samples"] += 1
            totals["pairs"] += count
            totals["true_edges"] += int(edge_targets.sum())
            totals["predicted_edges"] += int(edge_predictions.sum())
            totals["true_positive_edges"] += int(true_positive.sum())
            totals["skeleton_exact"] += int(torch.equal(edge_predictions, edge_targets))
            positive_count = int(edge_targets.sum())
            types_exact = bool(
                (display_correct[edge_targets] & chemical_correct[edge_targets] & direction_correct[edge_targets]).all()
            )
            totals["bond_object_exact"] += int(
                torch.equal(edge_predictions, edge_targets) and types_exact
            )
            totals["display_correct"] += int(display_correct[edge_targets].sum())
            totals["chemical_correct"] += int(chemical_correct[edge_targets].sum())
            totals["direction_correct"] += int(direction_correct[edge_targets].sum())
            for class_id, name in enumerate(BOND_TYPES):
                selected = edge_targets & (display_targets == class_id)
                class_totals[name] += int(selected.sum())
                class_edge_hits[name] += int((true_positive & selected).sum())
                class_object_hits[name] += int((object_correct & selected).sum())
            if not positive_count:
                raise ValueError("A molecule image contains no bonds")
            offset += count
    precision = totals["true_positive_edges"] / max(totals["predicted_edges"], 1)
    recall = totals["true_positive_edges"] / max(totals["true_edges"], 1)
    result = {
        "samples": totals["samples"],
        "pairs": totals["pairs"],
        "true_edges": totals["true_edges"],
        "threshold": threshold,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "skeleton_exact": totals["skeleton_exact"] / max(totals["samples"], 1),
        "bond_object_exact": totals["bond_object_exact"] / max(totals["samples"], 1),
        "display_accuracy": totals["display_correct"] / max(totals["true_edges"], 1),
        "chemical_accuracy": totals["chemical_correct"] / max(totals["true_edges"], 1),
        "direction_accuracy": totals["direction_correct"] / max(totals["true_edges"], 1),
        "by_display_type": {},
    }
    for name in BOND_TYPES:
        result["by_display_type"][name] = {
            "count": class_totals[name],
            "edge_recall": class_edge_hits[name] / max(class_totals[name], 1),
            "object_recall": class_object_hits[name] / max(class_totals[name], 1),
        }
    return result


def select_threshold(model, records, collator, device, batch_size) -> dict:
    candidates = [evaluate(model, records, collator, device, batch_size, value / 20) for value in range(2, 19)]
    return max(
        candidates,
        key=lambda item: (
            item["skeleton_exact"],
            item["bond_object_exact"],
            item["edge_f1"],
        ),
    )


def save(path: Path, model, optimizer, config, args, samples_seen, metrics) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "node_conditioned_bond": model.state_dict(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validate-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--endpoint-jitter-sigma-pixels", type=float, default=0.0)
    parser.add_argument("--endpoint-jitter-max-pixels", type=float, default=0.0)
    parser.add_argument("--endpoint-jitter-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    init_payload = (
        torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.init_checkpoint
        else None
    )
    config = NodeConditionedBondConfig(
        **(init_payload["config"] if init_payload is not None else {})
    )
    model = NodeConditionedBondObject(config).to(device)
    if init_payload is not None:
        model.load_state_dict(init_payload["node_conditioned_bond"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    train_collator = KnownEndpointBondCollator(
        endpoint_jitter_sigma_pixels=args.endpoint_jitter_sigma_pixels,
        endpoint_jitter_max_pixels=args.endpoint_jitter_max_pixels,
        endpoint_jitter_probability=args.endpoint_jitter_probability,
        seed=args.seed,
    )
    validation_collator = KnownEndpointBondCollator()

    def new_jitter_validation_collator() -> KnownEndpointBondCollator:
        return KnownEndpointBondCollator(
            endpoint_jitter_sigma_pixels=args.endpoint_jitter_sigma_pixels,
            endpoint_jitter_max_pixels=args.endpoint_jitter_max_pixels,
            endpoint_jitter_probability=args.endpoint_jitter_probability,
            seed=args.seed + 1,
        )

    def cache(shards, count):
        dataset = ChemVectorIterableDataset(
            expand_shards(shards), sequence_version="0.4"
        )
        return list(islice(iter(dataset), count))

    train_records = cache(args.train_shards, args.train_cache_samples)
    validation_records = cache(args.validation_shards, args.validation_samples)
    if len(train_records) != args.train_cache_samples or len(validation_records) != args.validation_samples:
        raise ValueError("Known-endpoint cache contains fewer samples than requested")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best = (-1.0, -1.0, -1.0)
    running_loss = 0.0
    running_samples = 0
    for _ in range(args.epochs):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                break
            indices = order[start : start + min(args.batch_size, remaining)]
            batch = train_collator([train_records[index] for index in indices])
            model.train()
            _, losses, _ = forward(model, batch, device)
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
                metrics = select_threshold(
                    model, validation_records, validation_collator, device, args.batch_size
                )
                if args.endpoint_jitter_sigma_pixels and args.endpoint_jitter_probability:
                    metrics["endpoint_jitter_validation"] = evaluate(
                        model,
                        validation_records,
                        new_jitter_validation_collator(),
                        device,
                        args.batch_size,
                        metrics["threshold"],
                    )
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save(checkpoint, model, optimizer, config, args, samples_seen, metrics)
                primary = metrics.get("endpoint_jitter_validation", metrics)
                score = (
                    primary["skeleton_exact"],
                    metrics["skeleton_exact"],
                    primary["edge_f1"],
                )
                if score > best:
                    best = score
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if samples_seen >= args.train_samples:
            break
    metrics = select_threshold(
        model, validation_records, validation_collator, device, args.batch_size
    )
    if args.endpoint_jitter_sigma_pixels and args.endpoint_jitter_probability:
        metrics["endpoint_jitter_validation"] = evaluate(
            model,
            validation_records,
            new_jitter_validation_collator(),
            device,
            args.batch_size,
            metrics["threshold"],
        )
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save(output_dir / "last.pt", model, optimizer, config, args, samples_seen, metrics)
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
