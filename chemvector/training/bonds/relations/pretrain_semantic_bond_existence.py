#!/usr/bin/env python3
"""Train general Bond relation existence with length-matched negatives."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import combinations
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPE_TO_ID, BOND_TYPES
from chemvector.data import ChemVectorIterableDataset
from chemvector.semantic_bond_relation import (
    SemanticBondExistence,
    SemanticBondRelationConfig,
)
from chemvector.train import expand_shards


def _crosses(first, second, third, fourth) -> bool:
    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    values = (
        orientation(first, second, third),
        orientation(first, second, fourth),
        orientation(third, fourth, first),
        orientation(third, fourth, second),
    )
    return values[0] * values[1] < 0.0 and values[2] * values[3] < 0.0


def relation_candidates(
    centers: dict[int, tuple[float, float]],
    positive_pairs: list[tuple[int, int]],
    *,
    negative_ratio: float,
    rng: random.Random,
    hard_negative_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> tuple[list[tuple[int, int]], list[int], list[str]]:
    positives = [tuple(sorted(pair)) for pair in positive_pairs]
    positive_set = set(positives)
    all_pairs = [tuple(pair) for pair in combinations(sorted(centers), 2)]
    negatives = [pair for pair in all_pairs if pair not in positive_set]

    def length(pair):
        first, second = centers[pair[0]], centers[pair[1]]
        return math.hypot(second[0] - first[0], second[1] - first[1])

    target_negatives = min(
        len(negatives), max(1, round(len(positives) * negative_ratio))
    )
    remaining = set(negatives)
    selected: list[tuple[int, int]] = []
    kinds: list[str] = []
    for pair in hard_negative_pairs:
        pair = tuple(sorted(pair))
        if pair not in remaining or len(selected) >= target_negatives:
            continue
        remaining.remove(pair)
        selected.append(pair)
        kinds.append("model_hard")
    for positive in sorted(positives, key=length):
        if not remaining or len(selected) >= target_negatives:
            break
        positive_length = length(positive)
        closest = min(
            remaining,
            key=lambda pair: (abs(length(pair) - positive_length), pair),
        )
        remaining.remove(closest)
        selected.append(closest)
        kinds.append("length_matched")

    crossing = []
    for pair in remaining:
        first, second = centers[pair[0]], centers[pair[1]]
        if any(
            not set(pair).intersection(bond)
            and _crosses(first, second, centers[bond[0]], centers[bond[1]])
            for bond in positives
        ):
            crossing.append(pair)
    rng.shuffle(crossing)
    for pair in crossing:
        if len(selected) >= target_negatives:
            break
        remaining.remove(pair)
        selected.append(pair)
        kinds.append("crossing")

    residual = list(remaining)
    rng.shuffle(residual)
    for pair in residual:
        if len(selected) >= target_negatives:
            break
        selected.append(pair)
        kinds.append("random")

    candidates = positives + selected
    labels = [1] * len(positives) + [0] * len(selected)
    candidate_kinds = ["positive"] * len(positives) + kinds
    order = list(range(len(candidates)))
    rng.shuffle(order)
    return (
        [candidates[index] for index in order],
        [labels[index] for index in order],
        [candidate_kinds[index] for index in order],
    )


class BondExistenceCollator:
    def __init__(
        self,
        image_size: int = 768,
        *,
        negative_ratio: float = 1.5,
        degradation_probability: float = 0.0,
        endpoint_jitter_sigma_pixels: float = 1.5,
        endpoint_jitter_max_pixels: float = 5.0,
        hard_negatives: dict[str, list[tuple[int, int]]] | None = None,
        predicted_junctions: dict[str, dict] | None = None,
        seed: int = 20260723,
    ) -> None:
        if negative_ratio <= 0.0:
            raise ValueError("negative ratio must be positive")
        self.image_size = image_size
        self.negative_ratio = negative_ratio
        self.degradation_probability = degradation_probability
        self.endpoint_jitter_sigma_pixels = endpoint_jitter_sigma_pixels
        self.endpoint_jitter_max_pixels = endpoint_jitter_max_pixels
        self.hard_negatives = hard_negatives or {}
        self.predicted_junctions = predicted_junctions or {}
        self.rng = random.Random(seed)

    def _pixels(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        if self.rng.random() < self.degradation_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.1))
            reduced = (round(self.image_size * 0.72), round(self.image_size * 0.72))
            image = image.resize(reduced, Image.Resampling.BILINEAR).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        values = np.asarray(image, dtype=np.float32) / 255.0
        return (torch.from_numpy(values).permute(2, 0, 1) - 0.5) / 0.5

    def _jitter(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.endpoint_jitter_sigma_pixels <= 0.0:
            return point
        angle = self.rng.random() * math.tau
        radius = min(
            abs(self.rng.gauss(0.0, self.endpoint_jitter_sigma_pixels)),
            self.endpoint_jitter_max_pixels,
        )
        return (
            min(max(point[0] + math.cos(angle) * radius, 0.0), self.image_size),
            min(max(point[1] + math.sin(angle) * radius, 0.0), self.image_size),
        )

    def __call__(self, samples: list[dict]) -> dict:
        prepared = []
        maximum_atoms = maximum_pairs = 0
        for sample in samples:
            predicted = self.predicted_junctions.get(sample["key"])
            if predicted is None:
                width = float(sample["layers"]["width"])
                height = float(sample["layers"]["height"])
                centers = {
                    atom["atom_id"]: self._jitter(
                        (
                            float(atom["center"][0]) * self.image_size / width,
                            float(atom["center"][1]) * self.image_size / height,
                        )
                    )
                    for atom in sample["layers"]["atoms"]
                }
                bond_lookup = {
                    tuple(sorted((bond["begin_atom"], bond["end_atom"]))): bond
                    for bond in sample["layers"]["bonds"]
                }
                manifest_hard_negatives = ()
            else:
                centers = {
                    index: self._jitter(
                        (
                            float(point[0]) * self.image_size,
                            float(point[1]) * self.image_size,
                        )
                    )
                    for index, point in enumerate(predicted["points"])
                }
                bond_lookup = {
                    tuple(bond["pair"]): {
                        "display_type": bond["display_type"],
                        "chemical_type": bond["chemical_type"],
                    }
                    for bond in predicted["positive_bonds"]
                }
                manifest_hard_negatives = [
                    tuple(pair) for pair in predicted["hard_negative_pairs"]
                ]
            pairs, labels, kinds = relation_candidates(
                centers,
                list(bond_lookup),
                negative_ratio=self.negative_ratio,
                rng=self.rng,
                hard_negative_pairs=(
                    manifest_hard_negatives
                    or self.hard_negatives.get(sample["key"], ())
                ),
            )
            prepared.append((sample, centers, bond_lookup, pairs, labels, kinds))
            maximum_atoms = max(maximum_atoms, max(centers) + 1)
            maximum_pairs = max(maximum_pairs, len(pairs))

        points = torch.zeros((len(samples), maximum_atoms, 2))
        pair_indices = torch.zeros(
            (len(samples), maximum_pairs, 2), dtype=torch.long
        )
        pair_mask = torch.zeros((len(samples), maximum_pairs), dtype=torch.bool)
        edge_targets = torch.zeros((len(samples), maximum_pairs))
        display_targets = torch.zeros((len(samples), maximum_pairs), dtype=torch.long)
        chemical_targets = torch.zeros_like(display_targets)
        lengths = torch.zeros((len(samples), maximum_pairs))
        kinds = []
        for batch_index, (sample, centers, bonds, pairs, labels, current_kinds) in enumerate(prepared):
            for atom_id, center in centers.items():
                points[batch_index, atom_id] = torch.tensor(center) / self.image_size
            for offset, (pair, label, kind) in enumerate(zip(pairs, labels, current_kinds)):
                pair_indices[batch_index, offset] = torch.tensor(pair)
                pair_mask[batch_index, offset] = True
                edge_targets[batch_index, offset] = label
                first, second = centers[pair[0]], centers[pair[1]]
                lengths[batch_index, offset] = math.hypot(
                    second[0] - first[0], second[1] - first[1]
                )
                if label:
                    bond = bonds[pair]
                    display_targets[batch_index, offset] = BOND_TYPE_TO_ID[
                        bond["display_type"]
                    ]
                    chemical_targets[batch_index, offset] = BOND_TYPE_TO_ID[
                        bond["chemical_type"]
                    ]
                kinds.append(kind)
        return {
            "pixel_values": torch.stack(
                [self._pixels(item[0]["image"]) for item in prepared]
            ),
            "points": points,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "edge_targets": edge_targets[pair_mask],
            "display_targets": display_targets[pair_mask],
            "chemical_targets": chemical_targets[pair_mask],
            "lengths_pixels": lengths[pair_mask],
            "pair_counts": pair_mask.sum(-1),
            "kinds": kinds,
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def forward(model, batch, device, type_loss_weight: float):
    current = move_batch(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            current["pixel_values"],
            current["points"],
            current["pair_indices"],
            current["pair_mask"],
        )
        targets = current["edge_targets"]
        positive_count = targets.sum().clamp_min(1.0)
        negative_count = (1.0 - targets).sum().clamp_min(1.0)
        edge_loss = F.binary_cross_entropy_with_logits(
            output["edge_logits"],
            targets,
            pos_weight=(negative_count / positive_count).clamp_max(4.0),
        )
        positives = targets.bool()
        display_loss = F.cross_entropy(
            output["display_logits"][positives],
            current["display_targets"][positives],
        )
        chemical_loss = F.cross_entropy(
            output["chemical_logits"][positives],
            current["chemical_targets"][positives],
        )
        loss = edge_loss + type_loss_weight * (display_loss + chemical_loss)
    return output, {
        "loss": loss,
        "edge_loss": edge_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
    }, current


@torch.no_grad()
def evaluate(model, records, collator, device, batch_size, threshold=0.5):
    model.eval()
    logits = []
    targets = []
    lengths = []
    sample_exact = 0
    samples = 0
    kind_totals = Counter()
    kind_false_positives = Counter()
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        output, _, current = forward(model, batch, device, 0.0)
        prediction = output["edge_logits"].sigmoid() >= threshold
        offset = 0
        for count in current["pair_counts"].tolist():
            sample_exact += int(
                torch.equal(
                    prediction[offset : offset + count],
                    current["edge_targets"][offset : offset + count].bool(),
                )
            )
            samples += 1
            offset += count
        for kind, predicted, target in zip(
            batch["kinds"], prediction.tolist(), current["edge_targets"].tolist()
        ):
            kind_totals[kind] += 1
            if not target and predicted:
                kind_false_positives[kind] += 1
        logits.append(output["edge_logits"].float().cpu())
        targets.append(current["edge_targets"].cpu())
        lengths.append(current["lengths_pixels"].cpu())
    logits = torch.cat(logits)
    targets = torch.cat(targets).bool()
    lengths = torch.cat(lengths)
    prediction = logits.sigmoid() >= threshold
    true_positive = int((prediction & targets).sum())
    precision = true_positive / max(int(prediction.sum()), 1)
    recall = true_positive / max(int(targets.sum()), 1)
    positive_lengths = lengths[targets]
    quantiles = torch.quantile(positive_lengths, torch.tensor((0.25, 0.5, 0.75)))
    boundaries = torch.cat(
        (torch.tensor([-float("inf")]), quantiles, torch.tensor([float("inf")]))
    )
    recall_by_length = {}
    for index in range(4):
        selected = targets & (lengths > boundaries[index]) & (lengths <= boundaries[index + 1])
        recall_by_length[f"q{index + 1}"] = float(prediction[selected].float().mean())
    return {
        "samples": samples,
        "pairs": len(targets),
        "positives": int(targets.sum()),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "sample_set_exact": sample_exact / max(samples, 1),
        "positive_length_quantiles_pixels": quantiles.tolist(),
        "recall_by_positive_length_quartile": recall_by_length,
        "candidate_kind_counts": dict(kind_totals),
        "false_positive_by_kind": dict(kind_false_positives),
        "positive_length_mean_pixels": float(positive_lengths.mean()),
        "negative_length_mean_pixels": float(lengths[~targets].mean()),
    }


def save(path, model, optimizer, config, args, samples_seen, metrics):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "semantic_bond_existence": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "training_args": vars(args),
            "samples_seen": samples_seen,
            "validation": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--relation-checkpoint", required=True)
    parser.add_argument("--initial-existence-checkpoint")
    parser.add_argument("--hard-negative-manifest")
    parser.add_argument("--predicted-junction-train-manifest")
    parser.add_argument("--predicted-junction-validation-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--edge-learning-rate", type=float, default=3e-4)
    parser.add_argument("--relation-learning-rate", type=float, default=2e-5)
    parser.add_argument("--type-loss-weight", type=float, default=0.1)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--degradation-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    def cache(shards, count, allowed_keys=None):
        dataset = ChemVectorIterableDataset(
            expand_shards(shards), sequence_version="0.4"
        )
        records = []
        for record in dataset:
            if allowed_keys is not None and record["key"] not in allowed_keys:
                continue
            records.append(record)
            if len(records) >= count:
                break
        if len(records) < count:
            raise ValueError(
                f"Only {len(records)} eligible records were found; expected {count}"
            )
        return records

    def load_predicted_manifest(path):
        if not path:
            return {}
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(item["key"]): item for item in payload["samples"]}

    predicted_train = load_predicted_manifest(
        args.predicted_junction_train_manifest
    )
    predicted_validation = load_predicted_manifest(
        args.predicted_junction_validation_manifest
    )
    train_records = cache(
        args.train_shards,
        args.train_cache_samples,
        set(predicted_train) if predicted_train else None,
    )
    validation_records = cache(
        args.validation_shards,
        args.validation_samples,
        set(predicted_validation) if predicted_validation else None,
    )
    relation_payload = torch.load(
        args.relation_checkpoint, map_location="cpu", weights_only=False
    )
    config = SemanticBondRelationConfig(**relation_payload["config"])
    model = SemanticBondExistence(config).to(device)
    model.relation.load_state_dict(relation_payload["semantic_bond_relation"])
    model.initialize_edge_visual_from_relation()
    if args.initial_existence_checkpoint:
        initial_payload = torch.load(
            args.initial_existence_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial_payload["semantic_bond_existence"])
    hard_negatives = {}
    if args.hard_negative_manifest:
        hard_payload = json.loads(
            Path(args.hard_negative_manifest).read_text(encoding="utf-8")
        )
        hard_negatives = {
            str(item["key"]): [tuple(pair) for pair in item["pairs"]]
            for item in hard_payload["samples"]
        }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.edge_visual.parameters())
                + list(model.edge.parameters()),
                "lr": args.edge_learning_rate,
            },
            {
                "params": model.relation.parameters(),
                "lr": args.relation_learning_rate,
            },
        ]
    )
    train_collator = BondExistenceCollator(
        negative_ratio=args.negative_ratio,
        degradation_probability=args.degradation_probability,
        hard_negatives=hard_negatives,
        predicted_junctions=predicted_train,
        seed=args.seed,
    )
    clean_collator = BondExistenceCollator(
        negative_ratio=(1e9 if predicted_validation else args.negative_ratio),
        endpoint_jitter_sigma_pixels=0.0,
        predicted_junctions=predicted_validation,
        seed=args.seed + 1,
    )
    degraded_collator = BondExistenceCollator(
        negative_ratio=(1e9 if predicted_validation else args.negative_ratio),
        degradation_probability=1.0,
        predicted_junctions=predicted_validation,
        seed=args.seed + 2,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best = (-1.0, -1.0, -1.0)
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
            _, losses, _ = forward(model, batch, device, args.type_loss_weight)
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
                metrics = {
                    "clean": evaluate(
                        model,
                        validation_records,
                        clean_collator,
                        device,
                        args.batch_size,
                    ),
                    "degraded": evaluate(
                        model,
                        validation_records,
                        degraded_collator,
                        device,
                        args.batch_size,
                    ),
                }
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save(checkpoint, model, optimizer, config, args, samples_seen, metrics)
                score = (
                    metrics["degraded"]["sample_set_exact"],
                    metrics["degraded"]["f1"],
                    metrics["clean"]["sample_set_exact"],
                )
                if score > best:
                    best = score
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
