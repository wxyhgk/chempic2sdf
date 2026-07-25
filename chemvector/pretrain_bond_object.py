#!/usr/bin/env python3
"""Train semantic Bond Object queries on ChemVision layers."""

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
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from chemvector.bond_object import (
    BOND_TYPE_TO_ID,
    DIRECTION_TO_ID,
    BondObjectConfig,
    BondObjectDetector,
    bond_object_loss,
    unordered_endpoint_cost,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.train import expand_shards


class BondObjectCollator:
    def __call__(self, samples: list[dict]) -> dict:
        maximum = max(len(sample["layers"]["bonds"]) for sample in samples)
        endpoints = torch.zeros((len(samples), maximum, 2, 2))
        boxes = torch.zeros((len(samples), maximum, 4))
        display = torch.zeros((len(samples), maximum), dtype=torch.long)
        chemical = torch.zeros((len(samples), maximum), dtype=torch.long)
        direction = torch.zeros((len(samples), maximum), dtype=torch.long)
        mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
        pixels = []
        for batch_index, sample in enumerate(samples):
            image = sample["image"]
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            pixels.append((torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5)
            width = sample["layers"]["width"]
            height = sample["layers"]["height"]
            for index, bond in enumerate(sample["layers"]["bonds"]):
                endpoints[batch_index, index] = torch.tensor((bond["begin"], bond["end"])) / torch.tensor((width, height))
                x0, y0, x1, y1 = bond["bbox"]
                boxes[batch_index, index] = torch.tensor(
                    ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)
                )
                display[batch_index, index] = BOND_TYPE_TO_ID[bond["display_type"]]
                chemical[batch_index, index] = BOND_TYPE_TO_ID[bond["chemical_type"]]
                direction[batch_index, index] = DIRECTION_TO_ID[bond["direction"]]
                mask[batch_index, index] = True
        return {
            "pixel_values": torch.stack(pixels),
            "endpoints": endpoints,
            "bbox": boxes,
            "display": display,
            "chemical": chemical,
            "direction": direction,
            "mask": mask,
            "keys": [sample["key"] for sample in samples],
        }


def stack_targets(batch: dict, device: torch.device) -> dict:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in ("endpoints", "bbox", "display", "chemical", "direction", "mask")
    }


@torch.no_grad()
def evaluate(model, detector, records, collator, device, batch_size: int) -> dict:
    detector.eval()
    totals = {
        "images": 0,
        "bonds": 0,
        "endpoint_pairs": 0,
        "display_correct": 0,
        "chemical_correct": 0,
        "known_exact": 0,
        "predicted_count_exact": 0,
        "predicted_exact": 0,
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = model.vision_encoder.forward_stroke_resolution(pixels)
            output = detector(features)
        for batch_index in range(len(pixels)):
            valid = targets["mask"][batch_index]
            expected = targets["endpoints"][batch_index, valid]
            count = len(expected)
            scores = output["presence_logits"][batch_index]
            known_queries = scores.topk(count).indices
            predicted_queries = torch.where(scores.sigmoid() >= 0.5)[0]

            def compare(queries):
                if len(queries) != count or not count:
                    return 0, 0, 0, False
                predicted = output["endpoints"][batch_index, queries]
                cost = unordered_endpoint_cost(predicted, expected)
                rows, columns = linear_sum_assignment(cost.float().cpu().numpy())
                matched = 0
                display_correct = 0
                chemical_correct = 0
                for row, column in zip(rows, columns):
                    forward = torch.linalg.vector_norm(
                        (predicted[row] - expected[column]) * 768.0, dim=-1
                    )
                    reverse = torch.linalg.vector_norm(
                        (predicted[row] - expected[column].flip(0)) * 768.0, dim=-1
                    )
                    distances = forward if float(forward.sum()) <= float(reverse.sum()) else reverse
                    if bool((distances <= 6.0).all()):
                        matched += 1
                        target_index = torch.where(valid)[0][column]
                        display_correct += int(
                            int(output["display_logits"][batch_index, queries[row]].argmax())
                            == int(targets["display"][batch_index, target_index])
                        )
                        chemical_correct += int(
                            int(output["chemical_logits"][batch_index, queries[row]].argmax())
                            == int(targets["chemical"][batch_index, target_index])
                        )
                return matched, display_correct, chemical_correct, matched == count

            matched, display_correct, chemical_correct, exact = compare(known_queries)
            predicted_matched, _, _, predicted_exact = compare(predicted_queries)
            totals["images"] += 1
            totals["bonds"] += count
            totals["endpoint_pairs"] += matched
            totals["display_correct"] += display_correct
            totals["chemical_correct"] += chemical_correct
            totals["known_exact"] += int(exact)
            totals["predicted_count_exact"] += int(len(predicted_queries) == count)
            totals["predicted_exact"] += int(predicted_exact and predicted_matched == count)
    bonds = max(totals["bonds"], 1)
    images = max(totals["images"], 1)
    return {
        "samples": totals["images"],
        "endpoint_pair_recall": totals["endpoint_pairs"] / bonds,
        "display_accuracy": totals["display_correct"] / max(totals["endpoint_pairs"], 1),
        "chemical_accuracy": totals["chemical_correct"] / max(totals["endpoint_pairs"], 1),
        "known_count_exact": totals["known_exact"] / images,
        "predicted_count_accuracy": totals["predicted_count_exact"] / images,
        "predicted_exact": totals["predicted_exact"] / images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=5000)
    args = parser.parse_args()
    device = torch.device("cuda")
    _, payload = load_grounded_graph_checkpoint(args.grounded_graph_checkpoint, device)
    model, _, _, _ = load_frozen_grounded_pipeline(Path(payload["grounded_selector_checkpoint"]), device)
    config = BondObjectConfig(input_channels=model.config.vision_dims[0])
    detector = BondObjectDetector(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = detector.load_state_dict(initial["bond_object"], strict=False)
        allowed = {
            name for name in detector.state_dict()
            if name.startswith("local_projection.")
            or name.startswith("endpoint_refinement.")
            or name.startswith("second_endpoint_refinement.")
        }
        if set(missing) - allowed or unexpected:
            raise ValueError(
                f"Bond Object initialization differs: missing={missing}, unexpected={unexpected}"
            )
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.learning_rate, weight_decay=0.0)
    collator = BondObjectCollator()

    def cache(shards, samples):
        dataset = ChemVectorIterableDataset(expand_shards(shards), sequence_version="0.4")
        return list(islice(iter(dataset), samples))

    train_records = cache(args.train_shards, args.train_cache_samples)
    validation_records = cache(args.validation_shards, args.validation_samples)
    if len(train_records) != args.train_cache_samples or len(validation_records) != args.validation_samples:
        raise ValueError("Bond Object cache contains fewer samples than requested")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    rng = random.Random(20260717)
    samples_seen = 0
    next_validation = args.validate_every
    best = (-1.0, -1.0, -1.0)
    running = 0.0
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
            batch = collator([train_records[index] for index in indices])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = model.vision_encoder.forward_stroke_resolution(pixels)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = detector(features)
                losses = bond_object_loss(output, targets)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running += float(losses["loss"].detach()) * len(indices)
            running_samples += len(indices)
            if samples_seen // args.log_every != (samples_seen - len(indices)) // args.log_every:
                print(f"samples={samples_seen} loss={running / running_samples:.6f}", flush=True)
                running = 0.0
                running_samples = 0
            if samples_seen >= next_validation:
                metrics = evaluate(model, detector, validation_records, collator, device, args.batch_size)
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                torch.save({"bond_object": detector.state_dict(), "config": asdict(config), "samples_seen": samples_seen, "validation": metrics, "grounded_graph_checkpoint": args.grounded_graph_checkpoint}, checkpoint)
                score = (metrics["known_count_exact"], metrics["endpoint_pair_recall"], metrics["display_accuracy"])
                if score > best:
                    best = score
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break
    metrics = evaluate(model, detector, validation_records, collator, device, args.batch_size)
    (output_dir / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"bond_object": detector.state_dict(), "config": asdict(config), "samples_seen": samples_seen, "validation": metrics, "grounded_graph_checkpoint": args.grounded_graph_checkpoint}, output_dir / "last.pt")
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
