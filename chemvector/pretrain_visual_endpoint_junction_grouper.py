#!/usr/bin/env python3
"""Train visual same-Junction grouping over semantic Bond endpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.audit_bond_endpoint_junction import (
    analyze_endpoint_junctions,
    load_samples,
    summarize,
)
from chemvector.audit_bond_first_decomposition import _atom_centers
from chemvector.audit_bond_oracle_rescue import (
    example_from_objects,
    oracle_select_objects,
)
from chemvector.bond_endpoint_junction import junctions_from_endpoint_groups
from chemvector.endpoint_junction_grouper import (
    VisualEndpointJunctionGrouper,
    VisualEndpointJunctionGrouperConfig,
    decode_endpoint_groups,
    endpoint_pair_features,
    sample_aligned_endpoint_ink_patches,
)


def image_pixels(sample: dict) -> torch.Tensor:
    array = np.asarray(sample["image"].convert("RGB"), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


def endpoint_atom_labels(sample: dict, objects: list[dict]) -> torch.Tensor:
    if not objects:
        return torch.empty((0,), dtype=torch.long)
    endpoints = torch.stack([item["endpoints"].float() for item in objects]).reshape(-1, 2)
    return torch.cdist(endpoints, _atom_centers(sample)).argmin(-1)


def endpoint_patches(
    sample: dict,
    objects: list[dict],
    *,
    patch_size: int,
) -> torch.Tensor:
    if not objects:
        return torch.empty((0, 1, patch_size, patch_size), dtype=torch.uint8)
    endpoints = torch.stack([item["endpoints"].float() for item in objects])
    patches = sample_aligned_endpoint_ink_patches(
        image_pixels(sample), endpoints, patch_size=patch_size
    )
    return patches.mul(255.0).round().to(torch.uint8)


def training_pairs(
    samples: list[dict],
    examples: list[dict],
    *,
    patch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    geometry_rows = []
    first_patch_rows = []
    second_patch_rows = []
    target_rows = []
    for sample, example in zip(samples, examples):
        if sample["key"] != example["key"]:
            raise ValueError("Bond candidate cache order differs from dataset")
        objects, _ = oracle_select_objects(sample, example, threshold=None)
        labels = endpoint_atom_labels(sample, objects)
        pairs, geometry = endpoint_pair_features(objects)
        if not len(pairs):
            continue
        targets = labels[pairs[:, 0]].eq(labels[pairs[:, 1]])
        positives = torch.where(targets)[0]
        negatives = torch.where(~targets)[0]
        hard = negatives[geometry[negatives, 4] <= (32.0 / 768.0 * 8.0)]
        remaining = negatives[geometry[negatives, 4] > (32.0 / 768.0 * 8.0)]
        random_count = min(len(remaining), max(4 * len(positives), 64))
        random_negative = (
            remaining[torch.randperm(len(remaining), generator=generator)[:random_count]]
            if random_count
            else torch.empty((0,), dtype=torch.long)
        )
        selected = torch.cat((positives, hard, random_negative)).unique()
        patches = endpoint_patches(sample, objects, patch_size=patch_size)
        selected_pairs = pairs[selected]
        geometry_rows.append(geometry[selected])
        first_patch_rows.append(patches[selected_pairs[:, 0]])
        second_patch_rows.append(patches[selected_pairs[:, 1]])
        target_rows.append(targets[selected])
    return (
        torch.cat(geometry_rows),
        torch.cat(first_patch_rows),
        torch.cat(second_patch_rows),
        torch.cat(target_rows),
    )


def grouped_example(
    key: str,
    objects: list[dict],
    groups: torch.Tensor,
) -> dict:
    result = junctions_from_endpoint_groups(objects, groups)
    grouped_objects = []
    for item in result.bond_objects:
        begin, end = item["junction_pair"]
        updated = dict(item)
        updated["endpoints"] = torch.stack((result.points[begin], result.points[end]))
        grouped_objects.append(updated)
    return example_from_objects(key, grouped_objects)


@torch.no_grad()
def predict_grouping(
    model: VisualEndpointJunctionGrouper,
    samples: list[dict],
    examples: list[dict],
    device: torch.device,
    *,
    patch_size: int,
) -> list[dict]:
    model.eval()
    predictions = []
    for sample, example in zip(samples, examples):
        if sample["key"] != example["key"]:
            raise ValueError("Bond candidate cache order differs from dataset")
        objects, covered = oracle_select_objects(sample, example, threshold=None)
        pairs, geometry = endpoint_pair_features(objects)
        patches = endpoint_patches(sample, objects, patch_size=patch_size)
        if len(pairs):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(
                    geometry.to(device),
                    patches[pairs[:, 0]].to(device).float().div_(255.0),
                    patches[pairs[:, 1]].to(device).float().div_(255.0),
                ).float().cpu()
        else:
            logits = torch.empty((0,))
        predictions.append(
            {
                "sample": sample,
                "objects": objects,
                "covered": covered,
                "pairs": pairs,
                "logits": logits,
            }
        )
    return predictions


def evaluate_predictions(
    predictions: list[dict],
    *,
    threshold: float,
    linkage: str,
) -> dict:
    rows = []
    covered_count = 0
    for prediction in predictions:
        sample = prediction["sample"]
        objects = prediction["objects"]
        covered_count += int(prediction["covered"])
        groups = decode_endpoint_groups(
            prediction["logits"],
            prediction["pairs"],
            len(objects) * 2,
            threshold=threshold,
            linkage=linkage,
        )
        metrics = analyze_endpoint_junctions(
            sample,
            grouped_example(sample["key"], objects, groups),
            threshold=0.0,
            cluster_radius_pixels=0.25,
            match_radius_pixels=8.0,
            compile_sdf=False,
        )
        rows.append({"metrics": metrics})
    metrics = summarize(rows)
    metrics["candidate_oracle_rate"] = covered_count / max(len(rows), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-*.tar"],
    )
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-*.tar"],
    )
    parser.add_argument(
        "--train-cache",
        default="outputs/checkpoints/chemvector-v0813-bond-set-selector-9k/train-candidates.pt",
    )
    parser.add_argument(
        "--validation-cache",
        default="outputs/checkpoints/chemvector-v0813-bond-set-selector-9k/validation-candidates.pt",
    )
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v0820-visual-endpoint-junction-grouper-capacity500",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples = load_samples(args.train_shards, args.train_samples)
    validation_samples = load_samples(args.validation_shards, args.validation_samples)
    train_examples = torch.load(
        args.train_cache, map_location="cpu", weights_only=False
    )[: args.train_samples]
    validation_examples = torch.load(
        args.validation_cache, map_location="cpu", weights_only=False
    )[: args.validation_samples]
    if len(train_samples) != args.train_samples or len(validation_samples) != args.validation_samples:
        raise ValueError("Endpoint grouper loaded fewer samples than requested")

    geometry, first_patches, second_patches, targets = training_pairs(
        train_samples,
        train_examples,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    config = VisualEndpointJunctionGrouperConfig()
    model = VisualEndpointJunctionGrouper(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    positives = targets.sum().clamp_min(1)
    negatives = (~targets).sum().clamp_min(1)
    positive_weight = (negatives / positives).to(device).clamp_max(8.0)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(geometry), generator=generator)
        model.train()
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_geometry = geometry[indices].to(device)
            batch_first = first_patches[indices].to(device).float().div_(255.0)
            batch_second = second_patches[indices].to(device).float().div_(255.0)
            batch_targets = targets[indices].float().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(batch_geometry, batch_first, batch_second)
                loss = F.binary_cross_entropy_with_logits(
                    logits, batch_targets, pos_weight=positive_weight
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(indices)
        if epoch % 2 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={total / len(geometry):.6f}", flush=True)

    train_predictions = predict_grouping(
        model,
        train_samples,
        train_examples,
        device,
        patch_size=args.patch_size,
    )
    candidates = []
    for linkage in ("complete", "single"):
        for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            metrics = evaluate_predictions(
                train_predictions, threshold=threshold, linkage=linkage
            )
            candidates.append(
                (metrics["topology_exact_rate"], linkage, threshold, metrics)
            )
            print(
                f"linkage={linkage} threshold={threshold:.2f} "
                f"train_topology={metrics['topology_exact_rate']:.4f}",
                flush=True,
            )
    _, selected_linkage, selected_threshold, train_metrics = max(candidates)
    validation_predictions = predict_grouping(
        model,
        validation_samples,
        validation_examples,
        device,
        patch_size=args.patch_size,
    )
    validation_metrics = evaluate_predictions(
        validation_predictions,
        threshold=selected_threshold,
        linkage=selected_linkage,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "visual_endpoint_junction_grouper": model.state_dict(),
        "config": asdict(config),
        "patch_size": args.patch_size,
        "threshold": selected_threshold,
        "linkage": selected_linkage,
        "training_args": vars(args),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    torch.save(payload, output_dir / "best.pt")
    summary = {
        "schema_version": "visual-endpoint-junction-grouper-capacity-0.1",
        "training_pairs": len(geometry),
        "positive_pairs": int(targets.sum()),
        "selected_threshold": selected_threshold,
        "selected_linkage": selected_linkage,
        "patch_size": args.patch_size,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "fixed_radius_baseline_topology_exact": 0.753,
        "go": validation_metrics["topology_exact_rate"] > 0.753,
        "protocol": {
            **vars(args),
            "patches_are_aligned_to_the_outgoing_bond_ray": True,
            "oracle_used_only_to_select_true_bonds_for_grouping_ablation": True,
            "gt_junction_coordinates_are_model_features": False,
            "uses_mr_tadf_for_tuning": False,
            "uses_uspto_for_tuning": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
