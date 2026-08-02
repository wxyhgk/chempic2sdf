#!/usr/bin/env python3
"""Train a learned same-Junction relation over semantic Bond endpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from chemvector.audit_bond_endpoint_junction import analyze_endpoint_junctions, load_samples, summarize
from chemvector.audit_bond_first_decomposition import _atom_centers
from chemvector.audit_bond_oracle_rescue import (
    example_from_objects,
    oracle_select_objects,
)
from chemvector.bond_endpoint_junction import junctions_from_endpoint_groups
from chemvector.endpoint_junction_grouper import (
    EndpointJunctionGrouper,
    EndpointJunctionGrouperConfig,
    decode_endpoint_groups,
    endpoint_pair_features,
)


def endpoint_atom_labels(sample: dict, objects: list[dict]) -> torch.Tensor:
    if not objects:
        return torch.empty((0,), dtype=torch.long)
    endpoints = torch.stack([item["endpoints"].float() for item in objects]).reshape(-1, 2)
    return torch.cdist(endpoints, _atom_centers(sample)).argmin(-1)


def training_pairs(
    samples: list[dict],
    examples: list[dict],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    feature_rows = []
    target_rows = []
    for sample, example in zip(samples, examples):
        objects, _ = oracle_select_objects(sample, example, threshold=None)
        labels = endpoint_atom_labels(sample, objects)
        pairs, features = endpoint_pair_features(objects)
        if not len(pairs):
            continue
        targets = labels[pairs[:, 0]].eq(labels[pairs[:, 1]])
        positives = torch.where(targets)[0]
        negatives = torch.where(~targets)[0]
        hard = negatives[features[negatives, 4] <= (32.0 / 768.0 * 8.0)]
        remaining = negatives[features[negatives, 4] > (32.0 / 768.0 * 8.0)]
        random_count = min(len(remaining), max(4 * len(positives), 64))
        random_negative = (
            remaining[torch.randperm(len(remaining), generator=generator)[:random_count]]
            if random_count
            else torch.empty((0,), dtype=torch.long)
        )
        selected = torch.cat((positives, hard, random_negative)).unique()
        feature_rows.append(features[selected])
        target_rows.append(targets[selected])
    return torch.cat(feature_rows), torch.cat(target_rows)


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
def evaluate(
    model: EndpointJunctionGrouper,
    samples: list[dict],
    examples: list[dict],
    device: torch.device,
    *,
    threshold: float,
    linkage: str = "complete",
) -> dict:
    model.eval()
    rows = []
    covered_count = 0
    for sample, example in zip(samples, examples):
        objects, covered = oracle_select_objects(sample, example, threshold=None)
        covered_count += int(covered)
        pairs, features = endpoint_pair_features(objects)
        logits = model(features.to(device)).cpu() if len(features) else torch.empty((0,))
        groups = decode_endpoint_groups(
            logits,
            pairs,
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v0819-endpoint-junction-grouper-capacity500",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_samples = load_samples(args.train_shards, args.train_samples)
    validation_samples = load_samples(args.validation_shards, args.validation_samples)
    train_examples = torch.load(args.train_cache, map_location="cpu", weights_only=False)[: args.train_samples]
    validation_examples = torch.load(args.validation_cache, map_location="cpu", weights_only=False)[: args.validation_samples]
    features, targets = training_pairs(
        train_samples, train_examples, seed=args.seed
    )
    config = EndpointJunctionGrouperConfig()
    model = EndpointJunctionGrouper(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    positives = targets.sum().clamp_min(1)
    negatives = (~targets).sum().clamp_min(1)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(features), generator=generator)
        model.train()
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            logits = model(features[indices].to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets[indices].float().to(device),
                pos_weight=(negatives / positives).to(device).clamp_max(32.0),
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(indices)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={total / len(features):.6f}", flush=True)

    candidates = []
    for linkage in ("complete", "single"):
        for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            metrics = evaluate(
                model,
                train_samples,
                train_examples,
                device,
                threshold=threshold,
                linkage=linkage,
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
    validation_metrics = evaluate(
        model,
        validation_samples,
        validation_examples,
        device,
        threshold=selected_threshold,
        linkage=selected_linkage,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "endpoint_junction_grouper": model.state_dict(),
            "config": asdict(config),
            "threshold": selected_threshold,
            "linkage": selected_linkage,
            "training_args": vars(args),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        },
        output_dir / "best.pt",
    )
    summary = {
        "schema_version": "endpoint-junction-grouper-capacity-0.1",
        "training_pairs": len(features),
        "positive_pairs": int(targets.sum()),
        "selected_threshold": selected_threshold,
        "selected_linkage": selected_linkage,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "go": validation_metrics["topology_exact_rate"] > 0.753,
        "protocol": {
            **vars(args),
            "oracle_used_only_to_select_true_bonds_for_grouping_ablation": True,
            "gt_junction_coordinates_are_model_features": False,
            "uses_mr_tadf_for_tuning": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
