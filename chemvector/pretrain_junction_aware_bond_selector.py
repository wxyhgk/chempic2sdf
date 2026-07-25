#!/usr/bin/env python3
"""Train deployable Bond keep/drop before visual Junction grouping."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from chemvector.audit_bond_endpoint_junction import (
    analyze_endpoint_junctions,
    load_samples,
    summarize,
)
from chemvector.audit_bond_oracle_rescue import example_from_objects
from chemvector.bond_endpoint_junction import (
    BondEndpointJunctionResult,
    project_unique_junction_pairs,
)
from chemvector.bond_set_selector import (
    BondSetSelector,
    BondSetSelectorConfig,
    bond_set_selector_loss,
)
from chemvector.endpoint_junction_grouper import (
    VisualEndpointJunctionGrouper,
    VisualEndpointJunctionGrouperConfig,
    decode_endpoint_groups,
    endpoint_pair_features,
)
from chemvector.pretrain_visual_endpoint_junction_grouper import (
    endpoint_patches,
)


def candidate_objects(example: dict, indices: list[int] | None = None) -> list[dict]:
    indices = list(range(len(example["endpoints"]))) if indices is None else indices
    return [
        {
            "endpoints": example["endpoints"][index].float(),
            "score": float(example["scores"][index]),
            "display": int(example["display_ids"][index]),
            "chemical": int(example["chemical_ids"][index]),
            "direction": int(example["direction_ids"][index]),
        }
        for index in indices
    ]


def load_visual_grouper(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[VisualEndpointJunctionGrouper, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = VisualEndpointJunctionGrouperConfig(**payload["config"])
    model = VisualEndpointJunctionGrouper(config).to(device)
    model.load_state_dict(payload["visual_endpoint_junction_grouper"])
    model.eval()
    model.requires_grad_(False)
    return model, payload


@torch.inference_mode()
def endpoint_context(
    sample: dict,
    example: dict,
    grouper: VisualEndpointJunctionGrouper,
    grouper_payload: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    objects = candidate_objects(example)
    patches = endpoint_patches(
        sample, objects, patch_size=int(grouper_payload["patch_size"])
    )
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        embeddings = grouper.encode_patches(
            patches.to(device).float().div_(255.0)
        ).float()
    pairs, geometry = endpoint_pair_features(objects)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = grouper.score_embeddings(
            geometry.to(device),
            embeddings[pairs[:, 0]],
            embeddings[pairs[:, 1]],
        ).float()
    probabilities = logits.sigmoid()
    endpoint_count = len(objects) * 2
    maximum_support = torch.zeros(endpoint_count, device=device)
    close_count = torch.zeros(endpoint_count, device=device)
    minimum_distance = torch.full((endpoint_count,), 8.0, device=device)
    for side in (0, 1):
        indices = pairs[:, side].to(device)
        maximum_support.scatter_reduce_(
            0, indices, probabilities, reduce="amax", include_self=True
        )
        close_count.scatter_add_(
            0,
            indices,
            (probabilities >= float(grouper_payload["threshold"])).float(),
        )
        minimum_distance.scatter_reduce_(
            0,
            indices,
            geometry[:, 4].to(device),
            reduce="amin",
            include_self=True,
        )
    local = torch.stack(
        (
            maximum_support,
            close_count.clamp_max(8.0).div(8.0),
            minimum_distance.clamp_max(8.0).div(8.0),
        ),
        dim=-1,
    )
    return embeddings.cpu(), local.cpu()


def deployable_base_features(example: dict) -> torch.Tensor:
    features = example["features"].float()
    if features.size(-1) < 154:
        raise ValueError("Bond candidate cache lacks frozen visual features")
    # Positions 23:26 are GT-nearest endpoint distances and an Oracle self-loop flag.
    return torch.cat((features[:, :23], features[:, 26:]), dim=-1)


@torch.inference_mode()
def build_junction_cache(
    samples: list[dict],
    examples: list[dict],
    grouper: VisualEndpointJunctionGrouper,
    grouper_payload: dict,
    device: torch.device,
) -> list[dict]:
    output = []
    for index, (sample, example) in enumerate(zip(samples, examples), start=1):
        if sample["key"] != example["key"]:
            raise ValueError("Bond cache order differs from source images")
        embeddings, local = endpoint_context(
            sample, example, grouper, grouper_payload, device
        )
        count = len(example["endpoints"])
        endpoint_features = torch.cat((embeddings, local), dim=-1).reshape(count, -1)
        output.append(
            {
                "key": sample["key"],
                "features": torch.cat(
                    (deployable_base_features(example), endpoint_features), dim=-1
                ).half(),
                "mask": example["mask"].clone(),
                "endpoints": example["endpoints"].clone(),
                "scores": example["scores"].clone(),
                "display_ids": example["display_ids"].clone(),
                "chemical_ids": example["chemical_ids"].clone(),
                "direction_ids": example["direction_ids"].clone(),
                "keep_targets": example["keep_targets"].clone(),
                "count_target": int(example["count_target"]),
                "display_targets": example["display_targets"].clone(),
                "chemical_targets": example["chemical_targets"].clone(),
                "candidate_oracle": bool(example["candidate_oracle"]),
                "endpoint_embeddings": embeddings.half(),
            }
        )
        if index % 100 == 0 or index == len(samples):
            print(f"junction_cached={index}/{len(samples)}", flush=True)
    return output


def load_or_build_cache(path: Path, build) -> list[dict]:
    if path.exists():
        print(f"loading_cache={path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    examples = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, path)
    return examples


def select_cached_examples(
    examples: list[dict],
    samples: list[dict],
    count: int,
    *,
    name: str,
) -> list[dict]:
    if len(examples) < count:
        raise ValueError(
            f"{name} cache has {len(examples)} examples, fewer than requested {count}"
        )
    selected = examples[:count]
    if len(samples) != count:
        raise ValueError(f"{name} dataset loaded {len(samples)} samples, expected {count}")
    for sample, example in zip(samples, selected):
        if sample["key"] != example["key"]:
            raise ValueError(f"{name} cache order differs from source images")
    return selected


def collate_examples(examples: list[dict]) -> dict:
    tensor_names = (
        "features",
        "mask",
        "display_ids",
        "chemical_ids",
        "keep_targets",
        "display_targets",
        "chemical_targets",
    )
    return {
        **{name: torch.stack([item[name] for item in examples]) for name in tensor_names},
        "count_targets": torch.tensor([item["count_target"] for item in examples]),
        "examples": examples,
    }


def threshold_indices(logits: torch.Tensor, mask: torch.Tensor, threshold: float) -> list[list[int]]:
    probabilities = logits.sigmoid().cpu()
    decoded = []
    for batch_index in range(len(mask)):
        order = probabilities[batch_index].argsort(descending=True)
        decoded.append(
            [
                index
                for index in order.tolist()
                if bool(mask[batch_index, index])
                and float(probabilities[batch_index, index]) >= threshold
            ]
        )
    return decoded


@torch.inference_mode()
def group_selected(
    example: dict,
    selected: list[int],
    keep_logits: torch.Tensor,
    grouper: VisualEndpointJunctionGrouper,
    grouper_payload: dict,
    device: torch.device,
) -> BondEndpointJunctionResult:
    if not selected:
        return project_unique_junction_pairs(
            [],
            torch.empty((0,), dtype=torch.long),
            ranking_scores=torch.empty((0,), dtype=torch.float32),
        )
    objects = candidate_objects(example, selected)
    pairs, geometry = endpoint_pair_features(objects)
    source_endpoint_indices = torch.tensor(
        [endpoint for index in selected for endpoint in (index * 2, index * 2 + 1)],
        dtype=torch.long,
    )
    embeddings = example["endpoint_embeddings"][source_endpoint_indices].float().to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = grouper.score_embeddings(
            geometry.to(device),
            embeddings[pairs[:, 0]],
            embeddings[pairs[:, 1]],
        ).float().cpu()
    groups = decode_endpoint_groups(
        logits,
        pairs,
        len(objects) * 2,
        threshold=float(grouper_payload["threshold"]),
        linkage=str(grouper_payload["linkage"]),
    )
    return project_unique_junction_pairs(
        objects,
        groups,
        ranking_scores=keep_logits[selected].float().sigmoid().cpu(),
    )


def projected_result_example(
    key: str,
    result: BondEndpointJunctionResult,
) -> dict:
    grouped_objects = []
    for item in result.bond_objects:
        begin, end = item["junction_pair"]
        updated = dict(item)
        updated["endpoints"] = torch.stack(
            (result.points[begin], result.points[end])
        )
        grouped_objects.append(updated)
    return example_from_objects(key, grouped_objects)


@torch.inference_mode()
def evaluate(
    selector: BondSetSelector,
    examples: list[dict],
    samples: list[dict],
    grouper: VisualEndpointJunctionGrouper,
    grouper_payload: dict,
    device: torch.device,
    *,
    threshold: float,
    batch_size: int,
) -> dict:
    selector.eval()
    rows = []
    count_exact = candidate_oracle = topology_on_oracle = target_exact = 0
    loader = DataLoader(
        list(range(len(examples))), batch_size=batch_size, shuffle=False
    )
    for indices in loader:
        batch_examples = [examples[int(index)] for index in indices]
        batch = collate_examples(batch_examples)
        features = batch["features"].to(device).float()
        mask = batch["mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = selector(
                features,
                mask,
                batch["display_ids"].to(device),
                batch["chemical_ids"].to(device),
            )
        selected_batches = threshold_indices(output["keep_logits"], mask.cpu(), threshold)
        for local_index, selected in enumerate(selected_batches):
            global_index = int(indices[local_index])
            example = examples[global_index]
            sample = samples[global_index]
            result = group_selected(
                example,
                selected,
                output["keep_logits"][local_index].float().cpu(),
                grouper,
                grouper_payload,
                device,
            )
            metrics = analyze_endpoint_junctions(
                sample,
                projected_result_example(sample["key"], result),
                threshold=0.0,
                cluster_radius_pixels=0.25,
                match_radius_pixels=8.0,
                compile_sdf=False,
            )
            rows.append({"metrics": metrics})
            count_exact += int(
                len(result.bond_objects) == int(example["count_target"])
            )
            candidate_oracle += int(example["candidate_oracle"])
            topology_on_oracle += int(
                example["candidate_oracle"] and metrics["topology_exact"]
            )
            target = set(torch.where(example["keep_targets"])[0].tolist())
            target_exact += int(set(selected) == target)
    metrics = summarize(rows)
    sample_count = max(len(examples), 1)
    metrics.update(
        {
            "keep_threshold": threshold,
            "count_accuracy": count_exact / sample_count,
            "candidate_oracle_rate": candidate_oracle / sample_count,
            "selector_efficiency_on_candidate_oracle": topology_on_oracle
            / max(candidate_oracle, 1),
            "raw_keep_target_exact_rate": target_exact / sample_count,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grouper-checkpoint",
        default="outputs/checkpoints/chemvector-v0820-visual-endpoint-junction-grouper-capacity500/best.pt",
    )
    parser.add_argument(
        "--train-base-cache",
        default="outputs/checkpoints/chemvector-v0815-line-band-bond-set-selector-9k/train-candidates.pt",
    )
    parser.add_argument(
        "--validation-base-cache",
        default="outputs/checkpoints/chemvector-v0815-line-band-bond-set-selector-9k/validation-candidates.pt",
    )
    parser.add_argument("--train-junction-cache")
    parser.add_argument("--validation-junction-cache")
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
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-capacity500",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouper, grouper_payload = load_visual_grouper(args.grouper_checkpoint, device)
    train_samples = load_samples(args.train_shards, args.train_samples)
    validation_samples = load_samples(args.validation_shards, args.validation_samples)
    train_base = torch.load(
        args.train_base_cache, map_location="cpu", weights_only=False
    )[: args.train_samples]
    validation_base = torch.load(
        args.validation_base_cache, map_location="cpu", weights_only=False
    )[: args.validation_samples]
    train_cache_path = Path(args.train_junction_cache) if args.train_junction_cache else (
        output_dir / "train-junction-candidates.pt"
    )
    validation_cache_path = (
        Path(args.validation_junction_cache)
        if args.validation_junction_cache
        else output_dir / "validation-junction-candidates.pt"
    )
    train_examples = select_cached_examples(
        load_or_build_cache(
            train_cache_path,
            lambda: build_junction_cache(
                train_samples, train_base, grouper, grouper_payload, device
            ),
        ),
        train_samples,
        args.train_samples,
        name="train",
    )
    validation_examples = select_cached_examples(
        load_or_build_cache(
            validation_cache_path,
            lambda: build_junction_cache(
                validation_samples, validation_base, grouper, grouper_payload, device
            ),
        ),
        validation_samples,
        args.validation_samples,
        name="validation",
    )
    config = BondSetSelectorConfig(
        input_dim=int(train_examples[0]["features"].size(-1)),
        d_model=64,
        layers=1,
        heads=4,
        ffn_dim=128,
    )
    selector = BondSetSelector(config).to(device)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        generator=torch.Generator().manual_seed(args.seed),
    )
    for epoch in range(1, args.epochs + 1):
        selector.train()
        total = 0.0
        batches = 0
        for batch in loader:
            features = batch["features"].to(device).float()
            mask = batch["mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = selector(
                    features,
                    mask,
                    batch["display_ids"].to(device),
                    batch["chemical_ids"].to(device),
                )
                losses = bond_set_selector_loss(
                    output,
                    mask,
                    batch["keep_targets"].to(device),
                    batch["count_targets"].to(device),
                    batch["display_targets"].to(device),
                    batch["chemical_targets"].to(device),
                    count_weight=0.0,
                    type_weight=0.0,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(losses["loss"].detach())
            batches += 1
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={total / max(batches, 1):.6f}", flush=True)

    thresholds = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    train_candidates = []
    for threshold in thresholds:
        metrics = evaluate(
            selector,
            train_examples,
            train_samples,
            grouper,
            grouper_payload,
            device,
            threshold=threshold,
            batch_size=args.batch_size,
        )
        train_candidates.append(
            (metrics["topology_exact_rate"], metrics["count_accuracy"], threshold, metrics)
        )
        print(
            f"threshold={threshold:.2f} train_topology={metrics['topology_exact_rate']:.4f} "
            f"train_count={metrics['count_accuracy']:.4f}",
            flush=True,
        )
    _, _, selected_threshold, train_metrics = max(train_candidates)
    validation_metrics = evaluate(
        selector,
        validation_examples,
        validation_samples,
        grouper,
        grouper_payload,
        device,
        threshold=selected_threshold,
        batch_size=args.batch_size,
    )
    torch.save(
        {
            "bond_set_selector": selector.state_dict(),
            "config": asdict(config),
            "keep_threshold": selected_threshold,
            "grouper_checkpoint": args.grouper_checkpoint,
            "training_args": vars(args),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        },
        output_dir / "best.pt",
    )
    summary = {
        "schema_version": "junction-aware-bond-selector-capacity-0.1",
        "selected_threshold": selected_threshold,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "fixed_threshold_baseline_topology_exact": 0.6767,
        "oracle_selected_visual_grouper_ceiling": 0.7791,
        "go": validation_metrics["topology_exact_rate"] > 0.6767,
        "protocol": {
            **vars(args),
            "learned_count_used_for_decode": False,
            "gt_junction_coordinates_are_model_features": False,
            "oracle_nearest_endpoint_features_removed": True,
            "junction_grouping_runs_after_keep_selection": True,
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
