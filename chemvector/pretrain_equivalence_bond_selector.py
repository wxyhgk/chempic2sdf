#!/usr/bin/env python3
"""Fine-tune Bond selection with GT atom-pair equivalence-group MIL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from chemvector.audit_bond_endpoint_junction import (
    analyze_endpoint_junctions,
    load_samples,
    summarize,
)
from chemvector.bond_set_selector import BondSetSelector, BondSetSelectorConfig
from chemvector.equivalence_bond_selector import (
    attach_equivalence_labels,
    build_equivalence_labels,
    equivalence_bond_selector_loss,
    equivalence_selection_metrics,
)
from chemvector.pretrain_junction_aware_bond_selector import (
    evaluate as evaluate_topology,
    group_selected,
    load_visual_grouper,
    projected_result_example,
    select_cached_examples,
)


DISPLAY_CLASS_WEIGHTS = torch.tensor((0.52, 1.07, 4.00, 0.94))
CHEMICAL_CLASS_WEIGHTS = torch.tensor((0.66, 1.87, 4.00, 0.76))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_overlay(base_examples: list[dict]) -> list[dict]:
    return [
        {"key": example["key"], **build_equivalence_labels(example)}
        for example in base_examples
    ]


def load_or_build_overlay(path: Path, base_examples: list[dict]) -> list[dict]:
    if path.exists():
        overlay = torch.load(path, map_location="cpu", weights_only=False)
    else:
        overlay = build_overlay(base_examples)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(overlay, path)
    if len(overlay) < len(base_examples):
        raise ValueError(f"Equivalence overlay {path} contains too few samples")
    selected = overlay[: len(base_examples)]
    for base, labels in zip(base_examples, selected):
        if base["key"] != labels["key"]:
            raise ValueError(f"Equivalence overlay {path} has a different sample order")
    return selected


def attach_overlay(junction_examples: list[dict], overlays: list[dict]) -> list[dict]:
    if len(junction_examples) != len(overlays):
        raise ValueError("Junction cache and equivalence overlay lengths differ")
    output = []
    for junction, labels in zip(junction_examples, overlays):
        if junction["key"] != labels["key"]:
            raise ValueError("Junction cache and equivalence overlay order differs")
        features = junction["features"]
        merged = dict(junction)
        merged.update({key: value for key, value in labels.items() if key != "key"})
        if merged["features"] is not features:
            raise AssertionError("Oracle equivalence labels entered model features")
        output.append(merged)
    return output


def build_examples(junction_examples: list[dict], base_examples: list[dict]) -> list[dict]:
    """Convenience path used by tests and one-off audits."""

    if len(junction_examples) != len(base_examples):
        raise ValueError("Junction and base cache lengths differ")
    return [
        attach_equivalence_labels(junction, base)
        for junction, base in zip(junction_examples, base_examples)
    ]


def collate_examples(examples: list[dict]) -> dict:
    tensor_names = (
        "features",
        "mask",
        "display_ids",
        "chemical_ids",
        "group_ids",
        "group_mask",
        "group_targets",
        "canonical_targets",
        "group_display_targets",
        "group_chemical_targets",
    )
    return {
        **{name: torch.stack([item[name] for item in examples]) for name in tensor_names},
        "representable_counts": torch.tensor(
            [item["representable_count"] for item in examples], dtype=torch.long
        ),
        "examples": examples,
    }


@torch.inference_mode()
def collect_keep_logits(
    selector: BondSetSelector,
    examples: list[dict],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    selector.eval()
    batches = []
    loader = DataLoader(
        examples, batch_size=batch_size, shuffle=False, collate_fn=collate_examples
    )
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
        batches.append(output["keep_logits"].float().cpu())
    return torch.cat(batches) if batches else torch.empty((0, 0))


def selection_metrics(
    logits: torch.Tensor,
    examples: list[dict],
    *,
    threshold: float,
) -> dict:
    return equivalence_selection_metrics(
        logits,
        torch.stack([item["mask"] for item in examples]),
        torch.stack([item["group_ids"] for item in examples]),
        torch.stack([item["group_mask"] for item in examples]),
        torch.stack([item["group_targets"] for item in examples]),
        threshold=threshold,
    )


def evaluate_at_threshold(
    selector: BondSetSelector,
    examples: list[dict],
    samples: list[dict],
    grouper,
    grouper_payload: dict,
    device: torch.device,
    *,
    threshold: float,
    batch_size: int,
    logits: torch.Tensor | None = None,
) -> dict:
    topology = evaluate_topology(
        selector,
        examples,
        samples,
        grouper,
        grouper_payload,
        device,
        threshold=threshold,
        batch_size=batch_size,
    )
    if logits is None:
        logits = collect_keep_logits(
            selector, examples, device, batch_size=batch_size
        )
    return {
        **topology,
        **selection_metrics(logits, examples, threshold=threshold),
    }


@torch.inference_mode()
def evaluate_oracle_group_representatives(
    examples: list[dict],
    samples: list[dict],
    logits: torch.Tensor,
    grouper,
    grouper_payload: dict,
    device: torch.device,
    *,
    threshold: float,
    canonical: bool,
) -> dict:
    """Oracle-only diagnostic for duplicate and representative headroom."""

    rows = []
    probabilities = logits.sigmoid()
    for example, sample, sample_logits, sample_probabilities in zip(
        examples, samples, logits, probabilities
    ):
        thresholded = example["mask"] & sample_probabilities.ge(threshold)
        selected = []
        for group_index in torch.where(example["group_mask"])[0].tolist():
            members = torch.where(
                thresholded & example["group_ids"].eq(group_index)
            )[0]
            if not len(members):
                continue
            if canonical and bool(example["group_targets"][group_index]):
                canonical_indices = torch.where(
                    example["mask"]
                    & example["group_ids"].eq(group_index)
                    & example["canonical_targets"]
                )[0]
                if len(canonical_indices) != 1:
                    raise ValueError("Oracle positive group lacks one canonical candidate")
                selected.append(int(canonical_indices[0]))
            else:
                best = members[sample_logits[members].argmax()]
                selected.append(int(best))
        result = group_selected(
            example,
            selected,
            sample_logits,
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
    return summarize(rows)


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("Threshold scan must contain values in [0, 1]")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-checkpoint",
        default="outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/best.pt",
    )
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
    parser.add_argument(
        "--train-junction-cache",
        default="outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/train-junction-candidates.pt",
    )
    parser.add_argument(
        "--validation-junction-cache",
        default="outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/validation-junction-candidates.pt",
    )
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--exactly-one-weight", type=float, default=1.0)
    parser.add_argument("--representative-weight", type=float, default=0.0)
    parser.add_argument("--cardinality-weight", type=float, default=0.0)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--type-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--positive-mode", choices=("canonical", "group"), default="canonical"
    )
    parser.add_argument(
        "--negative-mode", choices=("max", "group"), default="max"
    )
    parser.add_argument(
        "--cardinality-mode",
        choices=("candidate", "group", "none"),
        default="none",
    )
    parser.add_argument(
        "--thresholds",
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99",
    )
    parser.add_argument(
        "--threshold-selection", choices=("init", "train_scan"), default="init"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v0827-winner-anchored-bond-selector-capacity500",
    )
    args = parser.parse_args()
    if min(args.train_samples, args.validation_samples, args.epochs, args.batch_size) < 1:
        raise ValueError("Sample, epoch, and batch counts must be positive")
    if args.learning_rate <= 0.0 or args.temperature <= 0.0:
        raise ValueError("Learning rate and temperature must be positive")
    thresholds = parse_thresholds(args.thresholds)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouper, grouper_payload = load_visual_grouper(args.grouper_checkpoint, device)
    train_samples = load_samples(args.train_shards, args.train_samples)
    validation_samples = load_samples(args.validation_shards, args.validation_samples)
    train_base_full = torch.load(
        args.train_base_cache, map_location="cpu", weights_only=False
    )
    validation_base_full = torch.load(
        args.validation_base_cache, map_location="cpu", weights_only=False
    )
    train_junction_full = torch.load(
        args.train_junction_cache, map_location="cpu", weights_only=False
    )
    validation_junction_full = torch.load(
        args.validation_junction_cache, map_location="cpu", weights_only=False
    )
    train_base = select_cached_examples(
        train_base_full, train_samples, args.train_samples, name="train base"
    )
    validation_base = select_cached_examples(
        validation_base_full,
        validation_samples,
        args.validation_samples,
        name="validation base",
    )
    train_junction = select_cached_examples(
        train_junction_full, train_samples, args.train_samples, name="train junction"
    )
    validation_junction = select_cached_examples(
        validation_junction_full,
        validation_samples,
        args.validation_samples,
        name="validation junction",
    )
    train_overlay = load_or_build_overlay(
        output_dir / "train-equivalence-labels.pt", train_base
    )
    validation_overlay = load_or_build_overlay(
        output_dir / "validation-equivalence-labels.pt", validation_base
    )
    train_examples = attach_overlay(train_junction, train_overlay)
    validation_examples = attach_overlay(validation_junction, validation_overlay)

    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = BondSetSelectorConfig(**initial["config"])
    feature_widths = {
        int(item["features"].size(-1))
        for item in train_examples + validation_examples
    }
    if feature_widths != {config.input_dim}:
        raise ValueError(
            f"Cache feature widths {sorted(feature_widths)} differ from input_dim={config.input_dim}"
        )
    selector = BondSetSelector(config).to(device)
    selector.load_state_dict(initial["bond_set_selector"])

    initial_threshold = float(initial["keep_threshold"])
    initial_train_logits = collect_keep_logits(
        selector, train_examples, device, batch_size=args.batch_size
    )
    initial_validation_logits = collect_keep_logits(
        selector, validation_examples, device, batch_size=args.batch_size
    )
    baseline_train = evaluate_at_threshold(
        selector,
        train_examples,
        train_samples,
        grouper,
        grouper_payload,
        device,
        threshold=initial_threshold,
        batch_size=args.batch_size,
        logits=initial_train_logits,
    )
    baseline_validation = evaluate_at_threshold(
        selector,
        validation_examples,
        validation_samples,
        grouper,
        grouper_payload,
        device,
        threshold=initial_threshold,
        batch_size=args.batch_size,
        logits=initial_validation_logits,
    )
    print(
        "baseline "
        f"train_topology={baseline_train['topology_exact_rate']:.4f} "
        f"validation_topology={baseline_validation['topology_exact_rate']:.4f}",
        flush=True,
    )

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
    display_weights = DISPLAY_CLASS_WEIGHTS.to(device)
    chemical_weights = CHEMICAL_CLASS_WEIGHTS.to(device)
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        selector.train()
        totals: dict[str, float] = {}
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
                losses = equivalence_bond_selector_loss(
                    output,
                    mask,
                    batch["group_ids"].to(device),
                    batch["group_mask"].to(device),
                    batch["group_targets"].to(device),
                    batch["canonical_targets"].to(device),
                    batch["representable_counts"].to(device),
                    batch["group_display_targets"].to(device),
                    batch["group_chemical_targets"].to(device),
                    positive_weight=args.positive_weight,
                    negative_weight=args.negative_weight,
                    exactly_one_weight=args.exactly_one_weight,
                    representative_weight=args.representative_weight,
                    cardinality_weight=args.cardinality_weight,
                    margin_weight=args.margin_weight,
                    margin=args.margin,
                    type_weight=args.type_weight,
                    temperature=args.temperature,
                    positive_mode=args.positive_mode,
                    negative_mode=args.negative_mode,
                    cardinality_mode=args.cardinality_mode,
                    display_class_weights=display_weights,
                    chemical_class_weights=chemical_weights,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        epoch_metrics = {
            name: value / max(batches, 1) for name, value in totals.items()
        }
        epoch_metrics["epoch"] = epoch
        loss_history.append(epoch_metrics)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} loss={epoch_metrics['loss']:.6f} "
                f"positive={epoch_metrics['positive_group_loss']:.6f} "
                f"negative={epoch_metrics['negative_group_loss']:.6f} "
                f"one={epoch_metrics['exactly_one_loss']:.6f}",
                flush=True,
            )

    train_logits = collect_keep_logits(
        selector, train_examples, device, batch_size=args.batch_size
    )
    threshold_results = []
    for threshold in thresholds:
        metrics = evaluate_at_threshold(
            selector,
            train_examples,
            train_samples,
            grouper,
            grouper_payload,
            device,
            threshold=threshold,
            batch_size=args.batch_size,
            logits=train_logits,
        )
        threshold_results.append({"threshold": threshold, "metrics": metrics})
        print(
            f"threshold={threshold:.2f} topology={metrics['topology_exact_rate']:.4f} "
            f"group_recall={metrics['group_recall']:.4f} "
            f"exactly_one={metrics['exactly_one_rate']:.4f} "
            f"group_set={metrics['group_set_exact_rate']:.4f}",
            flush=True,
        )
    if args.threshold_selection == "init":
        selected_result = next(
            (
                item
                for item in threshold_results
                if abs(float(item["threshold"]) - initial_threshold) < 1e-9
            ),
            None,
        )
        if selected_result is None:
            raise ValueError(
                "Initial checkpoint threshold is absent from --thresholds"
            )
    else:
        selected_result = max(
            threshold_results,
            key=lambda item: (
                item["metrics"]["topology_exact_rate"],
                item["metrics"]["strict_group_selection_exact_rate"],
                item["metrics"]["edge_f1"],
                item["metrics"]["count_accuracy"],
                item["threshold"],
            ),
        )
    selected_threshold = float(selected_result["threshold"])
    train_metrics = selected_result["metrics"]
    validation_logits = collect_keep_logits(
        selector, validation_examples, device, batch_size=args.batch_size
    )
    validation_metrics = evaluate_at_threshold(
        selector,
        validation_examples,
        validation_samples,
        grouper,
        grouper_payload,
        device,
        threshold=selected_threshold,
        batch_size=args.batch_size,
        logits=validation_logits,
    )
    oracle_deduplicated = evaluate_oracle_group_representatives(
        validation_examples,
        validation_samples,
        validation_logits,
        grouper,
        grouper_payload,
        device,
        threshold=selected_threshold,
        canonical=False,
    )
    oracle_canonical = evaluate_oracle_group_representatives(
        validation_examples,
        validation_samples,
        validation_logits,
        grouper,
        grouper_payload,
        device,
        threshold=selected_threshold,
        canonical=True,
    )
    oracle_dedupe_gain = (
        oracle_deduplicated["topology_exact_rate"]
        - validation_metrics["topology_exact_rate"]
    )
    capacity_checks = {
        "validation_topology_preserves_baseline": validation_metrics[
            "topology_exact_rate"
        ]
        >= baseline_validation["topology_exact_rate"],
        "validation_same_group_extras_do_not_increase": validation_metrics[
            "same_group_extra_candidates_per_image"
        ]
        <= baseline_validation["same_group_extra_candidates_per_image"],
        "oracle_dedupe_gain_at_most_0_5pp": oracle_dedupe_gain <= 0.005,
    }
    capacity_go = all(capacity_checks.values())

    checkpoint = {
        "bond_set_selector": selector.state_dict(),
        "config": asdict(config),
        "keep_threshold": selected_threshold,
        "grouper_checkpoint": args.grouper_checkpoint,
        "epoch": args.epochs,
        "training_args": vars(args),
        "baseline_train_metrics": baseline_train,
        "baseline_validation_metrics": baseline_validation,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "oracle_deduplicated_validation_metrics": oracle_deduplicated,
        "oracle_canonical_validation_metrics": oracle_canonical,
        "loss_history": loss_history,
        "equivalence_training": True,
    }
    torch.save(checkpoint, output_dir / "best.pt")
    summary = {
        "schema_version": "equivalence-bond-selector-capacity-0.1",
        "selected_threshold": selected_threshold,
        "baseline_train_metrics": baseline_train,
        "baseline_validation_metrics": baseline_validation,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "oracle_diagnostics": {
            "deduplicated": oracle_deduplicated,
            "canonical": oracle_canonical,
            "dedupe_topology_gain": oracle_dedupe_gain,
            "canonical_topology_gain_over_dedupe": (
                oracle_canonical["topology_exact_rate"]
                - oracle_deduplicated["topology_exact_rate"]
            ),
        },
        "threshold_results": threshold_results,
        "capacity_checks": capacity_checks,
        "capacity_go": capacity_go,
        "protocol": {
            **vars(args),
            "device": str(device),
            "init_checkpoint_sha256": file_sha256(args.init_checkpoint),
            "grouper_checkpoint_sha256": file_sha256(args.grouper_checkpoint),
            "gt_atom_pairs_used_as_features": False,
            "representable_count_used_in_loss": (
                args.cardinality_weight > 0.0 and args.cardinality_mode != "none"
            ),
            "complete_gt_count_used_in_loss": False,
            "junction_grouping_runs_after_keep_selection": True,
            "inference_code_changed": False,
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
