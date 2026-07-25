#!/usr/bin/env python3
"""Train v0.9.7 typed affinities on the oracle usable proposal pool."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from chemvector.bond_group_proposal_membership import PHYSICAL_GROUP_NAMES
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    PROPOSAL_DESCRIPTOR_DIM,
    prepare_proposal_membership_pipeline,
)
from chemvector.bond_group_typed_affinity import (
    PAIR_HASHED,
    PAIR_NONE,
    PAIR_PARALLEL_2,
    PAIR_PARALLEL_3,
    BondGroupTypedAffinityConfig,
    BondGroupTypedAffinityPredictor,
    bond_group_typed_affinity_loss,
    build_typed_affinity_targets,
)
from chemvector.primitive_set import PrimitiveSetPredictor
from chemvector.train_bond_group_proposal_pointer import (
    DEFAULT_CAPACITY_SHARD,
    SOURCE_PRIMITIVE_CHECKPOINT,
    SOURCE_PRIMITIVE_SHA256,
    _device_tensor_dict,
    _finalize_coverage,
    _merge_coverage,
    _validate_batch_metadata,
    assert_frozen_flat_output,
    autocast_context,
    expand_shards,
    freeze_primitive_model,
    load_frozen_primitive_checkpoint,
    make_loader,
    oracle_pool_contract,
    reject_internal_test_metadata,
    reject_internal_test_shards,
)
from chemvector.train_bond_group_set import atomic_torch_save
from chemvector.train_primitive_set import file_sha256, prepare_output_directory


CHECKPOINT_SCHEMA_VERSION = (
    "bond-group-typed-affinity-conditional-checkpoint-0.1"
)
VALIDATION_SCHEMA_VERSION = (
    "bond-group-typed-affinity-conditional-validation-0.1"
)
TRAINING_SUMMARY_SCHEMA_VERSION = (
    "bond-group-typed-affinity-conditional-training-summary-0.1"
)
SEQUENCE_SCHEMA_VERSION = "visual-svg-0.2"
CONDITIONAL_SCOPE_NAME = "oracle_usable_proposal_pool"
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/"
    "chemvector-v097-bond-group-typed-affinity-capacity100-step500"
)
LOSS_FIELDS = (
    "loss",
    "node_type_loss",
    "node_type_positive_loss",
    "node_type_background_loss",
    "pair_loss",
    "pair_positive_loss",
    "pair_negative_loss",
)
POSITIVE_PAIR_CLASSES = (
    (PAIR_PARALLEL_2, "parallel_2"),
    (PAIR_PARALLEL_3, "parallel_3"),
    (PAIR_HASHED, "hashed"),
)


def typed_affinity_config(
    *,
    proposal_feature_dim: int,
    tiny: bool = False,
) -> BondGroupTypedAffinityConfig:
    common = {
        "proposal_feature_dim": proposal_feature_dim,
        "proposal_descriptor_dim": PROPOSAL_DESCRIPTOR_DIM,
        "flat_type_classes": 5,
        "dropout": 0.0,
    }
    if tiny:
        return BondGroupTypedAffinityConfig(
            **common,
            d_model=32,
            proposal_encoder_layers=1,
            heads=4,
            ffn_dim=64,
            pair_dim=16,
            pair_hidden_dim=32,
        )
    return BondGroupTypedAffinityConfig(
        **common,
        d_model=192,
        proposal_encoder_layers=2,
        heads=6,
        ffn_dim=768,
        pair_dim=96,
        pair_hidden_dim=192,
    )


def conditional_scope_contract(
    collator: BondGroupProposalMembershipCollator,
) -> dict[str, Any]:
    """Describe exactly which oracle information this experiment consumes."""

    return {
        "name": CONDITIONAL_SCOPE_NAME,
        "conditional": True,
        "proposal_mask": "targets.proposal_keep & frozen_proposal_mask",
        "oracle_information_used": [
            "which frozen proposals have complete usable GT association",
        ],
        "oracle_information_not_given_to_model": [
            "group identity",
            "group count",
            "pair labels",
            "group type labels",
        ],
        "reported_predictions": [
            "proposal node group type",
            "symmetric unordered typed pair affinity",
        ],
        "candidate_generator_in_scope": False,
        "exact_cover_solver_in_scope": False,
        "formal_sdf_in_scope": False,
        "end_to_end_metric": False,
        "frozen_proposal_pool": oracle_pool_contract(collator),
    }


def restrict_to_oracle_usable_pool(
    model_inputs: dict[str, Any],
    targets: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Replace the model mask with the exact usable membership union."""

    proposal_mask = model_inputs.get("proposal_mask")
    proposal_keep = targets.get("proposal_keep")
    if not isinstance(proposal_mask, torch.Tensor) or proposal_mask.ndim != 2:
        raise ValueError("Typed Affinity requires a 2-D proposal mask")
    if not isinstance(proposal_keep, torch.Tensor) or proposal_keep.shape != (
        proposal_mask.shape
    ):
        raise ValueError("Typed Affinity proposal_keep and proposal_mask differ")
    full_mask = proposal_mask.bool()
    if bool((proposal_keep.bool() & ~full_mask).any()):
        raise ValueError("Typed Affinity oracle usable pool references padding")
    usable_mask = full_mask & proposal_keep.bool()
    empty = torch.where(usable_mask.sum(dim=-1) == 0)[0]
    if len(empty):
        raise ValueError(
            "Typed Affinity conditional samples have no usable proposals: "
            f"{empty.tolist()}"
        )
    conditional_inputs = dict(model_inputs)
    conditional_inputs["proposal_mask"] = usable_mask
    return conditional_inputs, full_mask, usable_mask


def prepare_conditional_pipeline_batch(
    batch: dict[str, Any],
    flat_output: dict[str, torch.Tensor],
    primitive_model: PrimitiveSetPredictor,
    collator: BondGroupProposalMembershipCollator,
    *,
    max_group_count: int,
) -> dict[str, Any]:
    proposal_batch = collator.attach_flat_output(batch, flat_output)
    prepared = prepare_proposal_membership_pipeline(
        proposal_batch,
        flat_output,
        primitive_model.reference_points.detach(),
        group_slots=max_group_count,
    )
    conditional_inputs, full_mask, usable_mask = restrict_to_oracle_usable_pool(
        prepared["model_inputs"],
        prepared["targets"],
    )
    return {
        **prepared,
        "model_inputs": conditional_inputs,
        "full_proposal_mask": full_mask,
        "oracle_usable_mask": usable_mask,
    }


def compute_typed_affinity_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    node_type_weight: float,
    node_background_weight: float,
    pair_positive_weight: float,
    pair_negative_weight: float,
    negative_top_k: int | None,
) -> dict[str, Any]:
    return bond_group_typed_affinity_loss(
        output,
        targets,
        proposal_mask,
        node_type_weight=node_type_weight,
        node_background_weight=node_background_weight,
        pair_positive_weight=pair_positive_weight,
        pair_negative_weight=pair_negative_weight,
        negative_top_k=negative_top_k,
    )


def _empty_node_totals() -> dict[str, Any]:
    return {
        "correct": 0,
        "total": 0,
        "by_type": {
            PHYSICAL_GROUP_NAMES[group_type]: {"correct": 0, "total": 0}
            for group_type in range(1, len(PHYSICAL_GROUP_NAMES))
        },
    }


def _empty_pair_totals() -> dict[str, Any]:
    return {
        "correct": 0,
        "total": 0,
        "target_none": 0,
        "target_positive": 0,
        "predicted_positive": 0,
        "mined_hard_negative": 0,
        "by_type": {
            name: {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "target": 0,
                "predicted": 0,
            }
            for _, name in POSITIVE_PAIR_CLASSES
        },
    }


def _pair_count_for_top_k(negative_count: int, top_k: int | None) -> int:
    return negative_count if top_k is None else min(negative_count, top_k)


def update_conditional_metric_totals(
    node_totals: dict[str, Any],
    pair_totals: dict[str, Any],
    per_image_pair_counts: list[dict[str, Any]],
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    sample_keys: list[str],
    *,
    negative_top_k: int | None,
) -> None:
    """Accumulate node and strictly unordered pair metrics."""

    if len(sample_keys) != proposal_mask.size(0):
        raise ValueError("Typed Affinity sample keys and batch size differ")
    affinity_targets = build_typed_affinity_targets(targets, proposal_mask)
    node_labels = affinity_targets["node_group_type_labels"]
    node_predictions = output["node_group_type_logits"].float().argmax(dim=-1)
    pair_labels = affinity_targets["pair_labels"]
    pair_predictions = output["pair_logits"].float().argmax(dim=-1)
    proposals = proposal_mask.size(1)
    upper = torch.triu(
        torch.ones(
            (proposals, proposals),
            device=proposal_mask.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    for batch_index, sample_key in enumerate(sample_keys):
        node_scope = proposal_mask[batch_index].bool()
        current_labels = node_labels[batch_index]
        current_predictions = node_predictions[batch_index]
        node_totals["correct"] += int(
            (current_predictions[node_scope] == current_labels[node_scope]).sum()
        )
        node_totals["total"] += int(node_scope.sum())
        for group_type in range(1, len(PHYSICAL_GROUP_NAMES)):
            type_scope = node_scope & (current_labels == group_type)
            name = PHYSICAL_GROUP_NAMES[group_type]
            node_totals["by_type"][name]["correct"] += int(
                (current_predictions[type_scope] == group_type).sum()
            )
            node_totals["by_type"][name]["total"] += int(type_scope.sum())

        pair_scope = affinity_targets["pair_mask"][batch_index].bool() & upper
        true_pairs = pair_labels[batch_index, pair_scope]
        predicted_pairs = pair_predictions[batch_index, pair_scope]
        pair_count = len(true_pairs)
        target_none = int((true_pairs == PAIR_NONE).sum())
        target_positive = pair_count - target_none
        predicted_positive = int((predicted_pairs != PAIR_NONE).sum())
        mined_count = _pair_count_for_top_k(target_none, negative_top_k)
        pair_totals["correct"] += int((predicted_pairs == true_pairs).sum())
        pair_totals["total"] += pair_count
        pair_totals["target_none"] += target_none
        pair_totals["target_positive"] += target_positive
        pair_totals["predicted_positive"] += predicted_positive
        pair_totals["mined_hard_negative"] += mined_count

        target_by_type: dict[str, int] = {}
        predicted_by_type: dict[str, int] = {}
        for pair_class, name in POSITIVE_PAIR_CLASSES:
            target_class = true_pairs == pair_class
            predicted_class = predicted_pairs == pair_class
            true_positive = int((target_class & predicted_class).sum())
            false_positive = int((~target_class & predicted_class).sum())
            false_negative = int((target_class & ~predicted_class).sum())
            target_count = int(target_class.sum())
            predicted_count = int(predicted_class.sum())
            entry = pair_totals["by_type"][name]
            entry["true_positive"] += true_positive
            entry["false_positive"] += false_positive
            entry["false_negative"] += false_negative
            entry["target"] += target_count
            entry["predicted"] += predicted_count
            target_by_type[name] = target_count
            predicted_by_type[name] = predicted_count

        per_image_pair_counts.append(
            {
                "sample_key": str(sample_key),
                "usable_proposals": int(node_scope.sum()),
                "unordered_pairs": pair_count,
                "target_none_pairs": target_none,
                "target_positive_pairs": target_positive,
                "predicted_positive_pairs": predicted_positive,
                "mined_hard_negative_pairs": mined_count,
                "target_positive_by_type": target_by_type,
                "predicted_positive_by_type": predicted_by_type,
            }
        )


def finalize_node_type_metrics(totals: dict[str, Any]) -> dict[str, Any]:
    by_type = {}
    supported_accuracies = []
    for name, values in totals["by_type"].items():
        total = int(values["total"])
        accuracy = int(values["correct"]) / max(total, 1)
        by_type[name] = {**values, "accuracy": accuracy}
        if total:
            supported_accuracies.append(accuracy)
    return {
        "correct": int(totals["correct"]),
        "total": int(totals["total"]),
        "overall_accuracy": int(totals["correct"])
        / max(int(totals["total"]), 1),
        "macro_accuracy": sum(supported_accuracies)
        / max(len(supported_accuracies), 1),
        "macro_supported_type_count": len(supported_accuracies),
        "macro_scope": "physical node types with nonzero target support",
        "by_type": by_type,
    }


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def finalize_unordered_pair_metrics(totals: dict[str, Any]) -> dict[str, Any]:
    by_type = {}
    f1_values = []
    supported_f1_values = []
    for _, name in POSITIVE_PAIR_CLASSES:
        values = totals["by_type"][name]
        precision, recall, f1 = _precision_recall_f1(
            int(values["true_positive"]),
            int(values["false_positive"]),
            int(values["false_negative"]),
        )
        by_type[name] = {
            **values,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        if int(values["target"]):
            supported_f1_values.append(f1)
    return {
        "correct": int(totals["correct"]),
        "total": int(totals["total"]),
        "overall_accuracy_including_none": int(totals["correct"])
        / max(int(totals["total"]), 1),
        "macro_f1": sum(f1_values) / len(POSITIVE_PAIR_CLASSES),
        "supported_macro_f1": sum(supported_f1_values)
        / max(len(supported_f1_values), 1),
        "macro_scope": "parallel_2/parallel_3/hashed one-vs-rest",
        "target_none_pairs": int(totals["target_none"]),
        "target_positive_pairs": int(totals["target_positive"]),
        "predicted_positive_pairs": int(totals["predicted_positive"]),
        "mined_hard_negative_pairs": int(totals["mined_hard_negative"]),
        "by_type": by_type,
        "p2": by_type["parallel_2"],
        "p3": by_type["parallel_3"],
        "hashed": by_type["hashed"],
    }


@torch.inference_mode()
def validate(
    model: BondGroupTypedAffinityPredictor,
    primitive_model: PrimitiveSetPredictor,
    loader,
    device: torch.device,
    *,
    max_group_count: int,
    node_type_weight: float,
    node_background_weight: float,
    pair_positive_weight: float,
    pair_negative_weight: float,
    negative_top_k: int | None,
) -> dict[str, Any]:
    model.eval()
    primitive_model.eval()
    if any(parameter.requires_grad for parameter in primitive_model.parameters()):
        raise RuntimeError("Validation primitive model is not frozen")
    loss_totals = {name: 0.0 for name in LOSS_FIELDS}
    node_totals = _empty_node_totals()
    pair_totals = _empty_pair_totals()
    per_image_pair_counts: list[dict[str, Any]] = []
    coverage_totals: dict[str, Any] = {}
    usable_proposals = 0
    ignored_proposals = 0
    samples = 0
    started = time.time()

    for batch in loader:
        _validate_batch_metadata(batch)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with autocast_context(device):
            flat_output = primitive_model(pixels)
        assert_frozen_flat_output(primitive_model, flat_output)
        prepared = prepare_conditional_pipeline_batch(
            batch,
            flat_output,
            primitive_model,
            loader.collate_fn,
            max_group_count=max_group_count,
        )
        del flat_output
        model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
        targets = _device_tensor_dict(prepared["targets"], device)
        proposal_mask = model_inputs["proposal_mask"].bool()
        with autocast_context(device):
            output = model(**model_inputs)
            losses = compute_typed_affinity_loss(
                output,
                targets,
                proposal_mask,
                node_type_weight=node_type_weight,
                node_background_weight=node_background_weight,
                pair_positive_weight=pair_positive_weight,
                pair_negative_weight=pair_negative_weight,
                negative_top_k=negative_top_k,
            )

        batch_size = pixels.size(0)
        samples += batch_size
        for name in LOSS_FIELDS:
            loss_totals[name] += float(losses[name]) * batch_size
        update_conditional_metric_totals(
            node_totals,
            pair_totals,
            per_image_pair_counts,
            output,
            targets,
            proposal_mask,
            [str(key) for key in batch["keys"]],
            negative_top_k=negative_top_k,
        )
        _merge_coverage(coverage_totals, prepared["oracle_coverage"])
        full_mask = prepared["full_proposal_mask"].to(device).bool()
        usable_proposals += int(proposal_mask.sum())
        ignored_proposals += int((full_mask & ~proposal_mask).sum())

    if not samples:
        raise RuntimeError("Typed Affinity validation loader produced no samples")
    assert_frozen_flat_output(primitive_model, {})
    averaged_losses = {
        name: value / samples for name, value in loss_totals.items()
    }
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "samples": samples,
        **averaged_losses,
        "positive_pair_group_balanced_loss": averaged_losses[
            "pair_positive_loss"
        ],
        "hard_negative_pair_loss": averaged_losses["pair_negative_loss"],
        CONDITIONAL_SCOPE_NAME: {
            "node_type": finalize_node_type_metrics(node_totals),
            "unordered_pair": finalize_unordered_pair_metrics(pair_totals),
            "global_pair_counts": {
                "images": samples,
                "usable_proposals": usable_proposals,
                "unordered_pairs": int(pair_totals["total"]),
                "target_none_pairs": int(pair_totals["target_none"]),
                "target_positive_pairs": int(pair_totals["target_positive"]),
                "predicted_positive_pairs": int(
                    pair_totals["predicted_positive"]
                ),
                "mined_hard_negative_pairs": int(
                    pair_totals["mined_hard_negative"]
                ),
            },
            "per_image_pair_counts": per_image_pair_counts,
            "candidate_pool": {
                "usable_proposals": usable_proposals,
                "ignored_frozen_proposals": ignored_proposals,
                "model_attention_mask": CONDITIONAL_SCOPE_NAME,
            },
        },
        "oracle_coverage": _finalize_coverage(coverage_totals),
        "conditional_scope": conditional_scope_contract(loader.collate_fn),
        "elapsed_seconds": time.time() - started,
        "uses_internal_test": False,
    }


def save_checkpoint(
    path: Path,
    model: BondGroupTypedAffinityPredictor,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    samples_seen: int,
    args: argparse.Namespace,
    validation: dict[str, Any] | None,
    primitive_provenance: dict[str, Any],
    scope_contract: dict[str, Any],
) -> None:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    atomic_torch_save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "typed_affinity_config": asdict(model.config),
            "global_step": int(global_step),
            "samples_seen": int(samples_seen),
            "training_args": vars(args),
            "validation": validation,
            "conditional_scope": scope_contract,
            "source_primitive_checkpoint": primitive_provenance["checkpoint"],
            "source_primitive_checkpoint_sha256": primitive_provenance[
                "checkpoint_sha256"
            ],
            "primitive_provenance": primitive_provenance,
            "frozen_primitive_state_saved": False,
            "candidate_generator_state_saved": False,
            "exact_cover_solver_state_saved": False,
            "formal_sdf_evaluation_in_scope": False,
            "optimizer_mode": "fresh",
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "uses_internal_test": False,
        },
        path,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", default=[DEFAULT_CAPACITY_SHARD])
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=[DEFAULT_CAPACITY_SHARD],
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--primitive-checkpoint",
        default=str(SOURCE_PRIMITIVE_CHECKPOINT),
    )
    parser.add_argument(
        "--primitive-checkpoint-sha256",
        default=SOURCE_PRIMITIVE_SHA256,
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--max-group-count", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--node-type-weight", type=float, default=1.0)
    parser.add_argument("--node-background-weight", type=float, default=0.0)
    parser.add_argument("--pair-positive-weight", type=float, default=1.0)
    parser.add_argument("--pair-negative-weight", type=float, default=0.25)
    parser.add_argument("--negative-top-k", type=int, default=64)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def _device_from_args(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.image_size,
        args.batch_size,
        args.max_group_count,
        args.max_steps,
        args.validate_every,
        args.save_every,
        args.negative_top_k,
    )
    if min(positive) < 1:
        raise ValueError("Positive Typed Affinity dimensions are required")
    if args.workers < 0:
        raise ValueError("Workers must be nonnegative")
    if args.validation_limit is not None and args.validation_limit < 1:
        raise ValueError("Validation limit must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Learning rate and weight decay are invalid")
    if args.gradient_clip <= 0.0:
        raise ValueError("Gradient clip must be positive")
    loss_weights = (
        args.node_type_weight,
        args.node_background_weight,
        args.pair_positive_weight,
        args.pair_negative_weight,
    )
    if min(loss_weights) < 0.0 or not any(loss_weights):
        raise ValueError("Typed Affinity loss weights are invalid")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards)
    output_dir = Path(args.output_dir)
    prepare_output_directory(output_dir)
    device = _device_from_args(args)
    primitive_model, _, primitive_provenance = load_frozen_primitive_checkpoint(
        Path(args.primitive_checkpoint),
        expected_sha256=args.primitive_checkpoint_sha256,
        device=device,
    )
    config = typed_affinity_config(
        proposal_feature_dim=primitive_model.config.d_model,
        tiny=args.tiny,
    )
    model = BondGroupTypedAffinityPredictor(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        train_shards,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_shards,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        shuffle=False,
        limit=args.validation_limit,
    )
    scope_contract = conditional_scope_contract(train_loader.collate_fn)
    if scope_contract != conditional_scope_contract(validation_loader.collate_fn):
        raise RuntimeError("Train and validation conditional proposal pools differ")
    write_json(output_dir / "config.json", model.config_dict())
    write_json(output_dir / "training_args.json", vars(args))
    write_json(output_dir / "primitive_source.json", primitive_provenance)
    write_json(output_dir / "conditional_scope.json", scope_contract)

    global_step = 0
    samples_seen = 0
    best_loss = float("inf")
    last_validation: dict[str, Any] | None = None
    history = []
    while global_step < args.max_steps:
        batches = 0
        for batch in train_loader:
            batches += 1
            _validate_batch_metadata(batch)
            model.train()
            primitive_model.eval()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.inference_mode(), autocast_context(device):
                flat_output = primitive_model(pixels)
            assert_frozen_flat_output(primitive_model, flat_output)
            prepared = prepare_conditional_pipeline_batch(
                batch,
                flat_output,
                primitive_model,
                train_loader.collate_fn,
                max_group_count=args.max_group_count,
            )
            del flat_output
            model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
            targets = _device_tensor_dict(prepared["targets"], device)
            proposal_mask = model_inputs["proposal_mask"].bool()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(**model_inputs)
                losses = compute_typed_affinity_loss(
                    output,
                    targets,
                    proposal_mask,
                    node_type_weight=args.node_type_weight,
                    node_background_weight=args.node_background_weight,
                    pair_positive_weight=args.pair_positive_weight,
                    pair_negative_weight=args.pair_negative_weight,
                    negative_top_k=args.negative_top_k,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            if any(
                parameter.grad is not None
                for parameter in primitive_model.parameters()
            ):
                raise RuntimeError("Frozen primitive model received gradients")
            global_step += 1
            samples_seen += pixels.size(0)

            if global_step % args.validate_every == 0 or global_step >= args.max_steps:
                last_validation = validate(
                    model,
                    primitive_model,
                    validation_loader,
                    device,
                    max_group_count=args.max_group_count,
                    node_type_weight=args.node_type_weight,
                    node_background_weight=args.node_background_weight,
                    pair_positive_weight=args.pair_positive_weight,
                    pair_negative_weight=args.pair_negative_weight,
                    negative_top_k=args.negative_top_k,
                )
                history.append({"global_step": global_step, "metrics": last_validation})
                conditional = last_validation[CONDITIONAL_SCOPE_NAME]
                print(
                    f"step={global_step} loss={last_validation['loss']:.6f} "
                    f"node_macro={conditional['node_type']['macro_accuracy']:.4f} "
                    f"pair_macro_f1="
                    f"{conditional['unordered_pair']['macro_f1']:.4f} "
                    f"pairs={conditional['global_pair_counts']['unordered_pairs']}",
                    flush=True,
                )
                if float(last_validation["loss"]) < best_loss:
                    best_loss = float(last_validation["loss"])
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        global_step=global_step,
                        samples_seen=samples_seen,
                        args=args,
                        validation=last_validation,
                        primitive_provenance=primitive_provenance,
                        scope_contract=scope_contract,
                    )
            if global_step % args.save_every == 0 or global_step >= args.max_steps:
                save_checkpoint(
                    output_dir / f"step-{global_step:06d}.pt",
                    model,
                    optimizer,
                    global_step=global_step,
                    samples_seen=samples_seen,
                    args=args,
                    validation=last_validation,
                    primitive_provenance=primitive_provenance,
                    scope_contract=scope_contract,
                )
            if global_step >= args.max_steps:
                break
        if not batches:
            raise RuntimeError("Typed Affinity training loader produced no samples")

    if last_validation is None:
        raise RuntimeError("Typed Affinity training completed without validation")
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        global_step=global_step,
        samples_seen=samples_seen,
        args=args,
        validation=last_validation,
        primitive_provenance=primitive_provenance,
        scope_contract=scope_contract,
    )
    write_json(output_dir / "validation.json", last_validation)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
            "model_type": model.model_type,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "best_validation_loss": best_loss,
            "history": history,
            "config": asdict(config),
            "primitive_provenance": primitive_provenance,
            "conditional_scope": scope_contract,
            "frozen_primitive_state_saved": False,
            "candidate_generator_state_saved": False,
            "exact_cover_solver_state_saved": False,
            "formal_sdf_evaluation_in_scope": False,
            "optimizer_mode": "fresh",
            "trainable_parameter_count": model.parameter_count(),
            "uses_internal_test": False,
        },
    )


if __name__ == "__main__":
    main()
