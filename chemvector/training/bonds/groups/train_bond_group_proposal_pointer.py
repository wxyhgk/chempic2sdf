#!/usr/bin/env python3
"""Train v0.9.6 proposal-root pointers over the frozen v0.9.5 pool."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_proposal_pointer import (
    BondGroupProposalPointerConfig,
    BondGroupProposalPointerPredictor,
    bond_group_proposal_pointer_loss,
)
from chemvector.bond_group_proposal_pointer_pipeline import (
    PROPOSAL_DESCRIPTOR_DIM,
    compare_latent_root_assignments,
    decode_proposal_pointer_partition,
    latent_root_assignment_map,
    pointer_root_selection_diagnostics,
    prepare_proposal_pointer_pipeline,
)
from chemvector.primitive_set import PrimitiveSetPredictor
from chemvector.train_bond_group_proposal_membership import (
    DEFAULT_CAPACITY_SHARD,
    SOURCE_PRIMITIVE_CHECKPOINT,
    SOURCE_PRIMITIVE_SHA256,
    _device_tensor_dict,
    _finalize_coverage,
    _finalize_partition_metrics,
    _merge_coverage,
    _merge_integer_totals,
    _partition_counts,
    autocast_context,
    freeze_primitive_model,
    load_frozen_primitive_checkpoint as _load_frozen_primitive_checkpoint,
    make_loader,
)
from chemvector.train_bond_group_set import atomic_torch_save
from chemvector.train_primitive_set import (
    file_sha256,
    prepare_output_directory,
)


CHECKPOINT_SCHEMA_VERSION = "bond-group-proposal-pointer-checkpoint-0.1"
VALIDATION_SCHEMA_VERSION = "bond-group-proposal-pointer-validation-0.1"
TRAINING_SUMMARY_SCHEMA_VERSION = (
    "bond-group-proposal-pointer-training-summary-0.1"
)
SEQUENCE_SCHEMA_VERSION = "visual-svg-0.2"
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/"
    "chemvector-v096-bond-group-proposal-pointer-capacity100-step500"
)
LOSS_FIELDS = (
    "loss",
    "pointer_loss",
    "keep_loss",
    "positive_keep_loss",
    "negative_keep_loss",
    "group_type_loss",
    "background_type_loss",
    "member_count_loss",
    "background_count_loss",
    "group_count_loss",
)


def pointer_config(
    *,
    proposal_feature_dim: int,
    tiny: bool = False,
    max_members: int = 143,
    max_group_count: int = 192,
) -> BondGroupProposalPointerConfig:
    common = {
        "proposal_feature_dim": proposal_feature_dim,
        "proposal_descriptor_dim": PROPOSAL_DESCRIPTOR_DIM,
        "flat_type_classes": 5,
        "max_members": max_members,
        "max_group_count": max_group_count,
        "predict_group_count": True,
        "dropout": 0.0,
    }
    if tiny:
        return BondGroupProposalPointerConfig(
            **common,
            d_model=32,
            proposal_encoder_layers=1,
            heads=4,
            ffn_dim=64,
        )
    return BondGroupProposalPointerConfig(
        **common,
        d_model=192,
        proposal_encoder_layers=2,
        heads=6,
        ffn_dim=768,
    )


def _normalized_security_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    ).lower().replace("-", "_")


def reject_internal_test_metadata(meta: Any, sample_key: str = "") -> None:
    serialized = _normalized_security_text(meta)
    normalized_key = sample_key.lower().replace("-", "_")
    if "internal_test" in normalized_key or "internal_test" in serialized:
        raise ValueError(f"Refusing sealed internal_test sample: {sample_key}")


def reject_internal_test_shards(shards: list[str]) -> None:
    forbidden = [
        path
        for path in shards
        if "internal_test" in path.lower().replace("-", "_")
    ]
    if forbidden:
        raise ValueError(
            "Proposal Pointer training must not read sealed internal_test shards: "
            + ", ".join(forbidden)
        )


def expand_shards(patterns: list[str]) -> list[str]:
    shards = []
    for pattern in patterns:
        reject_internal_test_shards([pattern])
        matches = sorted(glob.glob(pattern))
        if matches:
            shards.extend(matches)
        elif Path(pattern).is_file():
            shards.append(pattern)
    unique = list(dict.fromkeys(shards))
    if not unique:
        raise ValueError(f"No shards matched: {patterns}")
    reject_internal_test_shards(unique)
    return unique


def _reject_checkpoint_lineage(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("uses_internal_test") not in (None, False):
        raise ValueError("Frozen primitive checkpoint used internal_test")
    training_args = checkpoint.get("training_args", {})
    if not isinstance(training_args, dict):
        raise ValueError("Frozen primitive checkpoint has invalid training_args")
    for field in ("train_shards", "validation_shards"):
        values = training_args.get(field, [])
        if values is None:
            continue
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"Frozen primitive checkpoint has invalid {field}")
        reject_internal_test_shards(values)


def load_frozen_primitive_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> tuple[PrimitiveSetPredictor, dict[str, Any], dict[str, Any]]:
    model, checkpoint, provenance = _load_frozen_primitive_checkpoint(
        path,
        expected_sha256=expected_sha256,
        device=device,
    )
    _reject_checkpoint_lineage(checkpoint)
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError("Frozen primitive model contract was not preserved")
    return model, checkpoint, provenance


def _validate_batch_metadata(batch: dict[str, Any]) -> None:
    keys = batch.get("keys")
    metadata = batch.get("meta")
    if not isinstance(keys, list) or not isinstance(metadata, list):
        raise ValueError("Proposal Pointer batch lacks keys or metadata")
    if len(keys) != len(metadata):
        raise ValueError("Proposal Pointer keys and metadata differ")
    for key, meta in zip(keys, metadata):
        reject_internal_test_metadata(meta, str(key))


def assert_frozen_flat_output(
    primitive_model: PrimitiveSetPredictor,
    flat_output: dict[str, Any],
) -> None:
    """Reject accidental training-mode or autograd leakage from flat inference."""

    if primitive_model.training:
        raise RuntimeError("Frozen primitive model left eval mode")
    if any(parameter.requires_grad for parameter in primitive_model.parameters()):
        raise RuntimeError("Frozen primitive model has trainable parameters")
    if any(parameter.grad is not None for parameter in primitive_model.parameters()):
        raise RuntimeError("Frozen primitive model received gradients")
    leaking = [
        name
        for name, value in flat_output.items()
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    if leaking:
        raise RuntimeError(f"Frozen flat outputs require gradients: {leaking}")


def oracle_pool_contract(
    collator: BondGroupProposalMembershipCollator,
) -> dict[str, Any]:
    return {
        "source": "v0.9.5_gt_associated_flat_proposals",
        "decode_mode": collator.decode_mode,
        "proposal_threshold": float(collator.proposal_threshold),
        "high_confidence_tolerance_q": float(
            collator.high_confidence_tolerance_q
        ),
        "association_tolerance_q": float(collator.association_tolerance_q),
        "shell_weight": float(collator.shell_weight),
        "max_proposals": int(collator.max_proposals),
        "unmatched_flat_false_positives_in_pool": False,
        "partial_fixed_groups_usable": False,
        "hashed_minimum_support": 2,
    }


def prepare_pipeline_batch(
    batch: dict[str, Any],
    flat_output: dict[str, torch.Tensor],
    primitive_model: PrimitiveSetPredictor,
    config: BondGroupProposalPointerConfig,
    collator: BondGroupProposalMembershipCollator,
) -> dict[str, Any]:
    proposal_batch = collator.attach_flat_output(batch, flat_output)
    return prepare_proposal_pointer_pipeline(
        proposal_batch,
        flat_output,
        primitive_model.reference_points.detach(),
        max_group_count=config.max_group_count,
    )


def compute_pointer_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    root_mode: str,
    softmin_temperature: float,
) -> dict[str, Any]:
    if "group_weights" not in targets:
        raise ValueError("Pointer targets must include group_weights")
    return bond_group_proposal_pointer_loss(
        output,
        targets,
        proposal_mask,
        group_weights=targets["group_weights"],
        root_mode=root_mode,
        softmin_temperature=softmin_temperature,
    )


def _merge_root_assignments(
    target: dict[str, int],
    values: dict[str, int],
) -> None:
    overlap = set(target) & set(values)
    if overlap:
        raise ValueError(f"Duplicate validation sample/group keys: {sorted(overlap)}")
    target.update(values)


def _finalize_root_diagnostics(values: dict[str, int]) -> dict[str, Any]:
    samples = max(values.get("samples", 0), 1)
    expected = values.get("expected_groups", 0)
    roots = values.get("selected_roots", 0)
    assigned = values.get("assigned_members", 0)
    return {
        **values,
        "distinct_target_group_coverage": values.get(
            "distinct_target_groups", 0
        )
        / max(expected, 1),
        "root_collision_rate": values.get("root_collisions", 0)
        / max(roots, 1),
        "distinct_group_coverage_exact_rate": values.get(
            "distinct_group_coverage_exact_samples", 0
        )
        / samples,
        "affinity_self_preference_rate": values.get(
            "affinity_self_preferred_roots", 0
        )
        / max(roots, 1),
        "affinity_majority_agreement_rate": values.get(
            "affinity_majority_agrees_roots", 0
        )
        / max(roots, 1),
        "root_group_member_purity": values.get(
            "assigned_members_same_as_root_group", 0
        )
        / max(assigned, 1),
    }


@torch.inference_mode()
def validate(
    model: BondGroupProposalPointerPredictor,
    primitive_model: PrimitiveSetPredictor,
    loader,
    device: torch.device,
    *,
    root_mode: str,
    softmin_temperature: float,
    previous_root_assignments: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    model.eval()
    primitive_model.eval()
    if any(parameter.requires_grad for parameter in primitive_model.parameters()):
        raise RuntimeError("Validation primitive model is not frozen")
    loss_totals = {name: 0.0 for name in LOSS_FIELDS}
    known_totals: dict[str, int] = {}
    learned_totals: dict[str, int] = {}
    known_root_totals: dict[str, int] = {}
    learned_root_totals: dict[str, int] = {}
    coverage_totals: dict[str, Any] = {}
    current_root_assignments: dict[str, int] = {}
    raw_count_infeasible = 0
    raw_count_absolute_error = 0
    feasible_count_absolute_error = 0
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
        prepared = prepare_pipeline_batch(
            batch,
            flat_output,
            primitive_model,
            model.config,
            loader.collate_fn,
        )
        del flat_output
        model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
        targets = _device_tensor_dict(prepared["targets"], device)
        proposal_mask = model_inputs["proposal_mask"].bool()
        usable_mask = prepared["oracle_usable_mask"].to(device).bool()
        known_counts = prepared["known_counts"].to(device).long()
        with autocast_context(device):
            output = model(**model_inputs)
            losses = compute_pointer_loss(
                output,
                targets,
                proposal_mask,
                root_mode=root_mode,
                softmin_temperature=softmin_temperature,
            )

        known_decoded, _ = decode_proposal_pointer_partition(
            output,
            proposal_mask,
            root_candidate_mask=usable_mask,
            member_candidate_mask=usable_mask,
            mode="known_count",
            known_counts=known_counts,
        )
        learned_decoded, learned_resolution = decode_proposal_pointer_partition(
            output,
            proposal_mask,
            root_candidate_mask=usable_mask,
            member_candidate_mask=usable_mask,
            mode="learned_count",
        )
        _merge_integer_totals(
            known_totals,
            _partition_counts(known_decoded, targets, usable_mask),
        )
        _merge_integer_totals(
            learned_totals,
            _partition_counts(learned_decoded, targets, usable_mask),
        )
        _merge_integer_totals(
            known_root_totals,
            pointer_root_selection_diagnostics(
                output,
                known_decoded,
                targets,
                usable_mask,
            ),
        )
        _merge_integer_totals(
            learned_root_totals,
            pointer_root_selection_diagnostics(
                output,
                learned_decoded,
                targets,
                usable_mask,
            ),
        )
        _merge_coverage(coverage_totals, prepared["oracle_coverage"])

        root_map = latent_root_assignment_map(
            [str(key) for key in batch["keys"]],
            targets,
            losses["selected_root_indices"],
            prepared["proposal_query_indices"].to(device),
        )
        _merge_root_assignments(current_root_assignments, root_map)

        batch_size = pixels.size(0)
        samples += batch_size
        for name in LOSS_FIELDS:
            loss_totals[name] += float(losses[name]) * batch_size
        raw_count_infeasible += int(learned_resolution.infeasible.sum())
        raw_count_absolute_error += int(
            (learned_resolution.raw_counts - known_counts).abs().sum()
        )
        feasible_count_absolute_error += int(
            (learned_resolution.feasible_counts - known_counts).abs().sum()
        )
        usable_proposals += int(usable_mask.sum())
        ignored_proposals += int((proposal_mask & ~usable_mask).sum())

    if not samples:
        raise RuntimeError("Proposal Pointer validation loader produced no samples")
    stability = compare_latent_root_assignments(
        previous_root_assignments,
        current_root_assignments,
    )
    metrics = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "samples": samples,
        **{name: value / samples for name, value in loss_totals.items()},
        "known_count_oracle_usable_pool": _finalize_partition_metrics(
            known_totals
        ),
        "learned_count_oracle_usable_pool": _finalize_partition_metrics(
            learned_totals
        ),
        "known_count_root_diagnostics": _finalize_root_diagnostics(
            known_root_totals
        ),
        "learned_count_root_diagnostics": _finalize_root_diagnostics(
            learned_root_totals
        ),
        "learned_count_resolution": {
            "raw_infeasible_samples": raw_count_infeasible,
            "raw_infeasible_rate": raw_count_infeasible / samples,
            "raw_count_mae": raw_count_absolute_error / samples,
            "feasible_count_mae": feasible_count_absolute_error / samples,
        },
        "candidate_pool": {
            "usable_proposals": usable_proposals,
            "ignored_proposals_excluded_from_decode": ignored_proposals,
            "root_mask": "oracle_usable_proposals",
            "member_mask": "oracle_usable_proposals",
        },
        "latent_root_assignment_stability": stability,
        "latent_root_assignment_count": len(current_root_assignments),
        "oracle_coverage": _finalize_coverage(coverage_totals),
        "elapsed_seconds": time.time() - started,
        "uses_internal_test": False,
    }
    return metrics, current_root_assignments


def save_checkpoint(
    path: Path,
    model: BondGroupProposalPointerPredictor,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    samples_seen: int,
    args: argparse.Namespace,
    validation: dict[str, Any] | None,
    primitive_provenance: dict[str, Any],
    pool_contract: dict[str, Any],
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
            "proposal_pointer_config": asdict(model.config),
            "global_step": int(global_step),
            "samples_seen": int(samples_seen),
            "training_args": vars(args),
            "validation": validation,
            "oracle_proposal_pool": pool_contract,
            "source_primitive_checkpoint": primitive_provenance["checkpoint"],
            "source_primitive_checkpoint_sha256": primitive_provenance[
                "checkpoint_sha256"
            ],
            "primitive_provenance": primitive_provenance,
            "frozen_primitive_state_saved": False,
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
    parser.add_argument("--max-members", type=int, default=143)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--root-mode", choices=("hard", "softmin"), default="softmin")
    parser.add_argument("--softmin-temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260722)
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
        args.max_members,
        args.max_steps,
        args.validate_every,
        args.save_every,
    )
    if min(positive) < 1:
        raise ValueError("Positive Proposal Pointer dimensions are required")
    if args.workers < 0:
        raise ValueError("Workers must be nonnegative")
    if args.validation_limit is not None and args.validation_limit < 1:
        raise ValueError("Validation limit must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Learning rate and weight decay are invalid")
    if args.gradient_clip <= 0.0:
        raise ValueError("Gradient clip must be positive")
    if args.softmin_temperature <= 0.0:
        raise ValueError("Softmin temperature must be positive")


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
    config = pointer_config(
        proposal_feature_dim=primitive_model.config.d_model,
        tiny=args.tiny,
        max_members=args.max_members,
        max_group_count=args.max_group_count,
    )
    model = BondGroupProposalPointerPredictor(config).to(device)
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
    pool_contract = oracle_pool_contract(train_loader.collate_fn)
    if pool_contract != oracle_pool_contract(validation_loader.collate_fn):
        raise RuntimeError("Train and validation oracle proposal pools differ")
    write_json(output_dir / "config.json", model.config_dict())
    write_json(output_dir / "training_args.json", vars(args))
    write_json(output_dir / "primitive_source.json", primitive_provenance)
    write_json(output_dir / "oracle_proposal_pool.json", pool_contract)

    global_step = 0
    samples_seen = 0
    best_loss = float("inf")
    last_validation: dict[str, Any] | None = None
    previous_root_assignments: dict[str, int] | None = None
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
            prepared = prepare_pipeline_batch(
                batch,
                flat_output,
                primitive_model,
                config,
                train_loader.collate_fn,
            )
            del flat_output
            model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
            targets = _device_tensor_dict(prepared["targets"], device)
            proposal_mask = model_inputs["proposal_mask"].bool()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(**model_inputs)
                losses = compute_pointer_loss(
                    output,
                    targets,
                    proposal_mask,
                    root_mode=args.root_mode,
                    softmin_temperature=args.softmin_temperature,
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
                last_validation, current_root_assignments = validate(
                    model,
                    primitive_model,
                    validation_loader,
                    device,
                    root_mode=args.root_mode,
                    softmin_temperature=args.softmin_temperature,
                    previous_root_assignments=previous_root_assignments,
                )
                previous_root_assignments = current_root_assignments
                history.append({"global_step": global_step, "metrics": last_validation})
                known = last_validation["known_count_oracle_usable_pool"]
                learned = last_validation["learned_count_oracle_usable_pool"]
                stability = last_validation["latent_root_assignment_stability"]
                print(
                    f"step={global_step} loss={last_validation['loss']:.6f} "
                    f"known_exact={known['sample_exact_rate']:.4f} "
                    f"learned_exact={learned['sample_exact_rate']:.4f} "
                    f"learned_f1={learned['membership_f1']:.4f} "
                    f"root_stability={stability['rate']}",
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
                        pool_contract=pool_contract,
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
                    pool_contract=pool_contract,
                )
            if global_step >= args.max_steps:
                break
        if not batches:
            raise RuntimeError("Proposal Pointer training loader produced no samples")

    if last_validation is None:
        raise RuntimeError("Proposal Pointer training completed without validation")
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        global_step=global_step,
        samples_seen=samples_seen,
        args=args,
        validation=last_validation,
        primitive_provenance=primitive_provenance,
        pool_contract=pool_contract,
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
            "oracle_proposal_pool": pool_contract,
            "validation_scope": (
                "capacity100_gt_associated_oracle_usable_proposal_pool"
            ),
            "frozen_primitive_state_saved": False,
            "optimizer_mode": "fresh",
            "trainable_parameter_count": model.parameter_count(),
            "uses_internal_test": False,
        },
    )


if __name__ == "__main__":
    main()
