#!/usr/bin/env python3
"""Run the train-only G3 structural P0 overfit32 capacity gate."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from chemvector.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3AtomPairBondIdentityPredictor,
    G3FixedLearnedBondSlotPredictor,
    G3HierarchicalGraphConfig,
    G3HierarchicalGraphPredictor,
    G3SetLearnedBondSlotPredictor,
    build_g3_bond_targets,
    build_g3_atom_pair_targets,
    g3_atom_only_loss,
    g3_canonical_atom_only_loss,
    g3_atom_pair_bond_loss,
    g3_confidence_aware_bond_assignments,
    g3_fixed_bond_slot_loss,
    g3_graph_loss,
    g3_set_bond_slot_loss,
    gt_bond_slot_assignments,
    select_g3_atom_pair_bonds,
)
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    file_sha256,
    g3_targets_to_device,
    load_g3_manifest_samples,
    load_g3_overfit32_manifest,
)
from chemvector.model import ChemVectorConfig, ChemVisionEncoder


DEFAULT_VISION_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v4-clean-50k/last.pt"
)
DEFAULT_VISION_CHECKPOINT_SHA256 = (
    "65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095"
)
DEFAULT_OUTPUT = "outputs/smoke/chemvector-g3-p0-overfit32"
GT_ATOM_SLOT_ORACLE_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0b-gt-atom-slot-oracle"
)
SHARED_BOND_ASSIGNMENT_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0c-shared-bond-assignment"
)
GT_BOND_SLOT_IDENTITY_ORACLE_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0d-gt-bond-slot-identity-oracle"
)
FIXED_LEARNED_BOND_SLOTS_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0e-fixed-learned-bond-slots"
)
SET_LEARNED_BOND_SLOTS_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0f-set-learned-bond-slots"
)
GT_ATOM_PAIR_BOND_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0g-gt-atom-pair-bond-capacity"
)
ATOM_ONLY_CAPACITY_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0h-atom-only-overfit32"
)
CANONICAL_CELL_SLOT_OUTPUT = (
    "outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32"
)


def _vision_config(payload: dict) -> ChemVectorConfig:
    valid = {field.name for field in fields(ChemVectorConfig)}
    config_payload = payload.get("vision_config", payload.get("config"))
    if not isinstance(config_payload, dict):
        raise ValueError("G3 vision checkpoint has no vision_config")
    return ChemVectorConfig(
        **{name: value for name, value in config_payload.items() if name in valid}
    )


def load_frozen_vision_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[ChemVisionEncoder, ChemVectorConfig, dict, str]:
    """Load only the committed primitive-set vision backbone."""

    path = Path(path)
    checkpoint_sha256 = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_type = payload.get("model_type")
    if model_type not in {None, "primitive-set-v0.1"}:
        raise ValueError("G3 vision source uses an unsupported checkpoint type")
    config = _vision_config(payload)
    vision = ChemVisionEncoder(config)
    state = {
        name.removeprefix("vision_encoder."): value
        for name, value in payload.get("model", {}).items()
        if name.startswith("vision_encoder.")
    }
    if not state:
        raise ValueError("G3 vision checkpoint contains no vision_encoder state")
    vision.load_state_dict(state, strict=True)
    vision.requires_grad_(False)
    vision.eval()
    return vision.to(device), config, payload, checkpoint_sha256


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"Unsupported G3 training precision: {precision}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def module_state_sha256(module: nn.Module) -> str:
    """Hash tensor names, metadata, and bytes in deterministic state order."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name]
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_p0h_atom_only_model(
    config: G3HierarchicalGraphConfig,
) -> tuple[G3AtomOnlyPredictor, dict[str, str | bool]]:
    """Reproduce and verify the Atom initialization of the original P0 model."""

    rng_state = torch.random.get_rng_state()
    try:
        reference = G3HierarchicalGraphPredictor(config)
        reference_sha256 = module_state_sha256(reference.atom_head)
        del reference
    finally:
        torch.random.set_rng_state(rng_state)
    model = G3AtomOnlyPredictor(config)
    candidate_sha256 = module_state_sha256(model.atom_head)
    if candidate_sha256 != reference_sha256:
        raise RuntimeError("G3 P0-H Atom initialization differs from original P0")
    return model, {
        "reference_model": G3HierarchicalGraphPredictor.model_type,
        "reference_atom_sha256": reference_sha256,
        "candidate_atom_sha256": candidate_sha256,
        "equal": True,
    }


@torch.no_grad()
def cache_frozen_vision_features(
    vision: nn.Module,
    pixel_values: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    precision: str,
) -> torch.Tensor:
    """Compute the immutable stride-4 feature cache used by every P0 step."""

    if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
        raise ValueError("G3 P0 pixels must have shape [batch, 3, h, w]")
    if batch_size < 1:
        raise ValueError("G3 vision cache batch size must be positive")
    vision.eval()
    features = []
    for start in range(0, len(pixel_values), batch_size):
        pixels = pixel_values[start : start + batch_size].to(
            device, non_blocking=True
        )
        with _autocast_context(device, precision):
            current = vision.forward_stroke_resolution(pixels)
        features.append(current.detach())
    if not features:
        raise ValueError("G3 vision cache requires at least one image")
    return torch.cat(features, dim=0)


def cyclic_batch_indices(
    step: int,
    *,
    sample_count: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the no-shuffle manifest-order batch for a one-indexed step."""

    if step < 1 or min(sample_count, batch_size) < 1:
        raise ValueError("G3 cyclic batch dimensions must be positive")
    start = ((step - 1) * batch_size) % sample_count
    return torch.tensor(
        [(start + offset) % sample_count for offset in range(batch_size)],
        device=device,
        dtype=torch.long,
    )


def g3_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    precision: str = "fp32",
    max_grad_norm: float = 1.0,
    gt_atom_slot_oracle: bool = False,
    shared_confidence_aware_bond_assignment: bool = False,
    gt_bond_slot_identity_oracle: bool = False,
    fixed_learned_bond_slots: bool = False,
    set_learned_bond_slots: bool = False,
    direct_gt_atom_pair_bonds: bool = False,
) -> dict[str, float]:
    """Run one structural step; only object counts are teacher-forced."""

    if max_grad_norm <= 0.0:
        raise ValueError("G3 max_grad_norm must be positive")
    if fixed_learned_bond_slots and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-E fixed Bond slots requires GT Atom-slot oracle")
    if fixed_learned_bond_slots and (
        shared_confidence_aware_bond_assignment
        or gt_bond_slot_identity_oracle
        or set_learned_bond_slots
        or direct_gt_atom_pair_bonds
    ):
        raise ValueError("G3 P0-E is mutually exclusive with other Bond protocols")
    if set_learned_bond_slots and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-F set Bond slots requires GT Atom-slot oracle")
    if set_learned_bond_slots and (
        shared_confidence_aware_bond_assignment
        or gt_bond_slot_identity_oracle
        or direct_gt_atom_pair_bonds
    ):
        raise ValueError("G3 P0-F is mutually exclusive with other Bond protocols")
    if direct_gt_atom_pair_bonds and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-G AtomPair Bonds require GT Atom-slot oracle")
    if direct_gt_atom_pair_bonds and (
        shared_confidence_aware_bond_assignment or gt_bond_slot_identity_oracle
    ):
        raise ValueError("G3 P0-G is mutually exclusive with P0-C and P0-D")
    atom_counts = targets["atom_mask"].sum(dim=1).long()
    bond_counts = targets["bond_mask"].sum(dim=1).long()
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(features.device, precision):
        normalized_bonds = build_g3_bond_targets(targets)
        model_kwargs = {"atom_counts": atom_counts}
        if not direct_gt_atom_pair_bonds:
            model_kwargs["bond_counts"] = bond_counts
        if gt_atom_slot_oracle:
            model_kwargs.update(
                {
                    "oracle_atom_anchors": targets["atom_anchors"],
                    "oracle_atom_mask": targets["atom_mask"],
                }
            )
        if gt_bond_slot_identity_oracle:
            model_kwargs.update(
                {
                    "oracle_bond_endpoints": normalized_bonds["endpoints"],
                    "oracle_bond_mask": normalized_bonds["mask"],
                }
            )
        output = model(features, **model_kwargs)
        if direct_gt_atom_pair_bonds:
            losses = g3_atom_pair_bond_loss(
                output,
                targets,
                normalized_bond_targets=normalized_bonds,
            )
        elif set_learned_bond_slots:
            losses = g3_set_bond_slot_loss(
                output,
                targets,
                normalized_bond_targets=normalized_bonds,
            )
        elif fixed_learned_bond_slots:
            losses = g3_fixed_bond_slot_loss(
                output,
                targets,
                normalized_bond_targets=normalized_bonds,
            )
        else:
            losses = g3_graph_loss(
                output,
                targets,
                shared_confidence_aware_bond_assignment=(
                    shared_confidence_aware_bond_assignment
                ),
                gt_bond_slot_identity_oracle=gt_bond_slot_identity_oracle,
                normalized_bond_targets=normalized_bonds,
            )
    losses["loss"].backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("G3 gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def validate_atom_only_optimizer(
    model: G3AtomOnlyPredictor,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = [parameter for parameter in model.atom_head.parameters()]
    actual = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(actual) != len(expected) or {id(value) for value in actual} != {
        id(value) for value in expected
    }:
        raise ValueError("G3 P0-H optimizer must contain only Atom-head parameters")


def g3_atom_only_training_step(
    model: G3AtomOnlyPredictor,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    precision: str = "fp32",
    max_grad_norm: float = 1.0,
    canonical_cell_slot_targets: bool = False,
) -> dict[str, float]:
    """Run one P0-H step without count teacher forcing or Bond target access."""

    if max_grad_norm <= 0.0:
        raise ValueError("G3 P0-H max_grad_norm must be positive")
    validate_atom_only_optimizer(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(features.device, precision):
        output = model(features)
        losses = (
            g3_canonical_atom_only_loss(output, targets)
            if canonical_cell_slot_targets
            else g3_atom_only_loss(output, targets)
        )
    losses["loss"].backward()
    trainable = list(model.atom_head.parameters())
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("G3 P0-H gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
    }


def maximum_cardinality_radius_match_count(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    *,
    radius_pixels: float,
    image_size: float,
) -> int:
    """Maximize one-to-one in-radius matches before minimizing distance."""

    if predicted.ndim != 2 or expected.ndim != 2:
        raise ValueError("G3 Atom matching requires rank-two coordinate tensors")
    if predicted.shape[-1] != 2 or expected.shape[-1] != 2:
        raise ValueError("G3 Atom matching coordinates must have two components")
    if min(radius_pixels, image_size) <= 0.0:
        raise ValueError("G3 Atom matching dimensions must be positive")
    if not len(predicted) or not len(expected):
        return 0
    distances = torch.cdist(expected.float(), predicted.float()) * image_size
    eligible_columns = (distances <= radius_pixels).any(dim=0)
    eligible = distances[:, eligible_columns]
    target_count = len(expected)
    unmatched_cost = float(target_count + 1)
    real_cost = torch.where(
        eligible <= radius_pixels,
        eligible / radius_pixels,
        torch.full_like(eligible, 2.0 * unmatched_cost),
    )
    dummy_cost = torch.full(
        (target_count, target_count),
        unmatched_cost,
        device=distances.device,
        dtype=distances.dtype,
    )
    cost = torch.cat((real_cost, dummy_cost), dim=1)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    real_columns = eligible.size(1)
    return int(
        sum(
            column < real_columns and float(eligible[row, column]) <= radius_pixels
            for row, column in zip(rows, columns)
        )
    )


def _atom_only_batch_statistics(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    image_size: float,
    atom_radius_pixels: float,
) -> dict[str, float]:
    totals = {
        "images": 0.0,
        "atom_count_exact": 0.0,
        "atom_set_exact_6px": 0.0,
        "atom_count_absolute_error": 0.0,
    }
    for batch_index in range(output["atom_mask"].size(0)):
        predicted = torch.where(output["atom_mask"][batch_index])[0]
        expected = torch.where(targets["atom_mask"][batch_index].bool())[0]
        count_exact = len(predicted) == len(expected)
        matched = maximum_cardinality_radius_match_count(
            output["atom_anchors"][batch_index, predicted],
            targets["atom_anchors"][batch_index, expected],
            radius_pixels=atom_radius_pixels,
            image_size=image_size,
        )
        totals["images"] += 1.0
        totals["atom_count_exact"] += float(count_exact)
        totals["atom_set_exact_6px"] += float(count_exact and matched == len(expected))
        totals["atom_count_absolute_error"] += abs(len(predicted) - len(expected))
    return totals


@torch.no_grad()
def evaluate_g3_atom_only_overfit(
    model: G3AtomOnlyPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
    count_mode: str,
    image_size: float = 768.0,
    atom_radius_pixels: float = 6.0,
) -> dict[str, float | int | str]:
    if count_mode not in {"gt_count", "deploy_count"}:
        raise ValueError(f"Unknown G3 P0-H count mode: {count_mode}")
    if batch_size < 1 or min(image_size, atom_radius_pixels) <= 0.0:
        raise ValueError("G3 P0-H evaluation dimensions must be positive")
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        atom_targets = {
            "atom_anchors": targets["atom_anchors"][start : start + batch_size],
            "atom_mask": targets["atom_mask"][start : start + batch_size],
        }
        forced_counts = (
            atom_targets["atom_mask"].sum(dim=1).long()
            if count_mode == "gt_count"
            else None
        )
        with _autocast_context(features.device, precision):
            output = model(current_features, atom_counts=forced_counts)
        current = _atom_only_batch_statistics(
            output,
            atom_targets,
            image_size=image_size,
            atom_radius_pixels=atom_radius_pixels,
        )
        for name, value in current.items():
            totals[name] = totals.get(name, 0.0) + value
    if was_training:
        model.train()
    images = max(totals.get("images", 0.0), 1.0)
    return {
        "count_mode": count_mode,
        "images": int(totals.get("images", 0.0)),
        "atom_count_exact": totals.get("atom_count_exact", 0.0) / images,
        "atom_count_exact_images": int(totals.get("atom_count_exact", 0.0)),
        "atom_set_exact_6px": totals.get("atom_set_exact_6px", 0.0) / images,
        "atom_set_exact_6px_images": int(
            totals.get("atom_set_exact_6px", 0.0)
        ),
        "atom_count_mae": totals.get("atom_count_absolute_error", 0.0) / images,
    }


def g3_p0h_atom_only_gate(metrics: dict, *, expected_images: int = 32) -> dict:
    gt_count = metrics["gt_count"]
    deploy = metrics["deploy_count"]
    checks = {
        "image_count": (
            gt_count["images"] == expected_images
            and deploy["images"] == expected_images
        ),
        "gt_atom_count_exact": gt_count["atom_count_exact_images"] == expected_images,
        "gt_atom_set_exact_6px": gt_count["atom_set_exact_6px_images"] >= 30,
        "deploy_atom_count_exact": deploy["atom_count_exact_images"] >= 30,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "expected_images": expected_images,
        "minimum_exact_images": 30,
        "checks": checks,
        "fixed_final_step_only": True,
    }


def evaluate_g3_atom_only_count_modes(
    model: G3AtomOnlyPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
    canonical_cell_slot_targets: bool = False,
) -> dict:
    metrics = {
        "scope": (
            "train-only-overfit32-g3-p0i-canonical-cell-slot-targets"
            if canonical_cell_slot_targets
            else "train-only-overfit32-g3-p0h-atom-only-capacity"
        ),
        "atom_slot_mode": "predicted",
        "atom_target_mode": (
            "canonical_cell_slot"
            if canonical_cell_slot_targets
            else "dynamic_predicted_anchor_hungarian"
        ),
        "training_count_teacher_forcing": False,
        "gt_count": evaluate_g3_atom_only_overfit(
            model,
            features,
            targets,
            batch_size=batch_size,
            precision=precision,
            count_mode="gt_count",
        ),
        "deploy_count": evaluate_g3_atom_only_overfit(
            model,
            features,
            targets,
            batch_size=batch_size,
            precision=precision,
            count_mode="deploy_count",
        ),
        "sdf_valid": {
            "status": "not_applicable",
            "reason": (
                "P0-I predicts only Atom anchors and count"
                if canonical_cell_slot_targets
                else "P0-H predicts only Atom anchors and count"
            ),
        },
    }
    gate_name = "p0i_gate" if canonical_cell_slot_targets else "p0h_gate"
    metrics[gate_name] = g3_p0h_atom_only_gate(metrics)
    return metrics


def _structural_batch_statistics(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    image_size: float,
    atom_radius_pixels: float,
    bond_assignments: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, float]:
    totals = {
        "images": 0.0,
        "atom_count_exact": 0.0,
        "bond_count_exact": 0.0,
        "count_pair_exact": 0.0,
        "atom_set_exact_6px": 0.0,
        "endpoint_pointer_topology_exact": 0.0,
        "bond_type_exact": 0.0,
        "structural_graph_exact": 0.0,
        "atom_count_absolute_error": 0.0,
        "bond_count_absolute_error": 0.0,
    }
    if bond_assignments is not None:
        if len(bond_assignments) != output["atom_mask"].size(0):
            raise ValueError("G3 evaluator Bond assignments have an incompatible batch")
        totals.update(
            {
                "bond_assignment_intersection_queries": 0.0,
                "bond_assignment_selected_queries": 0.0,
                "bond_assignment_matched_queries": 0.0,
                "bond_assignment_union_queries": 0.0,
                "bond_assignment_selected_only_queries": 0.0,
                "bond_assignment_matched_only_queries": 0.0,
                "bond_assignment_query_set_exact": 0.0,
                "bond_assignment_presence_cutoff_tie_images": 0.0,
            }
        )
    query_output = output.get("bond_query_output", {})
    oracle_coverage = query_output.get("oracle_full_coverage")
    if oracle_coverage is not None:
        totals.update(
            {
                "bond_oracle_full_coverage_images": 0.0,
                "bond_oracle_matched_targets": 0.0,
                "bond_oracle_target_count": 0.0,
            }
        )
    for batch_index in range(output["atom_mask"].size(0)):
        predicted_atoms = torch.where(output["atom_mask"][batch_index])[0]
        expected_atoms = torch.where(targets["atom_mask"][batch_index])[0]
        predicted_bonds = torch.where(output["bond_mask"][batch_index])[0]
        expected_bonds = torch.where(targets["bond_mask"][batch_index])[0]
        atom_count_exact = len(predicted_atoms) == len(expected_atoms)
        bond_count_exact = len(predicted_bonds) == len(expected_bonds)
        totals["images"] += 1.0
        totals["atom_count_exact"] += float(atom_count_exact)
        totals["bond_count_exact"] += float(bond_count_exact)
        totals["count_pair_exact"] += float(atom_count_exact and bond_count_exact)
        totals["atom_count_absolute_error"] += abs(
            len(predicted_atoms) - len(expected_atoms)
        )
        totals["bond_count_absolute_error"] += abs(
            len(predicted_bonds) - len(expected_bonds)
        )
        if bond_assignments is not None:
            assigned_queries = set(bond_assignments[batch_index][0].tolist())
            selected_queries = set(predicted_bonds.tolist())
            intersection = assigned_queries & selected_queries
            union = assigned_queries | selected_queries
            totals["bond_assignment_intersection_queries"] += len(intersection)
            totals["bond_assignment_selected_queries"] += len(selected_queries)
            totals["bond_assignment_matched_queries"] += len(assigned_queries)
            totals["bond_assignment_union_queries"] += len(union)
            totals["bond_assignment_selected_only_queries"] += len(
                selected_queries - assigned_queries
            )
            totals["bond_assignment_matched_only_queries"] += len(
                assigned_queries - selected_queries
            )
            totals["bond_assignment_query_set_exact"] += float(
                assigned_queries == selected_queries
            )
            if len(selected_queries):
                valid = output["bond_query_output"]["proposal_mask"][batch_index]
                logits = output["bond_presence_logits"][batch_index, valid].float()
                count = len(selected_queries)
                if count < len(logits):
                    cutoff = logits.topk(count).values[-1]
                    above = int((logits > cutoff).sum())
                    equal = int((logits == cutoff).sum())
                    totals["bond_assignment_presence_cutoff_tie_images"] += float(
                        above < count < above + equal
                    )
        if oracle_coverage is not None:
            totals["bond_oracle_full_coverage_images"] += float(
                bool(oracle_coverage[batch_index])
            )
            totals["bond_oracle_matched_targets"] += int(
                query_output["oracle_matched_target_count"][batch_index]
            )
            totals["bond_oracle_target_count"] += int(
                query_output["oracle_target_count"][batch_index]
            )

        predicted_to_target: dict[int, int] = {}
        target_identity = output.get("atom_target_indices")
        if target_identity is not None:
            for query in predicted_atoms.tolist():
                target = int(target_identity[batch_index, query])
                if target < 0 or target not in expected_atoms.tolist():
                    continue
                distance = torch.linalg.vector_norm(
                    output["atom_anchors"][batch_index, query].float()
                    - targets["atom_anchors"][batch_index, target].float()
                ) * image_size
                if float(distance) <= atom_radius_pixels:
                    predicted_to_target[query] = target
        elif len(predicted_atoms) and len(expected_atoms):
            distances = torch.cdist(
                output["atom_anchors"][batch_index, predicted_atoms].float(),
                targets["atom_anchors"][batch_index, expected_atoms].float(),
            ) * image_size
            rows, columns = linear_sum_assignment(distances.detach().cpu().numpy())
            for row, column in zip(rows, columns):
                if float(distances[row, column]) <= atom_radius_pixels:
                    predicted_to_target[int(predicted_atoms[row])] = int(
                        expected_atoms[column]
                    )
        atom_set_exact = atom_count_exact and len(predicted_to_target) == len(
            expected_atoms
        )
        totals["atom_set_exact_6px"] += float(atom_set_exact)

        expected_graph = {}
        for bond_index in expected_bonds.tolist():
            begin, end = targets["bond_atom_refs"][batch_index, bond_index].tolist()
            pair = tuple(sorted((int(begin), int(end))))
            expected_graph[pair] = int(targets["bond_types"][batch_index, bond_index])

        predicted_graph = {}
        valid_predicted_graph = atom_set_exact
        for bond_index in predicted_bonds.tolist():
            atom_queries = output["endpoint_pointer_logits"][
                batch_index, bond_index
            ].argmax(dim=-1)
            if any(int(query) not in predicted_to_target for query in atom_queries):
                valid_predicted_graph = False
                continue
            pair = tuple(
                sorted(predicted_to_target[int(query)] for query in atom_queries)
            )
            if pair[0] == pair[1] or pair in predicted_graph:
                valid_predicted_graph = False
                continue
            predicted_graph[pair] = int(
                output["bond_type_logits"][batch_index, bond_index].argmax()
            )
        topology_exact = (
            valid_predicted_graph
            and bond_count_exact
            and set(predicted_graph) == set(expected_graph)
        )
        type_exact = topology_exact and predicted_graph == expected_graph
        totals["endpoint_pointer_topology_exact"] += float(topology_exact)
        totals["bond_type_exact"] += float(type_exact)
        totals["structural_graph_exact"] += float(atom_set_exact and type_exact)
    return totals


def _atom_pair_batch_statistics(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
    normalized_bonds: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Report P0-G pair coverage, selected-edge quality, and positive type micro."""

    pair_targets = build_g3_atom_pair_targets(
        output,
        targets,
        normalized_bond_targets=normalized_bonds,
    )
    positives = pair_targets["presence"].bool()
    positive_types = output["flat_atom_pair_output"]["chemical_logits"][
        positives
    ].argmax(dim=-1)
    expected_types = pair_targets["chemical"][positives]
    selected = true_positive = 0
    for batch_index in range(targets["bond_mask"].size(0)):
        target_count = int(targets["bond_mask"][batch_index].sum())
        target_pairs = {
            tuple(sorted((int(refs[0]), int(refs[1]))))
            for refs in targets["bond_atom_refs"][
                batch_index, :target_count
            ].tolist()
        }
        selected_pairs = {
            tuple(int(value) for value in pair)
            for pair in output["atom_pair_indices"][batch_index][
                output["bond_mask"][batch_index]
            ].tolist()
        }
        selected += len(selected_pairs)
        true_positive += len(selected_pairs & target_pairs)
    return {
        "atom_pair_universe_pairs": float(
            output["flat_atom_pair_indices"].size(0)
        ),
        "atom_pair_target_bonds": float(pair_targets["target_bonds"]),
        "atom_pair_covered_bonds": float(pair_targets["covered_bonds"]),
        "atom_pair_selected_pairs": float(selected),
        "atom_pair_selected_true_positive": float(true_positive),
        "atom_pair_positive_type_correct": float(
            (positive_types == expected_types).sum()
        ),
    }


@torch.no_grad()
def evaluate_g3_structural_overfit(
    model: nn.Module,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
    count_mode: str,
    image_size: float = 768.0,
    atom_radius_pixels: float = 6.0,
    gt_atom_slot_oracle: bool = False,
    shared_confidence_aware_bond_assignment: bool = False,
    gt_bond_slot_identity_oracle: bool = False,
    fixed_learned_bond_slots: bool = False,
    set_learned_bond_slots: bool = False,
    direct_gt_atom_pair_bonds: bool = False,
) -> dict[str, float | int | str]:
    """Evaluate only the committed training images, without repair or SDF semantics."""

    if count_mode not in {"gt_count", "deploy_count"}:
        raise ValueError(f"Unknown G3 count evaluation mode: {count_mode}")
    if batch_size < 1 or min(image_size, atom_radius_pixels) <= 0.0:
        raise ValueError("G3 structural evaluation dimensions must be positive")
    if fixed_learned_bond_slots and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-E fixed Bond slots requires GT Atom-slot oracle")
    if set_learned_bond_slots and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-F set Bond slots requires GT Atom-slot oracle")
    if direct_gt_atom_pair_bonds and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-G AtomPair Bonds require GT Atom-slot oracle")
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size]
            for name, value in targets.items()
        }
        forced = count_mode == "gt_count"
        normalized_bonds = build_g3_bond_targets(current_targets)
        with _autocast_context(features.device, precision):
            model_kwargs = {
                "atom_counts": (
                    current_targets["atom_mask"].sum(dim=1).long()
                    if forced
                    else None
                ),
            }
            if not direct_gt_atom_pair_bonds:
                model_kwargs["bond_counts"] = (
                    current_targets["bond_mask"].sum(dim=1).long()
                    if forced
                    else None
                )
            if gt_atom_slot_oracle:
                model_kwargs.update(
                    {
                        "oracle_atom_anchors": current_targets["atom_anchors"],
                        "oracle_atom_mask": current_targets["atom_mask"],
                    }
                )
            if gt_bond_slot_identity_oracle:
                model_kwargs.update(
                    {
                        "oracle_bond_endpoints": normalized_bonds["endpoints"],
                        "oracle_bond_mask": normalized_bonds["mask"],
                    }
                )
            output = model(current_features, **model_kwargs)
            if direct_gt_atom_pair_bonds:
                output = select_g3_atom_pair_bonds(
                    output,
                    (
                        current_targets["bond_mask"].sum(dim=1).long()
                        if forced
                        else output["bond_counts"]
                    ),
                )
        bond_assignments = None
        if gt_bond_slot_identity_oracle:
            bond_assignments = gt_bond_slot_assignments(output, normalized_bonds)
        elif shared_confidence_aware_bond_assignment:
            bond_assignments = g3_confidence_aware_bond_assignments(
                output,
                normalized_bonds,
            )
        current = _structural_batch_statistics(
            output,
            current_targets,
            image_size=image_size,
            atom_radius_pixels=atom_radius_pixels,
            bond_assignments=bond_assignments,
        )
        if direct_gt_atom_pair_bonds:
            pair_statistics = _atom_pair_batch_statistics(
                output,
                current_targets,
                normalized_bonds,
            )
            current.update(pair_statistics)
        for name, value in current.items():
            totals[name] = totals.get(name, 0.0) + value
    if was_training:
        model.train()
    images = max(totals.get("images", 0.0), 1.0)
    result = {
        "count_mode": count_mode,
        "images": int(totals.get("images", 0.0)),
        "atom_count_exact": totals.get("atom_count_exact", 0.0) / images,
        "atom_count_exact_images": int(totals.get("atom_count_exact", 0.0)),
        "bond_count_exact": totals.get("bond_count_exact", 0.0) / images,
        "bond_count_exact_images": int(totals.get("bond_count_exact", 0.0)),
        "count_pair_exact": totals.get("count_pair_exact", 0.0) / images,
        "count_pair_exact_images": int(totals.get("count_pair_exact", 0.0)),
        "atom_set_exact_6px": totals.get("atom_set_exact_6px", 0.0) / images,
        "atom_set_exact_6px_images": int(
            totals.get("atom_set_exact_6px", 0.0)
        ),
        "endpoint_pointer_topology_exact": totals.get(
            "endpoint_pointer_topology_exact", 0.0
        )
        / images,
        "endpoint_pointer_topology_exact_images": int(
            totals.get("endpoint_pointer_topology_exact", 0.0)
        ),
        "bond_type_exact": totals.get("bond_type_exact", 0.0) / images,
        "bond_type_exact_images": int(totals.get("bond_type_exact", 0.0)),
        "structural_graph_exact": totals.get("structural_graph_exact", 0.0)
        / images,
        "structural_graph_exact_images": int(
            totals.get("structural_graph_exact", 0.0)
        ),
        "atom_count_mae": totals.get("atom_count_absolute_error", 0.0) / images,
        "bond_count_mae": totals.get("bond_count_absolute_error", 0.0) / images,
    }
    if shared_confidence_aware_bond_assignment:
        intersection = totals.get("bond_assignment_intersection_queries", 0.0)
        selected_total = totals.get("bond_assignment_selected_queries", 0.0)
        matched_total = totals.get("bond_assignment_matched_queries", 0.0)
        union_total = totals.get("bond_assignment_union_queries", 0.0)
        precision = intersection / max(selected_total, 1.0)
        recall = intersection / max(matched_total, 1.0)
        result.update(
            {
                "bond_assignment_mode": "shared_confidence_aware",
                "bond_assignment_presence_cost_weight": 2.0,
                "bond_assignment_intersection_queries": int(intersection),
                "bond_assignment_deploy_selected_queries": int(selected_total),
                "bond_assignment_shared_matched_queries": int(matched_total),
                "bond_assignment_selected_only_queries": int(
                    totals.get("bond_assignment_selected_only_queries", 0.0)
                ),
                "bond_assignment_matched_only_queries": int(
                    totals.get("bond_assignment_matched_only_queries", 0.0)
                ),
                "bond_assignment_union_queries": int(union_total),
                "bond_assignment_query_overlap_micro": recall,
                "bond_assignment_precision_micro": precision,
                "bond_assignment_recall_micro": recall,
                "bond_assignment_f1_micro": (
                    2.0 * precision * recall / max(precision + recall, 1e-12)
                ),
                "bond_assignment_jaccard_micro": intersection
                / max(union_total, 1.0),
                "bond_assignment_query_set_exact": totals.get(
                    "bond_assignment_query_set_exact", 0.0
                )
                / images,
                "bond_assignment_query_set_exact_images": int(
                    totals.get("bond_assignment_query_set_exact", 0.0)
                ),
                "bond_assignment_presence_cutoff_tie_images": int(
                    totals.get("bond_assignment_presence_cutoff_tie_images", 0.0)
                ),
            }
        )
    if gt_bond_slot_identity_oracle:
        result.update(
            {
                "bond_assignment_mode": "gt_visual_endpoint_router_prefix_identity",
                "bond_oracle_full_coverage": (
                    totals.get("bond_oracle_full_coverage_images", 0.0) / images
                ),
                "bond_oracle_full_coverage_images": int(
                    totals.get("bond_oracle_full_coverage_images", 0.0)
                ),
                "bond_oracle_matched_targets": int(
                    totals.get("bond_oracle_matched_targets", 0.0)
                ),
                "bond_oracle_target_count": int(
                    totals.get("bond_oracle_target_count", 0.0)
                ),
            }
        )
    if direct_gt_atom_pair_bonds:
        target_bonds = totals.get("atom_pair_target_bonds", 0.0)
        selected_pairs = totals.get("atom_pair_selected_pairs", 0.0)
        true_positive = totals.get("atom_pair_selected_true_positive", 0.0)
        precision = true_positive / max(selected_pairs, 1.0)
        recall = true_positive / max(target_bonds, 1.0)
        result.update(
            {
                "atom_pair_universe_pairs": int(
                    totals.get("atom_pair_universe_pairs", 0.0)
                ),
                "atom_pair_target_bonds": int(target_bonds),
                "atom_pair_covered_bonds": int(
                    totals.get("atom_pair_covered_bonds", 0.0)
                ),
                "atom_pair_target_coverage": totals.get(
                    "atom_pair_covered_bonds", 0.0
                )
                / max(target_bonds, 1.0),
                "atom_pair_presence_precision": precision,
                "atom_pair_presence_recall": recall,
                "atom_pair_presence_f1": (
                    2.0 * precision * recall / max(precision + recall, 1e-12)
                ),
                "atom_pair_positive_type_accuracy": totals.get(
                    "atom_pair_positive_type_correct", 0.0
                )
                / max(target_bonds, 1.0),
            }
        )
    return result


def g3_p0_gate_decision(metrics: dict, *, expected_images: int = 32) -> dict:
    """Apply the pre-registered structural P0 gates without checkpoint selection."""

    deploy = metrics["deploy_count"]
    gt_count = metrics["gt_count"]
    checks = {
        "image_count": (
            deploy["images"] == expected_images
            and gt_count["images"] == expected_images
        ),
        "deploy_count_pair_exact": deploy["count_pair_exact_images"] >= 30,
        "gt_atom_set_exact_6px": gt_count["atom_set_exact_6px_images"] >= 30,
        "gt_endpoint_pointer_topology_exact": (
            gt_count["endpoint_pointer_topology_exact_images"] >= 30
        ),
        "gt_bond_type_exact": gt_count["bond_type_exact_images"] >= 30,
        "gt_structural_graph_exact": (
            gt_count["structural_graph_exact_images"] >= 30
        ),
    }
    if metrics.get("bond_assignment_mode") == "shared_confidence_aware":
        checks["gt_atom_set_exact_6px"] = (
            gt_count["atom_set_exact_6px_images"] == expected_images
        )
        checks.update(
            {
                "deploy_shared_assignment_overlap_micro": (
                    deploy["bond_assignment_query_overlap_micro"] >= 0.95
                ),
                "deploy_shared_assignment_query_set_exact": (
                    deploy["bond_assignment_query_set_exact_images"] >= 30
                ),
            }
        )
    elif metrics.get("bond_assignment_mode") == "gt_visual_endpoint_router_prefix_identity":
        expected_router_targets = (
            1508
            if expected_images == 32
            else gt_count["bond_oracle_target_count"]
        )
        checks["gt_atom_set_exact_6px"] = (
            gt_count["atom_set_exact_6px_images"] == expected_images
        )
        checks.update(
            {
                "gt_bond_router_full_coverage": (
                    gt_count["bond_oracle_full_coverage_images"] == expected_images
                ),
                "gt_bond_router_target_coverage": (
                    gt_count["bond_oracle_matched_targets"]
                    == gt_count["bond_oracle_target_count"]
                    == expected_router_targets
                ),
            }
        )
    elif metrics.get("bond_assignment_mode") == "direct_gt_atom_pair_identity":
        checks["gt_atom_set_exact_6px"] = (
            gt_count["atom_set_exact_6px_images"] == expected_images
        )
        checks.update(
            {
                "gt_atom_pair_target_coverage": (
                    gt_count["atom_pair_target_bonds"]
                    == gt_count["atom_pair_covered_bonds"]
                    == 1508
                ),
                "gt_atom_pair_positive_type_micro": (
                    gt_count["atom_pair_positive_type_accuracy"] >= 0.99
                ),
                "deploy_endpoint_pointer_topology_exact": (
                    deploy["endpoint_pointer_topology_exact_images"] >= 30
                ),
                "deploy_bond_type_exact": deploy["bond_type_exact_images"] >= 30,
                "deploy_structural_graph_exact": (
                    deploy["structural_graph_exact_images"] >= 30
                ),
            }
        )
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "expected_images": expected_images,
        "minimum_exact_images": 30,
        "checks": checks,
        "fixed_final_step_only": True,
    }


def evaluate_g3_count_modes(
    model: nn.Module,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    precision: str,
    gt_atom_slot_oracle: bool = False,
    shared_confidence_aware_bond_assignment: bool = False,
    gt_bond_slot_identity_oracle: bool = False,
    fixed_learned_bond_slots: bool = False,
    set_learned_bond_slots: bool = False,
    direct_gt_atom_pair_bonds: bool = False,
) -> dict:
    if shared_confidence_aware_bond_assignment and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-C shared Bond assignment requires GT Atom-slot oracle")
    if gt_bond_slot_identity_oracle and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-D GT Bond-slot oracle requires GT Atom-slot oracle")
    if shared_confidence_aware_bond_assignment and gt_bond_slot_identity_oracle:
        raise ValueError("G3 P0-C and P0-D Bond identity protocols are mutually exclusive")
    if fixed_learned_bond_slots and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-E fixed Bond slots requires GT Atom-slot oracle")
    if set_learned_bond_slots and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-F set Bond slots requires GT Atom-slot oracle")
    if direct_gt_atom_pair_bonds and not gt_atom_slot_oracle:
        raise ValueError("G3 P0-G AtomPair Bonds require GT Atom-slot oracle")
    if fixed_learned_bond_slots and (
        shared_confidence_aware_bond_assignment or gt_bond_slot_identity_oracle
    ):
        raise ValueError("G3 P0-E is mutually exclusive with P0-C and P0-D")
    if direct_gt_atom_pair_bonds:
        scope = "train-only-overfit32-g3-p0g-gt-atom-pair-bond-capacity"
    elif set_learned_bond_slots:
        scope = "train-only-overfit32-g3-p0f-set-learned-bond-slots"
    elif fixed_learned_bond_slots:
        scope = "train-only-overfit32-g3-p0e-fixed-learned-bond-slots"
    elif gt_bond_slot_identity_oracle:
        scope = "train-only-overfit32-g3-p0d-gt-bond-slot-identity-oracle"
    elif shared_confidence_aware_bond_assignment:
        scope = "train-only-overfit32-g3-p0c-shared-bond-assignment"
    elif gt_atom_slot_oracle:
        scope = "train-only-overfit32-g3-p0b-gt-atom-slot-oracle"
    else:
        scope = "train-only-overfit32-structural-v0.1"
    metrics = {
        "scope": scope,
        "atom_slot_mode": (
            "gt_target_order_oracle" if gt_atom_slot_oracle else "predicted"
        ),
        "gt_count": evaluate_g3_structural_overfit(
            model,
            features,
            targets,
            batch_size=batch_size,
            precision=precision,
            count_mode="gt_count",
            gt_atom_slot_oracle=gt_atom_slot_oracle,
            shared_confidence_aware_bond_assignment=(
                shared_confidence_aware_bond_assignment
            ),
            gt_bond_slot_identity_oracle=gt_bond_slot_identity_oracle,
            fixed_learned_bond_slots=fixed_learned_bond_slots,
            set_learned_bond_slots=set_learned_bond_slots,
            direct_gt_atom_pair_bonds=direct_gt_atom_pair_bonds,
        ),
        "deploy_count": evaluate_g3_structural_overfit(
            model,
            features,
            targets,
            batch_size=batch_size,
            precision=precision,
            count_mode="deploy_count",
            gt_atom_slot_oracle=gt_atom_slot_oracle,
            shared_confidence_aware_bond_assignment=(
                shared_confidence_aware_bond_assignment
            ),
            gt_bond_slot_identity_oracle=gt_bond_slot_identity_oracle,
            fixed_learned_bond_slots=fixed_learned_bond_slots,
            set_learned_bond_slots=set_learned_bond_slots,
            direct_gt_atom_pair_bonds=direct_gt_atom_pair_bonds,
        ),
        "sdf_valid": {
            "status": "not_applicable",
            "reason": (
                "structural-v0.1 does not predict Atom element, charge, H, "
                "direction, or stereo; GT semantics are not injected"
            ),
        },
    }
    if shared_confidence_aware_bond_assignment:
        metrics["bond_assignment_mode"] = "shared_confidence_aware"
    elif gt_bond_slot_identity_oracle:
        metrics["bond_assignment_mode"] = "gt_visual_endpoint_router_prefix_identity"
        metrics["bond_slot_mode"] = "gt_visual_endpoint_router_prefix_identity"
    elif direct_gt_atom_pair_bonds:
        metrics["bond_assignment_mode"] = "direct_gt_atom_pair_identity"
        metrics["bond_slot_mode"] = "all_unordered_gt_atom_pairs"
    elif set_learned_bond_slots:
        metrics["bond_assignment_mode"] = "set_hungarian_train_presence_topk_deploy"
        metrics["bond_slot_mode"] = "fixed_learned_set_prediction"
    elif fixed_learned_bond_slots:
        metrics["bond_slot_mode"] = "fixed_learned_canonical_prefix"
    metrics["p0_gate"] = g3_p0_gate_decision(metrics)
    return metrics


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", default=str(OVERFIT32_MANIFEST))
    parser.add_argument("--vision-checkpoint", default=DEFAULT_VISION_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--evaluate-every", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--atom-slots-per-cell", type=int, default=3)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--bond-slots-per-cell", type=int, default=3)
    parser.add_argument("--maximum-bonds", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--decoder-ffn-dim", type=int, default=1536)
    parser.add_argument("--memory-stride", type=int, default=4)
    parser.add_argument(
        "--gt-atom-slot-oracle",
        action="store_true",
        help=(
            "Run the pre-registered P0-B causal isolation with GT-ordered Atom "
            "anchors; requires its independent output directory"
        ),
    )
    parser.add_argument(
        "--shared-confidence-aware-bond-assignment",
        action="store_true",
        help=(
            "Run the pre-registered P0-C with one presence-aware assignment "
            "shared by Bond object and pointer losses"
        ),
    )
    parser.add_argument(
        "--gt-bond-slot-identity-oracle",
        action="store_true",
        help=(
            "Run the pre-registered P0-D visual-endpoint router with direct "
            "GT-prefix Bond identity"
        ),
    )
    parser.add_argument(
        "--fixed-learned-bond-slots",
        action="store_true",
        help=(
            "Run the pre-registered P0-E fixed learned Bond slots with direct "
            "canonical-prefix supervision"
        ),
    )
    parser.add_argument(
        "--set-learned-bond-slots",
        action="store_true",
        help=(
            "Run the pre-registered P0-F fixed learned Bond queries with "
            "train-time set matching and deploy-time presence top-k"
        ),
    )
    parser.add_argument(
        "--direct-gt-atom-pair-bonds",
        action="store_true",
        help=(
            "Run the pre-registered P0-G capacity test over every unordered "
            "GT AtomPair without Bond queries or pointer prediction"
        ),
    )
    parser.add_argument(
        "--atom-only-capacity",
        action="store_true",
        help=(
            "Run the pre-registered P0-H Atom-only capacity isolation without "
            "Bond modules or training-time count teacher forcing"
        ),
    )
    parser.add_argument(
        "--canonical-cell-slot-targets",
        action="store_true",
        help=(
            "Run P0-I with deterministic coordinate-ordered GT cell slots; "
            "requires --atom-only-capacity"
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.max_steps,
        args.batch_size,
        args.vision_batch_size,
        args.learning_rate,
        args.max_grad_norm,
        args.log_every,
        args.evaluate_every,
        args.save_every,
    )
    if min(positive) <= 0:
        raise ValueError("G3 P0 training dimensions must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("G3 weight decay must be nonnegative")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("G3 CPU training must use fp32")
    output_dir = Path(args.output_dir).resolve()
    oracle_output = Path(GT_ATOM_SLOT_ORACLE_OUTPUT).resolve()
    shared_output = Path(SHARED_BOND_ASSIGNMENT_OUTPUT).resolve()
    bond_oracle_output = Path(GT_BOND_SLOT_IDENTITY_ORACLE_OUTPUT).resolve()
    fixed_slots_output = Path(FIXED_LEARNED_BOND_SLOTS_OUTPUT).resolve()
    set_slots_output = Path(SET_LEARNED_BOND_SLOTS_OUTPUT).resolve()
    atom_pair_output = Path(GT_ATOM_PAIR_BOND_OUTPUT).resolve()
    atom_only_output = Path(ATOM_ONLY_CAPACITY_OUTPUT).resolve()
    canonical_output = Path(CANONICAL_CELL_SLOT_OUTPUT).resolve()
    if args.canonical_cell_slot_targets and not args.atom_only_capacity:
        raise ValueError(
            "G3 P0-I canonical cell-slot targets require --atom-only-capacity"
        )
    if args.atom_only_capacity and any(
        (
            args.gt_atom_slot_oracle,
            args.shared_confidence_aware_bond_assignment,
            args.gt_bond_slot_identity_oracle,
            args.fixed_learned_bond_slots,
            args.set_learned_bond_slots,
            args.direct_gt_atom_pair_bonds,
        )
    ):
        raise ValueError("G3 P0-H Atom-only capacity is mutually exclusive")
    expected_atom_only_output = (
        canonical_output if args.canonical_cell_slot_targets else atom_only_output
    )
    if args.atom_only_capacity and output_dir != expected_atom_only_output:
        raise ValueError(
            "G3 Atom-only protocol requires its pre-registered output directory: "
            f"{expected_atom_only_output}"
        )
    if (
        args.shared_confidence_aware_bond_assignment
        and args.gt_bond_slot_identity_oracle
    ):
        raise ValueError("G3 P0-C and P0-D Bond identity protocols are mutually exclusive")
    if args.fixed_learned_bond_slots and (
        args.shared_confidence_aware_bond_assignment
        or args.gt_bond_slot_identity_oracle
        or args.set_learned_bond_slots
        or args.direct_gt_atom_pair_bonds
    ):
        raise ValueError("G3 P0-E is mutually exclusive with other Bond protocols")
    if args.set_learned_bond_slots and (
        args.shared_confidence_aware_bond_assignment
        or args.gt_bond_slot_identity_oracle
        or args.direct_gt_atom_pair_bonds
    ):
        raise ValueError("G3 P0-F is mutually exclusive with other Bond protocols")
    if args.direct_gt_atom_pair_bonds and (
        args.shared_confidence_aware_bond_assignment
        or args.gt_bond_slot_identity_oracle
    ):
        raise ValueError("G3 P0-G is mutually exclusive with P0-C and P0-D")
    if args.gt_bond_slot_identity_oracle:
        if not args.gt_atom_slot_oracle:
            raise ValueError(
                "G3 P0-D GT Bond-slot oracle requires --gt-atom-slot-oracle"
            )
        if output_dir != bond_oracle_output:
            raise ValueError(
                "G3 P0-D GT Bond-slot oracle requires the pre-registered "
                f"output directory: {GT_BOND_SLOT_IDENTITY_ORACLE_OUTPUT}"
            )
    if args.fixed_learned_bond_slots:
        if not args.gt_atom_slot_oracle:
            raise ValueError(
                "G3 P0-E fixed Bond slots requires --gt-atom-slot-oracle"
            )
        if output_dir != fixed_slots_output:
            raise ValueError(
                "G3 P0-E fixed Bond slots require the pre-registered output "
                f"directory: {FIXED_LEARNED_BOND_SLOTS_OUTPUT}"
            )
    if args.set_learned_bond_slots:
        if not args.gt_atom_slot_oracle:
            raise ValueError(
                "G3 P0-F set Bond slots requires --gt-atom-slot-oracle"
            )
        if output_dir != set_slots_output:
            raise ValueError(
                "G3 P0-F set Bond slots require the pre-registered output "
                f"directory: {SET_LEARNED_BOND_SLOTS_OUTPUT}"
            )
    if args.direct_gt_atom_pair_bonds:
        if not args.gt_atom_slot_oracle:
            raise ValueError(
                "G3 P0-G AtomPair Bonds require --gt-atom-slot-oracle"
            )
        if output_dir != atom_pair_output:
            raise ValueError(
                "G3 P0-G AtomPair Bonds require the pre-registered output "
                f"directory: {GT_ATOM_PAIR_BOND_OUTPUT}"
            )
    if args.shared_confidence_aware_bond_assignment:
        if not args.gt_atom_slot_oracle:
            raise ValueError(
                "G3 P0-C shared Bond assignment requires --gt-atom-slot-oracle"
            )
        if output_dir != shared_output:
            raise ValueError(
                "G3 P0-C shared Bond assignment requires the pre-registered "
                f"output directory: {SHARED_BOND_ASSIGNMENT_OUTPUT}"
            )
    elif (
        args.gt_atom_slot_oracle
        and not args.gt_bond_slot_identity_oracle
        and not args.fixed_learned_bond_slots
        and not args.set_learned_bond_slots
        and not args.direct_gt_atom_pair_bonds
        and not args.atom_only_capacity
        and output_dir != oracle_output
    ):
        raise ValueError(
            "G3 P0-B GT Atom-slot oracle requires the pre-registered output "
            f"directory: {GT_ATOM_SLOT_ORACLE_OUTPUT}"
        )
    if not args.gt_atom_slot_oracle and output_dir == oracle_output:
        raise ValueError("G3 P0-B output directory requires --gt-atom-slot-oracle")
    if not args.shared_confidence_aware_bond_assignment and output_dir == shared_output:
        raise ValueError(
            "G3 P0-C output directory requires "
            "--shared-confidence-aware-bond-assignment"
        )
    if not args.gt_bond_slot_identity_oracle and output_dir == bond_oracle_output:
        raise ValueError(
            "G3 P0-D output directory requires --gt-bond-slot-identity-oracle"
        )
    if not args.fixed_learned_bond_slots and output_dir == fixed_slots_output:
        raise ValueError(
            "G3 P0-E output directory requires --fixed-learned-bond-slots"
        )
    if not args.set_learned_bond_slots and output_dir == set_slots_output:
        raise ValueError(
            "G3 P0-F output directory requires --set-learned-bond-slots"
        )
    if not args.direct_gt_atom_pair_bonds and output_dir == atom_pair_output:
        raise ValueError(
            "G3 P0-G output directory requires --direct-gt-atom-pair-bonds"
        )
    if not args.atom_only_capacity and output_dir == atom_only_output:
        raise ValueError(
            "G3 P0-H output directory requires --atom-only-capacity"
        )
    if not args.canonical_cell_slot_targets and output_dir == canonical_output:
        raise ValueError(
            "G3 P0-I output directory requires --canonical-cell-slot-targets"
        )


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: G3HierarchicalGraphConfig,
    vision_config: ChemVectorConfig,
    args: argparse.Namespace,
    audit: dict,
    step: int,
    last_losses: dict[str, float],
    train_only_metrics: dict,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p0-structural-checkpoint-0.1",
        "model_type": model.model_type,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "g3_config": asdict(config),
        "vision_config": asdict(vision_config),
        "global_step": step,
        "samples_seen": step * args.batch_size,
        "last_losses": last_losses,
        "train_only_metrics": train_only_metrics,
        "source_vision_checkpoint": audit["vision"]["path"],
        "source_vision_checkpoint_sha256": audit["vision"]["sha256"],
        "sample_manifest": audit["sample_manifest"],
        "atom_initialization": audit.get("atom_initialization"),
        "training_args": vars(args),
        "structural_scope": {
            "atom_anchor_and_presence": True,
            "bond_refs_and_types": not args.atom_only_capacity,
            "count_teacher_forcing": not args.atom_only_capacity,
            "ocr_or_atom_semantics": False,
            "direction_or_stereo": False,
            "formal_evaluator": False,
            "gt_atom_slot_oracle": args.gt_atom_slot_oracle,
            "shared_confidence_aware_bond_assignment": (
                args.shared_confidence_aware_bond_assignment
            ),
            "gt_bond_slot_identity_oracle": args.gt_bond_slot_identity_oracle,
            "fixed_learned_bond_slots": args.fixed_learned_bond_slots,
            "set_learned_bond_slots": args.set_learned_bond_slots,
            "direct_gt_atom_pair_bonds": args.direct_gt_atom_pair_bonds,
            "atom_only_capacity": args.atom_only_capacity,
            "canonical_cell_slot_targets": args.canonical_cell_slot_targets,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("G3 P0 CUDA training requested but CUDA is unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"G3 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_audit = load_g3_overfit32_manifest(args.sample_manifest)
    records = load_g3_manifest_samples(manifest)
    vision, vision_config, vision_payload, vision_sha256 = (
        load_frozen_vision_checkpoint(args.vision_checkpoint, device)
    )
    if vision_sha256 != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError(
            "G3 P0 vision checkpoint SHA256 differs from the frozen protocol: "
            f"{vision_sha256}"
        )
    config = G3HierarchicalGraphConfig(
        input_channels=vision_config.vision_dims[0],
        hidden_channels=args.hidden_channels,
        d_model=args.d_model,
        atom_slots_per_cell=args.atom_slots_per_cell,
        maximum_atoms=args.maximum_atoms,
        bond_slots_per_cell=args.bond_slots_per_cell,
        maximum_bonds=args.maximum_bonds,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        memory_stride=args.memory_stride,
        dropout=0.0,
    )
    model: nn.Module
    atom_initialization = None
    if args.atom_only_capacity:
        atom_only_model, atom_initialization = build_p0h_atom_only_model(config)
        model = atom_only_model.to(device)
    elif args.direct_gt_atom_pair_bonds:
        model = G3AtomPairBondIdentityPredictor(config).to(device)
    elif args.set_learned_bond_slots:
        model = G3SetLearnedBondSlotPredictor(config).to(device)
    elif args.fixed_learned_bond_slots:
        model = G3FixedLearnedBondSlotPredictor(config).to(device)
    else:
        model = G3HierarchicalGraphPredictor(config).to(device)
    optimizer_parameters = (
        model.atom_head.parameters()
        if args.atom_only_capacity
        else model.parameters()
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.atom_only_capacity:
        validate_atom_only_optimizer(model, optimizer)
    collator = G3StructuralCollator(
        maximum_atoms=config.maximum_atoms,
        maximum_bonds=config.maximum_bonds,
        image_size=768,
    )
    full_batch = collator(records)
    features = cache_frozen_vision_features(
        vision,
        full_batch["pixel_values"],
        device=device,
        batch_size=args.vision_batch_size,
        precision=args.precision,
    )
    targets = g3_targets_to_device(full_batch, device)
    del full_batch["pixel_values"]

    graph_parameters = sum(parameter.numel() for parameter in model.parameters())
    vision_parameters = sum(parameter.numel() for parameter in vision.parameters())
    combined_parameters = vision_parameters + graph_parameters
    if (
        args.fixed_learned_bond_slots
        or args.set_learned_bond_slots
        or args.direct_gt_atom_pair_bonds
        or args.atom_only_capacity
    ) and combined_parameters >= 51_000_000:
        raise ValueError(
            "G3 fixed learned Bond-slot parameter count must stay below 51M: "
            f"{combined_parameters}"
        )
    if not (
        args.fixed_learned_bond_slots or args.set_learned_bond_slots
        or args.direct_gt_atom_pair_bonds or args.atom_only_capacity
    ) and not (
        48_000_000 <= combined_parameters <= 52_000_000
    ):
        raise ValueError(
            "G3 P0 combined parameter count is outside the frozen 48M-52M range: "
            f"{combined_parameters}"
        )
    audit = {
        "schema_version": "g3-p0-training-audit-0.1",
        "sample_manifest": manifest_audit,
        "vision": {
            "path": str(Path(args.vision_checkpoint).resolve()),
            "sha256": vision_sha256,
            "model_type": vision_payload.get("model_type"),
            "global_step": vision_payload.get("global_step"),
            "frozen": True,
        },
        "feature_cache": {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "device": str(features.device),
        },
        "sample_keys": full_batch["keys"],
        "molecule_ids": full_batch["molecule_ids"],
        "variants": full_batch["variants"],
        "parameter_count": {
            "vision_frozen": vision_parameters,
            "g3_trainable": graph_parameters,
            "combined": combined_parameters,
        },
        "atom_head": (
            {
                "parameter_names": [
                    f"atom_head.{name}"
                    for name, _ in model.atom_head.named_parameters()
                ],
                "parameter_tensor_count": sum(
                    1 for _ in model.atom_head.parameters()
                ),
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in model.atom_head.parameters()
                ),
                "trainable_names": [
                    f"atom_head.{name}"
                    for name, parameter in model.atom_head.named_parameters()
                    if parameter.requires_grad
                ],
            }
            if args.atom_only_capacity
            else None
        ),
        "atom_initialization": atom_initialization,
        "optimizer_scope": {
            "atom_only": args.atom_only_capacity,
            "parameter_names": [
                name for name, parameter in model.named_parameters()
                if any(
                    parameter is optimized
                    for group in optimizer.param_groups
                    for optimized in group["params"]
                )
            ],
        },
        "implementation": {
            "trainer_sha256": file_sha256(__file__),
            "g3_model_sha256": file_sha256(
                Path(__file__).parents[3]
                / "components/graph/joint/g3_hierarchical_graph.py"
            ),
            "data_adapter_sha256": file_sha256(
                Path(__file__).with_name("g3_structural_data.py")
            ),
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "protocol": {
            "name": (
                "g3-p0i-canonical-cell-slot-overfit32"
                if args.canonical_cell_slot_targets
                else (
                    "g3-p0h-atom-only-overfit32"
                    if args.atom_only_capacity
                    else (
                        "g3-p0g-gt-atom-pair-bond-capacity"
                        if args.direct_gt_atom_pair_bonds
                        else (
                            "g3-p0f-set-learned-bond-slots"
                            if args.set_learned_bond_slots
                            else (
                                "g3-p0e-fixed-learned-bond-slots"
                                if args.fixed_learned_bond_slots
                                else (
                                    "g3-p0d-gt-bond-slot-identity-oracle"
                                    if args.gt_bond_slot_identity_oracle
                                    else (
                                        "g3-p0c-shared-bond-assignment"
                                        if args.shared_confidence_aware_bond_assignment
                                        else (
                                            "g3-p0b-gt-atom-slot-oracle"
                                            if args.gt_atom_slot_oracle
                                            else "g3-p0-overfit32"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "gt_atom_slot_oracle": args.gt_atom_slot_oracle,
            "shared_confidence_aware_bond_assignment": (
                args.shared_confidence_aware_bond_assignment
            ),
            "gt_bond_slot_identity_oracle": args.gt_bond_slot_identity_oracle,
            "fixed_learned_bond_slots": args.fixed_learned_bond_slots,
            "set_learned_bond_slots": args.set_learned_bond_slots,
            "direct_gt_atom_pair_bonds": args.direct_gt_atom_pair_bonds,
            "atom_only_capacity": args.atom_only_capacity,
            "canonical_cell_slot_targets": args.canonical_cell_slot_targets,
            "bond_matching_presence_cost_weight": (
                2.0 if args.shared_confidence_aware_bond_assignment else None
            ),
        },
    }
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(
        output_dir / "config.json",
        {
            "g3_config": asdict(config),
            "vision_config": asdict(vision_config),
            "parameter_count": audit["parameter_count"],
        },
    )

    if args.atom_only_capacity:
        latest_metrics = evaluate_g3_atom_only_count_modes(
            model,
            features,
            targets,
            batch_size=args.batch_size,
            precision=args.precision,
            canonical_cell_slot_targets=args.canonical_cell_slot_targets,
        )
    else:
        latest_metrics = evaluate_g3_count_modes(
            model,
            features,
            targets,
            batch_size=args.batch_size,
            precision=args.precision,
            gt_atom_slot_oracle=args.gt_atom_slot_oracle,
            shared_confidence_aware_bond_assignment=(
                args.shared_confidence_aware_bond_assignment
            ),
            gt_bond_slot_identity_oracle=args.gt_bond_slot_identity_oracle,
            fixed_learned_bond_slots=args.fixed_learned_bond_slots,
            set_learned_bond_slots=args.set_learned_bond_slots,
            direct_gt_atom_pair_bonds=args.direct_gt_atom_pair_bonds,
        )
    latest_metrics["step"] = 0
    _atomic_json(output_dir / "initial_metrics.json", latest_metrics)
    _atomic_json(output_dir / "metrics.json", latest_metrics)
    print(json.dumps({"train_only_evaluation": latest_metrics}), flush=True)

    started = time.perf_counter()
    running: dict[str, float] = {}
    log_path = output_dir / "training_log.jsonl"
    model.train()
    for step in range(1, args.max_steps + 1):
        indices = cyclic_batch_indices(
            step,
            sample_count=len(records),
            batch_size=args.batch_size,
            device=device,
        )
        if args.atom_only_capacity:
            current_targets = {
                name: targets[name][indices]
                for name in ("atom_anchors", "atom_mask")
            }
            losses = g3_atom_only_training_step(
                model,
                optimizer,
                features[indices],
                current_targets,
                precision=args.precision,
                max_grad_norm=args.max_grad_norm,
                canonical_cell_slot_targets=args.canonical_cell_slot_targets,
            )
        else:
            current_targets = {
                name: value[indices] for name, value in targets.items()
            }
            losses = g3_training_step(
                model,
                optimizer,
                features[indices],
                current_targets,
                precision=args.precision,
                max_grad_norm=args.max_grad_norm,
                gt_atom_slot_oracle=args.gt_atom_slot_oracle,
                shared_confidence_aware_bond_assignment=(
                    args.shared_confidence_aware_bond_assignment
                ),
                gt_bond_slot_identity_oracle=args.gt_bond_slot_identity_oracle,
                fixed_learned_bond_slots=args.fixed_learned_bond_slots,
                set_learned_bond_slots=args.set_learned_bond_slots,
                direct_gt_atom_pair_bonds=args.direct_gt_atom_pair_bonds,
            )
        for name, value in losses.items():
            running[name] = running.get(name, 0.0) + value
        if step % args.log_every == 0 or step == args.max_steps:
            divisor = (
                args.log_every
                if step % args.log_every == 0
                else step % args.log_every
            )
            record = {
                "step": step,
                "samples_seen": step * args.batch_size,
                "elapsed_seconds": time.perf_counter() - started,
                **{name: value / divisor for name, value in running.items()},
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(record, separators=(",", ":")), flush=True)
            running.clear()
        if step % args.evaluate_every == 0 or step == args.max_steps:
            if args.atom_only_capacity:
                latest_metrics = evaluate_g3_atom_only_count_modes(
                    model,
                    features,
                    targets,
                    batch_size=args.batch_size,
                    precision=args.precision,
                    canonical_cell_slot_targets=args.canonical_cell_slot_targets,
                )
            else:
                latest_metrics = evaluate_g3_count_modes(
                    model,
                    features,
                    targets,
                    batch_size=args.batch_size,
                    precision=args.precision,
                    gt_atom_slot_oracle=args.gt_atom_slot_oracle,
                    shared_confidence_aware_bond_assignment=(
                        args.shared_confidence_aware_bond_assignment
                    ),
                    gt_bond_slot_identity_oracle=args.gt_bond_slot_identity_oracle,
                    fixed_learned_bond_slots=args.fixed_learned_bond_slots,
                    set_learned_bond_slots=args.set_learned_bond_slots,
                    direct_gt_atom_pair_bonds=args.direct_gt_atom_pair_bonds,
                )
            latest_metrics["step"] = step
            latest_metrics["elapsed_seconds"] = time.perf_counter() - started
            _atomic_json(output_dir / "metrics.json", latest_metrics)
            print(
                json.dumps(
                    {"train_only_evaluation": latest_metrics},
                    separators=(",", ":"),
                ),
                flush=True,
            )
        if step % args.save_every == 0 or step == args.max_steps:
            checkpoint = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                vision_config=vision_config,
                args=args,
                audit=audit,
                step=step,
                last_losses=losses,
                train_only_metrics=latest_metrics,
            )
            step_path = output_dir / f"step-{step:06d}.pt"
            _atomic_torch_save(step_path, checkpoint)
            _atomic_torch_save(output_dir / "last.pt", checkpoint)
            _atomic_json(
                output_dir / "status.json",
                {
                    "status": "completed" if step == args.max_steps else "running",
                    "global_step": step,
                    "samples_seen": step * args.batch_size,
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint": step_path.name,
                    "last_losses": losses,
                    "train_only_metrics": latest_metrics,
                },
            )


if __name__ == "__main__":
    main()
