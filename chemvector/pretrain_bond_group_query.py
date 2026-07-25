#!/usr/bin/env python3
"""Train proposal-conditioned Bond group queries on a controlled capacity set."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.atom_query import poisson_binomial_log_probs
from chemvector.bond_group_query import (
    BondGroupQueryConfig,
    BondGroupQueryDetector,
    bond_group_query_assignments,
    bond_group_query_geometry_loss,
    bond_group_query_loss,
    decode_bond_group_queries,
    extract_dense_bond_proposals,
    set_endpoint_local_only_trainable,
)
from chemvector.bond_object import unordered_endpoint_cost
from chemvector.data import ChemVectorIterableDataset
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_chemscene_bond_query import (
    ChemSceneBondCollator,
    _oriented_endpoint_errors,
    stack_targets,
)
from chemvector.pretrain_dense_bond_object import (
    _apply_detector,
    _extract_detector_features,
)
from chemvector.pretrain_dense_slot_matching import (
    build_collision_split,
    load_detector,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.train import expand_shards


IMAGE_SIZE = 768.0
CAPACITY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0829-bond-group-query-capacity100"
)
GENERALIZATION_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0829-bond-group-query-generalization"
)
MATCHED_GENERALIZATION_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0829-bond-group-query-generalization-matched20"
)
ENDPOINT_LOCAL_CAPACITY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0830-bond-group-query-endpoint-local-capacity100"
)
ENDPOINT_LOCAL_GENERALIZATION_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0830-bond-group-query-endpoint-local-generalization"
)
ENDPOINT_LOCAL_MATCHED_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0830-bond-group-query-endpoint-local-matched20"
)
ENDPOINT_LOCAL_EARLY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0830-bond-group-query-endpoint-local-early4k"
)
ENDPOINT_LOCAL_ONLY_CAPACITY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0830-bond-group-query-endpoint-local-only-capacity100"
)
ENDPOINT_LOCAL_ONLY_EARLY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0830-bond-group-query-endpoint-local-only-early4k"
)
ENDPOINT_LINE_LOCAL_ONLY_CAPACITY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0831-bond-group-query-line-local-only-capacity100"
)
ENDPOINT_LINE_LOCAL_ONLY_EARLY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0831-bond-group-query-line-local-only-early4k"
)
ENDPOINT_INK_LINE_LOCAL_ONLY_CAPACITY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0832-bond-group-query-ink-line-local-only-capacity100"
)
ENDPOINT_INK_LINE_LOCAL_ONLY_EARLY_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0832-bond-group-query-ink-line-local-only-early4k"
)


def apply_mode_protocol(args) -> None:
    """Resolve the fixed generalization protocol without changing capacity defaults."""

    if args.matched_exposure_control and args.early_validation_control:
        raise ValueError("Matched and early-validation controls are mutually exclusive")
    if args.early_validation_control and (
        args.mode != "generalization" or not args.endpoint_local
    ):
        raise ValueError(
            "Early-validation control requires generalization mode and endpoint local"
        )
    if args.mode != "generalization":
        if args.endpoint_local and args.output_dir == CAPACITY_OUTPUT_DIR:
            args.output_dir = (
                ENDPOINT_INK_LINE_LOCAL_ONLY_CAPACITY_OUTPUT_DIR
                if args.endpoint_ink_line and args.train_local_only
                else (
                    ENDPOINT_LINE_LOCAL_ONLY_CAPACITY_OUTPUT_DIR
                    if args.endpoint_line_band and args.train_local_only
                    else (
                        ENDPOINT_LOCAL_ONLY_CAPACITY_OUTPUT_DIR
                        if args.train_local_only
                        else ENDPOINT_LOCAL_CAPACITY_OUTPUT_DIR
                    )
                )
            )
        return
    if args.output_dir == CAPACITY_OUTPUT_DIR:
        args.output_dir = GENERALIZATION_OUTPUT_DIR
    args.objective = "geometry"
    args.collision_train_images = 400
    args.collision_holdout_images = 100
    args.ordinary_replay_images = 400
    args.train_exposures = 8000
    args.batch_size = 8
    args.learning_rate = 1e-4
    args.collision_bond_weight = 16.0
    args.endpoint_delta_scale = 0.1
    args.validate_every = 800
    args.initial_assignment_exposures = 1600
    if args.matched_exposure_control:
        if args.output_dir in {CAPACITY_OUTPUT_DIR, GENERALIZATION_OUTPUT_DIR}:
            args.output_dir = MATCHED_GENERALIZATION_OUTPUT_DIR
        args.train_exposures = 16000
        args.validate_every = 3200
        args.initial_assignment_exposures = 16000
    elif args.early_validation_control:
        args.output_dir = (
            ENDPOINT_INK_LINE_LOCAL_ONLY_EARLY_OUTPUT_DIR
            if args.endpoint_ink_line and args.train_local_only
            else (
                ENDPOINT_LINE_LOCAL_ONLY_EARLY_OUTPUT_DIR
                if args.endpoint_line_band and args.train_local_only
                else (
                    ENDPOINT_LOCAL_ONLY_EARLY_OUTPUT_DIR
                    if args.train_local_only
                    else ENDPOINT_LOCAL_EARLY_OUTPUT_DIR
                )
            )
        )
        args.train_exposures = 4000
        args.validate_every = 400
        args.initial_assignment_exposures = 4000
    if args.endpoint_local:
        if args.output_dir == GENERALIZATION_OUTPUT_DIR:
            args.output_dir = ENDPOINT_LOCAL_GENERALIZATION_OUTPUT_DIR
        elif args.output_dir == MATCHED_GENERALIZATION_OUTPUT_DIR:
            args.output_dir = ENDPOINT_LOCAL_MATCHED_OUTPUT_DIR


def load_records(shards: list[str], count: int) -> list[dict]:
    if count < 1:
        raise ValueError("Record count must be positive")
    records = []
    for record in ChemVectorIterableDataset(
        expand_shards(shards), sequence_version="0.4"
    ):
        records.append(record)
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"Requested {count} records, loaded {len(records)}")
    return records


def capacity_exposure_order(
    record_count: int,
    exposures: int,
    batch_size: int,
    seed: int,
) -> list[int]:
    """Repeat every capacity record equally with deterministic per-cycle shuffles."""

    if min(record_count, exposures, batch_size) < 1:
        raise ValueError("Capacity record, exposure, and batch counts must be positive")
    if exposures % record_count:
        raise ValueError("Capacity exposures must be divisible by the record count")
    if exposures % batch_size:
        raise ValueError("Capacity exposures must be divisible by the batch size")
    order = []
    for cycle in range(exposures // record_count):
        indices = list(range(record_count))
        random.Random(seed + cycle).shuffle(indices)
        order.extend(indices)
    if len(order) != exposures or Counter(order) != Counter(
        {index: exposures // record_count for index in range(record_count)}
    ):
        raise RuntimeError("Capacity exposure schedule is not balanced")
    return order


def mixed_exposure_order(
    collision_count: int,
    ordinary_count: int,
    exposures: int,
    batch_size: int,
    seed: int,
) -> list[int]:
    """Build exact half-collision, half-ordinary batches for every epoch."""

    if collision_count != ordinary_count or batch_size % 2:
        raise ValueError("Mixed Bond Query training requires balanced pools and batches")
    record_count = collision_count + ordinary_count
    if exposures % record_count or exposures % batch_size:
        raise ValueError("Mixed Bond Query exposures must contain complete epochs")
    half_batch = batch_size // 2
    order = []
    for epoch in range(exposures // record_count):
        collision = list(range(collision_count))
        ordinary = list(range(collision_count, record_count))
        random.Random(seed + 2 * epoch).shuffle(collision)
        random.Random(seed + 2 * epoch + 1).shuffle(ordinary)
        for start in range(0, collision_count, half_batch):
            batch = collision[start : start + half_batch] + ordinary[
                start : start + half_batch
            ]
            random.Random(seed + 10000 + epoch * collision_count + start).shuffle(
                batch
            )
            order.extend(batch)
    counts = Counter(order)
    expected = exposures // record_count
    if len(order) != exposures or any(
        counts[index] != expected for index in range(record_count)
    ):
        raise RuntimeError("Mixed Bond Query exposure schedule is unbalanced")
    return order


def _prepare_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        "training_args.json",
        "config.json",
        "split.json",
        "zero_step_parity.json",
        "initial.pt",
        "best.pt",
        "warmup.pt",
        "last.pt",
        "summary.json",
    )
    existing = [name for name in artifacts if (output_dir / name).exists()]
    existing.extend(path.name for path in output_dir.glob("checkpoint-*.pt"))
    if existing:
        raise FileExistsError(
            f"Bond Group Query output already contains artifacts: {output_dir} "
            f"({', '.join(sorted(existing)[:8])})"
        )


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


@torch.no_grad()
def _frozen_proposals(
    vision_model,
    dense_detector,
    pixels: torch.Tensor,
    *,
    queries: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    with _autocast(pixels.device):
        features = _extract_detector_features(vision_model, dense_detector, pixels)
        dense_output = _apply_detector(dense_detector, features)
        proposals = extract_dense_bond_proposals(
            dense_output, maximum_queries=queries
        )
    return features[0], dense_output, proposals


def _decoded_parity(
    dense_objects: list[dict],
    group_objects: list[dict],
    *,
    tolerance: float,
) -> tuple[bool, float, float]:
    if len(dense_objects) != len(group_objects):
        return False, float("inf"), float("inf")
    if not dense_objects:
        return True, 0.0, 0.0

    def sort_key(item: dict) -> tuple:
        endpoints = sorted(
            tuple(float(value) for value in endpoint)
            for endpoint in item["endpoints"].float().tolist()
        )
        return (
            *endpoints[0],
            *endpoints[1],
            int(item["display"]),
            int(item["chemical"]),
            int(item["direction"]),
            float(item["score"]),
        )

    dense_ranked = sorted(dense_objects, key=sort_key)
    group_ranked = sorted(group_objects, key=sort_key)
    maximum_endpoint_delta = 0.0
    maximum_score_delta = 0.0
    for dense_item, group_item in zip(dense_ranked, group_ranked):
        errors = _oriented_endpoint_errors(
            dense_item["endpoints"].float(),
            group_item["endpoints"].float(),
            1.0,
        )
        maximum_endpoint_delta = max(maximum_endpoint_delta, float(errors.max()))
        maximum_score_delta = max(
            maximum_score_delta,
            abs(float(dense_item["score"]) - float(group_item["score"])),
        )
        if any(
            int(dense_item[name]) != int(group_item[name])
            for name in ("display", "chemical", "direction")
        ):
            return False, maximum_endpoint_delta, maximum_score_delta
    passed = (
        maximum_endpoint_delta <= tolerance and maximum_score_delta <= tolerance
    )
    return passed, maximum_endpoint_delta, maximum_score_delta


@torch.inference_mode()
def audit_zero_step_parity(
    vision_model,
    dense_detector,
    group_detector,
    records: list[dict],
    collator: ChemSceneBondCollator,
    device: torch.device,
    *,
    batch_size: int,
    queries: int,
    threshold: float,
    tolerance: float = 1e-6,
) -> dict:
    """Prove zero-initialized group residuals preserve dense proposals exactly."""

    from chemvector.dense_bond_object import decode_dense_bond_objects

    group_detector.eval()
    field_map = {
        "presence_logits": "base_logits",
        "endpoints": "base_endpoints",
        "display_logits": "base_display_logits",
        "chemical_logits": "base_chemical_logits",
        "direction_logits": "base_direction_logits",
    }
    maximum_deltas = {name: 0.0 for name in field_map}
    nonfinite = 0
    mask_mismatches = 0
    decode_mismatches = 0
    decode_endpoint_delta = 0.0
    decode_score_delta = 0.0
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        stride4, dense_output, proposals = _frozen_proposals(
            vision_model, dense_detector, pixels, queries=queries
        )
        with _autocast(device):
            output = group_detector(stride4, proposals, pixel_values=pixels)
        for output_name, proposal_name in field_map.items():
            difference = (
                output[output_name].float() - proposals[proposal_name].float()
            ).abs()
            nonfinite += int((~torch.isfinite(difference)).sum())
            if difference.numel():
                maximum_deltas[output_name] = max(
                    maximum_deltas[output_name], float(difference.max())
                )
        mask_mismatches += int(
            (output["proposal_mask"] != proposals["mask"].bool()).sum()
        )
        dense_decoded = decode_dense_bond_objects(
            dense_output, threshold=threshold, maximum_objects=queries
        )
        group_decoded = decode_bond_group_queries(
            output, mode="threshold", threshold=threshold
        )
        for dense_objects, group_objects in zip(dense_decoded, group_decoded):
            passed, endpoint_delta, score_delta = _decoded_parity(
                dense_objects, group_objects, tolerance=tolerance
            )
            decode_mismatches += int(not passed)
            decode_endpoint_delta = max(decode_endpoint_delta, endpoint_delta)
            decode_score_delta = max(decode_score_delta, score_delta)
    passed = (
        nonfinite == 0
        and mask_mismatches == 0
        and decode_mismatches == 0
        and max(maximum_deltas.values(), default=0.0) <= tolerance
    )
    return {
        "schema_version": "bond-group-query-zero-step-parity-0.1",
        "status": "passed" if passed else "failed",
        "samples": len(records),
        "tolerance": tolerance,
        "maximum_absolute_delta": maximum_deltas,
        "nonfinite_values": nonfinite,
        "proposal_mask_mismatches": mask_mismatches,
        "threshold_decode_mismatches": decode_mismatches,
        "threshold_decode_max_endpoint_delta": decode_endpoint_delta,
        "threshold_decode_max_score_delta": decode_score_delta,
        "threshold": threshold,
    }


def _target_cells(
    targets: dict[str, torch.Tensor],
    batch_index: int,
    grid_size: int,
) -> dict[tuple[int, int], list[int]]:
    cells: dict[tuple[int, int], list[int]] = {}
    for target_index in torch.where(targets["mask"][batch_index])[0].tolist():
        center = targets["endpoints"][batch_index, target_index].mean(dim=0)
        cell_x = min(grid_size - 1, max(0, int(float(center[0]) * grid_size)))
        cell_y = min(grid_size - 1, max(0, int(float(center[1]) * grid_size)))
        cells.setdefault((cell_y, cell_x), []).append(target_index)
    for indices in cells.values():
        indices.sort(
            key=lambda index: tuple(
                targets["endpoints"][batch_index, index].flatten().tolist()
            )
        )
    return cells


def collision_target_weights(
    targets: dict[str, torch.Tensor],
    *,
    grid_size: int,
    collision_weight: float,
) -> torch.Tensor:
    if grid_size < 1 or collision_weight < 1.0:
        raise ValueError("Collision target weighting is invalid")
    weights = torch.ones_like(targets["mask"], dtype=torch.float32)
    for batch_index in range(targets["mask"].size(0)):
        for indices in _target_cells(targets, batch_index, grid_size).values():
            if len(indices) >= 2:
                weights[batch_index, indices] = collision_weight
    return weights


def _known_object_metrics(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    batch_index: int,
    *,
    endpoint_tolerance_pixels: float,
) -> tuple[int, bool]:
    valid_targets = torch.where(targets["mask"][batch_index])[0]
    count = len(valid_targets)
    valid_queries = torch.where(output["proposal_mask"][batch_index])[0]
    if not count or count > len(valid_queries):
        return 0, False
    logits = output["presence_logits"][batch_index, valid_queries]
    selected = valid_queries[logits.topk(count).indices]
    predicted = output["endpoints"][batch_index, selected].float()
    expected = targets["endpoints"][batch_index, valid_targets].float()
    rows, columns = linear_sum_assignment(
        unordered_endpoint_cost(predicted, expected).cpu().numpy()
    )
    correct = 0
    for row, column in zip(rows, columns):
        errors = _oriented_endpoint_errors(
            predicted[row], expected[column], IMAGE_SIZE
        )
        query_index = selected[row]
        target_index = valid_targets[column]
        classes_match = (
            int(output["display_logits"][batch_index, query_index].argmax())
            == int(targets["display"][batch_index, target_index])
            and int(output["chemical_logits"][batch_index, query_index].argmax())
            == int(targets["chemical"][batch_index, target_index])
        )
        correct += int(
            bool((errors <= endpoint_tolerance_pixels).all()) and classes_match
        )
    return correct, correct == count


@torch.inference_mode()
def evaluate_capacity(
    vision_model,
    dense_detector,
    group_detector,
    records: list[dict],
    collator: ChemSceneBondCollator,
    device: torch.device,
    *,
    batch_size: int,
    queries: int,
    grid_size: int,
    endpoint_tolerance_pixels: float = 6.0,
) -> dict:
    group_detector.eval()
    totals = Counter()
    occupancy_cells = Counter()
    pb_count_histogram = Counter()
    non_geometry_maximum_deltas = {
        "presence_logits": 0.0,
        "display_logits": 0.0,
        "chemical_logits": 0.0,
        "direction_logits": 0.0,
    }
    proposal_field_by_output = {
        "presence_logits": "base_logits",
        "display_logits": "base_display_logits",
        "chemical_logits": "base_chemical_logits",
        "direction_logits": "base_direction_logits",
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        stride4, _, proposals = _frozen_proposals(
            vision_model, dense_detector, pixels, queries=queries
        )
        with _autocast(device):
            output = group_detector(stride4, proposals, pixel_values=pixels)
        assignments = bond_group_query_assignments(output, targets)
        base_assignments = bond_group_query_assignments(
            {
                "endpoints": proposals["base_endpoints"],
                "proposal_mask": proposals["mask"].bool(),
            },
            targets,
        )
        masked_logits = output["presence_logits"].float().masked_fill(
            ~output["proposal_mask"], -30.0
        )
        pb_counts = poisson_binomial_log_probs(masked_logits).argmax(-1)
        base_masked_logits = proposals["base_logits"].float().masked_fill(
            ~proposals["mask"].bool(), -30.0
        )
        base_pb_counts = poisson_binomial_log_probs(base_masked_logits).argmax(-1)
        totals["pb_count_vs_base_mismatches"] += int(
            (pb_counts != base_pb_counts).sum()
        )
        totals["proposal_mask_mismatches"] += int(
            (output["proposal_mask"] != proposals["mask"].bool()).sum()
        )
        for output_name, proposal_name in proposal_field_by_output.items():
            difference = (
                output[output_name].float() - proposals[proposal_name].float()
            ).abs()
            if difference.numel():
                non_geometry_maximum_deltas[output_name] = max(
                    non_geometry_maximum_deltas[output_name],
                    float(difference.max()),
                )
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            query_by_target = {
                int(target): int(query)
                for query, target in zip(query_indices.tolist(), target_indices.tolist())
            }
            base_queries, base_targets = base_assignments[batch_index]
            base_query_by_target = {
                int(target): int(query)
                for query, target in zip(
                    base_queries.tolist(), base_targets.tolist()
                )
            }
            for target_index in torch.where(targets["mask"][batch_index])[0].tolist():
                query_index = query_by_target.get(target_index)
                base_query_index = base_query_by_target.get(target_index)
                if query_index is None or base_query_index is None:
                    continue
                expected = targets["endpoints"][batch_index, target_index].float()
                errors = _oriented_endpoint_errors(
                    output["endpoints"][batch_index, query_index].float(),
                    expected,
                    IMAGE_SIZE,
                )
                base_errors = _oriented_endpoint_errors(
                    proposals["base_endpoints"][batch_index, base_query_index].float(),
                    expected,
                    IMAGE_SIZE,
                )
                geometry = bool((errors <= endpoint_tolerance_pixels).all())
                totals["all_geometry_correct_bonds"] += int(geometry)
                if bool((base_errors <= endpoint_tolerance_pixels).all()):
                    totals["protected_base_bonds"] += 1
                    totals["protected_retained_bonds"] += int(geometry)
                totals.setdefault("endpoint_errors", []).append(float(errors.max()))
                displacement = _oriented_endpoint_errors(
                    output["endpoints"][batch_index, query_index].float(),
                    proposals["base_endpoints"][batch_index, query_index].float(),
                    IMAGE_SIZE,
                )
                totals.setdefault("endpoint_displacements", []).append(
                    float(displacement.max())
                )
            cells = _target_cells(targets, batch_index, grid_size)
            for indices in cells.values():
                if len(indices) < 2:
                    continue
                occupancy_cells[len(indices)] += 1
                totals["collision_cells"] += 1
                totals["collision_bonds"] += len(indices)
                geometry_correct = []
                object_correct = []
                for role_index, target_index in enumerate(indices):
                    query_index = query_by_target.get(target_index)
                    geometry = False
                    classes_match = False
                    if query_index is not None:
                        errors = _oriented_endpoint_errors(
                            output["endpoints"][batch_index, query_index].float(),
                            targets["endpoints"][batch_index, target_index].float(),
                            IMAGE_SIZE,
                        )
                        geometry = bool(
                            (errors <= endpoint_tolerance_pixels).all()
                        )
                        classes_match = (
                            int(
                                output["display_logits"][
                                    batch_index, query_index
                                ].argmax()
                            )
                            == int(targets["display"][batch_index, target_index])
                            and int(
                                output["chemical_logits"][
                                    batch_index, query_index
                                ].argmax()
                            )
                            == int(targets["chemical"][batch_index, target_index])
                        )
                    geometry_correct.append(geometry)
                    object_correct.append(geometry and classes_match)
                    totals["collision_geometry_correct_bonds"] += int(geometry)
                    totals["collision_object_correct_bonds"] += int(
                        geometry and classes_match
                    )
                    if role_index:
                        totals["secondary_bonds"] += 1
                        totals["secondary_geometry_correct"] += int(geometry)
                totals["collision_cell_geometry_exact"] += int(
                    all(geometry_correct)
                )
                totals["collision_cell_object_exact"] += int(all(object_correct))

            known_correct, known_exact = _known_object_metrics(
                output,
                targets,
                batch_index,
                endpoint_tolerance_pixels=endpoint_tolerance_pixels,
            )
            expected_count = int(targets["mask"][batch_index].sum())
            predicted_count = int(pb_counts[batch_index])
            totals["samples"] += 1
            totals["bonds"] += expected_count
            totals["known_object_correct"] += known_correct
            totals["known_object_exact"] += int(known_exact)
            totals["pb_count_exact"] += int(predicted_count == expected_count)
            totals["pb_count_absolute_error"] += abs(predicted_count - expected_count)
            pb_count_histogram[predicted_count] += 1
    samples = max(totals["samples"], 1)
    bonds = max(totals["bonds"], 1)
    collision_cells = max(totals["collision_cells"], 1)
    collision_bonds = max(totals["collision_bonds"], 1)
    secondary_bonds = max(totals["secondary_bonds"], 1)
    endpoint_errors = totals.get("endpoint_errors", [])
    endpoint_displacements = totals.get("endpoint_displacements", [])
    non_geometry_invariant = (
        totals["pb_count_vs_base_mismatches"] == 0
        and totals["proposal_mask_mismatches"] == 0
        and max(non_geometry_maximum_deltas.values(), default=0.0) <= 1e-6
    )
    return {
        "schema_version": "bond-group-query-capacity-metrics-0.1",
        "samples": totals["samples"],
        "bonds": totals["bonds"],
        "collision_cells": totals["collision_cells"],
        "collision_bonds": totals["collision_bonds"],
        "collision_occupancy_cells": {
            str(key): value for key, value in sorted(occupancy_cells.items())
        },
        "collision_cell_geometry_exact": totals[
            "collision_cell_geometry_exact"
        ],
        "collision_cell_geometry_exact_rate": totals[
            "collision_cell_geometry_exact"
        ]
        / collision_cells,
        "collision_cell_object_exact_rate": totals[
            "collision_cell_object_exact"
        ]
        / collision_cells,
        "collision_bond_geometry_recall": totals[
            "collision_geometry_correct_bonds"
        ]
        / collision_bonds,
        "collision_bond_object_recall": totals[
            "collision_object_correct_bonds"
        ]
        / collision_bonds,
        "secondary_geometry_recall": totals["secondary_geometry_correct"]
        / secondary_bonds,
        "all_bond_geometry_recall": totals["all_geometry_correct_bonds"] / bonds,
        "protected_geometry_retention": totals["protected_retained_bonds"]
        / max(totals["protected_base_bonds"], 1),
        "protected_base_bonds": totals["protected_base_bonds"],
        "endpoint_error_pixels": {
            "p50": float(np.percentile(endpoint_errors, 50)) if endpoint_errors else None,
            "p95": float(np.percentile(endpoint_errors, 95)) if endpoint_errors else None,
            "max": max(endpoint_errors) if endpoint_errors else None,
        },
        "endpoint_displacement_pixels": {
            "p50": (
                float(np.percentile(endpoint_displacements, 50))
                if endpoint_displacements
                else None
            ),
            "p95": (
                float(np.percentile(endpoint_displacements, 95))
                if endpoint_displacements
                else None
            ),
            "max": max(endpoint_displacements) if endpoint_displacements else None,
        },
        "known_object_accuracy": totals["known_object_correct"] / bonds,
        "known_object_exact": totals["known_object_exact"] / samples,
        "pb_count_accuracy": totals["pb_count_exact"] / samples,
        "pb_count_mae": totals["pb_count_absolute_error"] / samples,
        "pb_count_histogram": {
            str(key): value for key, value in sorted(pb_count_histogram.items())
        },
        "non_geometry_invariance": {
            "status": "passed" if non_geometry_invariant else "failed",
            "tolerance": 1e-6,
            "maximum_absolute_delta": non_geometry_maximum_deltas,
            "pb_count_vs_base_mismatches": totals[
                "pb_count_vs_base_mismatches"
            ],
            "proposal_mask_mismatches": totals["proposal_mask_mismatches"],
        },
        "endpoint_tolerance_pixels": endpoint_tolerance_pixels,
        "direction_in_object_metric": False,
    }


def _metric_score(metrics: dict) -> tuple[float, float, float, float]:
    return (
        float(metrics["collision_cell_geometry_exact_rate"]),
        float(metrics["secondary_geometry_recall"]),
        float(metrics["known_object_accuracy"]),
        float(metrics["pb_count_accuracy"]),
    )


def _generalization_score(metrics: dict) -> tuple[float, float, float]:
    holdout = metrics["collision_holdout"]
    return (
        float(holdout["collision_cell_geometry_exact_rate"]),
        float(holdout["secondary_geometry_recall"]),
        float(holdout["collision_bond_geometry_recall"]),
    )


def _generalization_non_geometry_invariant(metrics: dict) -> bool:
    return all(
        metrics[name]["non_geometry_invariance"]["status"] == "passed"
        for name in ("collision_holdout", "ordinary_validation")
    )


def _all_non_geometry_invariant(history: list[dict]) -> bool:
    return all(
        _generalization_non_geometry_invariant(item["metrics"])
        for item in history
    )


def generalization_checks(
    initial_metrics: dict,
    best_metrics: dict,
    final_metrics: dict,
    history: list[dict],
) -> dict:
    holdout = best_metrics["collision_holdout"]
    ordinary = best_metrics["ordinary_validation"]
    initial_ordinary = initial_metrics["ordinary_validation"]
    checks = {
        "holdout_secondary_at_least_0_50": (
            holdout["secondary_geometry_recall"] >= 0.50
        ),
        "holdout_cell_exact_at_least_0_18": (
            holdout["collision_cell_geometry_exact_rate"] >= 0.18
        ),
        "ordinary_known_accuracy_drop_at_most_0_005": (
            ordinary["known_object_accuracy"]
            >= initial_ordinary["known_object_accuracy"] - 0.005
        ),
        "protected_geometry_retention_at_least_0_95": (
            holdout["protected_geometry_retention"] >= 0.95
        ),
        "count_and_non_geometry_logits_unchanged": (
            _all_non_geometry_invariant(history)
            and final_metrics["collision_holdout"]["non_geometry_invariance"][
                "status"
            ]
            == "passed"
            and final_metrics["ordinary_validation"]["non_geometry_invariance"][
                "status"
            ]
            == "passed"
        ),
    }
    return {**checks, "go": all(checks.values())}


def _checkpoint(
    group_detector: BondGroupQueryDetector,
    config: BondGroupQueryConfig,
    args,
    *,
    samples_seen: int,
    metrics: dict,
    zero_step_parity: dict,
    dense_payload: dict,
) -> dict:
    return {
        "schema_version": "bond-group-query-checkpoint-0.1",
        "bond_group_query": group_detector.state_dict(),
        "config": asdict(config),
        "samples_seen": samples_seen,
        "metrics": metrics,
        "zero_step_parity": zero_step_parity,
        "training_args": vars(args),
        "dense_bond_checkpoint": args.init_checkpoint,
        "grounded_graph_checkpoint": dense_payload["grounded_graph_checkpoint"],
        "direction_loss_enabled": False,
        "assignment_phase": (
            "anchored"
            if args.mode == "generalization"
            and samples_seen <= args.initial_assignment_exposures
            else (
                "dynamic"
                if args.mode == "generalization"
                else args.assignment_mode
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train v0.8.29-v0.8.32 Bond group queries on controlled capacity "
            "or generalization protocols."
        )
    )
    parser.add_argument(
        "--mode", choices=("capacity100", "generalization"), default="capacity100"
    )
    parser.add_argument("--matched-exposure-control", action="store_true")
    parser.add_argument("--early-validation-control", action="store_true")
    parser.add_argument("--endpoint-local", action="store_true")
    parser.add_argument("--train-local-only", action="store_true")
    parser.add_argument("--endpoint-local-channels", type=int, default=32)
    parser.add_argument("--endpoint-patch-radius-cells", type=int, default=2)
    parser.add_argument("--endpoint-line-band", action="store_true")
    parser.add_argument("--endpoint-line-samples", type=int, default=32)
    parser.add_argument("--endpoint-line-radius-cells", type=int, default=2)
    parser.add_argument("--endpoint-line-feature-dim", type=int, default=64)
    parser.add_argument("--endpoint-ink-line", action="store_true")
    parser.add_argument("--endpoint-ink-line-samples", type=int, default=64)
    parser.add_argument("--endpoint-ink-line-band", type=int, default=25)
    parser.add_argument("--endpoint-ink-line-radius-pixels", type=float, default=12.0)
    parser.add_argument("--endpoint-ink-line-feature-dim", type=int, default=64)
    parser.add_argument(
        "--objective", choices=("joint", "geometry"), default="joint"
    )
    parser.add_argument(
        "--assignment-mode", choices=("dynamic", "initial"), default="dynamic"
    )
    parser.add_argument("--collision-bond-weight", type=float, default=16.0)
    parser.add_argument(
        "--init-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-*.tar"],
    )
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=[
            "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-*.tar"
        ],
    )
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument(
        "--output-dir",
        default=CAPACITY_OUTPUT_DIR,
    )
    parser.add_argument("--collision-train-images", type=int, default=100)
    parser.add_argument("--collision-holdout-images", type=int, default=100)
    parser.add_argument("--ordinary-replay-images", type=int, default=100)
    parser.add_argument("--train-exposures", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--endpoint-delta-scale", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--initial-assignment-exposures", type=int, default=1600)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    apply_mode_protocol(args)

    positive = (
        args.collision_train_images,
        args.collision_holdout_images,
        args.ordinary_replay_images,
        args.validation_samples,
        args.train_exposures,
        args.batch_size,
        args.learning_rate,
        args.queries,
        args.endpoint_delta_scale,
        args.grid_size,
        args.validate_every,
        args.initial_assignment_exposures,
        args.log_every,
        args.max_grad_norm,
    )
    if (
        min(positive) <= 0
        or args.weight_decay < 0.0
        or args.collision_bond_weight < 1.0
        or args.endpoint_local_channels < 1
        or args.endpoint_patch_radius_cells < 0
        or args.endpoint_line_samples < 2
        or args.endpoint_line_radius_cells < 0
        or args.endpoint_line_feature_dim < 1
        or args.endpoint_ink_line_samples < 2
        or args.endpoint_ink_line_band < 1
        or args.endpoint_ink_line_radius_pixels <= 0.0
        or args.endpoint_ink_line_feature_dim < 1
    ):
        raise ValueError("Bond Group Query training arguments are invalid")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.train_local_only and not args.endpoint_local:
        raise ValueError("--train-local-only requires --endpoint-local")
    if args.endpoint_line_band and not args.endpoint_local:
        raise ValueError("--endpoint-line-band requires --endpoint-local")
    if args.endpoint_ink_line and not args.endpoint_local:
        raise ValueError("--endpoint-ink-line requires --endpoint-local")
    if args.initial_assignment_exposures > args.train_exposures:
        raise ValueError("Initial-assignment exposures exceed total exposures")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    _prepare_output_directory(output_dir)
    collision_train, collision_holdout, ordinary_replay, split = build_collision_split(
        args.train_shards,
        grid_size=args.grid_size,
        collision_train_images=args.collision_train_images,
        collision_holdout_images=args.collision_holdout_images,
        ordinary_replay_images=args.ordinary_replay_images,
        seed=args.seed,
    )
    ordinary_validation = (
        load_records(args.validation_shards, args.validation_samples)
        if args.mode == "generalization"
        else []
    )
    train_records = (
        collision_train + ordinary_replay
        if args.mode == "generalization"
        else collision_train
    )
    exposure_order = (
        mixed_exposure_order(
            len(collision_train),
            len(ordinary_replay),
            args.train_exposures,
            args.batch_size,
            args.seed,
        )
        if args.mode == "generalization"
        else capacity_exposure_order(
            len(train_records), args.train_exposures, args.batch_size, args.seed
        )
    )

    device = torch.device(args.device)
    dense_payload = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    _, grounded_payload = load_grounded_graph_checkpoint(
        dense_payload["grounded_graph_checkpoint"], device
    )
    vision_model, _, _, _ = load_frozen_grounded_pipeline(
        Path(grounded_payload["grounded_selector_checkpoint"]), device
    )
    vision_model.requires_grad_(False).eval()
    dense_detector = load_detector(dense_payload, device)
    dense_detector.requires_grad_(False).eval()
    config = BondGroupQueryConfig(
        input_channels=int(vision_model.config.vision_dims[0]),
        proposal_channels=int(dense_payload["config"]["hidden_channels"]),
        queries=args.queries,
        endpoint_delta_scale=args.endpoint_delta_scale,
        endpoint_local_channels=(
            args.endpoint_local_channels if args.endpoint_local else 0
        ),
        endpoint_patch_radius_cells=args.endpoint_patch_radius_cells,
        endpoint_line_samples=(
            args.endpoint_line_samples if args.endpoint_line_band else 0
        ),
        endpoint_line_radius_cells=args.endpoint_line_radius_cells,
        endpoint_line_feature_dim=args.endpoint_line_feature_dim,
        endpoint_ink_line_samples=(
            args.endpoint_ink_line_samples if args.endpoint_ink_line else 0
        ),
        endpoint_ink_line_band=args.endpoint_ink_line_band,
        endpoint_ink_line_radius_pixels=args.endpoint_ink_line_radius_pixels,
        endpoint_ink_line_feature_dim=args.endpoint_ink_line_feature_dim,
    )
    group_detector = BondGroupQueryDetector(config).to(device)
    if args.train_local_only:
        set_endpoint_local_only_trainable(group_detector)
    collator = ChemSceneBondCollator()

    def evaluate_current() -> dict:
        if args.mode == "generalization":
            return {
                "collision_holdout": evaluate_capacity(
                    vision_model,
                    dense_detector,
                    group_detector,
                    collision_holdout,
                    collator,
                    device,
                    batch_size=args.batch_size,
                    queries=args.queries,
                    grid_size=args.grid_size,
                ),
                "ordinary_validation": evaluate_capacity(
                    vision_model,
                    dense_detector,
                    group_detector,
                    ordinary_validation,
                    collator,
                    device,
                    batch_size=args.batch_size,
                    queries=args.queries,
                    grid_size=args.grid_size,
                ),
            }
        return evaluate_capacity(
            vision_model,
            dense_detector,
            group_detector,
            collision_train,
            collator,
            device,
            batch_size=args.batch_size,
            queries=args.queries,
            grid_size=args.grid_size,
        )

    zero_step_parity = audit_zero_step_parity(
        vision_model,
        dense_detector,
        group_detector,
        collision_train,
        collator,
        device,
        batch_size=args.batch_size,
        queries=args.queries,
        threshold=args.threshold,
    )
    (output_dir / "zero_step_parity.json").write_text(
        json.dumps(zero_step_parity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if zero_step_parity["status"] != "passed":
        raise RuntimeError("Bond Group Query zero-step parity audit failed")

    initial_metrics = evaluate_current()
    if (
        args.mode == "generalization"
        and not _generalization_non_geometry_invariant(initial_metrics)
    ):
        raise RuntimeError("Initial non-geometry outputs differ from dense proposals")
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    initial_checkpoint = _checkpoint(
        group_detector,
        config,
        args,
        samples_seen=0,
        metrics=initial_metrics,
        zero_step_parity=zero_step_parity,
        dense_payload=dense_payload,
    )
    torch.save(initial_checkpoint, output_dir / "initial.pt")
    shutil.copy2(output_dir / "initial.pt", output_dir / "best.pt")

    trainable_parameters = [
        parameter for parameter in group_detector.parameters() if parameter.requires_grad
    ]
    total_parameter_count = sum(
        parameter.numel() for parameter in group_detector.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = [{"samples_seen": 0, "metrics": initial_metrics}]
    score_metrics = (
        _generalization_score if args.mode == "generalization" else _metric_score
    )
    best_score = score_metrics(initial_metrics)
    best_samples_seen = 0
    running_loss = 0.0
    running_batches = 0
    next_validation = args.validate_every
    for start in range(0, len(exposure_order), args.batch_size):
        indices = exposure_order[start : start + args.batch_size]
        batch = collator([train_records[index] for index in indices])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        stride4, _, proposals = _frozen_proposals(
            vision_model, dense_detector, pixels, queries=args.queries
        )
        group_detector.train()
        with _autocast(device):
            output = group_detector(stride4, proposals, pixel_values=pixels)
            if args.objective == "joint":
                losses = bond_group_query_loss(output, targets)
            else:
                use_initial_assignment = (
                    start < args.initial_assignment_exposures
                    if args.mode == "generalization"
                    else args.assignment_mode == "initial"
                )
                losses = bond_group_query_geometry_loss(
                    output,
                    targets,
                    matching_endpoints=(
                        proposals["base_endpoints"]
                        if use_initial_assignment
                        else None
                    ),
                    target_weights=collision_target_weights(
                        targets,
                        grid_size=args.grid_size,
                        collision_weight=args.collision_bond_weight,
                    ),
                    endpoint_barrier_weight=0.0,
                    endpoint_huber_beta_pixels=4.0,
                )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(group_detector.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        samples_seen = start + len(indices)
        running_loss += float(losses["loss"].detach())
        running_batches += 1
        if samples_seen % args.log_every == 0:
            print(
                f"samples={samples_seen} loss={running_loss / running_batches:.6f}",
                flush=True,
            )
            running_loss = 0.0
            running_batches = 0
        should_validate = samples_seen >= next_validation or samples_seen == len(
            exposure_order
        )
        if not should_validate:
            continue
        metrics = evaluate_current()
        if (
            args.mode == "generalization"
            and not _generalization_non_geometry_invariant(metrics)
        ):
            raise RuntimeError(
                "Geometry-only training changed count or non-geometry logits"
            )
        history.append({"samples_seen": samples_seen, "metrics": metrics})
        checkpoint = _checkpoint(
            group_detector,
            config,
            args,
            samples_seen=samples_seen,
            metrics=metrics,
            zero_step_parity=zero_step_parity,
            dense_payload=dense_payload,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        torch.save(checkpoint, output_dir / f"checkpoint-{samples_seen:06d}.pt")
        if (
            args.mode == "generalization"
            and samples_seen == args.initial_assignment_exposures
        ):
            shutil.copy2(output_dir / "last.pt", output_dir / "warmup.pt")
        score = score_metrics(metrics)
        if score > best_score:
            best_score = score
            best_samples_seen = samples_seen
            shutil.copy2(output_dir / "last.pt", output_dir / "best.pt")
        if args.mode == "generalization":
            holdout = metrics["collision_holdout"]
            ordinary = metrics["ordinary_validation"]
            message = (
                f"validation samples={samples_seen} "
                f"holdout_cell={holdout['collision_cell_geometry_exact_rate']:.4f} "
                f"holdout_secondary={holdout['secondary_geometry_recall']:.4f} "
                f"ordinary_known={ordinary['known_object_accuracy']:.4f}"
            )
        else:
            message = (
                f"validation samples={samples_seen} "
                f"cell_exact={metrics['collision_cell_geometry_exact_rate']:.4f} "
                f"secondary={metrics['secondary_geometry_recall']:.4f} "
                f"known_object={metrics['known_object_accuracy']:.4f} "
                f"pb_count={metrics['pb_count_accuracy']:.4f}"
            )
        print(message, flush=True)
        while next_validation <= samples_seen:
            next_validation += args.validate_every

    if not (output_dir / "last.pt").exists():
        final_metrics = initial_metrics
        shutil.copy2(output_dir / "initial.pt", output_dir / "last.pt")
    else:
        final_metrics = history[-1]["metrics"]
    best_metrics = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )["metrics"]
    summary = {
        "schema_version": (
            "bond-group-query-generalization-summary-0.1"
            if args.mode == "generalization"
            else "bond-group-query-capacity-summary-0.1"
        ),
        "mode": args.mode,
        "zero_step_parity": zero_step_parity,
        "split": split,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "best_samples_seen": best_samples_seen,
        "best_metrics": best_metrics,
        "history": history,
        "protocol": {
            **vars(args),
            "training_records": len(train_records),
            "capacity_records": (
                len(collision_train) if args.mode == "capacity100" else None
            ),
            "collision_training_records": len(collision_train),
            "ordinary_replay_records": (
                len(ordinary_replay) if args.mode == "generalization" else 0
            ),
            "collision_holdout_records": (
                len(collision_holdout) if args.mode == "generalization" else 0
            ),
            "ordinary_validation_records": len(ordinary_validation),
            "exposures_per_record": args.train_exposures // len(train_records),
            "optimizer_steps": args.train_exposures // args.batch_size,
            "model_parameters": total_parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "assignment_schedule": (
                {
                    "initial_until_exposure": args.initial_assignment_exposures,
                    "dynamic_after_exposure": args.initial_assignment_exposures,
                }
                if args.mode == "generalization"
                else {"mode": args.assignment_mode}
            ),
            "vision_encoder_frozen": True,
            "dense_detector_frozen": True,
            "direction_loss_enabled": False,
            "uses_mr_tadf_for_tuning": False,
            "uses_uspto_for_tuning": False,
        },
    }
    if args.mode == "generalization":
        summary["go_checks"] = generalization_checks(
            initial_metrics,
            best_metrics,
            final_metrics,
            history,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
