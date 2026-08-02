#!/usr/bin/env python3
"""Audit fixed canonical cell-slot identities in G3 P0-I checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.audit_g3_p0h_atom_only import (
    EXPECTED_STEPS,
    IMAGE_SIZE,
    MATCH_RADIUS_PIXELS,
    TOP128,
    conditional_localization_summary,
    config_from_payload,
    flatten_atom_candidates,
    hungarian_radius_match_count,
    identity_retention,
    localization_summary,
    model_from_checkpoint,
    ranked_set_retention,
    selector_identity_counts,
)
from chemvector.g3_hierarchical_graph import G3AtomOnlyPredictor
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    OVERFIT32_MANIFEST_SHA256,
    file_sha256,
    g3_targets_to_device,
    load_g3_manifest_samples,
    load_g3_overfit32_manifest,
)
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _autocast_context,
    cache_frozen_vision_features,
    load_frozen_vision_checkpoint,
)


DEFAULT_ARTIFACT_DIR = Path(
    "outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32"
)
EXPECTED_TARGET_ATOMS = 1424


def validate_p0i_checkpoint(payload: dict, *, step: int) -> None:
    if payload.get("global_step") != step:
        raise ValueError("P0-I checkpoint filename and global step differ")
    scope = payload.get("structural_scope", {})
    if (
        not scope.get("atom_only_capacity")
        or not scope.get("canonical_cell_slot_targets")
        or scope.get("bond_refs_and_types") is not False
        or scope.get("count_teacher_forcing") is not False
    ):
        raise ValueError("Atom audit accepts only the P0-I canonical cell-slot protocol")
    if any(
        scope.get(name)
        for name in (
            "gt_atom_slot_oracle",
            "shared_confidence_aware_bond_assignment",
            "gt_bond_slot_identity_oracle",
            "fixed_learned_bond_slots",
            "set_learned_bond_slots",
            "direct_gt_atom_pair_bonds",
        )
    ):
        raise ValueError("P0-I Atom audit rejects graph or oracle protocols")
    args = payload.get("training_args", {})
    if not args.get("atom_only_capacity") or not args.get(
        "canonical_cell_slot_targets"
    ):
        raise ValueError("P0-I checkpoint is missing its two required CLI flags")
    if payload.get("sample_manifest", {}).get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-I checkpoint is not bound to the committed overfit32")
    if payload.get("source_vision_checkpoint_sha256") != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-I checkpoint uses an unexpected vision source")
    metrics = payload.get("train_only_metrics", {})
    if not isinstance(metrics.get("p0i_gate"), dict) or "p0h_gate" in metrics:
        raise ValueError("P0-I checkpoint has an incompatible stored gate protocol")


def canonical_cell_slot_identities(
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    slots: int,
    height: int,
    width: int,
) -> tuple[list[list[tuple[int, int, int]]], dict[str, int]]:
    """Independently rebuild fixed identities from lexicographic GT anchors."""

    if target_anchors.ndim != 3 or target_anchors.shape[-1] != 2:
        raise ValueError("P0-I target anchors must have shape [batch, atoms, 2]")
    if target_mask.shape != target_anchors.shape[:2] or min(slots, height, width) < 1:
        raise ValueError("P0-I target mask or dense dimensions are incompatible")
    identities: list[list[tuple[int, int, int]]] = []
    occupancy_histogram: dict[int, int] = {}
    stats = {
        "images": len(target_anchors),
        "target_atoms": 0,
        "occupied_cells": 0,
        "collision_cells": 0,
        "atoms_in_collision_cells": 0,
        "extra_cell_occupancy": 0,
        "maximum_cell_occupancy": 0,
        "overflow_cells": 0,
        "overflow_atoms": 0,
    }
    for batch_index in range(len(target_anchors)):
        valid = torch.where(target_mask[batch_index].bool())[0].tolist()
        expected_prefix = list(range(len(valid)))
        if valid != expected_prefix:
            raise ValueError("P0-I canonical targets must use prefix Atom order")
        by_cell: dict[tuple[int, int], list[int]] = {}
        for target_index in valid:
            anchor = target_anchors[batch_index, target_index]
            if not bool(torch.isfinite(anchor).all()) or bool(
                ((anchor < 0.0) | (anchor > 1.0)).any()
            ):
                raise ValueError("P0-I canonical target anchor is invalid")
            x_index = min(width - 1, int(float(anchor[0]) * width))
            y_index = min(height - 1, int(float(anchor[1]) * height))
            by_cell.setdefault((y_index, x_index), []).append(target_index)
        sample: list[tuple[int, int, int] | None] = [None] * len(valid)
        for (y_index, x_index), target_indices in by_cell.items():
            coordinates = [
                tuple(
                    float(value)
                    for value in target_anchors[batch_index, target_index].tolist()
                )
                for target_index in target_indices
            ]
            if len(set(coordinates)) != len(coordinates):
                raise ValueError("P0-I canonical cell contains duplicate GT anchors")
            ordered = sorted(
                target_indices,
                key=lambda target_index: tuple(
                    float(value)
                    for value in target_anchors[batch_index, target_index].tolist()
                ),
            )
            occupancy = len(ordered)
            occupancy_histogram[occupancy] = occupancy_histogram.get(occupancy, 0) + 1
            stats["occupied_cells"] += 1
            stats["collision_cells"] += int(occupancy > 1)
            stats["atoms_in_collision_cells"] += occupancy if occupancy > 1 else 0
            stats["extra_cell_occupancy"] += max(occupancy - 1, 0)
            stats["maximum_cell_occupancy"] = max(
                stats["maximum_cell_occupancy"], occupancy
            )
            if occupancy > slots:
                stats["overflow_cells"] += 1
                stats["overflow_atoms"] += occupancy - slots
                continue
            for slot, target_index in enumerate(ordered):
                sample[target_index] = (slot, y_index, x_index)
        if any(identity is None for identity in sample):
            raise ValueError("P0-I canonical identity exceeds dense slot capacity")
        identities.append([identity for identity in sample if identity is not None])
        stats["target_atoms"] += len(sample)
    for occupancy, cells in sorted(occupancy_histogram.items()):
        stats[f"occupancy_{occupancy}_cells"] = cells
    return identities, stats


def selected_cell_statistics(
    selected: list[tuple[int, int, int]],
    fixed_positive: list[tuple[int, int, int]],
) -> dict[str, int]:
    """Separate raw same-cell selection from occupancy-relative errors."""

    selected_occupancy: dict[tuple[int, int], int] = {}
    target_occupancy: dict[tuple[int, int], int] = {}
    for _, y_index, x_index in selected:
        cell = (y_index, x_index)
        selected_occupancy[cell] = selected_occupancy.get(cell, 0) + 1
    for _, y_index, x_index in fixed_positive:
        cell = (y_index, x_index)
        target_occupancy[cell] = target_occupancy.get(cell, 0) + 1
    cells = set(selected_occupancy) | set(target_occupancy)
    return {
        "selected_multi_candidate_cells": sum(
            count > 1 for count in selected_occupancy.values()
        ),
        "selected_duplicate_candidates": sum(
            max(count - 1, 0) for count in selected_occupancy.values()
        ),
        "missing_target_cells": sum(
            cell not in selected_occupancy for cell in target_occupancy
        ),
        "missing_target_cell_atoms": sum(
            target_occupancy[cell]
            for cell in target_occupancy
            if cell not in selected_occupancy
        ),
        "excess_occupancy_cells": sum(
            selected_occupancy.get(cell, 0) > target_occupancy.get(cell, 0)
            for cell in cells
        ),
        "excess_occupancy_candidates": sum(
            max(
                selected_occupancy.get(cell, 0) - target_occupancy.get(cell, 0),
                0,
            )
            for cell in cells
        ),
        "missing_occupancy_cells": sum(
            selected_occupancy.get(cell, 0) < target_occupancy.get(cell, 0)
            for cell in cells
        ),
        "missing_occupancy_atoms": sum(
            max(
                target_occupancy.get(cell, 0) - selected_occupancy.get(cell, 0),
                0,
            )
            for cell in cells
        ),
    }


def cutoff_diagnostic(logits: torch.Tensor, count: int) -> dict[str, int | float | bool]:
    if logits.ndim != 1 or not 0 < count < len(logits):
        raise ValueError("P0-I cutoff diagnostic requires 0 < count < candidates")
    values = logits.float().sort(descending=True).values
    selected_cutoff = float(values[count - 1])
    rejected_cutoff = float(values[count])
    margin = selected_cutoff - rejected_cutoff
    above = int((logits.float() > selected_cutoff).sum())
    equal = int((logits.float() == selected_cutoff).sum())
    tie_crosses = above < count < above + equal
    return {
        "selected_cutoff_logit": selected_cutoff,
        "best_rejected_logit": rejected_cutoff,
        "cutoff_margin": margin,
        "equal_to_selected_cutoff": equal,
        "tie_crosses_cutoff": tie_crosses,
    }


def cutoff_summary(diagnostics: list[dict]) -> dict[str, int | float]:
    if not diagnostics:
        raise ValueError("P0-I audit requires cutoff diagnostics")
    margins = torch.tensor(
        [item["cutoff_margin"] for item in diagnostics], dtype=torch.float64
    )
    return {
        "images": len(diagnostics),
        "tie_crosses_cutoff_images": sum(
            bool(item["tie_crosses_cutoff"]) for item in diagnostics
        ),
        "zero_margin_images": int((margins == 0.0).sum()),
        "minimum_margin": float(margins.min()),
        "p10_margin": float(margins.quantile(0.1)),
        "median_margin": float(margins.quantile(0.5)),
        "p90_margin": float(margins.quantile(0.9)),
    }


@torch.no_grad()
def audit_checkpoint(
    model: G3AtomOnlyPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    sample_keys: list[str],
    fixed_identities: list[list[tuple[int, int, int]]],
    batch_size: int,
    precision: str,
) -> tuple[dict, list[list[tuple[int, int, int]]]]:
    model.eval()
    totals = {
        "images": 0,
        "target_atoms": 0,
        "deploy_count_exact_images": 0,
        "deploy_count_absolute_error": 0,
        "selector_identity_selected": 0,
        "selector_identity_fixed_positive": 0,
        "selector_identity_intersection": 0,
        "selector_identity_selected_only": 0,
        "selector_identity_positive_only": 0,
        "selector_identity_whole_set_exact_images": 0,
    }
    for mode in ("all_dense", "top128", "gt_count_topk"):
        totals[f"{mode}_matched_atoms"] = 0
        totals[f"{mode}_set_exact_images"] = 0
    for name in (
        "selected_multi_candidate_cells",
        "selected_duplicate_candidates",
        "missing_target_cells",
        "missing_target_cell_atoms",
        "excess_occupancy_cells",
        "excess_occupancy_candidates",
        "missing_occupancy_cells",
        "missing_occupancy_atoms",
    ):
        totals[name] = 0
    fixed_errors: list[float] = []
    selected_fixed_errors: list[float] = []
    cutoff_rows = []
    topk_snapshots: list[list[tuple[int, int, int]]] = []
    per_sample = []
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        with _autocast_context(features.device, precision):
            dense = model.atom_head(current_features)
        flat_logits, flat_anchors, flat_ids = flatten_atom_candidates(dense)
        deploy_counts = dense["count_logits"].argmax(dim=-1)
        for batch_index in range(len(current_features)):
            global_index = start + batch_index
            expected = current_targets["atom_anchors"][
                batch_index, current_targets["atom_mask"][batch_index].bool()
            ]
            count = len(expected)
            ranked = flat_logits[batch_index].argsort(descending=True)
            top128_indices = ranked[: min(TOP128, len(ranked))]
            gt_indices = ranked[:count]
            selected = [
                tuple(int(value) for value in flat_ids[index].tolist())
                for index in gt_indices.tolist()
            ]
            fixed = fixed_identities[global_index]
            selected_set = set(selected)
            identity_counts = selector_identity_counts(selected, fixed)
            cell_counts = selected_cell_statistics(selected, fixed)
            for name, value in cell_counts.items():
                totals[name] += value

            sample_fixed_errors = []
            for target_index, (slot, y_index, x_index) in enumerate(fixed):
                predicted = dense["anchors"][
                    batch_index, slot, :, y_index, x_index
                ]
                error = float(
                    torch.linalg.vector_norm(
                        predicted.float() - expected[target_index].float()
                    )
                    * IMAGE_SIZE
                )
                sample_fixed_errors.append(error)
                fixed_errors.append(error)
                if (slot, y_index, x_index) in selected_set:
                    selected_fixed_errors.append(error)
            mode_anchors = {
                "all_dense": flat_anchors[batch_index],
                "top128": flat_anchors[batch_index, top128_indices],
                "gt_count_topk": flat_anchors[batch_index, gt_indices],
            }
            matches = {
                mode: hungarian_radius_match_count(anchors, expected)
                for mode, anchors in mode_anchors.items()
            }
            cutoff = cutoff_diagnostic(flat_logits[batch_index], count)
            cutoff_rows.append(cutoff)
            deploy_count = int(deploy_counts[batch_index])
            totals["images"] += 1
            totals["target_atoms"] += count
            totals["deploy_count_exact_images"] += int(deploy_count == count)
            totals["deploy_count_absolute_error"] += abs(deploy_count - count)
            totals["selector_identity_selected"] += identity_counts["selected"]
            totals["selector_identity_fixed_positive"] += identity_counts["positive"]
            totals["selector_identity_intersection"] += identity_counts["intersection"]
            totals["selector_identity_selected_only"] += identity_counts["selected_only"]
            totals["selector_identity_positive_only"] += identity_counts["positive_only"]
            totals["selector_identity_whole_set_exact_images"] += identity_counts[
                "whole_set_exact"
            ]
            for mode, matched in matches.items():
                totals[f"{mode}_matched_atoms"] += matched
                totals[f"{mode}_set_exact_images"] += int(matched == count)
            topk_snapshots.append(selected)
            per_sample.append(
                {
                    "sample_key": sample_keys[global_index],
                    "target_atoms": count,
                    "deploy_count": deploy_count,
                    "fixed_positive_ids_by_target": [list(value) for value in fixed],
                    "gt_count_topk_ids_by_rank": [list(value) for value in selected],
                    "fixed_positive_errors_pixels": sample_fixed_errors,
                    "matches_6px": matches,
                    "selector_identity": identity_counts,
                    "selected_cell_statistics": cell_counts,
                    "cutoff": cutoff,
                }
            )
    images = max(totals["images"], 1)
    atoms = max(totals["target_atoms"], 1)
    totals.update(
        {
            "deploy_count_accuracy": totals["deploy_count_exact_images"] / images,
            "deploy_count_mae": totals["deploy_count_absolute_error"] / images,
            "selector_identity_micro_precision": totals[
                "selector_identity_intersection"
            ]
            / max(totals["selector_identity_selected"], 1),
            "selector_identity_micro_recall": totals[
                "selector_identity_intersection"
            ]
            / max(totals["selector_identity_fixed_positive"], 1),
            "selector_identity_whole_set_exact_rate": totals[
                "selector_identity_whole_set_exact_images"
            ]
            / images,
        }
    )
    for mode in ("all_dense", "top128", "gt_count_topk"):
        totals[f"{mode}_micro_recall_6px"] = totals[
            f"{mode}_matched_atoms"
        ] / atoms
        totals[f"{mode}_set_exact_rate_6px"] = totals[
            f"{mode}_set_exact_images"
        ] / images
    return {
        "metrics": totals,
        "fixed_positive_localization": localization_summary(fixed_errors),
        "selected_fixed_positive_conditional_localization": (
            conditional_localization_summary(selected_fixed_errors)
        ),
        "cutoff_margin": cutoff_summary(cutoff_rows),
        "samples": per_sample,
    }, topk_snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--output")
    return parser


def p0i_audit_gate(
    canonical_stats: dict,
    final_report: dict,
    identity_churn: dict,
    stored_gate: dict,
    *,
    expected_images: int = 32,
    expected_atoms: int = EXPECTED_TARGET_ATOMS,
) -> dict:
    """Apply the pre-registered P0-I protocol and fixed-final capability gates."""

    metrics = final_report["metrics"]
    localization = final_report["fixed_positive_localization"]
    minimum_exact_images = 30 if expected_images == 32 else expected_images
    checks = {
        "canonical_target_atoms": canonical_stats["target_atoms"] == expected_atoms,
        "canonical_occupied_cells": (
            canonical_stats["occupied_cells"] == expected_atoms
        ),
        "canonical_single_occupancy": (
            canonical_stats["collision_cells"] == 0
            and canonical_stats["extra_cell_occupancy"] == 0
            and canonical_stats["maximum_cell_occupancy"] == 1
        ),
        "canonical_no_overflow": (
            canonical_stats["overflow_cells"] == 0
            and canonical_stats["overflow_atoms"] == 0
        ),
        "canonical_identity_zero_churn": identity_churn["changed"] == 0,
        "stored_fixed_final_gate": stored_gate.get("status") == "passed",
        "deploy_atom_count_exact": (
            metrics["deploy_count_exact_images"] >= minimum_exact_images
        ),
        "all_dense_set_exact": (
            metrics["all_dense_set_exact_images"] == expected_images
        ),
        "top128_set_exact": (
            metrics["top128_set_exact_images"] == expected_images
        ),
        "fixed_positive_localization": (
            localization["within_6px"] == expected_atoms
        ),
        "selector_identity_micro": (
            metrics["selector_identity_micro_precision"] >= 0.99
            and metrics["selector_identity_micro_recall"] >= 0.99
        ),
        "selector_identity_set_exact": (
            metrics["selector_identity_whole_set_exact_images"]
            >= minimum_exact_images
        ),
        "gt_count_atom_set_exact": (
            metrics["gt_count_topk_set_exact_images"] >= minimum_exact_images
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "fixed_final_step_only": True,
        "expected_images": expected_images,
        "minimum_exact_images": minimum_exact_images,
        "expected_atoms": expected_atoms,
        "checks": checks,
    }


def main() -> None:
    args = build_parser().parse_args()
    if min(args.batch_size, args.vision_batch_size) < 1:
        raise ValueError("P0-I audit batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-I audit requested CUDA but CUDA is unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-I CPU audit requires fp32")
    artifact_dir = Path(args.artifact_dir)
    paths = [artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("P0-I audit requires all six fixed checkpoints")

    device = torch.device(args.device)
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    validate_p0i_checkpoint(first, step=EXPECTED_STEPS[0])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        first["source_vision_checkpoint"], device
    )
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-I audit vision bytes differ from the frozen protocol")
    config = config_from_payload(first)
    collator = G3StructuralCollator(
        maximum_atoms=config.maximum_atoms,
        maximum_bonds=config.maximum_bonds,
        image_size=768,
    )
    batch = collator(records)
    features = cache_frozen_vision_features(
        vision,
        batch["pixel_values"],
        device=device,
        batch_size=args.vision_batch_size,
        precision=args.precision,
    )
    targets = g3_targets_to_device(batch, device)
    sample_keys = list(batch["keys"])
    del vision, batch

    fixed, canonical_stats = canonical_cell_slot_identities(
        targets["atom_anchors"],
        targets["atom_mask"],
        slots=config.atom_slots_per_cell,
        height=features.shape[-2],
        width=features.shape[-1],
    )
    if canonical_stats["target_atoms"] != EXPECTED_TARGET_ATOMS:
        raise ValueError("P0-I canonical target coverage differs from 1424")
    if canonical_stats["overflow_cells"] or canonical_stats["overflow_atoms"]:
        raise ValueError("P0-I canonical targets exceed slot capacity")

    reports = []
    topk_snapshots = []
    stored_final_gate = None
    for step, path in zip(EXPECTED_STEPS, paths):
        payload = first if step == EXPECTED_STEPS[0] else torch.load(
            path, map_location="cpu", weights_only=False
        )
        validate_p0i_checkpoint(payload, step=step)
        if config_from_payload(payload) != config:
            raise ValueError("P0-I checkpoint model configs differ")
        model = model_from_checkpoint(payload, config, device)
        current, current_topk = audit_checkpoint(
            model,
            features,
            targets,
            sample_keys=sample_keys,
            fixed_identities=fixed,
            batch_size=args.batch_size,
            precision=args.precision,
        )
        current.update(
            {
                "step": step,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": file_sha256(path),
            }
        )
        reports.append(current)
        topk_snapshots.append(current_topk)
        if step == EXPECTED_STEPS[-1]:
            stored_final_gate = payload["train_only_metrics"]["p0i_gate"]
        del model, payload

    fixed_snapshots = [fixed for _ in EXPECTED_STEPS]
    fixed_transitions = []
    ranking_transitions = []
    for index in range(1, len(EXPECTED_STEPS)):
        fixed_retention = identity_retention(
            fixed_snapshots[index - 1], fixed_snapshots[index]
        )
        if fixed_retention["changed"] != 0:
            raise RuntimeError("P0-I canonical target identity churn is not zero")
        fixed_transitions.append(
            {
                "from_step": EXPECTED_STEPS[index - 1],
                "to_step": EXPECTED_STEPS[index],
                **fixed_retention,
            }
        )
        ranking_transitions.append(
            {
                "from_step": EXPECTED_STEPS[index - 1],
                "to_step": EXPECTED_STEPS[index],
                **ranked_set_retention(
                    topk_snapshots[index - 1], topk_snapshots[index]
                ),
            }
        )
    first_final_fixed = identity_retention(fixed, fixed)
    if first_final_fixed["changed"] != 0:
        raise RuntimeError("P0-I first-to-final canonical identity churn is not zero")
    if stored_final_gate is None:
        raise RuntimeError("P0-I audit did not observe the fixed-final stored gate")
    gate = p0i_audit_gate(
        canonical_stats,
        reports[-1],
        first_final_fixed,
        stored_final_gate,
    )
    report = {
        "schema_version": "g3-p0i-canonical-cell-slot-audit-0.1",
        "scope": "train-only-overfit32-no-grad-replay",
        "manifest": manifest_audit,
        "vision": {
            "path": str(Path(first["source_vision_checkpoint"]).resolve()),
            "sha256": vision_sha,
        },
        "implementation": {
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": file_sha256(__file__),
        },
        "expected_steps": list(EXPECTED_STEPS),
        "canonical_identity_rule": (
            "within each cell sort normalized anchors lexicographically by (x,y); "
            "duplicate coordinates are rejected"
        ),
        "canonical_target_audit": canonical_stats,
        "checkpoints": reports,
        "canonical_identity_adjacent_churn": fixed_transitions,
        "canonical_identity_first_to_final_churn": first_final_fixed,
        "gt_count_topk_adjacent_ranking_churn": ranking_transitions,
        "gt_count_topk_first_to_final_ranking_churn": ranked_set_retention(
            topk_snapshots[0], topk_snapshots[-1]
        ),
        "p0i_audit_gate": gate,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
