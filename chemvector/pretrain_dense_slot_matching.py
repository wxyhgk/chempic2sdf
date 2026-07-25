#!/usr/bin/env python3
"""Controlled collision pilot for permutation-invariant dense Bond slots."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import random
import shutil

from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_1 import build_chemscene_v0_1
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    MultiScaleDenseBondObjectDetector,
    _unordered_endpoint_l1_cost,
    build_dense_bond_targets,
    dense_bond_object_loss,
)
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_chemscene_bond_query import (
    ChemSceneBondCollator,
    stack_targets,
)
from chemvector.pretrain_dense_bond_object import (
    _apply_detector,
    _extract_detector_features,
    evaluate as evaluate_regression,
    prepare_output_directory,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.train import expand_shards


def record_cell_occupancies(record: dict, grid_size: int) -> Counter:
    scene = build_chemscene_v0_1(
        record["chemmol"], record["render"], record.get("layers")
    )
    occupancy: Counter = Counter()
    for bond in scene["bond_objects"]:
        (begin_x, begin_y), (end_x, end_y) = bond["endpoints"]
        center_x = (float(begin_x) + float(end_x)) / 2.0 / 1023.0
        center_y = (float(begin_y) + float(end_y)) / 2.0 / 1023.0
        cell_x = min(grid_size - 1, max(0, int(center_x * grid_size)))
        cell_y = min(grid_size - 1, max(0, int(center_y * grid_size)))
        occupancy[(cell_y, cell_x)] += 1
    return occupancy


def record_maximum_occupancy(record: dict, grid_size: int) -> int:
    return max(record_cell_occupancies(record, grid_size).values(), default=0)


def stable_rank(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_whole_molecule_groups(
    groups: dict[str, list[dict]],
    count: int,
    *,
    seed: int,
    prioritize_occupancy: bool,
) -> tuple[list[dict], set[str]]:
    ranked = sorted(
        groups,
        key=lambda molecule_id: (
            -max(item["maximum_occupancy"] for item in groups[molecule_id])
            if prioritize_occupancy
            else 0,
            stable_rank(seed, molecule_id),
            molecule_id,
        ),
    )
    selected: list[dict] = []
    molecules: set[str] = set()
    for molecule_id in ranked:
        items = sorted(groups[molecule_id], key=lambda item: item["record"]["key"])
        if len(selected) + len(items) > count:
            continue
        selected.extend(items)
        molecules.add(molecule_id)
        if len(selected) == count:
            return selected, molecules
    raise ValueError(f"Could not select exactly {count} collision images by molecule")


def build_collision_split(
    shards: list[str],
    *,
    grid_size: int,
    collision_train_images: int,
    collision_holdout_images: int,
    ordinary_replay_images: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    collision_groups: dict[str, list[dict]] = defaultdict(list)
    source_samples = 0
    for record in ChemVectorIterableDataset(
        expand_shards(shards), sequence_version="0.4"
    ):
        source_samples += 1
        maximum = record_maximum_occupancy(record, grid_size)
        if maximum < 2:
            continue
        molecule_id = str(record["meta"]["molecule_id"])
        collision_groups[molecule_id].append(
            {"record": record, "maximum_occupancy": maximum}
        )

    holdout_items, holdout_molecules = select_whole_molecule_groups(
        collision_groups,
        collision_holdout_images,
        seed=seed + 1009,
        prioritize_occupancy=False,
    )
    remaining_groups = {
        molecule_id: items
        for molecule_id, items in collision_groups.items()
        if molecule_id not in holdout_molecules
    }
    train_items, train_molecules = select_whole_molecule_groups(
        remaining_groups,
        collision_train_images,
        seed=seed + 2003,
        prioritize_occupancy=True,
    )

    ordinary_heap: list[tuple[int, str, dict]] = []
    for record in ChemVectorIterableDataset(
        expand_shards(shards), sequence_version="0.4"
    ):
        molecule_id = str(record["meta"]["molecule_id"])
        if molecule_id in holdout_molecules:
            continue
        if record_maximum_occupancy(record, grid_size) != 1:
            continue
        rank = stable_rank(seed + 3001, record["key"])
        item = (-rank, record["key"], record)
        if len(ordinary_heap) < ordinary_replay_images:
            heapq.heappush(ordinary_heap, item)
        elif rank < -ordinary_heap[0][0]:
            heapq.heapreplace(ordinary_heap, item)
    if len(ordinary_heap) != ordinary_replay_images:
        raise ValueError("Ordinary replay selection is incomplete")
    ordinary_records = [item[2] for item in sorted(ordinary_heap, reverse=True)]
    collision_train = [item["record"] for item in train_items]
    collision_holdout = [item["record"] for item in holdout_items]
    manifest = {
        "schema_version": "dense-slot-collision-split-0.1",
        "source_samples": source_samples,
        "collision_images_available": sum(map(len, collision_groups.values())),
        "collision_molecules_available": len(collision_groups),
        "grid_size": grid_size,
        "seed": seed,
        "collision_train": [
            {
                "key": item["record"]["key"],
                "molecule_id": item["record"]["meta"]["molecule_id"],
                "variant": item["record"]["render"]["variant"],
                "maximum_occupancy": item["maximum_occupancy"],
            }
            for item in train_items
        ],
        "collision_holdout": [
            {
                "key": item["record"]["key"],
                "molecule_id": item["record"]["meta"]["molecule_id"],
                "variant": item["record"]["render"]["variant"],
                "maximum_occupancy": item["maximum_occupancy"],
            }
            for item in holdout_items
        ],
        "ordinary_replay": [
            {
                "key": record["key"],
                "molecule_id": record["meta"]["molecule_id"],
                "variant": record["render"]["variant"],
            }
            for record in ordinary_records
        ],
        "molecule_overlap": {
            "train_holdout": len(train_molecules & holdout_molecules),
            "ordinary_holdout": len(
                {str(item["meta"]["molecule_id"]) for item in ordinary_records}
                & holdout_molecules
            ),
        },
    }
    return collision_train, collision_holdout, ordinary_records, manifest


def oriented_endpoint_errors(
    predicted: torch.Tensor, expected: torch.Tensor, image_size: float
) -> torch.Tensor:
    forward = torch.linalg.vector_norm((predicted - expected) * image_size, dim=-1)
    reverse = torch.linalg.vector_norm(
        (predicted - expected.flip(0)) * image_size, dim=-1
    )
    return forward if float(forward.sum()) <= float(reverse.sum()) else reverse


@torch.inference_mode()
def evaluate_collision_cells(
    model,
    detector,
    records: list[dict],
    collator: ChemSceneBondCollator,
    device: torch.device,
    *,
    batch_size: int,
    threshold: float,
    endpoint_tolerance_pixels: float = 6.0,
    duplicate_tolerance_pixels: float = 8.0,
) -> dict:
    detector.eval()
    totals = Counter()
    occupancy_cells = Counter()
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _apply_detector(
                detector, _extract_detector_features(model, detector, pixels)
            )
        dense = build_dense_bond_targets(
            targets,
            height=output["heatmap_logits"].size(-2),
            width=output["heatmap_logits"].size(-1),
            slots=output["heatmap_logits"].size(1),
        )
        for batch_index, cell_y, cell_x in zip(
            *(item.tolist() for item in torch.where(dense["occupancy"] >= 2))
        ):
            count = int(dense["occupancy"][batch_index, cell_y, cell_x])
            occupancy_cells[count] += 1
            totals["collision_cells"] += 1
            totals["collision_bonds"] += count
            expected = dense["endpoints"][
                batch_index, :count, :, :, cell_y, cell_x
            ].float()
            predicted = output["endpoints"][
                batch_index, :, :, :, cell_y, cell_x
            ].float()
            cost = _unordered_endpoint_l1_cost(predicted, expected)
            rows, columns = linear_sum_assignment(cost.cpu().numpy())
            matched = {int(column): int(row) for row, column in zip(rows, columns)}
            geometry_correct = []
            object_correct = []
            for target_index in range(count):
                slot = matched[target_index]
                errors = oriented_endpoint_errors(
                    predicted[slot], expected[target_index], 768.0
                )
                geometry = bool((errors <= endpoint_tolerance_pixels).all())
                display = int(
                    output["display_logits"][
                        batch_index, slot, :, cell_y, cell_x
                    ].argmax()
                ) == int(dense["display"][batch_index, target_index, cell_y, cell_x])
                chemical = int(
                    output["chemical_logits"][
                        batch_index, slot, :, cell_y, cell_x
                    ].argmax()
                ) == int(dense["chemical"][batch_index, target_index, cell_y, cell_x])
                correct = geometry and display and chemical
                geometry_correct.append(geometry)
                object_correct.append(correct)
                totals["geometry_correct_bonds"] += int(geometry)
                totals["object_correct_bonds"] += int(correct)
                role = "primary" if target_index == 0 else "secondary"
                totals[f"{role}_bonds"] += 1
                totals[f"{role}_geometry_correct"] += int(geometry)
            totals["cell_geometry_exact"] += int(all(geometry_correct))
            totals["cell_object_exact"] += int(all(object_correct))
            probabilities = output["heatmap_logits"][
                batch_index, :, cell_y, cell_x
            ].float().sigmoid()
            totals["cell_count_exact"] += int(int((probabilities >= threshold).sum()) == count)
            assigned_slots = set(matched.values())
            duplicate_slots = 0
            for slot in range(len(probabilities)):
                if slot in assigned_slots or float(probabilities[slot]) < threshold:
                    continue
                closest = min(
                    float(oriented_endpoint_errors(predicted[slot], target, 768.0).max())
                    for target in expected
                )
                duplicate_slots += int(closest <= duplicate_tolerance_pixels)
            totals["duplicate_cells"] += int(duplicate_slots > 0)
            totals["duplicate_slots"] += duplicate_slots
    cells = max(totals["collision_cells"], 1)
    bonds = max(totals["collision_bonds"], 1)
    secondary = max(totals["secondary_bonds"], 1)
    return {
        "samples": len(records),
        "collision_cells": totals["collision_cells"],
        "collision_bonds": totals["collision_bonds"],
        "occupancy_cells": {str(key): value for key, value in sorted(occupancy_cells.items())},
        "cell_geometry_exact_rate": totals["cell_geometry_exact"] / cells,
        "cell_object_exact_rate": totals["cell_object_exact"] / cells,
        "cell_count_accuracy": totals["cell_count_exact"] / cells,
        "bond_geometry_recall": totals["geometry_correct_bonds"] / bonds,
        "bond_object_recall": totals["object_correct_bonds"] / bonds,
        "primary_geometry_recall": totals["primary_geometry_correct"]
        / max(totals["primary_bonds"], 1),
        "secondary_geometry_recall": totals["secondary_geometry_correct"] / secondary,
        "duplicate_cell_rate": totals["duplicate_cells"] / cells,
        "duplicate_slots_per_cell": totals["duplicate_slots"] / cells,
    }


def load_detector(initial: dict, device: torch.device) -> MultiScaleDenseBondObjectDetector:
    config = DenseBondObjectConfig(**initial["config"])
    detector = MultiScaleDenseBondObjectDetector(config).to(device)
    detector.load_state_dict(initial["dense_bond_object"])
    return detector


def train_mode(
    mode: str,
    *,
    model,
    initial: dict,
    train_records: list[dict],
    collision_holdout: list[dict],
    regression_records: list[dict],
    collator: ChemSceneBondCollator,
    device: torch.device,
    args,
    output_dir: Path,
    initial_collision: dict,
    initial_regression: dict,
) -> dict:
    mode_dir = output_dir / mode
    prepare_output_directory(mode_dir)
    detector = load_detector(initial, device)
    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    orders = []
    for epoch in range(args.epochs):
        order = list(range(len(train_records)))
        random.Random(args.seed + epoch).shuffle(order)
        orders.append(order)

    best_score = (-1.0, -1.0, -1.0)
    best_metrics = None
    samples_seen = 0
    for epoch, order in enumerate(orders, start=1):
        detector.train()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            records = [train_records[index] for index in order[start : start + args.batch_size]]
            batch = collator(records)
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                features = _extract_detector_features(model, detector, pixels)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = _apply_detector(detector, features)
                losses = dense_bond_object_loss(
                    output,
                    targets,
                    endpoint_barrier_weight=args.endpoint_barrier_weight,
                    endpoint_barrier_margin_pixels=args.endpoint_barrier_margin_pixels,
                    heatmap_set_weight=args.heatmap_set_weight,
                    heatmap_set_margin=args.heatmap_set_margin,
                    slot_matching=mode,
                    cell_count_weight=args.cell_count_weight,
                    duplicate_weight=args.duplicate_weight,
                    duplicate_sigma_pixels=args.duplicate_sigma_pixels,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(records)
            total_loss += float(losses["loss"].detach())
            batches += 1
        metrics = evaluate_collision_cells(
            model,
            detector,
            collision_holdout,
            collator,
            device,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        score = (
            metrics["cell_geometry_exact_rate"],
            metrics["secondary_geometry_recall"],
            -metrics["duplicate_cell_rate"],
        )
        print(
            f"mode={mode} epoch={epoch} loss={total_loss / max(batches, 1):.6f} "
            f"cell_exact={metrics['cell_geometry_exact_rate']:.4f} "
            f"secondary={metrics['secondary_geometry_recall']:.4f}",
            flush=True,
        )
        checkpoint = {
            "dense_bond_object": detector.state_dict(),
            "config": initial["config"],
            "slot_matching": mode,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "collision_metrics": metrics,
            "training_args": vars(args),
            "init_checkpoint": args.init_checkpoint,
        }
        torch.save(checkpoint, mode_dir / "last.pt")
        if score > best_score:
            best_score = score
            best_metrics = metrics
            shutil.copy2(mode_dir / "last.pt", mode_dir / "best.pt")

    best = torch.load(mode_dir / "best.pt", map_location="cpu", weights_only=False)
    detector.load_state_dict(best["dense_bond_object"])
    final_collision = evaluate_collision_cells(
        model,
        detector,
        collision_holdout,
        collator,
        device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    final_regression = evaluate_regression(
        model,
        detector,
        regression_records,
        collator,
        device,
        args.batch_size,
        args.threshold,
    )
    summary = {
        "schema_version": "dense-slot-matching-pilot-0.1",
        "mode": mode,
        "initial_collision": initial_collision,
        "initial_regression": initial_regression,
        "collision": final_collision,
        "regression": final_regression,
        "best_epoch": best["epoch"],
        "best_training_collision": best_metrics,
        "training_args": vars(args),
    }
    (mode_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--regression-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-*.tar"],
    )
    parser.add_argument("--collision-train-images", type=int, default=400)
    parser.add_argument("--collision-holdout-images", type=int, default=100)
    parser.add_argument("--ordinary-replay-images", type=int, default=400)
    parser.add_argument("--regression-images", type=int, default=498)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--endpoint-barrier-weight", type=float, default=0.02)
    parser.add_argument("--endpoint-barrier-margin-pixels", type=float, default=4.0)
    parser.add_argument("--heatmap-set-weight", type=float, default=0.5)
    parser.add_argument("--heatmap-set-margin", type=float, default=1.0)
    parser.add_argument("--cell-count-weight", type=float, default=0.1)
    parser.add_argument("--duplicate-weight", type=float, default=0.05)
    parser.add_argument("--duplicate-sigma-pixels", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--modes", nargs="+", choices=("fixed", "hungarian"), default=("fixed", "hungarian")
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v0828-dense-slot-matching-collision500",
    )
    args = parser.parse_args()
    if min(
        args.collision_train_images,
        args.collision_holdout_images,
        args.ordinary_replay_images,
        args.regression_images,
        args.epochs,
        args.batch_size,
        args.grid_size,
    ) < 1:
        raise ValueError("Dense slot pilot counts must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"Dense slot pilot output already exists: {output_dir}")

    collision_train, collision_holdout, ordinary_replay, manifest = build_collision_split(
        args.train_shards,
        grid_size=args.grid_size,
        collision_train_images=args.collision_train_images,
        collision_holdout_images=args.collision_holdout_images,
        ordinary_replay_images=args.ordinary_replay_images,
        seed=args.seed,
    )
    (output_dir / "split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    regression_records = []
    for record in ChemVectorIterableDataset(
        expand_shards(args.regression_shards), sequence_version="0.4"
    ):
        regression_records.append(record)
        if len(regression_records) == args.regression_images:
            break
    if len(regression_records) != args.regression_images:
        raise ValueError("Dense slot regression set is incomplete")

    device = torch.device("cuda")
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    _, grounded_payload = load_grounded_graph_checkpoint(
        initial["grounded_graph_checkpoint"], device
    )
    model, _, _, _ = load_frozen_grounded_pipeline(
        Path(grounded_payload["grounded_selector_checkpoint"]), device
    )
    model.vision_encoder.requires_grad_(False).eval()
    collator = ChemSceneBondCollator()
    initial_detector = load_detector(initial, device)
    initial_collision = evaluate_collision_cells(
        model,
        initial_detector,
        collision_holdout,
        collator,
        device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    initial_regression = evaluate_regression(
        model,
        initial_detector,
        regression_records,
        collator,
        device,
        args.batch_size,
        args.threshold,
    )
    train_records = collision_train + ordinary_replay
    summaries = {}
    for mode in args.modes:
        summaries[mode] = train_mode(
            mode,
            model=model,
            initial=initial,
            train_records=train_records,
            collision_holdout=collision_holdout,
            regression_records=regression_records,
            collator=collator,
            device=device,
            args=args,
            output_dir=output_dir,
            initial_collision=initial_collision,
            initial_regression=initial_regression,
        )

    fixed = summaries.get("fixed", {"collision": initial_collision, "regression": initial_regression})
    hungarian = summaries.get("hungarian")
    comparison = None
    if hungarian is not None:
        fixed_collision = fixed["collision"]
        fixed_regression = fixed["regression"]
        treatment_collision = hungarian["collision"]
        treatment_regression = hungarian["regression"]
        checks = {
            "cell_geometry_exact_gain_at_least_10pp": treatment_collision[
                "cell_geometry_exact_rate"
            ]
            - fixed_collision["cell_geometry_exact_rate"]
            >= 0.10,
            "secondary_geometry_recall_at_least_50pct": treatment_collision[
                "secondary_geometry_recall"
            ]
            >= 0.50,
            "duplicate_cell_rate_halved": treatment_collision["duplicate_cell_rate"]
            <= fixed_collision["duplicate_cell_rate"] * 0.5,
            "regression_known_object_accuracy_drop_at_most_0_5pp": treatment_regression[
                "known_object_accuracy"
            ]
            >= fixed_regression["known_object_accuracy"] - 0.005,
        }
        comparison = {
            "fixed": fixed,
            "hungarian": hungarian,
            "checks": checks,
            "go": all(checks.values()),
        }
    result = {
        "schema_version": "dense-slot-matching-comparison-0.1",
        "split": manifest,
        "initial_collision": initial_collision,
        "initial_regression": initial_regression,
        "runs": summaries,
        "comparison": comparison,
        "protocol": {
            **vars(args),
            "uses_mr_tadf_for_tuning": False,
            "uses_uspto_for_tuning": False,
            "vision_encoder_frozen": True,
            "same_training_order_across_modes": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
