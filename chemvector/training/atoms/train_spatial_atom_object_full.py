#!/usr/bin/env python3
"""Train the spatial AtomObject head on the full ChemVision train split."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset, structure_labels
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
)
from chemvector.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
    spatial_atom_object_loss,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_spatial_atom_object import evaluate, load_grounded_vision
from chemvector.train_spatial_incidence import DEFAULT_ATOM, DEFAULT_GRAPH


DEFAULT_TRAIN_SHARDS = [
    f"Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-{index:06d}.tar"
    for index in range(3)
]
DEFAULT_VALIDATION_SHARDS = [
    "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar"
]
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0967-atom-object-full8967-step2000"
QUANTIZATION_MAX = 1023.0
ATOM_FEATURE_SIZE = 192


def freeze_atom_object_features(head: SpatialAtomObjectDetector) -> None:
    """Keep the downstream proposal feature space byte-for-byte stable."""

    for module in (head.input_projection, head.context):
        module.requires_grad_(False)


def _atom_target_cells(
    sample: dict,
    feature_size: int = ATOM_FEATURE_SIZE,
) -> list[tuple[int, int]]:
    """Return the exact cells used by ``spatial_atom_object_loss``.

    AtomBondSceneCollator quantizes anchors to q1023 before normalizing them.
    Auditing raw pixel coordinates with ``floor(x / 4)`` differs at feature-cell
    boundaries, so the preflight must reproduce that target protocol exactly.
    """

    render_atoms = sample.get("render", {}).get("atoms")
    if render_atoms is not None and all(
        "quantized_x" in atom and "quantized_y" in atom for atom in render_atoms
    ):
        anchors = [
            (int(atom["quantized_x"]), int(atom["quantized_y"]))
            for atom in render_atoms
        ]
    else:
        labels = structure_labels(sample)
        width = int(labels.get("width", 768))
        height = int(labels.get("height", 768))
        anchors = [
            (
                round(float(atom["center"][0]) / max(width - 1, 1) * QUANTIZATION_MAX),
                round(float(atom["center"][1]) / max(height - 1, 1) * QUANTIZATION_MAX),
            )
            for atom in labels["atoms"]
        ]
    return [
        (
            min(feature_size - 1, int(x / QUANTIZATION_MAX * feature_size)),
            min(feature_size - 1, int(y / QUANTIZATION_MAX * feature_size)),
        )
        for x, y in anchors
    ]


def spatial_cell_collision_count(
    sample: dict,
    feature_size: int = ATOM_FEATURE_SIZE,
) -> int:
    cells: dict[tuple[int, int], int] = {}
    for cell in _atom_target_cells(sample, feature_size):
        cells[cell] = cells.get(cell, 0) + 1
    return sum(count * (count - 1) // 2 for count in cells.values() if count > 1)


def scan_split(shards: list[str], expected_split: str) -> dict:
    samples = 0
    collisions = []
    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4"):
        split = sample.get("meta", {}).get("split")
        if split != expected_split:
            raise ValueError(f"Expected {expected_split} sample, found {split}")
        samples += 1
        collision_count = spatial_cell_collision_count(sample)
        if collision_count:
            collisions.append(
                {"key": sample["key"], "collision_pairs": collision_count}
            )
    return {
        "split": expected_split,
        "samples": samples,
        "collision_images": len(collisions),
        "collision_pairs": sum(item["collision_pairs"] for item in collisions),
        "collisions": collisions,
        "representable_samples": samples - len(collisions),
    }


def load_validation_records(shards: list[str], reject_keys: set[str]) -> list[dict]:
    records = []
    skipped = set()
    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4"):
        if spatial_cell_collision_count(sample):
            skipped.add(sample["key"])
            continue
        records.append(sample)
    if skipped != reject_keys:
        raise ValueError(
            "Validation collision keys differ from preflight: "
            f"expected={sorted(reject_keys)} actual={sorted(skipped)}"
        )
    if any(sample.get("meta", {}).get("split") != "validation" for sample in records):
        raise ValueError("Validation cache contains a non-validation sample")
    return records


def training_batches(
    shards: list[str],
    reject_keys: set[str],
    *,
    batch_size: int,
    seed: int,
    expected_samples: int | None = None,
):
    rng = random.Random(seed)
    epoch = 0
    while True:
        ordered = list(shards)
        rng.shuffle(ordered)
        dataset = ChemVectorIterableDataset(
            ordered,
            sequence_version="0.4",
            sample_shuffle_buffer=512,
        )
        batch = []
        accepted = 0
        skipped = set()
        for sample in dataset:
            if sample.get("meta", {}).get("split") != "train":
                raise ValueError("Training stream contains a non-train sample")
            if spatial_cell_collision_count(sample):
                skipped.add(sample["key"])
                continue
            accepted += 1
            batch.append(sample)
            if len(batch) == batch_size:
                yield epoch, batch
                batch = []
        if skipped != reject_keys:
            raise ValueError(
                "Training collision keys differ from preflight: "
                f"expected={sorted(reject_keys)} actual={sorted(skipped)}"
            )
        if expected_samples is not None and accepted != expected_samples:
            raise ValueError(
                f"Training epoch accepted {accepted} samples, expected {expected_samples}"
            )
        epoch += 1


@torch.no_grad()
def evaluate_proposals(
    vision: torch.nn.Module,
    head: SpatialAtomObjectDetector,
    records: list[dict],
    collator: AtomBondSceneCollator,
    device: torch.device,
    batch_size: int,
) -> dict:
    config = PredictedProposalConfig(maximum_atoms=128, maximum_bonds=1)
    totals = {
        "images": 0,
        "image_exact": 0,
        "atoms": 0,
        "matched_atoms": 0,
        "close_atoms_12_16": 0,
        "close_matched_12_16": 0,
        "close_images_12_16": 0,
        "close_image_exact_12_16": 0,
    }
    head.eval()
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            output = head(features)
            proposals = decode_predicted_atom_proposals(output, config)
        for batch_index in range(len(pixels)):
            atom_count = int(targets["atom_mask"][batch_index].sum())
            expected = targets["atom_anchors"][batch_index, :atom_count].float()
            proposal_indices = torch.where(proposals["atom_mask"][batch_index])[0]
            predicted = proposals["atom_geometry"][
                batch_index, proposal_indices, :2
            ].float()
            matched = set()
            if len(predicted) and atom_count:
                cost = torch.cdist(predicted, expected) * 768.0
                rows, columns = linear_sum_assignment(cost.cpu().numpy())
                matched = {
                    int(column)
                    for row, column in zip(rows, columns)
                    if float(cost[row, column]) <= config.atom_match_radius_px
                }
            distances = torch.cdist(expected, expected) * 768.0
            distances.fill_diagonal_(float("inf"))
            nearest = distances.amin(dim=1)
            close = (nearest >= 12.0) & (nearest < 16.0)
            close_indices = torch.where(close)[0].tolist()
            totals["images"] += 1
            totals["image_exact"] += int(len(matched) == atom_count)
            totals["atoms"] += atom_count
            totals["matched_atoms"] += len(matched)
            totals["close_atoms_12_16"] += len(close_indices)
            totals["close_matched_12_16"] += sum(
                atom_index in matched for atom_index in close_indices
            )
            if close_indices:
                totals["close_images_12_16"] += 1
                totals["close_image_exact_12_16"] += int(
                    all(atom_index in matched for atom_index in close_indices)
                )
    return {
        **totals,
        "proposal_atom_recall_6px": totals["matched_atoms"]
        / max(totals["atoms"], 1),
        "proposal_image_exact": totals["image_exact"] / max(totals["images"], 1),
        "close_recall_12_16": totals["close_matched_12_16"]
        / max(totals["close_atoms_12_16"], 1),
        "close_image_exact_12_16_rate": totals["close_image_exact_12_16"]
        / max(totals["close_images_12_16"], 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", default=DEFAULT_TRAIN_SHARDS)
    parser.add_argument(
        "--validation-shards", nargs="+", default=DEFAULT_VALIDATION_SHARDS
    )
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--init-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--freeze-object-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full AtomObject training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_scan = scan_split(args.train_shards, "train")
    validation_scan = scan_split(args.validation_shards, "validation")
    preflight = {"train": train_scan, "validation": validation_scan}
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n"
    )
    train_reject = {item["key"] for item in train_scan["collisions"]}
    validation_reject = {item["key"] for item in validation_scan["collisions"]}
    validation_records = load_validation_records(
        args.validation_shards, validation_reject
    )
    if len(validation_records) != validation_scan["representable_samples"]:
        raise ValueError("Validation cache count differs from preflight")
    device = torch.device("cuda")
    vision, vision_config, vision_payload = load_grounded_vision(
        Path(args.grounded_graph_checkpoint), device
    )
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = SpatialAtomObjectConfig(**initial["config"])
    if config.input_channels != vision_config.vision_dims[0]:
        raise ValueError("Atom head and shared vision channels differ")
    head = SpatialAtomObjectDetector(config).to(device)
    head.load_state_dict(initial["head"], strict=True)
    if args.freeze_object_features:
        freeze_atom_object_features(head)
    trainable_parameters = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=0.0
    )
    collator = AtomBondSceneCollator(AtomBondSceneConfig())
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n"
    )
    initial_metrics = {
        **evaluate(vision, head, validation_records, collator, device, args.batch_size),
        **evaluate_proposals(
            vision, head, validation_records, collator, device, args.batch_size
        ),
        "step": 0,
    }
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial_metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"initial_validation": initial_metrics}), flush=True)
    stream = training_batches(
        args.train_shards,
        train_reject,
        batch_size=args.batch_size,
        seed=args.seed,
        expected_samples=train_scan["representable_samples"],
    )
    best_score = None
    best_metrics = None
    started = time.perf_counter()
    running = Counter()
    for step in range(1, args.max_steps + 1):
        epoch, samples = next(stream)
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = head(features)
            losses = spatial_atom_object_loss(output, targets)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        for name, value in losses.items():
            running[name] += float(value.detach())
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "epoch": epoch,
                        **{
                            name: value / args.log_every
                            for name, value in running.items()
                        },
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running.clear()
        if step % args.validate_every == 0 or step == args.max_steps:
            metrics = {
                **evaluate(
                    vision, head, validation_records, collator, device, args.batch_size
                ),
                **evaluate_proposals(
                    vision, head, validation_records, collator, device, args.batch_size
                ),
                "step": step,
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "spatial-atom-object-checkpoint-0.2",
                "head": head.state_dict(),
                "config": asdict(config),
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "validation": metrics,
                "initial_validation": initial_metrics,
                "source_vision": vision_payload,
                "source_atom_checkpoint": args.init_checkpoint,
                "training_args": vars(args),
                "preflight": preflight,
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["proposal_image_exact"],
                metrics["close_recall_12_16"],
                metrics["known_exact"],
                metrics["anchor_accuracy_6px"],
                metrics["role_accuracy"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {"latest": metrics, "best": best_metrics},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
