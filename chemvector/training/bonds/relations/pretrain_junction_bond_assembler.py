#!/usr/bin/env python3
"""Train endpoint-to-Junction assignment on frozen hybrid predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import shutil

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.data import ChemVectorIterableDataset
from chemvector.infer_chemscene_hybrid import ChemSceneHybridPipeline
from chemvector.junction_bond_assembler import (
    JunctionBondAssembler,
    JunctionBondAssemblerConfig,
    decode_junction_assignments,
    junction_bond_assembler_loss,
)
from chemvector.pretrain_chemscene_bond_query import ChemSceneBondCollator
from chemvector.train import expand_shards


def _match_junctions(
    predicted: torch.Tensor,
    sample: dict,
    *,
    radius_pixels: float,
) -> dict[int, int]:
    """Map predicted Junction indices to source atom ids within a strict radius."""

    if not len(predicted):
        return {}
    width = float(sample["layers"]["width"])
    height = float(sample["layers"]["height"])
    atoms = sample["layers"]["atoms"]
    expected = torch.tensor(
        [
            (float(atom["center"][0]) / width, float(atom["center"][1]) / height)
            for atom in atoms
        ],
        dtype=torch.float32,
    )
    distances = torch.cdist(predicted.float().cpu(), expected)
    rows, columns = linear_sum_assignment(distances.numpy())
    scale = max(width, height)
    return {
        int(row): int(atoms[column]["atom_id"])
        for row, column in zip(rows, columns)
        if float(distances[row, column]) * scale <= radius_pixels
    }


def build_assembler_targets(
    points: torch.Tensor,
    bond_objects: list[dict],
    sample: dict,
    *,
    junction_radius_pixels: float = 6.0,
    bond_radius_pixels: float = 12.0,
) -> tuple[torch.Tensor, dict]:
    """Assign predicted BondObject endpoints to predicted Junctions or NULL."""

    null_id = len(points)
    targets = torch.full((len(bond_objects), 2), null_id, dtype=torch.long)
    if not len(points) or not bond_objects:
        return targets, {
            "matched_junctions": 0,
            "expected_junctions": len(sample["layers"]["atoms"]),
            "matched_bond_objects": 0,
            "expected_bond_objects": len(sample["layers"]["bonds"]),
            "nonnull_endpoint_targets": 0,
        }

    predicted_to_atom = _match_junctions(
        points,
        sample,
        radius_pixels=junction_radius_pixels,
    )
    atom_to_predicted = {atom_id: index for index, atom_id in predicted_to_atom.items()}
    predicted_endpoints = torch.stack(
        [item["endpoints"].float().cpu() for item in bond_objects]
    )
    width = float(sample["layers"]["width"])
    height = float(sample["layers"]["height"])
    scale = torch.tensor((width, height), dtype=torch.float32)
    expected_bonds = sample["layers"]["bonds"]
    if not expected_bonds:
        return targets, {
            "matched_junctions": len(predicted_to_atom),
            "expected_junctions": len(sample["layers"]["atoms"]),
            "matched_bond_objects": 0,
            "expected_bond_objects": 0,
            "nonnull_endpoint_targets": 0,
        }
    expected_endpoints = torch.tensor(
        [[bond["begin"], bond["end"]] for bond in expected_bonds],
        dtype=torch.float32,
    ) / scale
    forward = torch.linalg.vector_norm(
        (predicted_endpoints[:, None] - expected_endpoints[None]) * scale,
        dim=-1,
    )
    reverse = torch.linalg.vector_norm(
        (predicted_endpoints[:, None] - expected_endpoints.flip(1)[None]) * scale,
        dim=-1,
    )
    forward_cost = forward.sum(-1)
    reverse_cost = reverse.sum(-1)
    cost = torch.minimum(forward_cost, reverse_cost)
    rows, columns = linear_sum_assignment(cost.numpy())
    matched_bonds = 0
    for row, column in zip(rows, columns):
        use_reverse = bool(reverse_cost[row, column] < forward_cost[row, column])
        endpoint_errors = reverse[row, column] if use_reverse else forward[row, column]
        if float(endpoint_errors.max()) > bond_radius_pixels:
            continue
        bond = expected_bonds[column]
        atom_ids = (
            (int(bond["end_atom"]), int(bond["begin_atom"]))
            if use_reverse
            else (int(bond["begin_atom"]), int(bond["end_atom"]))
        )
        targets[row, 0] = atom_to_predicted.get(atom_ids[0], null_id)
        targets[row, 1] = atom_to_predicted.get(atom_ids[1], null_id)
        matched_bonds += 1
    return targets, {
        "matched_junctions": len(predicted_to_atom),
        "expected_junctions": len(sample["layers"]["atoms"]),
        "matched_bond_objects": matched_bonds,
        "expected_bond_objects": len(expected_bonds),
        "nonnull_endpoint_targets": int((targets != null_id).sum()),
    }


def _prediction_example(
    sample: dict,
    points: torch.Tensor,
    roles: torch.Tensor,
    bond_objects: list[dict],
    *,
    junction_radius_pixels: float,
    bond_radius_pixels: float,
) -> dict:
    targets, diagnostics = build_assembler_targets(
        points,
        bond_objects,
        sample,
        junction_radius_pixels=junction_radius_pixels,
        bond_radius_pixels=bond_radius_pixels,
    )
    return {
        "key": sample["key"],
        "points": points.float().cpu(),
        "roles": roles.long().cpu(),
        "bond_endpoints": torch.stack(
            [item["endpoints"].float().cpu() for item in bond_objects]
        ) if bond_objects else torch.empty((0, 2, 2)),
        "bond_scores": torch.tensor(
            [float(item["score"]) for item in bond_objects], dtype=torch.float32
        ),
        "targets": targets,
        "diagnostics": diagnostics,
    }


@torch.inference_mode()
def build_prediction_cache(
    pipeline: ChemSceneHybridPipeline,
    shards: list[str],
    *,
    samples: int,
    real_threshold: float,
    bond_threshold: float,
    junction_radius_pixels: float,
    bond_radius_pixels: float,
    batch_size: int,
) -> list[dict]:
    dataset = ChemVectorIterableDataset(shards, sequence_version="0.4")
    collator = ChemSceneBondCollator()
    examples = []
    iterator = islice(iter(dataset), samples)
    completed = 0
    while completed < samples:
        batch_records = list(islice(iterator, batch_size))
        if not batch_records:
            break
        pixels = collator(batch_records)["pixel_values"].to(pipeline.device)
        try:
            predictions = pipeline.predict_components_batch(
                pixels,
                real_threshold=real_threshold,
                bond_threshold=bond_threshold,
                include_ocr=False,
            )
        except ValueError as error:
            print(
                f"cache_skip start={completed} count={len(batch_records)} error={error}",
                flush=True,
            )
            completed += len(batch_records)
            continue
        for sample, (points, roles, _, bond_objects) in zip(
            batch_records, predictions
        ):
            examples.append(
                _prediction_example(
                    sample,
                    points,
                    roles,
                    bond_objects,
                    junction_radius_pixels=junction_radius_pixels,
                    bond_radius_pixels=bond_radius_pixels,
                )
            )
        completed += len(batch_records)
        if completed % 100 < batch_size or completed == samples:
            print(
                f"cached={completed}/{samples} usable={len(examples)}",
                flush=True,
            )
    return examples


def _nearest_predictions(example: dict) -> torch.Tensor:
    endpoints = example["bond_endpoints"]
    points = example["points"]
    if not len(endpoints):
        return torch.empty((0, 2), dtype=torch.long)
    return torch.cdist(endpoints, points).argmin(-1)


@torch.no_grad()
def evaluate(model: JunctionBondAssembler, examples: list[dict], device: torch.device) -> dict:
    model.eval()
    totals = {
        "examples": 0,
        "objects": 0,
        "endpoints": 0,
        "endpoint_correct": 0,
        "nonnull_endpoints": 0,
        "nonnull_correct": 0,
        "null_endpoints": 0,
        "null_correct": 0,
        "object_target_exact": 0,
        "decoded_pair_exact": 0,
        "nearest_endpoint_correct": 0,
        "nearest_object_target_exact": 0,
        "assembler_null_objects": 0,
        "assembler_same_junction_objects": 0,
        "assembler_duplicate_objects": 0,
    }
    for example in examples:
        if not len(example["bond_endpoints"]) or not len(example["points"]):
            continue
        points = example["points"].to(device)
        roles = example["roles"].to(device)
        endpoints = example["bond_endpoints"].to(device)
        scores = example["bond_scores"].to(device)
        targets = example["targets"].to(device)
        logits = model(points, roles, endpoints, scores)
        predictions = logits.argmax(-1)
        null_id = len(points)
        endpoint_correct = predictions.eq(targets)
        nonnull = targets.ne(null_id)
        null = ~nonnull
        assignments, diagnostics = decode_junction_assignments(logits, scores)
        nearest = _nearest_predictions(example).to(device)
        nearest_correct = nearest.eq(targets)
        totals["examples"] += 1
        totals["objects"] += len(targets)
        totals["endpoints"] += targets.numel()
        totals["endpoint_correct"] += int(endpoint_correct.sum())
        totals["nonnull_endpoints"] += int(nonnull.sum())
        totals["nonnull_correct"] += int((endpoint_correct & nonnull).sum())
        totals["null_endpoints"] += int(null.sum())
        totals["null_correct"] += int((endpoint_correct & null).sum())
        totals["object_target_exact"] += int(endpoint_correct.all(-1).sum())
        totals["nearest_endpoint_correct"] += int(nearest_correct.sum())
        totals["nearest_object_target_exact"] += int(nearest_correct.all(-1).sum())
        for index, assignment in enumerate(assignments):
            expected = tuple(sorted(targets[index].tolist()))
            if null_id in expected or expected[0] == expected[1]:
                totals["decoded_pair_exact"] += int(assignment is None)
            else:
                totals["decoded_pair_exact"] += int(assignment == expected)
        for name in (
            "assembler_null_objects",
            "assembler_same_junction_objects",
            "assembler_duplicate_objects",
        ):
            totals[name] += diagnostics[name]
    return {
        **totals,
        "endpoint_accuracy": totals["endpoint_correct"] / max(totals["endpoints"], 1),
        "nonnull_endpoint_accuracy": totals["nonnull_correct"] / max(totals["nonnull_endpoints"], 1),
        "null_endpoint_accuracy": totals["null_correct"] / max(totals["null_endpoints"], 1),
        "object_target_exact_rate": totals["object_target_exact"] / max(totals["objects"], 1),
        "decoded_pair_exact_rate": totals["decoded_pair_exact"] / max(totals["objects"], 1),
        "nearest_endpoint_accuracy": totals["nearest_endpoint_correct"] / max(totals["endpoints"], 1),
        "nearest_object_target_exact_rate": totals["nearest_object_target_exact"] / max(totals["objects"], 1),
    }


def _load_or_build_cache(
    path: Path,
    pipeline: ChemSceneHybridPipeline,
    shards: list[str],
    **kwargs,
) -> list[dict]:
    if path.exists():
        print(f"loading_cache={path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    examples = build_prediction_cache(pipeline, shards, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, path)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", default="outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt")
    parser.add_argument("--bond-checkpoint", default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--same-junction-weight", type=float, default=0.2)
    parser.add_argument("--junction-radius-pixels", type=float, default=6.0)
    parser.add_argument("--bond-radius-pixels", type=float, default=12.0)
    parser.add_argument("--real-threshold", type=float, default=0.2)
    parser.add_argument("--bond-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint), Path(args.bond_checkpoint), device
    )
    cache_arguments = {
        "real_threshold": args.real_threshold,
        "bond_threshold": args.bond_threshold,
        "junction_radius_pixels": args.junction_radius_pixels,
        "bond_radius_pixels": args.bond_radius_pixels,
        "batch_size": args.cache_batch_size,
    }
    train_examples = _load_or_build_cache(
        output / "train-predictions.pt",
        pipeline,
        expand_shards(args.train_shards),
        samples=args.train_samples,
        **cache_arguments,
    )
    validation_examples = _load_or_build_cache(
        output / "validation-predictions.pt",
        pipeline,
        expand_shards(args.validation_shards),
        samples=args.validation_samples,
        **cache_arguments,
    )
    del pipeline
    torch.cuda.empty_cache()
    if not train_examples or not validation_examples:
        raise ValueError("Assembler cache contains no usable predictions")

    config = JunctionBondAssemblerConfig(hidden_dim=args.hidden_dim)
    model = JunctionBondAssembler(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    baseline = evaluate(model, validation_examples, device)
    print("untrained=" + json.dumps(baseline), flush=True)
    best_score = (-1.0, -1.0)
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_examples)
        running_loss = 0.0
        used = 0
        optimizer.zero_grad(set_to_none=True)
        for example in train_examples:
            if not len(example["bond_endpoints"]) or not len(example["points"]):
                continue
            logits = model(
                example["points"].to(device),
                example["roles"].to(device),
                example["bond_endpoints"].to(device),
                example["bond_scores"].to(device),
            )
            losses = junction_bond_assembler_loss(
                logits,
                example["targets"].to(device),
                same_junction_weight=args.same_junction_weight,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += float(losses["loss"].detach())
            used += 1
        metrics = evaluate(model, validation_examples, device)
        print(
            f"epoch={epoch} loss={running_loss / max(used, 1):.6f} "
            f"endpoint={metrics['endpoint_accuracy']:.4f} "
            f"nonnull={metrics['nonnull_endpoint_accuracy']:.4f} "
            f"pair={metrics['decoded_pair_exact_rate']:.4f}",
            flush=True,
        )
        checkpoint = {
            "junction_bond_assembler": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "validation": metrics,
            "selector_checkpoint": args.selector_checkpoint,
            "bond_checkpoint": args.bond_checkpoint,
            "protocol": vars(args),
        }
        torch.save(checkpoint, output / "last.pt")
        score = (metrics["decoded_pair_exact_rate"], metrics["nonnull_endpoint_accuracy"])
        if score > best_score:
            best_score = score
            best_metrics = metrics
            shutil.copy2(output / "last.pt", output / "best.pt")

    result = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "baseline": baseline,
        "best": best_metrics,
        "best_score": list(best_score),
        "configuration": vars(args),
    }
    (output / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
