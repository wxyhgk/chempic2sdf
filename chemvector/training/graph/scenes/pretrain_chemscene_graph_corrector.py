#!/usr/bin/env python3
"""Train a graph-level ChemScene BondObject set corrector."""

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

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.chemscene_graph_corrector import (
    ELEMENT_TO_ID,
    ChemSceneGraphCorrector,
    ChemSceneGraphCorrectorConfig,
    graph_corrector_loss,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.infer_chemscene_hybrid import ChemSceneHybridPipeline
from chemvector.ocr import parse_atom_display_text
from chemvector.pretrain_chemscene_bond_query import ChemSceneBondCollator
from chemvector.pretrain_junction_bond_assembler import _match_junctions
from chemvector.train import expand_shards


def element_id(label: str, role: int) -> int:
    if role == 0:
        return ELEMENT_TO_ID["C"]
    parsed = parse_atom_display_text(label)
    element = str(parsed["element"]) if parsed is not None else "OTHER"
    return ELEMENT_TO_ID.get(element, ELEMENT_TO_ID["OTHER"])


def build_graph_targets(
    points: torch.Tensor,
    bond_objects: list[dict],
    sample: dict,
    *,
    junction_radius_pixels: float = 6.0,
    bond_radius_pixels: float = 12.0,
) -> dict[str, torch.Tensor | int]:
    predicted_to_atom = _match_junctions(
        points, sample, radius_pixels=junction_radius_pixels
    )
    node_targets = torch.tensor(
        [int(index in predicted_to_atom) for index in range(len(points))],
        dtype=torch.float32,
    )
    count = len(bond_objects)
    edge_targets = torch.zeros(count, dtype=torch.float32)
    display_targets = torch.full((count,), -100, dtype=torch.long)
    chemical_targets = torch.full((count,), -100, dtype=torch.long)
    expected_bonds = sample["layers"]["bonds"]
    if not count or not expected_bonds:
        return {
            "node_targets": node_targets,
            "edge_targets": edge_targets,
            "display_targets": display_targets,
            "chemical_targets": chemical_targets,
            "expected_bonds": len(expected_bonds),
            "matched_bonds": 0,
        }
    width = float(sample["layers"]["width"])
    height = float(sample["layers"]["height"])
    scale = torch.tensor((width, height), dtype=torch.float32)
    predicted = torch.stack([item["endpoints"].float().cpu() for item in bond_objects])
    expected = torch.tensor(
        [[bond["begin"], bond["end"]] for bond in expected_bonds],
        dtype=torch.float32,
    ) / scale
    forward = torch.linalg.vector_norm(
        (predicted[:, None] - expected[None]) * scale, dim=-1
    )
    reverse = torch.linalg.vector_norm(
        (predicted[:, None] - expected.flip(1)[None]) * scale, dim=-1
    )
    cost = torch.minimum(forward.sum(-1), reverse.sum(-1))
    rows, columns = linear_sum_assignment(cost.numpy())
    matched = 0
    for row, column in zip(rows, columns):
        errors = (
            reverse[row, column]
            if float(reverse[row, column].sum()) < float(forward[row, column].sum())
            else forward[row, column]
        )
        if float(errors.max()) > bond_radius_pixels:
            continue
        target = expected_bonds[column]
        edge_targets[row] = 1.0
        display_targets[row] = BOND_TYPE_TO_ID[str(target["display_type"])]
        chemical_targets[row] = BOND_TYPE_TO_ID[str(target["chemical_type"])]
        matched += 1
    return {
        "node_targets": node_targets,
        "edge_targets": edge_targets,
        "display_targets": display_targets,
        "chemical_targets": chemical_targets,
        "expected_bonds": len(expected_bonds),
        "matched_bonds": matched,
    }


def prediction_example(
    sample: dict,
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    bond_objects: list[dict],
    *,
    junction_radius_pixels: float,
    bond_radius_pixels: float,
) -> dict:
    targets = build_graph_targets(
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
        "elements": torch.tensor(
            [element_id(label, int(role)) for label, role in zip(labels, roles)],
            dtype=torch.long,
        ),
        "bond_endpoints": torch.stack(
            [item["endpoints"].float().cpu() for item in bond_objects]
        ) if bond_objects else torch.empty((0, 2, 2)),
        "bond_scores": torch.tensor(
            [float(item["score"]) for item in bond_objects], dtype=torch.float32
        ),
        "display_ids": torch.tensor(
            [int(item["display"]) for item in bond_objects], dtype=torch.long
        ),
        "chemical_ids": torch.tensor(
            [int(item["chemical"]) for item in bond_objects], dtype=torch.long
        ),
        **targets,
    }


@torch.inference_mode()
def build_cache(
    pipeline: ChemSceneHybridPipeline,
    shards: list[str],
    *,
    samples: int,
    batch_size: int,
    real_threshold: float,
    bond_threshold: float,
    junction_radius_pixels: float,
    bond_radius_pixels: float,
    include_ocr_elements: bool,
) -> list[dict]:
    iterator = islice(
        iter(ChemVectorIterableDataset(shards, sequence_version="0.4")), samples
    )
    collator = ChemSceneBondCollator()
    examples = []
    completed = 0
    while completed < samples:
        records = list(islice(iterator, batch_size))
        if not records:
            break
        pixels = collator(records)["pixel_values"].to(pipeline.device)
        predictions = pipeline.predict_components_batch(
            pixels,
            real_threshold=real_threshold,
            bond_threshold=bond_threshold,
            include_ocr=include_ocr_elements,
        )
        for sample, (points, roles, labels, bond_objects) in zip(records, predictions):
            examples.append(
                prediction_example(
                    sample,
                    points,
                    roles,
                    labels,
                    bond_objects,
                    junction_radius_pixels=junction_radius_pixels,
                    bond_radius_pixels=bond_radius_pixels,
                )
            )
        completed += len(records)
        if completed % 100 < batch_size or completed == samples:
            print(f"cached={completed}/{samples}", flush=True)
    return examples


def load_or_build_cache(path: Path, pipeline, shards, **kwargs) -> list[dict]:
    if path.exists():
        print(f"loading_cache={path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    examples = build_cache(pipeline, shards, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, path)
    return examples


def forward_example(model, example: dict, device: torch.device) -> dict:
    return model(
        example["points"].to(device),
        example["roles"].to(device),
        example["elements"].to(device),
        example["bond_endpoints"].to(device),
        example["bond_scores"].to(device),
        example["display_ids"].to(device),
        example["chemical_ids"].to(device),
    )


@torch.no_grad()
def evaluate(
    model: ChemSceneGraphCorrector,
    examples: list[dict],
    device: torch.device,
    *,
    keep_threshold: float,
) -> dict:
    model.eval()
    totals = {
        "examples": 0,
        "edges": 0,
        "positive_edges": 0,
        "kept_edges": 0,
        "true_kept_edges": 0,
        "baseline_display_correct": 0,
        "corrected_display_correct": 0,
        "baseline_chemical_correct": 0,
        "corrected_chemical_correct": 0,
        "coverage_exact": 0,
        "baseline_set_exact": 0,
        "type_only_set_exact": 0,
        "full_set_exact": 0,
        "node_correct": 0,
        "nodes": 0,
    }
    for example in examples:
        if not len(example["bond_endpoints"]) or not len(example["points"]):
            continue
        output = forward_example(model, example, device)
        edge_targets = example["edge_targets"].to(device).bool()
        display_targets = example["display_targets"].to(device)
        chemical_targets = example["chemical_targets"].to(device)
        display = output["display_logits"].argmax(-1)
        chemical = output["chemical_logits"].argmax(-1)
        kept = output["edge_keep_logits"].sigmoid() >= keep_threshold
        baseline_display = example["display_ids"].to(device)
        baseline_chemical = example["chemical_ids"].to(device)
        expected = int(example["expected_bonds"])
        coverage = int(example["matched_bonds"]) == expected
        baseline_types = (
            baseline_display.eq(display_targets) & baseline_chemical.eq(chemical_targets)
        )
        corrected_types = display.eq(display_targets) & chemical.eq(chemical_targets)
        baseline_exact = (
            coverage
            and len(edge_targets) == expected
            and bool(edge_targets.all())
            and bool(baseline_types.all())
        )
        type_only_exact = (
            coverage
            and len(edge_targets) == expected
            and bool(edge_targets.all())
            and bool(corrected_types.all())
        )
        full_exact = (
            coverage
            and int(kept.sum()) == expected
            and bool(edge_targets[kept].all())
            and bool(corrected_types[kept].all())
        )
        node_predictions = output["node_keep_logits"] >= 0.0
        node_targets = example["node_targets"].to(device).bool()
        totals["examples"] += 1
        totals["edges"] += len(edge_targets)
        totals["positive_edges"] += int(edge_targets.sum())
        totals["kept_edges"] += int(kept.sum())
        totals["true_kept_edges"] += int((kept & edge_targets).sum())
        totals["baseline_display_correct"] += int(
            baseline_display[edge_targets].eq(display_targets[edge_targets]).sum()
        )
        totals["corrected_display_correct"] += int(
            display[edge_targets].eq(display_targets[edge_targets]).sum()
        )
        totals["baseline_chemical_correct"] += int(
            baseline_chemical[edge_targets].eq(chemical_targets[edge_targets]).sum()
        )
        totals["corrected_chemical_correct"] += int(
            chemical[edge_targets].eq(chemical_targets[edge_targets]).sum()
        )
        totals["coverage_exact"] += int(coverage)
        totals["baseline_set_exact"] += int(baseline_exact)
        totals["type_only_set_exact"] += int(type_only_exact)
        totals["full_set_exact"] += int(full_exact)
        totals["node_correct"] += int(node_predictions.eq(node_targets).sum())
        totals["nodes"] += len(node_targets)
    true_kept = totals["true_kept_edges"]
    precision = true_kept / max(totals["kept_edges"], 1)
    recall = true_kept / max(totals["positive_edges"], 1)
    return {
        **totals,
        "edge_keep_precision": precision,
        "edge_keep_recall": recall,
        "edge_keep_f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        "baseline_display_accuracy": totals["baseline_display_correct"] / max(totals["positive_edges"], 1),
        "corrected_display_accuracy": totals["corrected_display_correct"] / max(totals["positive_edges"], 1),
        "baseline_chemical_accuracy": totals["baseline_chemical_correct"] / max(totals["positive_edges"], 1),
        "corrected_chemical_accuracy": totals["corrected_chemical_correct"] / max(totals["positive_edges"], 1),
        "coverage_exact_rate": totals["coverage_exact"] / max(totals["examples"], 1),
        "baseline_set_exact_rate": totals["baseline_set_exact"] / max(totals["examples"], 1),
        "type_only_set_exact_rate": totals["type_only_set_exact"] / max(totals["examples"], 1),
        "full_set_exact_rate": totals["full_set_exact"] / max(totals["examples"], 1),
        "node_accuracy": totals["node_correct"] / max(totals["nodes"], 1),
        "keep_threshold": keep_threshold,
    }


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
    parser.add_argument("--include-ocr-elements", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--node-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--type-weight", type=float, default=1.0)
    parser.add_argument("--junction-radius-pixels", type=float, default=6.0)
    parser.add_argument("--bond-radius-pixels", type=float, default=12.0)
    parser.add_argument("--real-threshold", type=float, default=0.2)
    parser.add_argument("--bond-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint), Path(args.bond_checkpoint), device
    )
    cache_args = {
        "batch_size": args.cache_batch_size,
        "real_threshold": args.real_threshold,
        "bond_threshold": args.bond_threshold,
        "junction_radius_pixels": args.junction_radius_pixels,
        "bond_radius_pixels": args.bond_radius_pixels,
        "include_ocr_elements": args.include_ocr_elements,
    }
    train_examples = load_or_build_cache(
        output_dir / "train-predictions.pt",
        pipeline,
        expand_shards(args.train_shards),
        samples=args.train_samples,
        **cache_args,
    )
    validation_examples = load_or_build_cache(
        output_dir / "validation-predictions.pt",
        pipeline,
        expand_shards(args.validation_shards),
        samples=args.validation_samples,
        **cache_args,
    )
    del pipeline
    torch.cuda.empty_cache()
    config = ChemSceneGraphCorrectorConfig(d_model=args.d_model, layers=args.layers)
    model = ChemSceneGraphCorrector(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_score = (-1.0, -1.0, -1.0)
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_examples)
        loss_sum = 0.0
        used = 0
        for example in train_examples:
            if not len(example["points"]) or not len(example["bond_endpoints"]):
                continue
            output = forward_example(model, example, device)
            losses = graph_corrector_loss(
                output,
                example["node_targets"].to(device),
                example["edge_targets"].to(device),
                example["display_targets"].to(device),
                example["chemical_targets"].to(device),
                node_weight=args.node_weight,
                edge_weight=args.edge_weight,
                type_weight=args.type_weight,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(losses["loss"].detach())
            used += 1
        threshold_results = [
            evaluate(model, validation_examples, device, keep_threshold=threshold)
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)
        ]
        metrics = max(
            threshold_results,
            key=lambda item: (
                item["type_only_set_exact_rate"],
                item["full_set_exact_rate"],
                item["corrected_chemical_accuracy"],
            ),
        )
        print(
            f"epoch={epoch} loss={loss_sum / max(used, 1):.6f} "
            f"display={metrics['corrected_display_accuracy']:.4f} "
            f"chemical={metrics['corrected_chemical_accuracy']:.4f} "
            f"type_set={metrics['type_only_set_exact_rate']:.4f} "
            f"full_set={metrics['full_set_exact_rate']:.4f}",
            flush=True,
        )
        payload = {
            "chemscene_graph_corrector": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "validation": metrics,
            "threshold_results": threshold_results,
            "selector_checkpoint": args.selector_checkpoint,
            "bond_checkpoint": args.bond_checkpoint,
            "protocol": vars(args),
        }
        torch.save(payload, output_dir / "last.pt")
        score = (
            metrics["type_only_set_exact_rate"],
            metrics["full_set_exact_rate"],
            metrics["corrected_chemical_accuracy"],
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            shutil.copy2(output_dir / "last.pt", output_dir / "best.pt")
    result = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "best": best_metrics,
        "best_score": list(best_score),
        "configuration": vars(args),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
