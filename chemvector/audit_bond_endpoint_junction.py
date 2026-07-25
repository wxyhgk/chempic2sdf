#!/usr/bin/env python3
"""Audit Junctions derived only from semantic Bond Object endpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path

from rdkit import Chem
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_1 import build_chemscene_v0_1, mol_from_chemscene_v0_1
from chemvector.audit_bond_first_decomposition import (
    _atom_centers,
    _expected_bonds,
    sample_slices,
)
from chemvector.bond_endpoint_junction import junctions_from_bond_endpoints
from chemvector.data import ChemVectorIterableDataset
from chemvector.pretrain_dense_bond_object import _scene_from_dense_objects
from chemvector.train import expand_shards


def objects_from_cache(example: dict, threshold: float) -> list[dict]:
    return [
        {
            "endpoints": endpoints,
            "score": float(score),
            "display": int(display),
            "chemical": int(chemical),
            "direction": int(direction),
        }
        for endpoints, score, display, chemical, direction in zip(
            example["endpoints"],
            example["scores"],
            example["display_ids"],
            example["chemical_ids"],
            example["direction_ids"],
        )
        if float(score) >= threshold
    ]


def analyze_endpoint_junctions(
    sample: dict,
    example: dict,
    *,
    threshold: float,
    cluster_radius_pixels: float,
    match_radius_pixels: float,
    compile_sdf: bool,
) -> dict:
    width = int(sample["layers"]["width"])
    height = int(sample["layers"]["height"])
    objects = objects_from_cache(example, threshold)
    result = junctions_from_bond_endpoints(
        objects,
        width=width,
        height=height,
        radius_pixels=cluster_radius_pixels,
    )
    expected_points = _atom_centers(sample)
    predicted_points = result.points
    scale = torch.tensor((float(width), float(height)))
    matches: dict[int, int] = {}
    match_distances = []
    if len(predicted_points) and len(expected_points):
        costs = torch.cdist(predicted_points * scale, expected_points * scale)
        rows, columns = linear_sum_assignment(costs.numpy())
        for row, column in zip(rows, columns):
            distance = float(costs[row, column])
            if distance <= match_radius_pixels:
                matches[int(row)] = int(column)
                match_distances.append(distance)

    expected_by_pair = {item["pair"]: item for item in _expected_bonds(sample)}
    expected_pairs = set(expected_by_pair)
    predicted_pairs = set()
    chemical_correct = display_correct = 0
    unmapped_bonds = 0
    mapped_objects = []
    for item in result.bond_objects:
        begin, end = item["junction_pair"]
        if begin not in matches or end not in matches:
            unmapped_bonds += 1
            continue
        pair = tuple(sorted((matches[begin], matches[end])))
        predicted_pairs.add(pair)
        target = expected_by_pair.get(pair)
        if target is not None:
            chemical_correct += int(int(item["chemical"]) == target["chemical"])
            display_correct += int(int(item["display"]) == target["display"])
        mapped_objects.append(item)

    expected_nodes = len(expected_points)
    predicted_nodes = len(predicted_points)
    true_nodes = len(matches)
    node_count_exact = predicted_nodes == expected_nodes
    vertex_exact = node_count_exact and true_nodes == expected_nodes
    topology_exact = (
        vertex_exact
        and unmapped_bonds == 0
        and predicted_pairs == expected_pairs
    )
    chemical_exact = topology_exact and chemical_correct == len(expected_pairs)
    display_exact = topology_exact and display_correct == len(expected_pairs)
    sdf_exact = False
    if compile_sdf and topology_exact:
        scene = build_chemscene_v0_1(
            sample["chemmol"], sample["render"], sample.get("layers")
        )
        predicted_scene = _scene_from_dense_objects(scene, list(result.bond_objects))
        if predicted_scene is not None:
            try:
                sdf_exact = (
                    Chem.MolToInchiKey(mol_from_chemscene_v0_1(predicted_scene))
                    == sample["chemmol"]["inchikey"]
                )
            except Exception:
                sdf_exact = False

    true_edges = len(predicted_pairs & expected_pairs)
    false_edges = len(predicted_pairs - expected_pairs) + unmapped_bonds
    missed_edges = len(expected_pairs - predicted_pairs)
    return {
        "expected_nodes": expected_nodes,
        "predicted_nodes": predicted_nodes,
        "matched_nodes": true_nodes,
        "node_count_exact": node_count_exact,
        "vertex_exact": vertex_exact,
        "expected_bonds": len(expected_pairs),
        "predicted_bonds": len(result.bond_objects),
        "true_edges": true_edges,
        "false_edges": false_edges,
        "missed_edges": missed_edges,
        "topology_exact": topology_exact,
        "chemical_exact": chemical_exact,
        "display_exact": display_exact,
        "sdf_exact": sdf_exact,
        "raw_bond_candidates": len(objects),
        "raw_endpoint_hypotheses": len(objects) * 2,
        "raw_clusters": result.raw_clusters,
        "self_loops": result.self_loops,
        "duplicate_pairs": result.duplicate_pairs,
        "unmapped_bonds": unmapped_bonds,
        "match_distance_mean_pixels": (
            sum(match_distances) / len(match_distances) if match_distances else None
        ),
        "match_distance_max_pixels": max(match_distances, default=None),
    }


def summarize(rows: list[dict]) -> dict:
    samples = max(len(rows), 1)
    expected_nodes = sum(row["metrics"]["expected_nodes"] for row in rows)
    predicted_nodes = sum(row["metrics"]["predicted_nodes"] for row in rows)
    matched_nodes = sum(row["metrics"]["matched_nodes"] for row in rows)
    true_edges = sum(row["metrics"]["true_edges"] for row in rows)
    false_edges = sum(row["metrics"]["false_edges"] for row in rows)
    missed_edges = sum(row["metrics"]["missed_edges"] for row in rows)
    node_precision = matched_nodes / max(predicted_nodes, 1)
    node_recall = matched_nodes / max(expected_nodes, 1)
    edge_precision = true_edges / max(true_edges + false_edges, 1)
    edge_recall = true_edges / max(true_edges + missed_edges, 1)
    return {
        "samples": len(rows),
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": 2 * node_precision * node_recall / max(node_precision + node_recall, 1e-12),
        "node_count_accuracy": sum(row["metrics"]["node_count_exact"] for row in rows) / samples,
        "vertex_exact_rate": sum(row["metrics"]["vertex_exact"] for row in rows) / samples,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12),
        "topology_exact_rate": sum(row["metrics"]["topology_exact"] for row in rows) / samples,
        "chemical_exact_rate": sum(row["metrics"]["chemical_exact"] for row in rows) / samples,
        "display_exact_rate": sum(row["metrics"]["display_exact"] for row in rows) / samples,
        "oracle_label_sdf_exact_rate": sum(row["metrics"]["sdf_exact"] for row in rows) / samples,
        "average_expected_nodes": expected_nodes / samples,
        "average_predicted_nodes": predicted_nodes / samples,
        "average_raw_bond_candidates": sum(row["metrics"]["raw_bond_candidates"] for row in rows) / samples,
        "average_raw_endpoint_hypotheses": sum(row["metrics"]["raw_endpoint_hypotheses"] for row in rows) / samples,
        "average_raw_clusters": sum(row["metrics"]["raw_clusters"] for row in rows) / samples,
        "average_self_loops": sum(row["metrics"]["self_loops"] for row in rows) / samples,
        "average_duplicate_pairs": sum(row["metrics"]["duplicate_pairs"] for row in rows) / samples,
    }


def evaluate(
    samples: list[dict],
    examples: list[dict],
    *,
    threshold: float,
    cluster_radius_pixels: float,
    match_radius_pixels: float,
    compile_sdf: bool,
) -> list[dict]:
    rows = []
    for sample, example in zip(samples, examples):
        if sample["key"] != example["key"]:
            raise ValueError("Bond candidate cache order differs from dataset")
        rows.append(
            {
                "key": sample["key"],
                "slices": sample_slices(sample),
                "metrics": analyze_endpoint_junctions(
                    sample,
                    example,
                    threshold=threshold,
                    cluster_radius_pixels=cluster_radius_pixels,
                    match_radius_pixels=match_radius_pixels,
                    compile_sdf=compile_sdf,
                ),
            }
        )
    return rows


def summarize_slices(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for name in row["slices"]:
            grouped[name].append(row)
    return {name: summarize(items) for name, items in sorted(grouped.items())}


def load_samples(patterns: list[str], count: int) -> list[dict]:
    return list(
        islice(
            iter(
                ChemVectorIterableDataset(
                    expand_shards(patterns), sequence_version="0.4"
                )
            ),
            count,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--train-cache",
        default="outputs/checkpoints/chemvector-v0813-bond-set-selector-9k/train-candidates.pt",
    )
    parser.add_argument(
        "--validation-cache",
        default="outputs/checkpoints/chemvector-v0813-bond-set-selector-9k/validation-candidates.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/audits/chemvector-v0816-bond-endpoint-junction-498",
    )
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=498)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--match-radius-pixels", type=float, default=8.0)
    parser.add_argument(
        "--cluster-radius-candidates",
        type=float,
        nargs="+",
        default=[4.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    )
    args = parser.parse_args()

    train_examples = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    validation_examples = torch.load(
        args.validation_cache, map_location="cpu", weights_only=False
    )
    train_samples = load_samples(args.train_shards, args.calibration_samples)
    validation_samples = load_samples(args.validation_shards, args.validation_samples)
    train_examples = train_examples[: args.calibration_samples]
    validation_examples = validation_examples[: args.validation_samples]
    if len(train_samples) != len(train_examples) or len(validation_samples) != len(validation_examples):
        raise ValueError("Dataset and Bond candidate cache sizes differ")

    calibration = []
    for radius in args.cluster_radius_candidates:
        rows = evaluate(
            train_samples,
            train_examples,
            threshold=args.threshold,
            cluster_radius_pixels=radius,
            match_radius_pixels=args.match_radius_pixels,
            compile_sdf=False,
        )
        metrics = summarize(rows)
        calibration.append({"cluster_radius_pixels": radius, "metrics": metrics})
        print(
            f"radius={radius:.1f} count={metrics['node_count_accuracy']:.4f} "
            f"vertex={metrics['vertex_exact_rate']:.4f} "
            f"topology={metrics['topology_exact_rate']:.4f}",
            flush=True,
        )
    best = max(
        calibration,
        key=lambda item: (
            item["metrics"]["topology_exact_rate"],
            item["metrics"]["vertex_exact_rate"],
            item["metrics"]["node_count_accuracy"],
            -item["cluster_radius_pixels"],
        ),
    )
    selected_radius = float(best["cluster_radius_pixels"])
    rows = evaluate(
        validation_samples,
        validation_examples,
        threshold=args.threshold,
        cluster_radius_pixels=selected_radius,
        match_radius_pixels=args.match_radius_pixels,
        compile_sdf=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "bond-endpoint-junction-audit-0.1",
        "overall": summarize(rows),
        "slices": summarize_slices(rows),
        "calibration": calibration,
        "protocol": {
            **vars(args),
            "selected_cluster_radius_pixels": selected_radius,
            "cluster_radius_selected_on_train_only": True,
            "uses_vertex_detector_for_implicit_junctions": False,
            "semantic_endpoints_per_bond_object": 2,
            "uses_mr_tadf_for_tuning": False,
            "uses_oracle_labels_for_sdf": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
