#!/usr/bin/env python3
"""Evaluate the semantic bond reader on fully predicted dual-branch vertices."""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.bond_object import (
    BOND_TYPES,
    BOND_TYPE_TO_ID,
    DIRECTION_TO_ID,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.geometry import stable_pairwise_distances
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
    enumerate_all_pairs,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.pretrain_node_conditioned_bond import KnownEndpointBondCollator
from chemvector.text_object import TextObjectCenterHead, decode_text_objects
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


def outside_text_objects(
    vertex: dict, text_objects: list[dict], expansion_pixels: float
) -> bool:
    expansion = expansion_pixels / 768.0
    return not any(
        obj["x"] - obj["half_width"] - expansion
        <= vertex["x"]
        <= obj["x"] + obj["half_width"] + expansion
        and obj["y"] - obj["half_height"] - expansion
        <= vertex["y"]
        <= obj["y"] + obj["half_height"] + expansion
        for obj in text_objects
    )


def match_vertices(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    radius_pixels: float = 6.0,
) -> dict[int, int]:
    if not len(predicted) or not len(expected):
        return {}
    distances = stable_pairwise_distances(predicted, expected) * 768.0
    rows, columns = linear_sum_assignment(distances.cpu().numpy())
    return {
        int(row): int(column)
        for row, column in zip(rows, columns)
        if float(distances[row, column]) <= radius_pixels
    }


def point_to_segment_distance(
    point: torch.Tensor, begin: torch.Tensor, end: torch.Tensor
) -> float:
    delta = end - begin
    denominator = float(delta.square().sum())
    if denominator <= 1e-12:
        return float(torch.linalg.vector_norm(point - begin))
    projection = float(((point - begin) * delta).sum()) / denominator
    closest = begin + delta * min(1.0, max(0.0, projection))
    return float(torch.linalg.vector_norm(point - closest))


def unmatched_vertex_diagnostics(
    predicted_points: torch.Tensor,
    assignment: dict[int, int],
    sample: dict,
) -> list[dict]:
    width = float(sample["layers"]["width"])
    height = float(sample["layers"]["height"])
    atoms = sample["layers"]["atoms"]
    bonds = sample["layers"]["bonds"]
    expected_pixels = torch.tensor([atom["center"] for atom in atoms])
    rows = []
    for predicted_index in sorted(set(range(len(predicted_points))) - set(assignment)):
        point = predicted_points[predicted_index] * torch.tensor((width, height))
        atom_distances = torch.linalg.vector_norm(expected_pixels - point, dim=-1)
        nearest_atom_distance = float(atom_distances.min()) if len(atom_distances) else None
        nearest_bond = None
        nearest_bond_distance = float("inf")
        for bond in bonds:
            distance = point_to_segment_distance(
                point,
                torch.tensor(bond["begin"], dtype=torch.float32),
                torch.tensor(bond["end"], dtype=torch.float32),
            )
            if distance < nearest_bond_distance:
                nearest_bond_distance = distance
                nearest_bond = bond
        rows.append(
            {
                "predicted_index": predicted_index,
                "x": float(point[0]),
                "y": float(point[1]),
                "nearest_gt_atom_distance_pixels": nearest_atom_distance,
                "nearest_bond_id": (
                    int(nearest_bond["bond_id"]) if nearest_bond is not None else None
                ),
                "nearest_bond_distance_pixels": (
                    nearest_bond_distance if nearest_bond is not None else None
                ),
                "nearest_bond_display_type": (
                    nearest_bond["display_type"] if nearest_bond is not None else None
                ),
                "near_multiple_bond": bool(
                    nearest_bond is not None
                    and nearest_bond["display_type"] in {"DOUBLE", "TRIPLE"}
                    and nearest_bond_distance <= 12.0
                ),
            }
        )
    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond-checkpoint", required=True)
    parser.add_argument("--text-object-checkpoint", required=True)
    parser.add_argument("--implicit-vertex-checkpoint", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--implicit-threshold", type=float, default=0.52)
    parser.add_argument("--bond-threshold", type=float)
    parser.add_argument("--bbox-expansion", type=float, default=8.0)
    args = parser.parse_args()

    device = torch.device("cuda")
    text_payload = torch.load(
        args.text_object_checkpoint, map_location="cpu", weights_only=False
    )
    implicit_payload = torch.load(
        args.implicit_vertex_checkpoint, map_location="cpu", weights_only=False
    )
    graph_checkpoint = implicit_payload["grounded_graph_checkpoint"]
    if text_payload["grounded_graph_checkpoint"] != graph_checkpoint:
        raise ValueError("Text and implicit vertex heads use different visual backbones")
    _, graph_payload = load_grounded_graph_checkpoint(graph_checkpoint, device)
    base, _, _, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    base.eval()
    text_head = TextObjectCenterHead(text_payload["input_channels"]).to(device)
    text_head.load_state_dict(text_payload["text_object_head"])
    text_head.eval()
    implicit_head = deepcopy(base.vertex_head).to(device)
    implicit_head.load_state_dict(implicit_payload["implicit_vertex_head"])
    implicit_head.eval()
    bond_payload = torch.load(
        args.bond_checkpoint, map_location="cpu", weights_only=False
    )
    bond_config = NodeConditionedBondConfig(**bond_payload["config"])
    bond_head = NodeConditionedBondObject(bond_config).to(device)
    bond_head.load_state_dict(bond_payload["node_conditioned_bond"])
    bond_head.eval()
    bond_threshold = (
        args.bond_threshold
        if args.bond_threshold is not None
        else float(bond_payload["validation"]["threshold"])
    )

    dataset = ChemVectorIterableDataset(
        expand_shards(args.validation_shards), sequence_version="0.4"
    )
    records = list(islice(iter(dataset), args.samples))
    if len(records) != args.samples:
        raise ValueError("Validation dataset contains fewer samples than requested")
    image_collator = KnownEndpointBondCollator()
    totals = {
        "samples": 0,
        "predicted_vertices": 0,
        "expected_vertices": 0,
        "matched_vertices": 0,
        "vertex_exact": 0,
        "predicted_edges": 0,
        "expected_edges": 0,
        "true_positive_edges": 0,
        "matchable_edges": 0,
        "skeleton_exact": 0,
        "vertex_exact_samples": 0,
        "edge_exact_given_vertex_exact": 0,
        "display_type_correct_on_recovered_edges": 0,
        "chemical_type_correct_on_recovered_edges": 0,
        "direction_correct_on_recovered_edges": 0,
        "bond_object_correct": 0,
    }
    class_expected = {name: 0 for name in BOND_TYPES}
    class_edge_hits = {name: 0 for name in BOND_TYPES}
    class_display_hits = {name: 0 for name in BOND_TYPES}
    class_object_hits = {name: 0 for name in BOND_TYPES}
    sample_rows = []
    for start in range(0, len(records), args.batch_size):
        samples = records[start : start + args.batch_size]
        pixels = torch.stack(
            [image_collator._pixels(sample["image"]) for sample in samples]
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = base.vision_encoder.forward_stroke_resolution(pixels)
            text_output = text_head(features)
            implicit_output = implicit_head(features)
        text_batches = decode_text_objects(
            text_output, threshold=args.text_threshold
        )
        implicit_batches = decode_vertices(
            implicit_output,
            threshold=args.implicit_threshold,
            maximum_vertices=256,
            nms_kernel_size=1,
        )
        merged_batches = []
        for text_objects, implicit_vertices in zip(text_batches, implicit_batches):
            filtered = [
                vertex
                for vertex in implicit_vertices
                if outside_text_objects(vertex, text_objects, args.bbox_expansion)
            ]
            merged_batches.append(
                [(float(item["x"]), float(item["y"])) for item in filtered]
                + [(float(item["x"]), float(item["y"])) for item in text_objects]
            )
        maximum_points = max(max(len(points), 1) for points in merged_batches)
        points = torch.zeros((len(samples), maximum_points, 2), device=device)
        point_mask = torch.zeros(
            (len(samples), maximum_points), device=device, dtype=torch.bool
        )
        for batch_index, current_points in enumerate(merged_batches):
            if current_points:
                points[batch_index, : len(current_points)] = torch.tensor(
                    current_points, device=device
                )
                point_mask[batch_index, : len(current_points)] = True
        pair_indices, pair_mask = enumerate_all_pairs(point_mask)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bond_output = bond_head(pixels, points, pair_indices, pair_mask)
        flat_offset = 0
        for batch_index, sample in enumerate(samples):
            predicted_points = points[batch_index, point_mask[batch_index]].float().cpu()
            expected_points = torch.tensor(
                [
                    (
                        atom["center"][0] / sample["layers"]["width"],
                        atom["center"][1] / sample["layers"]["height"],
                    )
                    for atom in sample["layers"]["atoms"]
                ],
                dtype=torch.float32,
            )
            assignment = match_vertices(predicted_points, expected_points)
            expected_edges = {
                tuple(sorted((bond["begin_atom"], bond["end_atom"])))
                for bond in sample["layers"]["bonds"]
            }
            edge_attributes = {
                tuple(sorted((bond["begin_atom"], bond["end_atom"]))): {
                    "display": BOND_TYPE_TO_ID[bond["display_type"]],
                    "chemical": BOND_TYPE_TO_ID[bond["chemical_type"]],
                    "direction": DIRECTION_TO_ID[bond["direction"]],
                }
                for bond in sample["layers"]["bonds"]
            }
            current_pair_count = int(pair_mask[batch_index].sum())
            current_pairs = pair_indices[batch_index, :current_pair_count].cpu()
            edge_probabilities = bond_output["edge_logits"][
                flat_offset : flat_offset + current_pair_count
            ].sigmoid().float().cpu()
            predicted_display = bond_output["display_logits"][
                flat_offset : flat_offset + current_pair_count
            ].argmax(-1).cpu()
            predicted_chemical = bond_output["chemical_logits"][
                flat_offset : flat_offset + current_pair_count
            ].argmax(-1).cpu()
            predicted_direction = bond_output["direction_logits"][
                flat_offset : flat_offset + current_pair_count
            ].argmax(-1).cpu()
            predicted_edge_offsets = torch.where(
                edge_probabilities >= bond_threshold
            )[0]
            mapped_edges = set()
            false_edges = 0
            display_hits = set()
            chemical_hits = set()
            direction_hits = set()
            object_hits = set()
            for pair_offset in predicted_edge_offsets.tolist():
                begin, end = current_pairs[pair_offset].tolist()
                if begin not in assignment or end not in assignment:
                    false_edges += 1
                    continue
                mapped = tuple(sorted((assignment[begin], assignment[end])))
                if mapped not in expected_edges or mapped in mapped_edges:
                    false_edges += 1
                    continue
                mapped_edges.add(mapped)
                attributes = edge_attributes[mapped]
                display_correct = int(predicted_display[pair_offset]) == attributes["display"]
                chemical_correct = int(predicted_chemical[pair_offset]) == attributes["chemical"]
                direction_correct = int(predicted_direction[pair_offset]) == attributes["direction"]
                if display_correct:
                    display_hits.add(mapped)
                if chemical_correct:
                    chemical_hits.add(mapped)
                if direction_correct:
                    direction_hits.add(mapped)
                if display_correct and chemical_correct and direction_correct:
                    object_hits.add(mapped)
            matched_gt = set(assignment.values())
            matchable_edges = {
                edge for edge in expected_edges if set(edge).issubset(matched_gt)
            }
            true_positive = len(mapped_edges)
            predicted_edge_count = len(predicted_edge_offsets)
            vertex_exact = (
                len(assignment) == len(expected_points) == len(predicted_points)
            )
            edge_exact = mapped_edges == expected_edges and false_edges == 0
            totals["samples"] += 1
            totals["predicted_vertices"] += len(predicted_points)
            totals["expected_vertices"] += len(expected_points)
            totals["matched_vertices"] += len(assignment)
            totals["vertex_exact"] += int(vertex_exact)
            totals["predicted_edges"] += predicted_edge_count
            totals["expected_edges"] += len(expected_edges)
            totals["true_positive_edges"] += true_positive
            totals["matchable_edges"] += len(matchable_edges)
            totals["skeleton_exact"] += int(vertex_exact and edge_exact)
            totals["vertex_exact_samples"] += int(vertex_exact)
            totals["edge_exact_given_vertex_exact"] += int(vertex_exact and edge_exact)
            totals["display_type_correct_on_recovered_edges"] += len(display_hits)
            totals["chemical_type_correct_on_recovered_edges"] += len(chemical_hits)
            totals["direction_correct_on_recovered_edges"] += len(direction_hits)
            totals["bond_object_correct"] += len(object_hits)
            for edge, attributes in edge_attributes.items():
                name = BOND_TYPES[attributes["display"]]
                class_expected[name] += 1
                class_edge_hits[name] += int(edge in mapped_edges)
                class_display_hits[name] += int(edge in display_hits)
                class_object_hits[name] += int(edge in object_hits)
            assigned_targets = set(assignment.values())
            sample_rows.append(
                {
                    "key": sample["key"],
                    "predicted_vertices": len(predicted_points),
                    "expected_vertices": len(expected_points),
                    "matched_vertices": len(assignment),
                    "predicted_edges": predicted_edge_count,
                    "expected_edges": len(expected_edges),
                    "true_positive_edges": true_positive,
                    "matchable_edges": len(matchable_edges),
                    "vertex_exact": vertex_exact,
                    "skeleton_exact": vertex_exact and edge_exact,
                    "correct_display_types": len(display_hits),
                    "correct_chemical_types": len(chemical_hits),
                    "correct_directions": len(direction_hits),
                    "correct_bond_objects": len(object_hits),
                    "unmatched_vertices": unmatched_vertex_diagnostics(
                        predicted_points, assignment, sample
                    ),
                    "missing_vertices": [
                        {
                            "atom_id": int(atom["atom_id"]),
                            "x": float(atom["center"][0]),
                            "y": float(atom["center"][1]),
                            "visible_text": bool(atom["visible_text"]),
                        }
                        for atom in sample["layers"]["atoms"]
                        if atom["atom_id"] not in assigned_targets
                    ],
                }
            )
            flat_offset += current_pair_count
        if totals["samples"] % 100 < args.batch_size:
            print(f"evaluated={totals['samples']}", flush=True)

    vertex_precision = totals["matched_vertices"] / max(
        totals["predicted_vertices"], 1
    )
    vertex_recall = totals["matched_vertices"] / max(totals["expected_vertices"], 1)
    edge_precision = totals["true_positive_edges"] / max(totals["predicted_edges"], 1)
    edge_recall = totals["true_positive_edges"] / max(totals["expected_edges"], 1)
    summary = {
        "samples": totals["samples"],
        "thresholds": {
            "text": args.text_threshold,
            "implicit": args.implicit_threshold,
            "bond": bond_threshold,
        },
        "vertex": {
            "precision": vertex_precision,
            "recall": vertex_recall,
            "f1": 2
            * vertex_precision
            * vertex_recall
            / max(vertex_precision + vertex_recall, 1e-12),
            "exact": totals["vertex_exact"] / max(totals["samples"], 1),
        },
        "edge": {
            "precision": edge_precision,
            "recall": edge_recall,
            "f1": 2
            * edge_precision
            * edge_recall
            / max(edge_precision + edge_recall, 1e-12),
            "recall_given_matched_endpoints": totals["true_positive_edges"]
            / max(totals["matchable_edges"], 1),
            "exact_given_vertex_exact": totals["edge_exact_given_vertex_exact"]
            / max(totals["vertex_exact_samples"], 1),
            "display_type_accuracy_on_recovered_true_edges": totals[
                "display_type_correct_on_recovered_edges"
            ]
            / max(totals["true_positive_edges"], 1),
            "chemical_type_accuracy_on_recovered_true_edges": totals[
                "chemical_type_correct_on_recovered_edges"
            ]
            / max(totals["true_positive_edges"], 1),
            "direction_accuracy_on_recovered_true_edges": totals[
                "direction_correct_on_recovered_edges"
            ]
            / max(totals["true_positive_edges"], 1),
            "bond_object_recall": totals["bond_object_correct"]
            / max(totals["expected_edges"], 1),
            "edge_recall_by_gt_display_type": {
                name: class_edge_hits[name] / max(class_expected[name], 1)
                for name in BOND_TYPES
            },
            "display_type_recall_by_gt_display_type": {
                name: class_display_hits[name] / max(class_expected[name], 1)
                for name in BOND_TYPES
            },
            "bond_object_recall_by_gt_display_type": {
                name: class_object_hits[name] / max(class_expected[name], 1)
                for name in BOND_TYPES
            },
        },
        "skeleton_exact": totals["skeleton_exact"] / max(totals["samples"], 1),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
