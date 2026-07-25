#!/usr/bin/env python3
"""Evaluate v0.7.37 grouped vertices with the v0.7.36 bond object."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

import torch

from chemvector.bond_object import BOND_TYPES, BOND_TYPE_TO_ID, DIRECTION_TO_ID
from chemvector.data import ChemVectorIterableDataset
from chemvector.dual_vertex_selector import (
    IMPLICIT_ROLE,
    TEXT_ROLE,
    dual_candidate_auxiliary,
    merge_dual_candidates,
    padded_candidate_roles,
    role_suppressed_role_threshold_indices,
)
from chemvector.dual_graph_denoiser import (
    DualGraphDenoiserConfig,
    GroundedDualGraphDenoiser,
)
from chemvector.edge_evidence import load_edge_teacher, score_full_adjacency_edges
from chemvector.endpoint_evidence import endpoint_ray_features
from chemvector.evaluate_node_conditioned_bond import (
    match_vertices,
    outside_text_objects,
    unmatched_vertex_diagnostics,
)
from chemvector.grounded_selector import (
    fuse_dense_candidates_with_queries,
    sample_candidate_features,
    sample_candidate_ink_patches,
)
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
    enumerate_all_pairs,
)
from chemvector.pretrain_dual_vertex_selector import (
    load_dual_pipeline,
    selector_forward,
)
from chemvector.pretrain_node_conditioned_bond import KnownEndpointBondCollator
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
)
from chemvector.text_object import decode_text_objects
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


ROLE_NAMES = {IMPLICIT_ROLE: "implicit", TEXT_ROLE: "text"}
LENGTH_BUCKETS = ("short_le_48", "medium_48_96", "long_gt_96")


def bond_length_bucket(length_pixels: float) -> str:
    if length_pixels <= 48.0:
        return "short_le_48"
    if length_pixels <= 96.0:
        return "medium_48_96"
    return "long_gt_96"


def normalized_thresholds(
    checkpoint_threshold: float, requested: list[float] | None
) -> list[float]:
    values = requested or [checkpoint_threshold, 0.80, 0.85, 0.90, 0.95]
    values = sorted(set(float(value) for value in values))
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("Bond thresholds must lie in (0, 1)")
    if checkpoint_threshold not in values:
        values.append(checkpoint_threshold)
        values.sort()
    return values


def selector_candidate_protocol(payload: dict) -> dict:
    """Read the deployed candidate protocol, including pre-metadata checkpoints."""

    defaults = {
        "text_threshold": 0.05,
        "implicit_threshold": 0.05,
        "text_filter_threshold": 0.30,
        "matching_radius_pixels": 6.0,
        "suppression_radius_pixels": 6.0,
    }
    protocol = payload.get("candidate_protocol")
    if protocol is None:
        return defaults
    return {**defaults, **protocol}


def load_evaluation_selector(checkpoint: str, device: torch.device):
    """Load either a structured selector or its full-adjacency denoiser."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "dual_graph_denoiser" not in payload:
        config = StructuredSelectorConfig(**payload["structured_selector_config"])
        selector = GroundedStructuredSelector(config).to(device)
        selector.load_state_dict(payload["dual_vertex_selector"])
        selector.eval()
        return selector, config, payload, None

    base_checkpoint = payload["base_selector_checkpoint"]
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    config = StructuredSelectorConfig(**base_payload["structured_selector_config"])
    base_selector = GroundedStructuredSelector(config).to(device)
    base_selector.load_state_dict(base_payload["dual_vertex_selector"])
    graph_config = DualGraphDenoiserConfig(**payload["dual_graph_denoiser_config"])
    selector = GroundedDualGraphDenoiser(base_selector, graph_config).to(device)
    selector.load_state_dict(payload["dual_graph_denoiser"])
    selector.eval()
    return selector, config, base_payload, base_checkpoint


def resolved_real_thresholds(
    shared: float, implicit: float | None, text: float | None
) -> torch.Tensor:
    values = torch.tensor(
        [shared if implicit is None else implicit, shared if text is None else text],
        dtype=torch.float32,
    )
    if bool(((values <= 0.0) | (values >= 1.0)).any()):
        raise ValueError("REAL thresholds must lie in (0, 1)")
    return values


def new_totals() -> dict:
    return {
        "samples": 0,
        "predicted_vertices": 0,
        "expected_vertices": 0,
        "matched_vertices": 0,
        "vertex_exact": 0,
        "role_matched_vertices": 0,
        "role_vertex_exact": 0,
        "predicted_edges": 0,
        "expected_edges": 0,
        "true_positive_edges": 0,
        "matchable_edges": 0,
        "skeleton_exact": 0,
        "vertex_exact_samples": 0,
        "edge_exact_given_vertex_exact": 0,
        "display_hits": 0,
        "chemical_hits": 0,
        "direction_hits": 0,
        "bond_object_hits": 0,
        "by_display": {
            name: {"expected": 0, "edge_hits": 0, "display_hits": 0, "object_hits": 0}
            for name in BOND_TYPES
        },
        "by_length": {
            name: {"expected": 0, "matchable": 0, "edge_hits": 0, "object_hits": 0}
            for name in LENGTH_BUCKETS
        },
    }


def ground_truth_edges(sample: dict) -> tuple[set[tuple[int, int]], dict]:
    edges = set()
    attributes = {}
    for bond in sample["layers"]["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        begin = torch.tensor(bond["begin"], dtype=torch.float32)
        end = torch.tensor(bond["end"], dtype=torch.float32)
        length_pixels = float(torch.linalg.vector_norm(end - begin))
        edges.add(edge)
        attributes[edge] = {
            "display": BOND_TYPE_TO_ID[bond["display_type"]],
            "chemical": BOND_TYPE_TO_ID[bond["chemical_type"]],
            "direction": DIRECTION_TO_ID[bond["direction"]],
            "length_pixels": length_pixels,
            "length_bucket": bond_length_bucket(length_pixels),
        }
    return edges, attributes


def match_vertices_by_role(
    predicted: torch.Tensor,
    predicted_roles: torch.Tensor,
    expected: torch.Tensor,
    expected_roles: torch.Tensor,
    radius_pixels: float = 6.0,
) -> dict[int, int]:
    if predicted_roles.shape != (len(predicted),):
        raise ValueError("Predicted roles do not match predicted vertices")
    if expected_roles.shape != (len(expected),):
        raise ValueError("Expected roles do not match expected vertices")
    assignment = {}
    for role in ROLE_NAMES:
        predicted_indices = torch.where(predicted_roles == role)[0]
        expected_indices = torch.where(expected_roles == role)[0]
        local = match_vertices(
            predicted[predicted_indices],
            expected[expected_indices],
            radius_pixels=radius_pixels,
        )
        assignment.update(
            {
                int(predicted_indices[predicted_index]): int(expected_indices[expected_index])
                for predicted_index, expected_index in local.items()
            }
        )
    return assignment


def role_candidate_coverage(
    candidate_points: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_roles: torch.Tensor,
    expected_points: torch.Tensor,
    expected_roles: torch.Tensor,
    *,
    radius_pixels: float,
    image_size: int,
) -> list[dict]:
    """Report whether each expected atom had a same-role candidate before selection."""

    if candidate_points.shape != (*candidate_mask.shape, 2):
        raise ValueError("Candidate points and mask differ")
    if candidate_roles.shape != candidate_mask.shape:
        raise ValueError("Candidate roles and mask differ")
    if expected_points.shape != (len(expected_roles), 2):
        raise ValueError("Expected points and roles differ")
    rows = []
    for point, role in zip(expected_points, expected_roles):
        valid = candidate_mask & (candidate_roles == role)
        distances = (
            torch.linalg.vector_norm(candidate_points[valid] - point, dim=-1)
            * image_size
        )
        count = int((distances <= radius_pixels).sum()) if len(distances) else 0
        rows.append(
            {
                "candidate_available": count > 0,
                "candidate_count_within_radius": count,
                "nearest_same_role_candidate_distance_pixels": (
                    float(distances.min()) if len(distances) else None
                ),
            }
        )
    return rows


def summarize(totals: dict, threshold: float) -> dict:
    vertex_precision = totals["matched_vertices"] / max(totals["predicted_vertices"], 1)
    vertex_recall = totals["matched_vertices"] / max(totals["expected_vertices"], 1)
    edge_precision = totals["true_positive_edges"] / max(totals["predicted_edges"], 1)
    edge_recall = totals["true_positive_edges"] / max(totals["expected_edges"], 1)
    return {
        "samples": totals["samples"],
        "bond_threshold": threshold,
        "vertex": {
            "precision": vertex_precision,
            "recall": vertex_recall,
            "f1": 2.0 * vertex_precision * vertex_recall / max(vertex_precision + vertex_recall, 1e-12),
            "exact": totals["vertex_exact"] / max(totals["samples"], 1),
            "role_aware_precision": totals["role_matched_vertices"]
            / max(totals["predicted_vertices"], 1),
            "role_aware_recall": totals["role_matched_vertices"]
            / max(totals["expected_vertices"], 1),
            "role_aware_exact": totals["role_vertex_exact"]
            / max(totals["samples"], 1),
        },
        "edge": {
            "precision": edge_precision,
            "recall": edge_recall,
            "f1": 2.0 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12),
            "recall_given_matched_endpoints": totals["true_positive_edges"]
            / max(totals["matchable_edges"], 1),
            "exact_given_vertex_exact": totals["edge_exact_given_vertex_exact"]
            / max(totals["vertex_exact_samples"], 1),
            "display_type_accuracy_on_recovered_true_edges": totals["display_hits"]
            / max(totals["true_positive_edges"], 1),
            "chemical_type_accuracy_on_recovered_true_edges": totals["chemical_hits"]
            / max(totals["true_positive_edges"], 1),
            "direction_accuracy_on_recovered_true_edges": totals["direction_hits"]
            / max(totals["true_positive_edges"], 1),
            "bond_object_recall": totals["bond_object_hits"]
            / max(totals["expected_edges"], 1),
            "by_gt_display_type": {
                name: {
                    "count": values["expected"],
                    "edge_recall": values["edge_hits"] / max(values["expected"], 1),
                    "display_type_recall": values["display_hits"] / max(values["expected"], 1),
                    "bond_object_recall": values["object_hits"] / max(values["expected"], 1),
                }
                for name, values in totals["by_display"].items()
            },
            "by_gt_length_pixels": {
                name: {
                    "count": values["expected"],
                    "matchable_count": values["matchable"],
                    "edge_recall": values["edge_hits"] / max(values["expected"], 1),
                    "recall_given_matched_endpoints": values["edge_hits"]
                    / max(values["matchable"], 1),
                    "bond_object_recall": values["object_hits"] / max(values["expected"], 1),
                }
                for name, values in totals["by_length"].items()
            },
        },
        "skeleton_exact": totals["skeleton_exact"] / max(totals["samples"], 1),
    }


def update_totals(
    totals: dict,
    *,
    predicted_vertex_count: int,
    expected_vertex_count: int,
    assignment: dict[int, int],
    role_assignment: dict[int, int],
    expected_edges: set[tuple[int, int]],
    edge_attributes: dict,
    current_pairs: torch.Tensor,
    edge_probabilities: torch.Tensor,
    predicted_display: torch.Tensor,
    predicted_chemical: torch.Tensor,
    predicted_direction: torch.Tensor,
    threshold: float,
) -> dict:
    predicted_offsets = torch.where(edge_probabilities >= threshold)[0]
    mapped_edges = set()
    false_edges = 0
    display_hits = set()
    chemical_hits = set()
    direction_hits = set()
    object_hits = set()
    for pair_offset in predicted_offsets.tolist():
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

    matched_targets = set(assignment.values())
    matchable_edges = {edge for edge in expected_edges if set(edge).issubset(matched_targets)}
    vertex_exact = len(assignment) == expected_vertex_count == predicted_vertex_count
    edge_exact = mapped_edges == expected_edges and false_edges == 0
    totals["samples"] += 1
    totals["predicted_vertices"] += predicted_vertex_count
    totals["expected_vertices"] += expected_vertex_count
    totals["matched_vertices"] += len(assignment)
    totals["vertex_exact"] += int(vertex_exact)
    totals["role_matched_vertices"] += len(role_assignment)
    totals["role_vertex_exact"] += int(
        len(role_assignment) == expected_vertex_count == predicted_vertex_count
    )
    totals["predicted_edges"] += len(predicted_offsets)
    totals["expected_edges"] += len(expected_edges)
    totals["true_positive_edges"] += len(mapped_edges)
    totals["matchable_edges"] += len(matchable_edges)
    totals["skeleton_exact"] += int(vertex_exact and edge_exact)
    totals["vertex_exact_samples"] += int(vertex_exact)
    totals["edge_exact_given_vertex_exact"] += int(vertex_exact and edge_exact)
    totals["display_hits"] += len(display_hits)
    totals["chemical_hits"] += len(chemical_hits)
    totals["direction_hits"] += len(direction_hits)
    totals["bond_object_hits"] += len(object_hits)
    for edge, attributes in edge_attributes.items():
        display_name = BOND_TYPES[attributes["display"]]
        display_values = totals["by_display"][display_name]
        display_values["expected"] += 1
        display_values["edge_hits"] += int(edge in mapped_edges)
        display_values["display_hits"] += int(edge in display_hits)
        display_values["object_hits"] += int(edge in object_hits)
        length_values = totals["by_length"][attributes["length_bucket"]]
        length_values["expected"] += 1
        length_values["matchable"] += int(edge in matchable_edges)
        length_values["edge_hits"] += int(edge in mapped_edges)
        length_values["object_hits"] += int(edge in object_hits)
    return {
        "predicted_edges": len(predicted_offsets),
        "true_positive_edges": len(mapped_edges),
        "matchable_edges": len(matchable_edges),
        "false_edges": false_edges,
        "vertex_exact": vertex_exact,
        "skeleton_exact": vertex_exact and edge_exact,
        "correct_display_types": len(display_hits),
        "correct_chemical_types": len(chemical_hits),
        "correct_directions": len(direction_hits),
        "correct_bond_objects": len(object_hits),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--bond-checkpoint", required=True)
    parser.add_argument("--edge-teacher-checkpoint")
    parser.add_argument("--edge-teacher-chunk-size", type=int, default=2048)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--real-threshold", type=float, default=0.70)
    parser.add_argument("--implicit-real-threshold", type=float)
    parser.add_argument("--text-real-threshold", type=float)
    parser.add_argument("--bond-thresholds", nargs="+", type=float)
    parser.add_argument("--bbox-expansion-pixels", type=float, default=8.0)
    args = parser.parse_args()

    device = torch.device("cuda")
    selector, selector_config, selector_payload, base_selector_checkpoint = (
        load_evaluation_selector(args.selector_checkpoint, device)
    )
    base, detector, grounded_selector, graph, text_head, implicit_head = load_dual_pipeline(
        selector_payload["text_object_checkpoint"],
        selector_payload["implicit_vertex_checkpoint"],
        device,
        maximum_candidates=selector_config.maximum_candidates,
    )
    protocol = selector_candidate_protocol(selector_payload)
    edge_teacher = (
        load_edge_teacher(args.edge_teacher_checkpoint, device)
        if args.edge_teacher_checkpoint
        else None
    )
    real_thresholds = resolved_real_thresholds(
        args.real_threshold,
        args.implicit_real_threshold,
        args.text_real_threshold,
    )

    bond_payload = torch.load(args.bond_checkpoint, map_location="cpu", weights_only=False)
    bond_config = NodeConditionedBondConfig(**bond_payload["config"])
    bond_head = NodeConditionedBondObject(bond_config).to(device)
    bond_head.load_state_dict(bond_payload["node_conditioned_bond"])
    bond_head.eval()
    checkpoint_bond_threshold = float(bond_payload["validation"]["threshold"])
    bond_thresholds = normalized_thresholds(checkpoint_bond_threshold, args.bond_thresholds)
    totals_by_threshold = {threshold: new_totals() for threshold in bond_thresholds}
    known_endpoint_totals = {threshold: new_totals() for threshold in bond_thresholds}

    dataset = ChemVectorIterableDataset(
        expand_shards(args.validation_shards), sequence_version="0.4"
    )
    records = list(islice(iter(dataset), args.samples))
    if len(records) != args.samples:
        raise ValueError("Validation dataset contains fewer samples than requested")
    image_collator = KnownEndpointBondCollator()
    primary_rows = []

    for start in range(0, len(records), args.batch_size):
        samples = records[start : start + args.batch_size]
        pixels = torch.stack([image_collator._pixels(sample["image"]) for sample in samples]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = base.vision_encoder.forward_stroke_resolution(pixels)
            query_output = detector(features)
            text_output = text_head(features)
            implicit_output = implicit_head(features)
        text_batches = decode_text_objects(
            text_output,
            threshold=float(protocol["text_threshold"]),
            maximum_objects=selector_config.maximum_candidates,
        )
        filter_text_batches = decode_text_objects(
            text_output,
            threshold=float(protocol["text_filter_threshold"]),
            maximum_objects=selector_config.maximum_candidates,
        )
        implicit_batches = decode_vertices(
            implicit_output,
            threshold=float(protocol["implicit_threshold"]),
            maximum_vertices=selector_config.maximum_candidates,
            nms_kernel_size=1,
        )
        merged_batches = []
        role_batches = []
        for text_items, filter_text, implicit_items in zip(
            text_batches, filter_text_batches, implicit_batches
        ):
            filtered_implicit = [
                item
                for item in implicit_items
                if outside_text_objects(item, filter_text, args.bbox_expansion_pixels)
            ]
            merged, roles = merge_dual_candidates(
                text_items,
                filtered_implicit,
                maximum_candidates=selector_config.maximum_candidates,
            )
            merged_batches.append(merged)
            role_batches.append(roles)

        grounded = fuse_dense_candidates_with_queries(
            query_output,
            merged_batches,
            maximum_candidates=selector_config.maximum_candidates,
            image_size=pixels.size(-1),
        )
        candidate_roles = padded_candidate_roles(role_batches, grounded["candidate_mask"])
        candidate_auxiliary = dual_candidate_auxiliary(
            grounded["candidate_auxiliary"], merged_batches, grounded["candidate_mask"]
        )
        candidate_visual = sample_candidate_features(
            features, grounded["grounded_centers"], grounded["candidate_mask"]
        )
        candidate_pixel_patches = None
        if getattr(selector.config, "patch_mode", "none") in {
            "pixel",
            "semantic_pixel",
        }:
            candidate_pixel_patches = sample_candidate_ink_patches(
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                patch_size=selector.config.patch_size,
            )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            candidate_hidden = grounded_selector.encode(
                grounded["query_features"],
                grounded["scalar_features"],
                grounded["grounded_centers"],
                grounded["candidate_mask"],
            )
            graph_output = graph(
                candidate_hidden,
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                grounded["base_logits"],
            )
            selector_edge_logits = graph_output["edge_logits"]
        if edge_teacher is not None:
            selector_edge_logits = score_full_adjacency_edges(
                edge_teacher,
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                graph_output["pair_indices"],
                chunk_size=args.edge_teacher_chunk_size,
            )
        selector_edge_auxiliary = None
        if getattr(selector.config, "edge_auxiliary_dim", 0):
            selector_edge_auxiliary = endpoint_ray_features(
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                graph_output["pair_indices"],
            )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            selector_logits = selector_forward(
                selector,
                {
                    "candidate_hidden": candidate_hidden,
                    "base_logits": grounded["base_logits"],
                    "points": grounded["grounded_centers"],
                    "candidate_mask": grounded["candidate_mask"],
                    "edge_logits": selector_edge_logits,
                    "candidate_auxiliary": candidate_auxiliary,
                    "candidate_visual": candidate_visual,
                    "candidate_roles": candidate_roles,
                    "edge_auxiliary": selector_edge_auxiliary,
                    "candidate_pixel_patches": candidate_pixel_patches,
                },
                graph_output["pair_indices"],
            )

        selected_indices = []
        selected_batches = []
        for batch_index in range(len(samples)):
            indices = role_suppressed_role_threshold_indices(
                selector_logits[batch_index],
                grounded["grounded_centers"][batch_index],
                grounded["candidate_mask"][batch_index],
                candidate_roles[batch_index],
                probability_thresholds=real_thresholds,
                suppression_radius_pixels=float(protocol["suppression_radius_pixels"]),
                image_size=pixels.size(-1),
            )
            selected_indices.append(indices)
            selected_batches.append(grounded["grounded_centers"][batch_index, indices])

        maximum_points = max(max(len(points), 1) for points in selected_batches)
        points = torch.zeros((len(samples), maximum_points, 2), device=device)
        point_mask = torch.zeros((len(samples), maximum_points), device=device, dtype=torch.bool)
        for batch_index, selected in enumerate(selected_batches):
            if len(selected):
                points[batch_index, : len(selected)] = selected
                point_mask[batch_index, : len(selected)] = True
        pair_indices, pair_mask = enumerate_all_pairs(point_mask)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            bond_output = bond_head(pixels, points, pair_indices, pair_mask)

        maximum_gt_points = max(len(sample["layers"]["atoms"]) for sample in samples)
        gt_points = torch.zeros((len(samples), maximum_gt_points, 2), device=device)
        gt_point_mask = torch.zeros(
            (len(samples), maximum_gt_points), device=device, dtype=torch.bool
        )
        for batch_index, sample in enumerate(samples):
            width = float(sample["layers"]["width"])
            height = float(sample["layers"]["height"])
            count = len(sample["layers"]["atoms"])
            gt_points[batch_index, :count] = torch.tensor(
                [
                    (atom["center"][0] / width, atom["center"][1] / height)
                    for atom in sample["layers"]["atoms"]
                ],
                device=device,
            )
            gt_point_mask[batch_index, :count] = True
        gt_pair_indices, gt_pair_mask = enumerate_all_pairs(gt_point_mask)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gt_bond_output = bond_head(
                pixels, gt_points, gt_pair_indices, gt_pair_mask
            )

        flat_offset = 0
        gt_flat_offset = 0
        for batch_index, sample in enumerate(samples):
            predicted_points = points[batch_index, point_mask[batch_index]].float().cpu()
            expected_points = torch.tensor(
                [
                    (atom["center"][0] / sample["layers"]["width"], atom["center"][1] / sample["layers"]["height"])
                    for atom in sample["layers"]["atoms"]
                ],
                dtype=torch.float32,
            )
            assignment = match_vertices(predicted_points, expected_points)
            selected = selected_indices[batch_index]
            roles = candidate_roles[batch_index, selected].cpu()
            expected_roles = torch.tensor(
                [
                    TEXT_ROLE if atom["visible_text"] else IMPLICIT_ROLE
                    for atom in sample["layers"]["atoms"]
                ],
                dtype=torch.long,
            )
            candidate_coverage_rows = role_candidate_coverage(
                grounded["grounded_centers"][batch_index].float().cpu(),
                grounded["candidate_mask"][batch_index].cpu(),
                candidate_roles[batch_index].cpu(),
                expected_points,
                expected_roles,
                radius_pixels=float(protocol["matching_radius_pixels"]),
                image_size=int(sample["layers"]["width"]),
            )
            candidate_coverage = {
                int(atom["atom_id"]): row
                for atom, row in zip(
                    sample["layers"]["atoms"], candidate_coverage_rows
                )
            }
            role_assignment = match_vertices_by_role(
                predicted_points, roles, expected_points, expected_roles
            )
            expected_edges, edge_attributes = ground_truth_edges(sample)
            current_pair_count = int(pair_mask[batch_index].sum())
            current_pairs = pair_indices[batch_index, :current_pair_count].cpu()
            current_slice = slice(flat_offset, flat_offset + current_pair_count)
            edge_probabilities = bond_output["edge_logits"][current_slice].sigmoid().float().cpu()
            predicted_display = bond_output["display_logits"][current_slice].argmax(-1).cpu()
            predicted_chemical = bond_output["chemical_logits"][current_slice].argmax(-1).cpu()
            predicted_direction = bond_output["direction_logits"][current_slice].argmax(-1).cpu()
            gt_pair_count = int(gt_pair_mask[batch_index].sum())
            gt_pairs = gt_pair_indices[batch_index, :gt_pair_count].cpu()
            gt_slice = slice(gt_flat_offset, gt_flat_offset + gt_pair_count)
            gt_edge_probabilities = (
                gt_bond_output["edge_logits"][gt_slice].sigmoid().float().cpu()
            )
            gt_predicted_display = (
                gt_bond_output["display_logits"][gt_slice].argmax(-1).cpu()
            )
            gt_predicted_chemical = (
                gt_bond_output["chemical_logits"][gt_slice].argmax(-1).cpu()
            )
            gt_predicted_direction = (
                gt_bond_output["direction_logits"][gt_slice].argmax(-1).cpu()
            )
            identity_assignment = {index: index for index in range(len(expected_points))}
            threshold_rows = {}
            for threshold in bond_thresholds:
                threshold_rows[f"{threshold:.2f}"] = update_totals(
                    totals_by_threshold[threshold],
                    predicted_vertex_count=len(predicted_points),
                    expected_vertex_count=len(expected_points),
                    assignment=assignment,
                    role_assignment=role_assignment,
                    expected_edges=expected_edges,
                    edge_attributes=edge_attributes,
                    current_pairs=current_pairs,
                    edge_probabilities=edge_probabilities,
                    predicted_display=predicted_display,
                    predicted_chemical=predicted_chemical,
                    predicted_direction=predicted_direction,
                    threshold=threshold,
                )
                update_totals(
                    known_endpoint_totals[threshold],
                    predicted_vertex_count=len(expected_points),
                    expected_vertex_count=len(expected_points),
                    assignment=identity_assignment,
                    role_assignment=identity_assignment,
                    expected_edges=expected_edges,
                    edge_attributes=edge_attributes,
                    current_pairs=gt_pairs,
                    edge_probabilities=gt_edge_probabilities,
                    predicted_display=gt_predicted_display,
                    predicted_chemical=gt_predicted_chemical,
                    predicted_direction=gt_predicted_direction,
                    threshold=threshold,
                )
            probabilities = selector_logits[batch_index, selected].sigmoid().float().cpu()
            assigned_targets = set(assignment.values())
            matched_vertex_errors = [
                {
                    "predicted_index": predicted_index,
                    "expected_index": expected_index,
                    "dx_pixels": float(
                        (predicted_points[predicted_index, 0] - expected_points[expected_index, 0])
                        * sample["layers"]["width"]
                    ),
                    "dy_pixels": float(
                        (predicted_points[predicted_index, 1] - expected_points[expected_index, 1])
                        * sample["layers"]["height"]
                    ),
                    "distance_pixels": float(
                        torch.linalg.vector_norm(
                            (predicted_points[predicted_index] - expected_points[expected_index])
                            * torch.tensor(
                                (sample["layers"]["width"], sample["layers"]["height"])
                            )
                        )
                    ),
                }
                for predicted_index, expected_index in sorted(assignment.items())
            ]
            primary_rows.append(
                {
                    "key": sample["key"],
                    "predicted_vertices": len(predicted_points),
                    "expected_vertices": len(expected_points),
                    "matched_vertices": len(assignment),
                    "role_matched_vertices": len(role_assignment),
                    "selected_role_counts": {
                        ROLE_NAMES[role]: int((roles == role).sum()) for role in ROLE_NAMES
                    },
                    "selected_vertices": [
                        {
                            "x": float(point[0] * sample["layers"]["width"]),
                            "y": float(point[1] * sample["layers"]["height"]),
                            "role": ROLE_NAMES[int(role)],
                            "real_probability": float(probability),
                        }
                        for point, role, probability in zip(predicted_points, roles, probabilities)
                    ],
                    "by_bond_threshold": threshold_rows,
                    "matched_vertex_errors": matched_vertex_errors,
                    "unmatched_vertices": unmatched_vertex_diagnostics(predicted_points, assignment, sample),
                    "missing_vertices": [
                        {
                            "atom_id": int(atom["atom_id"]),
                            "x": float(atom["center"][0]),
                            "y": float(atom["center"][1]),
                            "visible_text": bool(atom["visible_text"]),
                            **candidate_coverage[int(atom["atom_id"])],
                        }
                        for atom in sample["layers"]["atoms"]
                        if atom["atom_id"] not in assigned_targets
                    ],
                }
            )
            flat_offset += current_pair_count
            gt_flat_offset += gt_pair_count
        if len(primary_rows) % 100 < args.batch_size:
            print(f"evaluated={len(primary_rows)}", flush=True)

    by_threshold = {
        f"{threshold:.2f}": summarize(totals_by_threshold[threshold], threshold)
        for threshold in bond_thresholds
    }
    known_endpoint_by_threshold = {
        f"{threshold:.2f}": summarize(known_endpoint_totals[threshold], threshold)
        for threshold in bond_thresholds
    }
    checkpoint_key = f"{checkpoint_bond_threshold:.2f}"
    best_key = max(
        by_threshold,
        key=lambda key: (
            by_threshold[key]["skeleton_exact"],
            by_threshold[key]["edge"]["f1"],
        ),
    )
    summary = {
        "samples": len(records),
        "checkpoints": {
            "selector": args.selector_checkpoint,
            "base_selector": base_selector_checkpoint,
            "edge_teacher": args.edge_teacher_checkpoint,
            "bond": args.bond_checkpoint,
        },
        "candidate_protocol": {
            **protocol,
            "bbox_expansion_pixels": args.bbox_expansion_pixels,
            "real_threshold": args.real_threshold,
            "implicit_real_threshold": float(real_thresholds[IMPLICIT_ROLE]),
            "text_real_threshold": float(real_thresholds[TEXT_ROLE]),
        },
        "checkpoint_bond_threshold": checkpoint_bond_threshold,
        "checkpoint_result": by_threshold[checkpoint_key],
        "best_validation_bond_threshold": float(best_key),
        "best_validation_result": by_threshold[best_key],
        "by_bond_threshold": by_threshold,
        "known_endpoint_reference_by_bond_threshold": known_endpoint_by_threshold,
        "length_bucket_protocol": {
            "short_le_48": "length <= 48 px",
            "medium_48_96": "48 px < length <= 96 px",
            "long_gt_96": "length > 96 px",
            "use": "reporting only; never filters candidate bonds",
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in primary_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
