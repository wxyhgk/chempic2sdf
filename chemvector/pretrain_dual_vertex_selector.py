#!/usr/bin/env python3
"""Train a role-aware whole-set selector over dual-head vertex candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.dual_vertex_selector import (
    ROLE_COUNT,
    dual_candidate_auxiliary,
    grouped_candidate_selection_loss,
    merge_dual_candidates,
    padded_candidate_roles,
    role_aware_candidate_targets,
    role_aware_candidate_groups,
    line_interior_candidate_mask,
    role_aware_match_count,
    role_suppressed_topk_indices,
    role_suppressed_threshold_indices,
)
from chemvector.evaluate_node_conditioned_bond import outside_text_objects
from chemvector.grounded_selector import (
    decode_grounded_queries,
    fuse_dense_candidates_with_queries,
    sample_candidate_features,
    sample_candidate_ink_patches,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.pretrain_structured_selector import load_resized_graph
from chemvector.pretrain_vertex import (
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.proposal_graph import ProposalGraphCollator
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
    structured_selector_loss,
)
from chemvector.text_object import TextObjectCenterHead, decode_text_objects
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


def load_dual_pipeline(
    text_checkpoint: str,
    implicit_checkpoint: str,
    device: torch.device,
    *,
    maximum_candidates: int,
):
    text_payload = torch.load(text_checkpoint, map_location="cpu", weights_only=False)
    implicit_payload = torch.load(
        implicit_checkpoint, map_location="cpu", weights_only=False
    )
    graph_checkpoint = implicit_payload["grounded_graph_checkpoint"]
    if text_payload["grounded_graph_checkpoint"] != graph_checkpoint:
        raise ValueError("Text and implicit vertex heads use different backbones")
    graph, graph_payload = load_resized_graph(
        graph_checkpoint, device, maximum_candidates
    )
    base, detector, grounded_selector, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    text_head = TextObjectCenterHead(text_payload["input_channels"]).to(device)
    text_head.load_state_dict(text_payload["text_object_head"])
    implicit_head = deepcopy(base.vertex_head).to(device)
    implicit_head.load_state_dict(implicit_payload["implicit_vertex_head"])
    for module in (base, detector, grounded_selector, graph, text_head, implicit_head):
        module.eval().requires_grad_(False)
    return base, detector, grounded_selector, graph, text_head, implicit_head


@torch.inference_mode()
def prepare_cache(
    base,
    detector,
    grounded_selector,
    graph,
    text_head,
    implicit_head,
    loader,
    device: torch.device,
    *,
    samples: int,
    text_candidate_threshold: float,
    text_filter_threshold: float,
    implicit_candidate_threshold: float,
    bbox_expansion_pixels: float,
    matching_radius_pixels: float,
    cache_name: str,
    edge_teacher=None,
    edge_teacher_chunk_size: int = 2048,
    include_endpoint_ray_features: bool = False,
    candidate_pixel_patch_size: int = 0,
    include_line_interior_mask: bool = False,
    line_interior_distance_pixels: float = 3.0,
    line_interior_endpoint_distance_pixels: float = 8.0,
    line_interior_ambiguous_atom_radius_pixels: float = 10.0,
) -> tuple[list[dict], torch.Tensor]:
    records: list[dict] = []
    pair_indices = None
    for batch in loader:
        if len(records) >= samples:
            break
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = base.vision_encoder.forward_stroke_resolution(pixels)
            query_output = detector(features)
            text_output = text_head(features)
            implicit_output = implicit_head(features)
        text_batches = decode_text_objects(
            text_output,
            threshold=text_candidate_threshold,
            maximum_objects=graph.config.maximum_candidates,
        )
        filter_text_batches = decode_text_objects(
            text_output,
            threshold=text_filter_threshold,
            maximum_objects=graph.config.maximum_candidates,
        )
        implicit_batches = decode_vertices(
            implicit_output,
            threshold=implicit_candidate_threshold,
            maximum_vertices=graph.config.maximum_candidates,
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
                if outside_text_objects(item, filter_text, bbox_expansion_pixels)
            ]
            merged, roles = merge_dual_candidates(
                text_items,
                filtered_implicit,
                maximum_candidates=graph.config.maximum_candidates,
            )
            merged_batches.append(merged)
            role_batches.append(roles)

        grounded = fuse_dense_candidates_with_queries(
            query_output,
            merged_batches,
            maximum_candidates=graph.config.maximum_candidates,
            image_size=pixels.size(-1),
        )
        candidate_roles = padded_candidate_roles(
            role_batches, grounded["candidate_mask"]
        )
        candidate_auxiliary = dual_candidate_auxiliary(
            grounded["candidate_auxiliary"],
            merged_batches,
            grounded["candidate_mask"],
        )
        candidate_visual = sample_candidate_features(
            features,
            grounded["grounded_centers"],
            grounded["candidate_mask"],
        )
        candidate_pixel_patches = None
        if candidate_pixel_patch_size:
            candidate_pixel_patches = sample_candidate_ink_patches(
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                patch_size=candidate_pixel_patch_size,
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
        edge_logits = graph_output["edge_logits"]
        if edge_teacher is not None:
            from chemvector.edge_evidence import score_full_adjacency_edges

            edge_logits = score_full_adjacency_edges(
                edge_teacher,
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                graph_output["pair_indices"],
                chunk_size=edge_teacher_chunk_size,
            )
        edge_auxiliary = None
        if include_endpoint_ray_features:
            from chemvector.endpoint_evidence import endpoint_ray_features

            edge_auxiliary = endpoint_ray_features(
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                graph_output["pair_indices"],
            )
        targets, assignments = role_aware_candidate_targets(
            grounded["grounded_centers"].float(),
            grounded["candidate_mask"],
            candidate_roles,
            batch["anchor_points"].to(device),
            batch["anchor_mask"].to(device),
            batch["role_targets"].to(device),
            matching_radius_pixels=matching_radius_pixels,
            image_size=pixels.size(-1),
        )
        begin = graph_output["pair_indices"][:, 0]
        end = graph_output["pair_indices"][:, 1]
        begin_targets = assignments[:, begin]
        end_targets = assignments[:, end]
        mapped_pairs = (begin_targets >= 0) & (end_targets >= 0)
        edge_targets = torch.zeros_like(edge_logits, dtype=torch.float32)
        batch_indices = torch.arange(pixels.size(0), device=device)[:, None]
        edge_targets[mapped_pairs] = batch["adjacency"].to(device)[
            batch_indices.expand_as(mapped_pairs),
            begin_targets.clamp_min(0),
            end_targets.clamp_min(0),
        ][mapped_pairs].float()
        candidate_groups = role_aware_candidate_groups(
            grounded["grounded_centers"].float(),
            grounded["candidate_mask"],
            candidate_roles,
            batch["anchor_points"].to(device),
            batch["anchor_mask"].to(device),
            batch["role_targets"].to(device),
            matching_radius_pixels=matching_radius_pixels,
            image_size=pixels.size(-1),
        )
        line_interior_mask = None
        if include_line_interior_mask:
            line_interior_mask = line_interior_candidate_mask(
                grounded["grounded_centers"].float(),
                grounded["candidate_mask"],
                candidate_groups,
                batch["anchor_points"].to(device),
                batch["anchor_mask"].to(device),
                batch["adjacency"].to(device),
                image_size=pixels.size(-1),
                maximum_line_distance_pixels=line_interior_distance_pixels,
                minimum_endpoint_distance_pixels=(
                    line_interior_endpoint_distance_pixels
                ),
                ambiguous_atom_radius_pixels=(
                    line_interior_ambiguous_atom_radius_pixels
                ),
            )
        current_pairs = graph_output["pair_indices"].cpu()
        if pair_indices is None:
            pair_indices = current_pairs
        elif not torch.equal(pair_indices, current_pairs):
            raise ValueError("Pair ordering changed while preparing dual cache")
        for batch_index in range(pixels.size(0)):
            valid = batch["anchor_mask"][batch_index]
            records.append(
                {
                    "key": batch["keys"][batch_index],
                    "candidate_hidden": candidate_hidden[batch_index].bfloat16().cpu(),
                    "base_logits": grounded["base_logits"][batch_index].half().cpu(),
                    "points": grounded["grounded_centers"][batch_index].float().cpu(),
                    "candidate_mask": grounded["candidate_mask"][batch_index].cpu(),
                    "candidate_roles": candidate_roles[batch_index].cpu(),
                    "candidate_auxiliary": candidate_auxiliary[batch_index].half().cpu(),
                    "candidate_visual": candidate_visual[batch_index].bfloat16().cpu(),
                    "edge_logits": edge_logits[batch_index].half().cpu(),
                    "edge_targets": edge_targets[batch_index].cpu(),
                    "aromatic_logits": grounded["aromatic_logits"][batch_index].half().cpu(),
                    "targets": targets[batch_index].cpu(),
                    "candidate_groups": candidate_groups[batch_index].cpu(),
                    "assignments": assignments[batch_index].cpu(),
                    "target_count": int(valid.sum()),
                    "target_role_counts": torch.bincount(
                        batch["role_targets"][batch_index][valid].long(),
                        minlength=ROLE_COUNT,
                    ),
                    "anchor_points": batch["anchor_points"][batch_index][valid].cpu(),
                    "anchor_roles": batch["role_targets"][batch_index][valid].cpu(),
                    "aromatic_targets": batch["aromatic_targets"][batch_index][valid].cpu(),
                }
            )
            if edge_auxiliary is not None:
                records[-1]["edge_auxiliary"] = edge_auxiliary[
                    batch_index
                ].half().cpu()
            if candidate_pixel_patches is not None:
                records[-1]["candidate_pixel_patches"] = (
                    candidate_pixel_patches[batch_index]
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                )
            if line_interior_mask is not None:
                records[-1]["line_interior_mask"] = line_interior_mask[
                    batch_index
                ].cpu()
            if len(records) >= samples:
                break
        if len(records) % 25 == 0:
            print(f"{cache_name}_cached={len(records)}", flush=True)
    if len(records) != samples or pair_indices is None:
        raise ValueError(f"Requested {samples} {cache_name} samples, found {len(records)}")
    return records, pair_indices


STACK_FIELDS = (
    "candidate_hidden",
    "base_logits",
    "points",
    "candidate_mask",
    "candidate_roles",
    "candidate_auxiliary",
    "candidate_visual",
    "edge_logits",
    "edge_targets",
    "aromatic_logits",
    "targets",
    "candidate_groups",
)


def stack_records(
    records: list[dict], indices: list[int], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = {
        name: torch.stack([records[index][name] for index in indices]).to(
            device, non_blocking=True
        )
        for name in STACK_FIELDS
    }
    if "edge_auxiliary" in records[indices[0]]:
        batch["edge_auxiliary"] = torch.stack(
            [records[index]["edge_auxiliary"] for index in indices]
        ).to(device, non_blocking=True)
    for name in ("candidate_pixel_patches", "line_interior_mask"):
        if name in records[indices[0]]:
            batch[name] = torch.stack(
                [records[index][name] for index in indices]
            ).to(device, non_blocking=True)
    return batch


def selector_forward(
    selector,
    batch: dict[str, torch.Tensor],
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    arguments = (
        batch["candidate_hidden"],
        batch["base_logits"],
        batch["points"],
        batch["candidate_mask"],
        batch["edge_logits"].detach(),
        pair_indices,
        batch["candidate_auxiliary"],
        batch["candidate_visual"],
        batch["candidate_roles"],
    )
    optional = {}
    if batch.get("edge_auxiliary") is not None:
        optional["edge_auxiliary"] = batch["edge_auxiliary"]
    if batch.get("candidate_pixel_patches") is not None:
        optional["candidate_pixel_patches"] = batch["candidate_pixel_patches"]
    return selector(*arguments, **optional)


@torch.no_grad()
def evaluate_cached(
    selector,
    records: list[dict],
    pair_indices: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    loss_kwargs: dict,
    loss_mode: str,
) -> dict:
    selector.eval()
    selected_totals = new_detection_totals()
    base_totals = new_detection_totals()
    oracle_totals = new_detection_totals()
    loss_totals = {
        name: 0.0
        for name in (
            "loss",
            "selection_loss",
            "whole_set_loss",
            "extreme_set_loss",
            "line_interior_ranking_loss",
        )
    }
    strict = {
        name: {"matches": 0, "predicted": 0, "exact": 0}
        for name in ("selected", "base_known_count", "candidate_oracle")
    }
    per_role_strict = {
        name: {"matches": 0, "predicted": 0, "exact": 0}
        for name in ("selected", "base_known_count", "candidate_oracle")
    }
    expected_total = 0
    spatial = {
        mode: {
            name: {"matches": 0, "predicted": 0, "exact": 0}
            for name in ("selected", "base_known_count", "candidate_oracle")
        }
        for mode in ("global_topk", "per_role_topk")
    }
    spatial["per_role_nms_topk"] = {
        name: {"matches": 0, "predicted": 0, "exact": 0}
        for name in ("selected", "base_known_count", "candidate_oracle")
    }
    threshold_values = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    threshold_totals = {
        f"{threshold:.2f}": {
            "matches": 0,
            "predicted": 0,
            "expected": 0,
            "count_exact": 0,
            "exact": 0,
        }
        for threshold in threshold_values
    }
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = stack_records(records, indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = selector_forward(selector, batch, pair_indices)
            losses = (
                grouped_candidate_selection_loss(
                    logits,
                    batch["candidate_groups"],
                    batch["candidate_mask"],
                    ranking_weight=loss_kwargs["group_ranking_weight"],
                    ranking_margin=loss_kwargs["group_ranking_margin"],
                    background_weight=loss_kwargs["group_background_weight"],
                    absolute_boundary_weight=loss_kwargs[
                        "absolute_boundary_weight"
                    ],
                    positive_floor_probability=loss_kwargs[
                        "positive_floor_probability"
                    ],
                    background_ceiling_probability=loss_kwargs[
                        "background_ceiling_probability"
                    ],
                    hard_background_count=loss_kwargs["hard_background_count"],
                    line_interior_mask=batch.get("line_interior_mask"),
                    line_interior_weight=loss_kwargs.get(
                        "line_interior_weight", 0.0
                    ),
                    line_interior_margin=loss_kwargs.get(
                        "line_interior_margin", 1.0
                    ),
                )
                if loss_mode == "target_group"
                else structured_selector_loss(
                    logits,
                    batch["targets"],
                    batch["candidate_mask"],
                    **{
                        name: loss_kwargs[name]
                        for name in (
                            "whole_set_weight",
                            "whole_set_margin",
                            "boundary_size",
                            "extreme_set_weight",
                            "extreme_set_margin",
                        )
                    },
                )
            )
        counts = torch.tensor(
            [records[index]["target_count"] for index in indices],
            device=device,
            dtype=torch.long,
        )
        decoded_by_name = {
            "selected": decode_grounded_queries(
                logits,
                batch["points"],
                batch["aromatic_logits"],
                counts=counts,
                candidate_mask=batch["candidate_mask"],
            ),
            "base": decode_grounded_queries(
                batch["base_logits"],
                batch["points"],
                batch["aromatic_logits"],
                counts=counts,
                candidate_mask=batch["candidate_mask"],
            ),
            "oracle": decode_grounded_queries(
                (batch["targets"] * 2.0 - 1.0) * 100.0,
                batch["points"],
                batch["aromatic_logits"],
                counts=counts,
                candidate_mask=batch["candidate_mask"],
            ),
        }
        for offset, count in enumerate(counts.tolist()):
            valid = batch["candidate_mask"][offset]
            available = int(valid.sum())
            selected_count = min(count, available)
            expected_total += count
            for name, scores in (
                ("selected", logits[offset]),
                ("base_known_count", batch["base_logits"][offset]),
            ):
                selected_indices = scores.masked_fill(~valid, -30.0).topk(
                    selected_count
                ).indices
                matches = int(batch["targets"][offset, selected_indices].sum())
                strict[name]["matches"] += matches
                strict[name]["predicted"] += selected_count
                strict[name]["exact"] += int(
                    selected_count == count and matches == count
                )
            global_indices = {
                "selected": logits[offset].masked_fill(~valid, -30.0).topk(
                    selected_count
                ).indices,
                "base_known_count": batch["base_logits"][offset]
                .masked_fill(~valid, -30.0)
                .topk(selected_count)
                .indices,
                "candidate_oracle": ((batch["targets"][offset] * 2.0 - 1.0) * 100.0)
                .masked_fill(~valid, -30.0)
                .topk(selected_count)
                .indices,
            }
            oracle_matches = int(batch["targets"][offset, valid].sum())
            strict["candidate_oracle"]["matches"] += oracle_matches
            strict["candidate_oracle"]["predicted"] += selected_count
            strict["candidate_oracle"]["exact"] += int(
                selected_count == count and oracle_matches == count
            )
            role_counts = records[indices[offset]]["target_role_counts"]
            per_role_matches = {
                "selected": 0,
                "base_known_count": 0,
                "candidate_oracle": 0,
            }
            per_role_predicted = 0
            role_capacity_exact = True
            per_role_indices = {
                "selected": [],
                "base_known_count": [],
                "candidate_oracle": [],
            }
            for role, role_count in enumerate(role_counts.tolist()):
                role_valid = valid & (batch["candidate_roles"][offset] == role)
                role_selected_count = min(int(role_count), int(role_valid.sum()))
                per_role_predicted += role_selected_count
                role_capacity_exact &= role_selected_count == int(role_count)
                for name, scores in (
                    ("selected", logits[offset]),
                    ("base_known_count", batch["base_logits"][offset]),
                ):
                    selected_indices = scores.masked_fill(~role_valid, -30.0).topk(
                        role_selected_count
                    ).indices
                    per_role_matches[name] += int(
                        batch["targets"][offset, selected_indices].sum()
                    )
                    per_role_indices[name].append(selected_indices)
                per_role_matches["candidate_oracle"] += int(
                    batch["targets"][offset, role_valid].sum()
                )
                oracle_indices = ((batch["targets"][offset] * 2.0 - 1.0) * 100.0)
                oracle_indices = oracle_indices.masked_fill(~role_valid, -30.0).topk(
                    role_selected_count
                ).indices
                per_role_indices["candidate_oracle"].append(oracle_indices)
            for name, matches in per_role_matches.items():
                per_role_strict[name]["matches"] += matches
                per_role_strict[name]["predicted"] += per_role_predicted
                per_role_strict[name]["exact"] += int(
                    role_capacity_exact and matches == count
                )
            record = records[indices[offset]]
            target_points = record["anchor_points"].to(device)
            target_roles = record["anchor_roles"].to(device)
            for mode, selections in (
                ("global_topk", global_indices),
                (
                    "per_role_topk",
                    {
                        name: (
                            torch.cat(parts)
                            if parts
                            else torch.empty(0, device=device, dtype=torch.long)
                        )
                        for name, parts in per_role_indices.items()
                    },
                ),
            ):
                for name, selected_indices in selections.items():
                    matches = role_aware_match_count(
                        batch["points"][offset, selected_indices],
                        batch["candidate_roles"][offset, selected_indices],
                        target_points,
                        target_roles,
                        matching_radius_pixels=6.0,
                        image_size=768,
                    )
                    spatial[mode][name]["matches"] += matches
                    spatial[mode][name]["predicted"] += len(selected_indices)
                    spatial[mode][name]["exact"] += int(
                        len(selected_indices) == count and matches == count
                    )
            nms_selections = {
                name: role_suppressed_topk_indices(
                    scores,
                    batch["points"][offset],
                    valid,
                    batch["candidate_roles"][offset],
                    role_counts.to(device),
                    suppression_radius_pixels=6.0,
                    image_size=768,
                )
                for name, scores in (
                    ("selected", logits[offset]),
                    ("base_known_count", batch["base_logits"][offset]),
                    (
                        "candidate_oracle",
                        (batch["targets"][offset] * 2.0 - 1.0) * 100.0,
                    ),
                )
            }
            for name, selected_indices in nms_selections.items():
                matches = role_aware_match_count(
                    batch["points"][offset, selected_indices],
                    batch["candidate_roles"][offset, selected_indices],
                    target_points,
                    target_roles,
                    matching_radius_pixels=6.0,
                    image_size=768,
                )
                values = spatial["per_role_nms_topk"][name]
                values["matches"] += matches
                values["predicted"] += len(selected_indices)
                values["exact"] += int(
                    len(selected_indices) == count and matches == count
                )
            for threshold in threshold_values:
                selected_indices = role_suppressed_threshold_indices(
                    logits[offset],
                    batch["points"][offset],
                    valid,
                    batch["candidate_roles"][offset],
                    probability_threshold=threshold,
                    suppression_radius_pixels=6.0,
                    image_size=768,
                )
                matches = role_aware_match_count(
                    batch["points"][offset, selected_indices],
                    batch["candidate_roles"][offset, selected_indices],
                    target_points,
                    target_roles,
                    matching_radius_pixels=6.0,
                    image_size=768,
                )
                values = threshold_totals[f"{threshold:.2f}"]
                values["matches"] += matches
                values["predicted"] += len(selected_indices)
                values["expected"] += count
                values["count_exact"] += int(len(selected_indices) == count)
                values["exact"] += int(
                    len(selected_indices) == count and matches == count
                )
        for name in loss_totals:
            loss_totals[name] += float(losses.get(name, logits.sum() * 0.0)) * len(
                indices
            )
        for offset, record_index in enumerate(indices):
            record = records[record_index]
            anchors = record["anchor_points"].unsqueeze(0)
            mask = torch.ones((1, anchors.size(1)), dtype=torch.bool)
            aromatic = record["aromatic_targets"].unsqueeze(0)
            for totals, decoded in (
                (selected_totals, decoded_by_name["selected"][offset]),
                (base_totals, decoded_by_name["base"][offset]),
                (oracle_totals, decoded_by_name["oracle"][offset]),
            ):
                update_detection_metrics(
                    totals,
                    [decoded],
                    anchors,
                    mask,
                    aromatic,
                    768,
                    6.0,
                )
    role_aware = {}
    for mode, mode_totals in (
        ("global_topk", strict),
        ("per_role_topk", per_role_strict),
    ):
        role_aware[mode] = {}
        for name, values in mode_totals.items():
            precision = values["matches"] / max(values["predicted"], 1)
            recall = values["matches"] / max(expected_total, 1)
            role_aware[mode][name] = {
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "exact_rate": values["exact"] / max(len(records), 1),
            }
    role_spatial = {}
    for mode, mode_totals in spatial.items():
        role_spatial[mode] = {}
        for name, values in mode_totals.items():
            precision = values["matches"] / max(values["predicted"], 1)
            recall = values["matches"] / max(expected_total, 1)
            role_spatial[mode][name] = {
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "exact_rate": values["exact"] / max(len(records), 1),
            }
    threshold_metrics = {}
    for threshold, values in threshold_totals.items():
        precision = values["matches"] / max(values["predicted"], 1)
        recall = values["matches"] / max(values["expected"], 1)
        threshold_metrics[threshold] = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "count_exact_rate": values["count_exact"] / max(len(records), 1),
            "exact_rate": values["exact"] / max(len(records), 1),
        }
    return {
        "samples": len(records),
        "role_aware": role_aware,
        "role_spatial": role_spatial,
        "real_thresholds": threshold_metrics,
        "coordinate_only": {
            "selected": summarize_detection_metrics(selected_totals),
            "base_known_count": summarize_detection_metrics(base_totals),
            "candidate_oracle": summarize_detection_metrics(oracle_totals),
        },
        "matched_candidate_positives": strict["candidate_oracle"]["matches"],
        "losses": {
            name: value / max(len(records), 1)
            for name, value in loss_totals.items()
        },
    }


def validation_key(result: dict) -> tuple[float, float, float]:
    selected = result["real_thresholds"]["0.70"]
    return (
        selected["exact_rate"],
        selected["count_exact_rate"],
        selected["f1"],
    )


def save_checkpoint(
    path: Path,
    selector,
    optimizer,
    config,
    args,
    *,
    samples_seen: int,
    validation: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "dual_vertex_selector": selector.state_dict(),
            "structured_selector_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "text_object_checkpoint": args.text_object_checkpoint,
            "implicit_vertex_checkpoint": args.implicit_vertex_checkpoint,
            "samples_seen": samples_seen,
            "validation": validation,
            "loss_mode": args.loss_mode,
            "group_ranking_weight": args.group_ranking_weight,
            "group_ranking_margin": args.group_ranking_margin,
            "group_background_weight": args.group_background_weight,
            "candidate_protocol": {
                "text_threshold": args.text_candidate_threshold,
                "implicit_threshold": args.implicit_candidate_threshold,
                "text_filter_threshold": args.text_filter_threshold,
                "matching_radius_pixels": args.matching_radius_pixels,
                "suppression_radius_pixels": 6.0,
            },
            "training_kind": "dual_branch_role_aware_whole_set_selector",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-object-checkpoint", required=True)
    parser.add_argument("--implicit-vertex-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--maximum-candidates", type=int, default=256)
    parser.add_argument("--text-candidate-threshold", type=float, default=0.05)
    parser.add_argument("--text-filter-threshold", type=float, default=0.3)
    parser.add_argument("--implicit-candidate-threshold", type=float, default=0.05)
    parser.add_argument("--bbox-expansion-pixels", type=float, default=8.0)
    parser.add_argument("--matching-radius-pixels", type=float, default=6.0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--incident-topk", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=12.0)
    parser.add_argument("--whole-set-weight", type=float, default=10.0)
    parser.add_argument("--whole-set-margin", type=float, default=1.0)
    parser.add_argument("--boundary-size", type=int, default=16)
    parser.add_argument("--extreme-set-weight", type=float, default=2.0)
    parser.add_argument("--extreme-set-margin", type=float, default=1.0)
    parser.add_argument(
        "--loss-mode",
        choices=("single_assignment", "target_group"),
        default="target_group",
    )
    parser.add_argument("--group-ranking-weight", type=float, default=4.0)
    parser.add_argument("--group-ranking-margin", type=float, default=1.0)
    parser.add_argument("--group-background-weight", type=float, default=1.0)
    parser.add_argument("--absolute-boundary-weight", type=float, default=0.0)
    parser.add_argument("--positive-floor-probability", type=float, default=0.8)
    parser.add_argument("--background-ceiling-probability", type=float, default=0.2)
    parser.add_argument("--hard-background-count", type=int, default=16)
    parser.add_argument("--train-final-score-only", action="store_true")
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dual vertex selector training")
    if min(
        args.train_cache_samples,
        args.validation_samples,
        args.train_samples,
        args.epochs,
        args.batch_size,
        args.cache_batch_size,
        args.maximum_candidates,
    ) < 1:
        raise ValueError("Training counts and capacities must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    base, detector, grounded_selector, graph, text_head, implicit_head = (
        load_dual_pipeline(
            args.text_object_checkpoint,
            args.implicit_vertex_checkpoint,
            device,
            maximum_candidates=args.maximum_candidates,
        )
    )
    config = StructuredSelectorConfig(
        candidate_dim=grounded_selector.config.d_model,
        auxiliary_dim=32,
        visual_dim=base.config.vision_dims[0] * 9,
        d_model=args.d_model,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        layers=args.layers,
        maximum_candidates=args.maximum_candidates,
        incident_topk=args.incident_topk,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        role_count=ROLE_COUNT,
    )
    selector = GroundedStructuredSelector(config).to(device)
    if args.init_checkpoint:
        initial_payload = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        initial_config = StructuredSelectorConfig(
            **initial_payload["structured_selector_config"]
        )
        if initial_config != config:
            raise ValueError("Initial dual selector config does not match")
        selector.load_state_dict(initial_payload["dual_vertex_selector"])
    if args.train_final_score_only:
        if not args.init_checkpoint:
            raise ValueError("Final-score-only training requires an initial checkpoint")
        selector.requires_grad_(False)
        selector.residual_score.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in selector.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def loader(shards):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                sequence_version="0.4",
            ),
            batch_size=args.cache_batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=ProposalGraphCollator(),
        )

    cache_kwargs = {
        "text_candidate_threshold": args.text_candidate_threshold,
        "text_filter_threshold": args.text_filter_threshold,
        "implicit_candidate_threshold": args.implicit_candidate_threshold,
        "bbox_expansion_pixels": args.bbox_expansion_pixels,
        "matching_radius_pixels": args.matching_radius_pixels,
    }
    train_cache, pair_indices = prepare_cache(
        base,
        detector,
        grounded_selector,
        graph,
        text_head,
        implicit_head,
        loader(args.train_shards),
        device,
        samples=args.train_cache_samples,
        cache_name="train",
        **cache_kwargs,
    )
    if (
        expand_shards(args.train_shards) == expand_shards(args.validation_shards)
        and args.train_cache_samples == args.validation_samples
    ):
        validation_cache = train_cache
    else:
        validation_cache, validation_pairs = prepare_cache(
            base,
            detector,
            grounded_selector,
            graph,
            text_head,
            implicit_head,
            loader(args.validation_shards),
            device,
            samples=args.validation_samples,
            cache_name="validation",
            **cache_kwargs,
        )
        if not torch.equal(pair_indices, validation_pairs):
            raise ValueError("Train and validation pair ordering differs")
    pair_indices = pair_indices.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    loss_kwargs = {
        "whole_set_weight": args.whole_set_weight,
        "whole_set_margin": args.whole_set_margin,
        "boundary_size": args.boundary_size,
        "extreme_set_weight": args.extreme_set_weight,
        "extreme_set_margin": args.extreme_set_margin,
        "group_ranking_weight": args.group_ranking_weight,
        "group_ranking_margin": args.group_ranking_margin,
        "group_background_weight": args.group_background_weight,
        "absolute_boundary_weight": args.absolute_boundary_weight,
        "positive_floor_probability": args.positive_floor_probability,
        "background_ceiling_probability": args.background_ceiling_probability,
        "hard_background_count": args.hard_background_count,
    }
    initial = evaluate_cached(
        selector,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
        loss_kwargs=loss_kwargs,
        loss_mode=args.loss_mode,
    )
    print("initial=" + json.dumps(initial), flush=True)
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"trainable_parameters={sum(parameter.numel() for parameter in selector.parameters()):,}",
        flush=True,
    )
    if args.eval_only:
        (output_dir / "validation.json").write_text(
            json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best_score = (-1.0, -1.0, -1.0)
    running = {
        "loss": 0.0,
        "selection_loss": 0.0,
        "whole_set_loss": 0.0,
        "extreme_set_loss": 0.0,
    }
    running_samples = 0
    started = time.perf_counter()
    stop = False
    for _ in range(args.epochs):
        order = list(range(len(train_cache)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            indices = order[start : start + min(args.batch_size, remaining)]
            batch = stack_records(train_cache, indices, device)
            selector.train()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = selector_forward(selector, batch, pair_indices)
                losses = (
                    grouped_candidate_selection_loss(
                        logits,
                        batch["candidate_groups"],
                        batch["candidate_mask"],
                        ranking_weight=args.group_ranking_weight,
                        ranking_margin=args.group_ranking_margin,
                        background_weight=args.group_background_weight,
                        absolute_boundary_weight=args.absolute_boundary_weight,
                        positive_floor_probability=args.positive_floor_probability,
                        background_ceiling_probability=args.background_ceiling_probability,
                        hard_background_count=args.hard_background_count,
                    )
                    if args.loss_mode == "target_group"
                    else structured_selector_loss(
                        logits,
                        batch["targets"],
                        batch["candidate_mask"],
                        whole_set_weight=args.whole_set_weight,
                        whole_set_margin=args.whole_set_margin,
                        boundary_size=args.boundary_size,
                        extreme_set_weight=args.extreme_set_weight,
                        extreme_set_margin=args.extreme_set_margin,
                    )
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_samples += len(indices)
            for name in running:
                running[name] += float(losses[name].detach()) * len(indices)
            if samples_seen // args.log_every != (
                samples_seen - len(indices)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} "
                    + " ".join(
                        f"{name}={value / max(running_samples, 1):.6f}"
                        for name, value in running.items()
                    )
                    + f" samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                running_samples = 0
            if args.validate_every > 0 and samples_seen >= next_validation:
                validation = evaluate_cached(
                    selector,
                    validation_cache,
                    pair_indices,
                    device,
                    batch_size=args.batch_size,
                    loss_kwargs=loss_kwargs,
                    loss_mode=args.loss_mode,
                )
                print("validation=" + json.dumps(validation), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save_checkpoint(
                    checkpoint,
                    selector,
                    optimizer,
                    config,
                    args,
                    samples_seen=samples_seen,
                    validation=validation,
                )
                if validation_key(validation) > best_score:
                    best_score = validation_key(validation)
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break

    validation = evaluate_cached(
        selector,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
        loss_kwargs=loss_kwargs,
        loss_mode=args.loss_mode,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        output_dir / "last.pt",
        selector,
        optimizer,
        config,
        args,
        samples_seen=samples_seen,
        validation=validation,
    )
    print("validation=" + json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
