"""Select structural Junction batches before OCR label decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chemvector.dual_vertex_selector import (
    dual_candidate_auxiliary,
    merge_dual_candidates,
    padded_candidate_roles,
    role_suppressed_role_threshold_indices,
)
from chemvector.endpoint_evidence import endpoint_ray_features
from chemvector.evaluate_group_selector_bond import (
    outside_text_objects,
    resolved_real_thresholds,
)
from chemvector.grounded_selector import (
    fuse_dense_candidates_with_queries,
    sample_candidate_features,
    sample_candidate_ink_patches,
)
from chemvector.pretrain_dual_vertex_selector import selector_forward
from chemvector.text_object import decode_text_objects
from chemvector.vertex import decode_vertices


@dataclass(frozen=True, slots=True)
class JunctionBatchSelection:
    text_batches: list[list[dict]]
    point_batches: list[torch.Tensor]
    role_batches: list[torch.Tensor]
    padded_points: torch.Tensor
    point_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class JunctionCandidatePreparation:
    text_batches: list[list[dict]]
    grounded: dict[str, torch.Tensor]
    candidate_roles: torch.Tensor
    candidate_auxiliary: torch.Tensor
    candidate_visual: torch.Tensor
    candidate_pixel_patches: torch.Tensor | None


def prepare_junction_candidates(
    *,
    pixels: torch.Tensor,
    features: torch.Tensor,
    query_output: dict[str, torch.Tensor],
    text_output: Any,
    implicit_output: Any,
    protocol: dict,
    selector_config: Any,
    scoring_config: Any,
    bbox_expansion_pixels: float,
) -> JunctionCandidatePreparation:
    """Decode, merge, ground, and featurize structural Junction candidates."""

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
    source_role_batches = []
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
            maximum_candidates=selector_config.maximum_candidates,
        )
        merged_batches.append(merged)
        source_role_batches.append(roles)
    grounded = fuse_dense_candidates_with_queries(
        query_output,
        merged_batches,
        maximum_candidates=selector_config.maximum_candidates,
        image_size=pixels.size(-1),
    )
    candidate_roles = padded_candidate_roles(
        source_role_batches, grounded["candidate_mask"]
    )
    candidate_auxiliary = dual_candidate_auxiliary(
        grounded["candidate_auxiliary"],
        merged_batches,
        grounded["candidate_mask"],
    )
    candidate_visual = sample_candidate_features(
        features, grounded["grounded_centers"], grounded["candidate_mask"]
    )
    candidate_pixel_patches = None
    if getattr(scoring_config, "patch_mode", "none") in {
        "pixel",
        "semantic_pixel",
    }:
        candidate_pixel_patches = sample_candidate_ink_patches(
            pixels,
            grounded["grounded_centers"],
            grounded["candidate_mask"],
            patch_size=scoring_config.patch_size,
        )
    return JunctionCandidatePreparation(
        text_batches=text_batches,
        grounded=grounded,
        candidate_roles=candidate_roles,
        candidate_auxiliary=candidate_auxiliary,
        candidate_visual=candidate_visual,
        candidate_pixel_patches=candidate_pixel_patches,
    )


def finalize_junction_batches(
    *,
    text_batches: list[list[dict]],
    logits: torch.Tensor,
    grounded_centers: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_roles: torch.Tensor,
    batch_size: int,
    real_threshold: float,
    suppression_radius_pixels: float,
    image_size: int,
    device: torch.device,
) -> JunctionBatchSelection:
    """Threshold, suppress, and pad selected Junctions deterministically."""

    thresholds = resolved_real_thresholds(real_threshold, None, None)
    point_batches = []
    selected_role_batches = []
    for batch_index in range(batch_size):
        selected = role_suppressed_role_threshold_indices(
            logits[batch_index],
            grounded_centers[batch_index],
            candidate_mask[batch_index],
            candidate_roles[batch_index],
            probability_thresholds=thresholds,
            suppression_radius_pixels=suppression_radius_pixels,
            image_size=image_size,
        )
        points = grounded_centers[batch_index, selected]
        if not len(points):
            raise ValueError(
                f"Junction selector returned an empty set for batch item {batch_index}"
            )
        point_batches.append(points)
        selected_role_batches.append(candidate_roles[batch_index, selected])
    maximum_points = max(len(points) for points in point_batches)
    point_batch = torch.zeros(
        (batch_size, maximum_points, 2),
        device=device,
        dtype=point_batches[0].dtype,
    )
    point_mask = torch.zeros(
        (batch_size, maximum_points), device=device, dtype=torch.bool
    )
    for batch_index, points in enumerate(point_batches):
        point_batch[batch_index, : len(points)] = points
        point_mask[batch_index, : len(points)] = True
    return JunctionBatchSelection(
        text_batches=text_batches,
        point_batches=point_batches,
        role_batches=selected_role_batches,
        padded_points=point_batch,
        point_mask=point_mask,
    )


def select_junction_batches(
    *,
    pixels: torch.Tensor,
    features: torch.Tensor,
    device: torch.device,
    query_detector: Any,
    text_head: Any,
    implicit_head: Any,
    protocol: dict,
    selector_config: Any,
    selector: Any,
    grounded_selector: Any,
    graph: Any,
    real_threshold: float,
    bbox_expansion_pixels: float,
) -> JunctionBatchSelection:
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        query_output = query_detector(features)
        text_output = text_head(features)
        implicit_output = implicit_head(features)
    prepared = prepare_junction_candidates(
        pixels=pixels,
        features=features,
        query_output=query_output,
        text_output=text_output,
        implicit_output=implicit_output,
        protocol=protocol,
        selector_config=selector_config,
        scoring_config=selector.config,
        bbox_expansion_pixels=bbox_expansion_pixels,
    )
    grounded = prepared.grounded
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        hidden = grounded_selector.encode(
            grounded["query_features"],
            grounded["scalar_features"],
            grounded["grounded_centers"],
            grounded["candidate_mask"],
        )
        graph_output = graph(
            hidden,
            pixels,
            grounded["grounded_centers"],
            grounded["candidate_mask"],
            grounded["base_logits"],
        )
        edge_auxiliary = None
        if getattr(selector.config, "edge_auxiliary_dim", 0):
            edge_auxiliary = endpoint_ray_features(
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                graph_output["pair_indices"],
            )
        logits = selector_forward(
            selector,
            {
                "candidate_hidden": hidden,
                "base_logits": grounded["base_logits"],
                "points": grounded["grounded_centers"],
                "candidate_mask": grounded["candidate_mask"],
                "edge_logits": graph_output["edge_logits"],
                "candidate_auxiliary": prepared.candidate_auxiliary,
                "candidate_visual": prepared.candidate_visual,
                "candidate_roles": prepared.candidate_roles,
                "edge_auxiliary": edge_auxiliary,
                "candidate_pixel_patches": prepared.candidate_pixel_patches,
            },
            graph_output["pair_indices"],
        )
    return finalize_junction_batches(
        text_batches=prepared.text_batches,
        logits=logits,
        grounded_centers=grounded["grounded_centers"],
        candidate_mask=grounded["candidate_mask"],
        candidate_roles=prepared.candidate_roles,
        batch_size=len(pixels),
        real_threshold=real_threshold,
        suppression_radius_pixels=float(protocol["suppression_radius_pixels"]),
        image_size=pixels.size(-1),
        device=device,
    )
