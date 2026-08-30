"""Decode and route OCR labels for an already selected Junction set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chemvector.ctc_ocr import (
    decode_ctc_with_score,
    detected_text_crops,
)
from chemvector.dual_vertex_selector import TEXT_ROLE
from chemvector.hybrid_junction_selection import JunctionBatchSelection
from chemvector.hybrid_ocr_routing import route_ocr_predictions


@dataclass(frozen=True, slots=True)
class HybridOCRRuntime:
    """The explicit subset of runtime state needed by Junction OCR."""

    device: torch.device
    base: Any
    codec: Any
    primary_ctc: Any
    expert_ctc: Any
    utility_router: Any
    utility_feature_names: Any
    utility_feature_mean: Any
    utility_feature_std: Any
    utility_threshold: float
    anchor_fallback_margin: float | None


def decode_ctc_crops(
    model: Any,
    shifted_crops: torch.Tensor,
    anchor_crops: torch.Tensor | None,
    *,
    device: torch.device,
    anchor_fallback_margin: float | None,
    include_visual: bool = False,
) -> tuple[list[tuple[str, float]], torch.Tensor | None, torch.Tensor]:
    """Decode shifted crops, optionally falling back to anchor-centred crops."""

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        if include_visual:
            shifted_logits, shifted_visual = model.forward_with_visual_descriptor(
                shifted_crops
            )
        else:
            shifted_logits = model(shifted_crops)
            shifted_visual = None
    shifted = [decode_ctc_with_score(row.float()) for row in shifted_logits]
    if anchor_crops is None:
        return shifted, shifted_visual, shifted_logits
    if anchor_fallback_margin is None:
        raise ValueError("Anchor OCR crops require a fallback margin")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        if include_visual:
            anchor_logits, anchor_visual = model.forward_with_visual_descriptor(
                anchor_crops
            )
        else:
            anchor_logits = model(anchor_crops)
            anchor_visual = None
    anchor = [decode_ctc_with_score(row.float()) for row in anchor_logits]
    use_shifted = [
        shifted_item[1] >= anchor_item[1] + float(anchor_fallback_margin)
        for shifted_item, anchor_item in zip(shifted, anchor)
    ]
    selected = [
        shifted_item if choose_shifted else anchor_item
        for shifted_item, anchor_item, choose_shifted in zip(
            shifted, anchor, use_shifted
        )
    ]
    selected_logits = torch.stack(
        [
            shifted_logits[index] if choose_shifted else anchor_logits[index]
            for index, choose_shifted in enumerate(use_shifted)
        ]
    )
    if not include_visual:
        return selected, None, selected_logits
    selected_visual = torch.stack(
        [
            shifted_visual[index] if choose_shifted else anchor_visual[index]
            for index, choose_shifted in enumerate(use_shifted)
        ]
    )
    return selected, selected_visual, selected_logits


def decode_junction_labels(
    *,
    selection: JunctionBatchSelection,
    pixels: torch.Tensor,
    features: torch.Tensor,
    include_ocr: bool,
    runtime: HybridOCRRuntime,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, list[str]]], dict]:
    """Attach OCR labels without changing the selected structural Junction set."""

    if include_ocr and runtime.primary_ctc is not None:
        label_batches = []
        utility_disagreements = 0
        utility_routes = 0
        include_visual = bool(
            runtime.utility_feature_names
            and any(
                name.startswith("old_visual_")
                for name in runtime.utility_feature_names
            )
        )
        for batch_index, (points, selected_roles) in enumerate(
            zip(selection.point_batches, selection.role_batches)
        ):
            labels = ["C"] * len(points)
            crops, label_indices = detected_text_crops(
                pixels[batch_index],
                points,
                selected_roles,
                selection.text_batches[batch_index],
                text_role=TEXT_ROLE,
            )
            if len(crops):
                anchor_crops = None
                if runtime.anchor_fallback_margin is not None:
                    anchor_objects = [
                        {
                            key: value
                            for key, value in item.items()
                            if key not in ("bbox_x", "bbox_y")
                        }
                        for item in selection.text_batches[batch_index]
                    ]
                    anchor_crops, anchor_indices = detected_text_crops(
                        pixels[batch_index],
                        points,
                        selected_roles,
                        anchor_objects,
                        text_role=TEXT_ROLE,
                    )
                    if anchor_indices != label_indices:
                        raise ValueError(
                            "Anchor and shifted OCR crop assignments differ"
                        )
                predictions, primary_visual, primary_logits = decode_ctc_crops(
                    runtime.primary_ctc,
                    crops,
                    anchor_crops,
                    device=runtime.device,
                    anchor_fallback_margin=runtime.anchor_fallback_margin,
                    include_visual=include_visual,
                )
                if runtime.expert_ctc is not None:
                    expert_predictions, expert_visual, expert_logits = (
                        decode_ctc_crops(
                            runtime.expert_ctc,
                            crops,
                            anchor_crops,
                            device=runtime.device,
                            anchor_fallback_margin=runtime.anchor_fallback_margin,
                            include_visual=include_visual,
                        )
                    )
                    predictions, disagreements, routes = route_ocr_predictions(
                        predictions,
                        expert_predictions,
                        runtime=runtime,
                        primary_visual=primary_visual,
                        expert_visual=expert_visual,
                        primary_logits=primary_logits,
                        expert_logits=expert_logits,
                    )
                    utility_disagreements += disagreements
                    utility_routes += routes
                for point_index, (label, _) in zip(label_indices, predictions):
                    labels[point_index] = label
            label_batches.append(labels)
        diagnostics = {
            "ocr_utility_router_enabled": runtime.utility_router is not None,
            "ocr_utility_disagreements": utility_disagreements,
            "ocr_utility_routes": utility_routes,
            "ocr_utility_threshold": (
                runtime.utility_threshold
                if runtime.utility_router is not None
                else None
            ),
        }
        flat_labels = [label for labels in label_batches for label in labels]
    elif include_ocr:
        ocr_features = runtime.base.ocr_source(pixels, features)
        with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16):
            token_ids = runtime.base.ocr_head.generate(
                ocr_features,
                selection.padded_points,
                selection.point_mask,
                pixels.shape[-2:],
                roi_radius=20.0,
            )
        flat_labels = [runtime.codec.decode(row.tolist()) for row in token_ids.cpu()]
        diagnostics = {}
    else:
        flat_labels = ["C"] * int(selection.point_mask.sum())
        diagnostics = {}

    results = []
    offset = 0
    for points, selected_roles in zip(
        selection.point_batches, selection.role_batches
    ):
        labels = flat_labels[offset : offset + len(points)]
        offset += len(points)
        labels = [
            label if int(role) == TEXT_ROLE else "C"
            for label, role in zip(labels, selected_roles)
        ]
        results.append((points, selected_roles, labels))
    if offset != len(flat_labels):
        raise ValueError("OCR output count differs from selected Junction count")
    return results, diagnostics
