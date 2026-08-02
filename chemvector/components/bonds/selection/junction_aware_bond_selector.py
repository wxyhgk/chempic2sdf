"""Deployable Bond selection followed by visual endpoint grouping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from chemvector.bond_endpoint_junction import (
    BondEndpointJunctionResult,
    junctions_from_endpoint_groups,
    project_unique_junction_pairs,
)
from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.bond_set_selector import BondSetSelector, BondSetSelectorConfig
from chemvector.endpoint_junction_grouper import (
    VisualEndpointJunctionGrouper,
    VisualEndpointJunctionGrouperConfig,
    decode_endpoint_groups,
    endpoint_pair_features,
    sample_aligned_endpoint_ink_patches,
)


@dataclass(frozen=True, slots=True)
class JunctionAwareSelectionDiagnostics:
    raw_candidates: int
    threshold_selected: int
    grouped_bonds: int
    predicted_junctions: int
    keep_threshold: float
    projection_score_source: str
    bond_type_source: str
    self_loop_objects: int
    duplicate_pair_objects: int


def load_bond_set_selector(
    checkpoint_path,
    device: torch.device,
) -> tuple[BondSetSelector, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BondSetSelector(BondSetSelectorConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["bond_set_selector"])
    model.eval().requires_grad_(False)
    return model, payload


def load_endpoint_grouper(
    checkpoint_path,
    device: torch.device,
) -> tuple[VisualEndpointJunctionGrouper, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VisualEndpointJunctionGrouper(
        VisualEndpointJunctionGrouperConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["visual_endpoint_junction_grouper"])
    model.eval().requires_grad_(False)
    return model, payload


def deployable_candidate_features(
    objects: list[dict],
    line_band_features: torch.Tensor,
) -> torch.Tensor:
    if not objects:
        return torch.empty((0, 0))
    if line_band_features.shape[0] != len(objects):
        raise ValueError("Line-band features differ from Bond candidates")
    endpoints = torch.stack([item["endpoints"].detach().float().cpu() for item in objects])
    scores = torch.tensor([float(item["score"]) for item in objects])
    display_ids = torch.tensor([int(item["display"]) for item in objects])
    chemical_ids = torch.tensor([int(item["chemical"]) for item in objects])
    direction_ids = torch.tensor([int(item["direction"]) for item in objects])
    object_features = torch.stack(
        [item["feature"].detach().float().cpu() for item in objects]
    )
    center = endpoints.mean(1)
    delta = endpoints[:, 1] - endpoints[:, 0]
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return torch.cat(
        (
            torch.logit(scores.clamp(1e-5, 1.0 - 1e-5))[:, None],
            endpoints.flatten(1),
            center,
            delta.abs(),
            length,
            F.one_hot(display_ids, num_classes=len(BOND_TYPES)).float(),
            F.one_hot(chemical_ids, num_classes=len(BOND_TYPES)).float(),
            F.one_hot(direction_ids, num_classes=len(DIRECTIONS)).float(),
            object_features,
            line_band_features.detach().float().cpu(),
        ),
        dim=-1,
    )


@torch.inference_mode()
def endpoint_visual_context(
    pixels: torch.Tensor,
    objects: list[dict],
    grouper: VisualEndpointJunctionGrouper,
    grouper_payload: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pixels.ndim != 4 or pixels.size(0) != 1:
        raise ValueError("Junction-aware selection expects one image")
    endpoints = torch.stack([item["endpoints"].detach().float() for item in objects])
    patches = sample_aligned_endpoint_ink_patches(
        pixels,
        endpoints,
        patch_size=int(grouper_payload["patch_size"]),
    )
    with torch.autocast(device_type=pixels.device.type, dtype=torch.bfloat16):
        embeddings = grouper.encode_patches(patches).float()
    pairs, geometry = endpoint_pair_features(objects)
    geometry = geometry.to(pixels.device)
    with torch.autocast(device_type=pixels.device.type, dtype=torch.bfloat16):
        logits = grouper.score_embeddings(
            geometry,
            embeddings[pairs[:, 0]],
            embeddings[pairs[:, 1]],
        ).float()
    probabilities = logits.sigmoid()
    endpoint_count = len(objects) * 2
    maximum_support = torch.zeros(endpoint_count, device=pixels.device)
    close_count = torch.zeros(endpoint_count, device=pixels.device)
    minimum_distance = torch.full((endpoint_count,), 8.0, device=pixels.device)
    for side in (0, 1):
        indices = pairs[:, side].to(pixels.device)
        maximum_support.scatter_reduce_(
            0, indices, probabilities, reduce="amax", include_self=True
        )
        close_count.scatter_add_(
            0,
            indices,
            (probabilities >= float(grouper_payload["threshold"])).float(),
        )
        minimum_distance.scatter_reduce_(
            0,
            indices,
            geometry[:, 4],
            reduce="amin",
            include_self=True,
        )
    local = torch.stack(
        (
            maximum_support,
            close_count.clamp_max(8.0).div(8.0),
            minimum_distance.clamp_max(8.0).div(8.0),
        ),
        dim=-1,
    )
    return embeddings, local


@torch.inference_mode()
def select_and_group_bonds(
    pixels: torch.Tensor,
    objects: list[dict],
    line_band_features: torch.Tensor,
    selector: BondSetSelector,
    selector_payload: dict,
    grouper: VisualEndpointJunctionGrouper,
    grouper_payload: dict,
    *,
    projection_score_source: str = "detector",
    bond_type_source: str = "detector",
) -> tuple[BondEndpointJunctionResult, JunctionAwareSelectionDiagnostics]:
    if not objects:
        raise ValueError("Bond detector returned no candidates")
    if projection_score_source not in {"detector", "selector"}:
        raise ValueError(
            f"Unknown Junction projection score source: {projection_score_source}"
        )
    if bond_type_source not in {"detector", "line_band"}:
        raise ValueError(f"Unknown Bond type source: {bond_type_source}")
    embeddings, local = endpoint_visual_context(
        pixels, objects, grouper, grouper_payload
    )
    endpoint_features = torch.cat((embeddings, local), dim=-1).reshape(len(objects), -1)
    features = torch.cat(
        (
            deployable_candidate_features(objects, line_band_features).to(pixels.device),
            endpoint_features,
        ),
        dim=-1,
    )
    if features.size(-1) != selector.config.input_dim:
        raise ValueError(
            f"Live selector feature width {features.size(-1)} differs from "
            f"checkpoint width {selector.config.input_dim}"
        )
    mask = torch.ones((1, len(objects)), device=pixels.device, dtype=torch.bool)
    display = torch.tensor(
        [[int(item["display"]) for item in objects]],
        device=pixels.device,
    )
    chemical = torch.tensor(
        [[int(item["chemical"]) for item in objects]],
        device=pixels.device,
    )
    with torch.autocast(device_type=pixels.device.type, dtype=torch.bfloat16):
        output = selector(features[None], mask, display, chemical)
    keep_threshold = float(selector_payload["keep_threshold"])
    probabilities = output["keep_logits"][0].sigmoid()
    selected = [
        index
        for index in probabilities.argsort(descending=True).tolist()
        if float(probabilities[index]) >= keep_threshold
    ]
    if not selected:
        raise ValueError("Bond selector rejected every candidate")
    selected_objects = [dict(objects[index]) for index in selected]
    if bond_type_source == "line_band":
        display_offset = 1
        chemical_offset = display_offset + len(BOND_TYPES)
        for item, source_index in zip(selected_objects, selected):
            features = line_band_features[source_index]
            item["display"] = int(
                features[display_offset : display_offset + len(BOND_TYPES)].argmax()
            )
            item["chemical"] = int(
                features[
                    chemical_offset : chemical_offset + len(BOND_TYPES)
                ].argmax()
            )
    source_endpoint_indices = torch.tensor(
        [endpoint for index in selected for endpoint in (index * 2, index * 2 + 1)],
        device=pixels.device,
        dtype=torch.long,
    )
    selected_embeddings = embeddings[source_endpoint_indices]
    pairs, geometry = endpoint_pair_features(selected_objects)
    with torch.autocast(device_type=pixels.device.type, dtype=torch.bfloat16):
        group_logits = grouper.score_embeddings(
            geometry.to(pixels.device),
            selected_embeddings[pairs[:, 0]],
            selected_embeddings[pairs[:, 1]],
        ).float().cpu()
    groups = decode_endpoint_groups(
        group_logits,
        pairs,
        len(selected_objects) * 2,
        threshold=float(grouper_payload["threshold"]),
        linkage=str(grouper_payload["linkage"]),
    )
    result = (
        project_unique_junction_pairs(
            selected_objects,
            groups,
            ranking_scores=probabilities[selected].detach().float().cpu(),
        )
        if projection_score_source == "selector"
        else junctions_from_endpoint_groups(selected_objects, groups)
    )
    diagnostics = JunctionAwareSelectionDiagnostics(
        raw_candidates=len(objects),
        threshold_selected=len(selected_objects),
        grouped_bonds=len(result.bond_objects),
        predicted_junctions=len(result.points),
        keep_threshold=keep_threshold,
        projection_score_source=projection_score_source,
        bond_type_source=bond_type_source,
        self_loop_objects=result.self_loops,
        duplicate_pair_objects=result.duplicate_pairs,
    )
    return result, diagnostics
