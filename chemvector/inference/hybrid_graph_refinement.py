"""Refine predicted bonds with graph-level and semantic model heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chemvector.chemscene_assembly import element_id_from_label
from chemvector.chemscene_graph_corrector import ELEMENT_TO_ID
from chemvector.graph_prediction_correction import apply_graph_correction
from chemvector.semantic_bond_relation import apply_aromatic_ring_semantics


@dataclass(frozen=True, slots=True)
class GraphCorrectionRuntime:
    device: torch.device
    corrector: Any
    default_keep_threshold: float
    uses_elements: bool


@dataclass(frozen=True, slots=True)
class SemanticBondRuntime:
    device: torch.device
    relation: Any
    relation_mode: str
    aromatic_mean_threshold: float
    aromatic_minimum_threshold: float
    existence: Any
    existence_threshold: float
    existence_chunk_size: int


def correct_bond_objects(
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    bond_objects: list[dict],
    *,
    mode: str,
    graph_keep_threshold: float | None,
    runtime: GraphCorrectionRuntime,
) -> tuple[list[dict], dict]:
    """Apply the learned graph corrector to an existing BondObject set."""

    if runtime.corrector is None:
        raise ValueError("Graph correction requires a graph corrector checkpoint")
    keep_threshold = (
        runtime.default_keep_threshold
        if graph_keep_threshold is None
        else graph_keep_threshold
    )
    if not bond_objects:
        return bond_objects, {
            "graph_correction_mode": mode,
            "graph_removed_bond_objects": 0,
            "graph_display_type_changes": 0,
            "graph_chemical_type_changes": 0,
            "graph_keep_threshold": keep_threshold,
        }
    graph_output = runtime.corrector(
        points,
        roles,
        torch.tensor(
            [
                (
                    element_id_from_label(label, int(role))
                    if runtime.uses_elements
                    else ELEMENT_TO_ID["C"]
                )
                for label, role in zip(labels, roles)
            ],
            device=runtime.device,
        ),
        torch.stack(
            [item["endpoints"].to(runtime.device) for item in bond_objects]
        ),
        torch.tensor(
            [float(item["score"]) for item in bond_objects],
            device=runtime.device,
        ),
        torch.tensor(
            [int(item["display"]) for item in bond_objects],
            device=runtime.device,
        ),
        torch.tensor(
            [int(item["chemical"]) for item in bond_objects],
            device=runtime.device,
        ),
    )
    return apply_graph_correction(
        bond_objects,
        graph_output,
        mode=mode,
        keep_threshold=keep_threshold,
    )


def apply_semantic_bond_types(
    pixels: torch.Tensor,
    points: torch.Tensor,
    bond_objects: list[dict] | tuple[dict, ...],
    *,
    runtime: SemanticBondRuntime,
) -> tuple[list[dict], dict]:
    """Predict chemical bond types on an already fixed Junction graph."""

    if runtime.relation is None or not bond_objects:
        return list(bond_objects), {
            "semantic_bond_relation_enabled": runtime.relation is not None,
            "semantic_bond_relation_mode": runtime.relation_mode,
            "semantic_chemical_type_changes": 0,
        }
    pair_indices = torch.tensor(
        [[item["junction_pair"] for item in bond_objects]],
        device=runtime.device,
        dtype=torch.long,
    )
    pair_mask = torch.ones(
        pair_indices.shape[:2], device=runtime.device, dtype=torch.bool
    )
    with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16):
        semantic_output = runtime.relation(
            pixels,
            points.to(runtime.device)[None],
            pair_indices,
            pair_mask,
        )
    if runtime.relation_mode == "aromatic_ring_only":
        updated, diagnostics = apply_aromatic_ring_semantics(
            bond_objects,
            semantic_output["chemical_logits"],
            mean_threshold=runtime.aromatic_mean_threshold,
            minimum_threshold=runtime.aromatic_minimum_threshold,
        )
    else:
        chemical = semantic_output["chemical_logits"].argmax(-1).tolist()
        updated = []
        changes = 0
        for item, chemical_type in zip(bond_objects, chemical):
            value = dict(item)
            changes += int(int(value["chemical"]) != chemical_type)
            value["chemical"] = int(chemical_type)
            updated.append(value)
        diagnostics = {"semantic_chemical_type_changes": changes}
    return updated, {
        "semantic_bond_relation_enabled": True,
        "semantic_bond_relation_mode": runtime.relation_mode,
        **diagnostics,
    }


@torch.inference_mode()
def semantic_relation_bonds(
    pixels: torch.Tensor,
    points: torch.Tensor,
    *,
    runtime: SemanticBondRuntime,
) -> tuple[list[dict], dict]:
    """Predict sparse bond existence and type from a fixed Junction set."""

    if runtime.existence is None:
        raise ValueError(
            "Semantic relation Junction source requires an existence checkpoint"
        )
    if len(points) < 2:
        raise ValueError("Semantic relation graph requires at least two Junctions")
    pairs = torch.combinations(
        torch.arange(len(points), device=runtime.device), r=2
    )
    selected_objects = []
    selected_logits = []
    scored_pairs = 0
    for start in range(0, len(pairs), runtime.existence_chunk_size):
        current_pairs = pairs[start : start + runtime.existence_chunk_size]
        pair_indices = current_pairs[None]
        pair_mask = torch.ones(
            pair_indices.shape[:2], device=runtime.device, dtype=torch.bool
        )
        with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16):
            output = runtime.existence(
                pixels,
                points.to(runtime.device)[None],
                pair_indices,
                pair_mask,
            )
        probabilities = output["edge_logits"].float().sigmoid()
        display = output["display_logits"].float().argmax(-1)
        chemical_logits = output["chemical_logits"].float()
        for index in torch.where(
            probabilities >= runtime.existence_threshold
        )[0].tolist():
            pair = tuple(int(value) for value in current_pairs[index].tolist())
            display_type = int(display[index])
            selected_objects.append(
                {
                    "score": float(probabilities[index]),
                    "feature": torch.empty(0),
                    "endpoints": points[list(pair)].detach().float().cpu(),
                    "junction_pair": pair,
                    "display": display_type,
                    "chemical": display_type,
                    "direction": 0,
                }
            )
            selected_logits.append(chemical_logits[index].cpu())
        scored_pairs += len(current_pairs)
    if not selected_objects:
        raise ValueError("Semantic Bond existence rejected every Junction pair")
    updated, ring_diagnostics = apply_aromatic_ring_semantics(
        selected_objects,
        torch.stack(selected_logits),
        mean_threshold=runtime.aromatic_mean_threshold,
        minimum_threshold=runtime.aromatic_minimum_threshold,
    )
    diagnostics = {
        "semantic_bond_existence_enabled": True,
        "semantic_bond_existence_threshold": runtime.existence_threshold,
        "semantic_bond_pairs_scored": scored_pairs,
        "semantic_bond_pairs_selected": len(updated),
        **ring_diagnostics,
    }
    return updated, diagnostics
