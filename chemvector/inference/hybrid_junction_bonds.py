"""Select BondObjects and derive Junctions for junction-aware inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from chemvector.bond_endpoint_junction import BondEndpointJunctionResult
from chemvector.hybrid_graph_refinement import (
    SemanticBondRuntime,
    apply_semantic_bond_types,
)
from chemvector.junction_aware_bond_selector import select_and_group_bonds
from chemvector.pretrain_bond_set_selector import read_line_band_features


@dataclass(frozen=True, slots=True)
class HybridJunctionBondRuntime:
    bond_set_selector: Any
    bond_set_selector_payload: Any
    endpoint_grouper: Any
    endpoint_grouper_payload: Any
    line_band_reader: Any
    projection_score_source: str
    bond_type_source: str
    semantic: SemanticBondRuntime


def junction_aware_bonds(
    pixels: torch.Tensor,
    bond_objects: list[dict],
    *,
    runtime: HybridJunctionBondRuntime,
) -> tuple[BondEndpointJunctionResult, dict]:
    """Select visual bonds, cluster endpoints, then refine chemical types."""

    if (
        runtime.bond_set_selector is None
        or runtime.bond_set_selector_payload is None
        or runtime.endpoint_grouper is None
        or runtime.endpoint_grouper_payload is None
        or runtime.line_band_reader is None
    ):
        raise ValueError(
            "Junction-aware Bond source requires all v0.8.21 checkpoints"
        )
    line_band = read_line_band_features(
        runtime.line_band_reader, pixels, [bond_objects]
    )[0]
    result, selection_diagnostics = select_and_group_bonds(
        pixels,
        bond_objects,
        line_band,
        runtime.bond_set_selector,
        runtime.bond_set_selector_payload,
        runtime.endpoint_grouper,
        runtime.endpoint_grouper_payload,
        projection_score_source=runtime.projection_score_source,
        bond_type_source=runtime.bond_type_source,
    )
    updated_objects, semantic_diagnostics = apply_semantic_bond_types(
        pixels,
        result.points,
        result.bond_objects,
        runtime=runtime.semantic,
    )
    if runtime.semantic.relation is not None and result.bond_objects:
        result = BondEndpointJunctionResult(
            points=result.points,
            bond_objects=tuple(updated_objects),
            raw_clusters=result.raw_clusters,
            self_loops=result.self_loops,
            duplicate_pairs=result.duplicate_pairs,
        )
    return result, {
        **asdict(selection_diagnostics),
        **semantic_diagnostics,
    }
