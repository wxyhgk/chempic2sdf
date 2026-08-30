"""Route predicted Bond and Junction sources into graph components.

This module owns source dispatch only. Neural inference remains behind explicit
callbacks so the routing policy can be tested without loading checkpoints or a
pipeline object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias

import torch

from chemvector.bond_endpoint_junction import junctions_from_bond_endpoints
from chemvector.dual_vertex_selector import IMPLICIT_ROLE
from chemvector.junction_text_attachment import (
    attach_text_to_bond_junctions,
    components_from_external_junctions,
)


JunctionPrediction: TypeAlias = tuple[torch.Tensor, torch.Tensor, list[str]]
ComponentPrediction: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[dict],
]


class EndpointJunctionResult(Protocol):
    points: torch.Tensor
    bond_objects: tuple[dict, ...] | list[dict]


BondDecoder: TypeAlias = Callable[
    [torch.Tensor, float],
    tuple[list[list[dict]], dict],
]
JunctionDetector: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, float, float, bool],
    tuple[list[JunctionPrediction], dict],
]
JunctionAwareDecoder: TypeAlias = Callable[
    [torch.Tensor, list[dict]],
    tuple[EndpointJunctionResult, dict],
]
SemanticBondDecoder: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[list[dict], dict],
]
TextInteriorBondVeto: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, list[dict]],
    tuple[list[dict], int, list[dict]],
]


@dataclass(frozen=True, slots=True)
class HybridComponentRoutingRuntime:
    """Callbacks and device required by component source routing."""

    device: torch.device
    decode_bonds: BondDecoder
    detect_junctions: JunctionDetector
    junction_aware_bonds: JunctionAwareDecoder
    semantic_relation_bonds: SemanticBondDecoder
    text_interior_bond_veto_enabled: bool = False
    text_interior_bond_veto: TextInteriorBondVeto | None = None


@dataclass(frozen=True, slots=True)
class RoutedComponents:
    components: ComponentPrediction
    diagnostics: dict


@dataclass(frozen=True, slots=True)
class RoutedComponentBatch:
    components: list[ComponentPrediction]
    diagnostics: dict


def _detect_junctions(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    features: torch.Tensor,
    *,
    real_threshold: float,
    bbox_expansion_pixels: float,
    include_ocr: bool,
) -> tuple[list[JunctionPrediction], dict]:
    return runtime.detect_junctions(
        pixels,
        features,
        real_threshold,
        bbox_expansion_pixels,
        include_ocr,
    )


def _single_junction_prediction(
    predictions: list[JunctionPrediction],
    *,
    enforce_single_image: bool,
) -> JunctionPrediction:
    if enforce_single_image and len(predictions) != 1:
        raise ValueError("Single-image Junction inference received a batch")
    return predictions[0]


def _implicit_atoms(
    points: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    roles = torch.full(
        (len(points),),
        IMPLICIT_ROLE,
        device=device,
        dtype=torch.long,
    )
    return roles, ["C"] * len(points)


def _single_veto_diagnostics(
    *,
    enabled: bool,
    text_objects: int = 0,
    removed: list[dict] | None = None,
) -> dict:
    details = [] if removed is None else removed
    return {
        "text_interior_bond_veto_enabled": enabled,
        "text_interior_bond_veto_text_objects": text_objects,
        "text_interior_bond_veto_removed": len(details),
        "text_interior_bond_veto_details": details,
    }


def _apply_text_interior_veto(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    features: torch.Tensor,
    bond_objects: list[dict],
) -> tuple[list[dict], dict]:
    if runtime.text_interior_bond_veto_enabled and bond_objects:
        if runtime.text_interior_bond_veto is None:
            raise ValueError("Text interior Bond veto callback is unavailable")
        bond_objects, text_count, removed = runtime.text_interior_bond_veto(
            pixels,
            features,
            bond_objects,
        )
        return bond_objects, _single_veto_diagnostics(
            enabled=True,
            text_objects=text_count,
            removed=removed,
        )
    return bond_objects, _single_veto_diagnostics(
        enabled=runtime.text_interior_bond_veto_enabled,
    )


def _route_bond_endpoint_sources(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    features: torch.Tensor,
    bond_objects: list[dict],
    junction_source: str,
    endpoint_cluster_radius_pixels: float,
    text_attach_radius_pixels: float,
    real_threshold: float,
    bbox_expansion_pixels: float,
    include_ocr: bool,
    diagnostics: dict,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
    if junction_source in {
        "junction_aware_bonds",
        "junction_semantic_relations",
    }:
        endpoint_result, endpoint_diagnostics = runtime.junction_aware_bonds(
            pixels,
            bond_objects,
        )
        diagnostics.update(endpoint_diagnostics)
    else:
        endpoint_result = junctions_from_bond_endpoints(
            bond_objects,
            width=pixels.size(-1),
            height=pixels.size(-2),
            radius_pixels=endpoint_cluster_radius_pixels,
        )
    points = endpoint_result.points.to(runtime.device)
    bond_objects = list(endpoint_result.bond_objects)
    if not len(points):
        raise ValueError("Bond endpoint clustering returned no Junctions")
    if include_ocr:
        detected, junction_diagnostics = _detect_junctions(
            runtime,
            pixels,
            features,
            real_threshold=real_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=True,
        )
        diagnostics.update(junction_diagnostics)
        detected_points, detected_roles, detected_labels = (
            _single_junction_prediction(
                detected,
                enforce_single_image=True,
            )
        )
        roles, labels = attach_text_to_bond_junctions(
            points,
            detected_points,
            detected_roles,
            detected_labels,
            image_size=pixels.size(-1),
            maximum_distance_pixels=text_attach_radius_pixels,
        )
    else:
        roles, labels = _implicit_atoms(points, device=runtime.device)
    if junction_source == "junction_semantic_relations":
        bond_objects, semantic_diagnostics = runtime.semantic_relation_bonds(
            pixels,
            points,
        )
        diagnostics.update(semantic_diagnostics)
    return points, roles, labels, bond_objects


def _route_external_source(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    features: torch.Tensor,
    bond_objects: list[dict],
    external_junctions: torch.Tensor,
    text_attach_radius_pixels: float,
    real_threshold: float,
    bbox_expansion_pixels: float,
    include_ocr: bool,
    diagnostics: dict,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
    points = external_junctions.to(runtime.device)
    if include_ocr:
        detected, junction_diagnostics = _detect_junctions(
            runtime,
            pixels,
            features,
            real_threshold=real_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=True,
        )
        diagnostics.update(junction_diagnostics)
        detected_points, detected_roles, detected_labels = (
            _single_junction_prediction(
                detected,
                enforce_single_image=True,
            )
        )
    else:
        detected_points = detected_roles = detected_labels = None
    points, roles, labels, bond_objects = components_from_external_junctions(
        points,
        bond_objects,
        include_ocr=include_ocr,
        detected_points=detected_points,
        detected_roles=detected_roles,
        detected_labels=detected_labels,
        image_size=pixels.size(-1),
        text_attach_radius_pixels=text_attach_radius_pixels,
    )
    return points, roles, labels, bond_objects


def _route_vertex_sources(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    features: torch.Tensor,
    bond_objects: list[dict],
    junction_source: str,
    real_threshold: float,
    bbox_expansion_pixels: float,
    include_ocr: bool,
    diagnostics: dict,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
    detected, junction_diagnostics = _detect_junctions(
        runtime,
        pixels,
        features,
        real_threshold=real_threshold,
        bbox_expansion_pixels=bbox_expansion_pixels,
        include_ocr=include_ocr,
    )
    diagnostics.update(junction_diagnostics)
    points, roles, labels = _single_junction_prediction(
        detected,
        enforce_single_image=include_ocr,
    )
    if junction_source == "semantic_relations":
        bond_objects, semantic_diagnostics = runtime.semantic_relation_bonds(
            pixels,
            points,
        )
        diagnostics.update(semantic_diagnostics)
    return points, roles, labels, bond_objects


def _route_single_source(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    features: torch.Tensor,
    bond_objects: list[dict],
    junction_source: str,
    endpoint_cluster_radius_pixels: float,
    text_attach_radius_pixels: float,
    real_threshold: float,
    bbox_expansion_pixels: float,
    include_ocr: bool,
    external_junctions: torch.Tensor | None,
    diagnostics: dict,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
    if junction_source in {
        "bond_endpoints",
        "junction_aware_bonds",
        "junction_semantic_relations",
    }:
        return _route_bond_endpoint_sources(
            runtime, pixels, features, bond_objects, junction_source,
            endpoint_cluster_radius_pixels, text_attach_radius_pixels,
            real_threshold, bbox_expansion_pixels, include_ocr,
            diagnostics,
        )
    if junction_source == "external":
        return _route_external_source(
            runtime, pixels, features, bond_objects, external_junctions,
            text_attach_radius_pixels, real_threshold,
            bbox_expansion_pixels, include_ocr, diagnostics,
        )
    if junction_source in {"vertex", "semantic_relations"}:
        return _route_vertex_sources(
            runtime, pixels, features, bond_objects, junction_source,
            real_threshold, bbox_expansion_pixels, include_ocr,
            diagnostics,
        )
    raise ValueError(f"Unknown Junction source: {junction_source}")


def route_single_components(
    pixels: torch.Tensor,
    features: torch.Tensor,
    *,
    runtime: HybridComponentRoutingRuntime,
    real_threshold: float = 0.2,
    bond_threshold: float = 0.25,
    bbox_expansion_pixels: float = 8.0,
    include_ocr: bool = True,
    junction_source: str = "vertex",
    endpoint_cluster_radius_pixels: float = 12.0,
    text_attach_radius_pixels: float = 24.0,
    external_junctions: torch.Tensor | None = None,
) -> RoutedComponents:
    """Route one encoded image while preserving the legacy source semantics."""

    if (junction_source == "external") != (external_junctions is not None):
        raise ValueError(
            "External Junction source and external_junctions must be provided together"
        )

    diagnostics = {}
    if junction_source == "semantic_relations":
        bond_batches = [[]]
    else:
        threshold = (
            0.0
            if junction_source
            in {"junction_aware_bonds", "junction_semantic_relations"}
            else bond_threshold
        )
        bond_batches, bond_diagnostics = runtime.decode_bonds(features, threshold)
        diagnostics.update(bond_diagnostics)
    bond_objects = bond_batches[0]

    bond_objects, veto_diagnostics = _apply_text_interior_veto(
        runtime, pixels, features, bond_objects,
    )
    diagnostics.update(veto_diagnostics)

    points, roles, labels, bond_objects = _route_single_source(
        runtime, pixels, features, bond_objects, junction_source,
        endpoint_cluster_radius_pixels, text_attach_radius_pixels,
        real_threshold, bbox_expansion_pixels, include_ocr,
        external_junctions, diagnostics,
    )

    return RoutedComponents(
        components=(points, roles, labels, bond_objects),
        diagnostics=diagnostics,
    )


def _route_batch_endpoint_source(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    detected: list[JunctionPrediction],
    bond_batches: list[list[dict]],
    *,
    junction_source: str,
    endpoint_cluster_radius_pixels: float,
    text_attach_radius_pixels: float,
    include_ocr: bool,
    diagnostics: dict,
) -> list[ComponentPrediction]:
    results = []
    for batch_index, (
        (detected_points, detected_roles, detected_labels),
        bonds,
    ) in enumerate(zip(detected, bond_batches)):
        if junction_source == "junction_aware_bonds":
            endpoint_result, endpoint_diagnostics = runtime.junction_aware_bonds(
                pixels[batch_index : batch_index + 1],
                bonds,
            )
            diagnostics.update(endpoint_diagnostics)
        else:
            endpoint_result = junctions_from_bond_endpoints(
                bonds,
                width=pixels.size(-1),
                height=pixels.size(-2),
                radius_pixels=endpoint_cluster_radius_pixels,
            )
        points = endpoint_result.points.to(runtime.device)
        if not len(points):
            raise ValueError("Bond endpoint clustering returned no Junctions")
        if include_ocr:
            roles, labels = attach_text_to_bond_junctions(
                points,
                detected_points,
                detected_roles,
                detected_labels,
                image_size=pixels.size(-1),
                maximum_distance_pixels=text_attach_radius_pixels,
            )
        else:
            roles, labels = _implicit_atoms(points, device=runtime.device)
        results.append(
            (points, roles, labels, list(endpoint_result.bond_objects))
        )
    return results


def _route_batch_sources(
    runtime: HybridComponentRoutingRuntime,
    pixels: torch.Tensor,
    bond_batches: list[list[dict]],
    detected_junctions: list[JunctionPrediction],
    *,
    junction_source: str,
    endpoint_cluster_radius_pixels: float,
    text_attach_radius_pixels: float,
    include_ocr: bool,
    diagnostics: dict,
) -> RoutedComponentBatch:
    if junction_source == "vertex":
        return RoutedComponentBatch(
            components=[
                (points, roles, labels, bonds)
                for (points, roles, labels), bonds in zip(
                    detected_junctions,
                    bond_batches,
                )
            ],
            diagnostics=diagnostics,
        )
    if junction_source not in {"bond_endpoints", "junction_aware_bonds"}:
        raise ValueError(f"Unknown Junction source: {junction_source}")
    return RoutedComponentBatch(
        components=_route_batch_endpoint_source(
            runtime,
            pixels,
            detected_junctions,
            bond_batches,
            junction_source=junction_source,
            endpoint_cluster_radius_pixels=endpoint_cluster_radius_pixels,
            text_attach_radius_pixels=text_attach_radius_pixels,
            include_ocr=include_ocr,
            diagnostics=diagnostics,
        ),
        diagnostics=diagnostics,
    )


def route_component_batches(
    pixels: torch.Tensor,
    features: torch.Tensor,
    *,
    runtime: HybridComponentRoutingRuntime,
    real_threshold: float = 0.2,
    bond_threshold: float = 0.25,
    bbox_expansion_pixels: float = 8.0,
    include_ocr: bool = True,
    junction_source: str = "vertex",
    endpoint_cluster_radius_pixels: float = 12.0,
    text_attach_radius_pixels: float = 24.0,
) -> RoutedComponentBatch:
    """Route an encoded batch with the legacy batch source restrictions."""

    diagnostics = {}
    threshold = 0.0 if junction_source == "junction_aware_bonds" else bond_threshold
    bond_batches, bond_diagnostics = runtime.decode_bonds(features, threshold)
    diagnostics.update(bond_diagnostics)
    detected_junctions, junction_diagnostics = _detect_junctions(
        runtime,
        pixels,
        features,
        real_threshold=real_threshold,
        bbox_expansion_pixels=bbox_expansion_pixels,
        include_ocr=include_ocr,
    )
    diagnostics.update(junction_diagnostics)
    return _route_batch_sources(
        runtime,
        pixels,
        bond_batches,
        detected_junctions,
        junction_source=junction_source,
        endpoint_cluster_radius_pixels=endpoint_cluster_radius_pixels,
        text_attach_radius_pixels=text_attach_radius_pixels,
        include_ocr=include_ocr,
        diagnostics=diagnostics,
    )
