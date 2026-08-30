"""Compile predicted hybrid components into a ChemScene and diagnostics.

This module owns the post-component inference orchestration. Learned stages are
provided explicitly through ``HybridSceneCompilationRuntime`` so this layer
does not depend on the hybrid pipeline or checkpoint loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from chemvector.chemscene_assembly import (
    assemble_chemscene,
    compile_scene_with_aromatic_fallback,
)
from chemvector.junction_bond_assembler import (
    decode_junction_assignment_rescue,
    decode_junction_assignments,
)
from chemvector.structured_atom_decoder import (
    AtomStateCandidate,
    rerank_invalid_scene_atom_states,
)


class GraphCorrectionCallback(Protocol):
    def __call__(
        self,
        points: torch.Tensor,
        roles: torch.Tensor,
        labels: list[str],
        bond_objects: list[dict],
        *,
        mode: str,
        graph_keep_threshold: float | None,
    ) -> tuple[list[dict], dict]: ...


class SemanticBondCallback(Protocol):
    def __call__(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
        bond_objects: list[dict] | tuple[dict, ...],
    ) -> tuple[list[dict], dict]: ...


class AtomCandidateCallback(Protocol):
    def __call__(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
        roles: torch.Tensor,
        labels: list[str],
    ) -> Sequence[Sequence[AtomStateCandidate]]: ...


@dataclass(frozen=True, slots=True)
class HybridSceneCompilationRuntime:
    """Learned runtime hooks required after component prediction."""

    device: torch.device
    assembler: Any | None
    default_graph_keep_threshold: float
    correct_bonds: GraphCorrectionCallback
    apply_semantic_bond_types: SemanticBondCallback
    structured_atom_candidates: AtomCandidateCallback


@dataclass(frozen=True, slots=True)
class HybridSceneCompilationSettings:
    molecule_id: str
    assignment_mode: str = "nearest"
    graph_correction_mode: str = "none"
    graph_keep_threshold: float | None = None
    junction_source: str = "vertex"
    atom_decode_mode: str = "greedy"
    real_threshold: float = 0.2
    bond_threshold: float = 0.25
    endpoint_cluster_radius_pixels: float = 12.0
    text_attach_radius_pixels: float = 24.0


@dataclass(frozen=True, slots=True)
class HybridSceneCompilationResult:
    scene: dict
    diagnostics: dict


@dataclass(frozen=True, slots=True)
class _AssignmentPlan:
    mode: str
    assignments: list[tuple[int, int] | None] | None
    diagnostics: dict


@dataclass(frozen=True, slots=True)
class _AtomDecodingResult:
    scene: dict
    candidates: Sequence[Sequence[AtomStateCandidate]] | None
    diagnostics: dict


@dataclass(frozen=True, slots=True)
class _InvalidOnlyResult:
    scene: dict
    assembly_diagnostics: dict
    atom_diagnostics: dict
    graph_diagnostics: dict


_EXPLICIT_ASSIGNMENT_SOURCES = frozenset(
    {
        "bond_endpoints",
        "junction_aware_bonds",
        "semantic_relations",
        "junction_semantic_relations",
    }
)


def _decode_assignments(
    points: torch.Tensor,
    roles: torch.Tensor,
    bond_objects: list[dict],
    *,
    assignment_mode: str,
    runtime: HybridSceneCompilationRuntime,
) -> tuple[list[tuple[int, int] | None] | None, dict]:
    if assignment_mode not in {"learned", "learned_rescue"}:
        return None, {}
    if runtime.assembler is None:
        raise ValueError("Learned assignment requires an assembler checkpoint")
    if not bond_objects:
        return [], {}

    bond_endpoints = torch.stack(
        [item["endpoints"].to(runtime.device) for item in bond_objects]
    )
    bond_scores = torch.tensor(
        [float(item["score"]) for item in bond_objects],
        device=runtime.device,
    )
    logits = runtime.assembler(points, roles, bond_endpoints, bond_scores)
    if assignment_mode == "learned_rescue":
        return decode_junction_assignment_rescue(
            logits,
            bond_scores,
            bond_endpoints,
            points,
        )
    return decode_junction_assignments(logits, bond_scores)


def _assemble(
    *,
    settings: HybridSceneCompilationSettings,
    pixels: torch.Tensor,
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    bond_objects: list[dict],
    assignment_mode: str,
    junction_assignments: list[tuple[int, int] | None] | None,
) -> tuple[dict, dict]:
    return assemble_chemscene(
        settings.molecule_id,
        points,
        roles,
        labels,
        bond_objects,
        width=pixels.size(-1),
        height=pixels.size(-2),
        assignment_mode=assignment_mode,
        junction_assignments=junction_assignments,
        unresolved_as_dummy=settings.atom_decode_mode != "greedy",
    )


def _correct_initial_bonds(
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    bond_objects: list[dict],
    *,
    settings: HybridSceneCompilationSettings,
    runtime: HybridSceneCompilationRuntime,
) -> tuple[list[dict], dict]:
    if settings.graph_correction_mode not in {"type_only", "full"}:
        return bond_objects, {}
    return runtime.correct_bonds(
        points,
        roles,
        labels,
        bond_objects,
        mode=settings.graph_correction_mode,
        graph_keep_threshold=settings.graph_keep_threshold,
    )


def _assignment_plan(
    points: torch.Tensor,
    roles: torch.Tensor,
    bond_objects: list[dict],
    *,
    settings: HybridSceneCompilationSettings,
    runtime: HybridSceneCompilationRuntime,
) -> _AssignmentPlan:
    if settings.junction_source in _EXPLICIT_ASSIGNMENT_SOURCES:
        return _AssignmentPlan(
            mode="explicit",
            assignments=[item["junction_pair"] for item in bond_objects],
            diagnostics={},
        )
    assignments, diagnostics = _decode_assignments(
        points,
        roles,
        bond_objects,
        assignment_mode=settings.assignment_mode,
        runtime=runtime,
    )
    return _AssignmentPlan(settings.assignment_mode, assignments, diagnostics)


def _decode_atom_states(
    scene: dict,
    *,
    pixels: torch.Tensor,
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    settings: HybridSceneCompilationSettings,
    runtime: HybridSceneCompilationRuntime,
) -> _AtomDecodingResult:
    if settings.atom_decode_mode != "structured_invalid_only":
        return _AtomDecodingResult(
            scene=scene,
            candidates=None,
            diagnostics={"atom_decode_mode": settings.atom_decode_mode},
        )
    candidates = runtime.structured_atom_candidates(pixels, points, roles, labels)
    scene, diagnostics = rerank_invalid_scene_atom_states(scene, candidates)
    return _AtomDecodingResult(scene, candidates, diagnostics)


def _inactive_invalid_only_diagnostics(
    settings: HybridSceneCompilationSettings,
    runtime: HybridSceneCompilationRuntime,
) -> dict:
    return {
        "graph_correction_mode": "invalid_only",
        "graph_correction_triggered": False,
        "graph_removed_bond_objects": 0,
        "graph_display_type_changes": 0,
        "graph_chemical_type_changes": 0,
        "graph_keep_threshold": (
            runtime.default_graph_keep_threshold
            if settings.graph_keep_threshold is None
            else settings.graph_keep_threshold
        ),
    }


def _recover_invalid_scene(
    scene: dict,
    assembly_diagnostics: dict,
    atom_decoding: _AtomDecodingResult,
    *,
    pixels: torch.Tensor,
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    bond_objects: list[dict],
    assignment: _AssignmentPlan,
    settings: HybridSceneCompilationSettings,
    runtime: HybridSceneCompilationRuntime,
) -> _InvalidOnlyResult:
    graph_diagnostics = _inactive_invalid_only_diagnostics(settings, runtime)
    try:
        compile_scene_with_aromatic_fallback(scene)
    except Exception as base_error:
        corrected_bonds, correction = runtime.correct_bonds(
            points,
            roles,
            labels,
            bond_objects,
            mode="type_only",
            graph_keep_threshold=settings.graph_keep_threshold,
        )
        corrected_bonds, post_semantic = runtime.apply_semantic_bond_types(
            pixels, points, corrected_bonds
        )
        scene, assembly_diagnostics = _assemble(
            settings=settings,
            pixels=pixels,
            points=points,
            roles=roles,
            labels=labels,
            bond_objects=corrected_bonds,
            assignment_mode=assignment.mode,
            junction_assignments=assignment.assignments,
        )
        atom_diagnostics = atom_decoding.diagnostics
        if atom_decoding.candidates is not None:
            scene, atom_diagnostics = rerank_invalid_scene_atom_states(
                scene, atom_decoding.candidates
            )
        graph_diagnostics = {
            **correction,
            "graph_correction_mode": "invalid_only",
            "graph_correction_triggered": True,
            "graph_correction_base_error": str(base_error),
            **{f"post_graph_{key}": value for key, value in post_semantic.items()},
        }
        return _InvalidOnlyResult(
            scene, assembly_diagnostics, atom_diagnostics, graph_diagnostics
        )
    return _InvalidOnlyResult(
        scene,
        assembly_diagnostics,
        atom_decoding.diagnostics,
        graph_diagnostics,
    )


def _merge_diagnostics(
    assembly: dict,
    assignment: dict,
    graph: dict,
    atom: dict,
    component: dict | None,
    *,
    labels: list[str],
    settings: HybridSceneCompilationSettings,
) -> dict:
    diagnostics = dict(assembly)
    diagnostics.update(assignment)
    diagnostics.update(graph)
    diagnostics.update(atom)
    diagnostics.update(component or {})
    diagnostics["ocr_labels"] = labels
    diagnostics["real_threshold"] = settings.real_threshold
    diagnostics["bond_threshold"] = settings.bond_threshold
    diagnostics["junction_source"] = settings.junction_source
    diagnostics["endpoint_cluster_radius_pixels"] = (
        settings.endpoint_cluster_radius_pixels
    )
    diagnostics["text_attach_radius_pixels"] = settings.text_attach_radius_pixels
    return diagnostics


def compile_hybrid_scene(
    *,
    pixels: torch.Tensor,
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    bond_objects: list[dict],
    settings: HybridSceneCompilationSettings,
    runtime: HybridSceneCompilationRuntime,
    component_diagnostics: dict | None = None,
) -> HybridSceneCompilationResult:
    """Compile predicted Junctions and bonds without accessing pipeline state."""

    if settings.atom_decode_mode not in {"greedy", "structured_invalid_only"}:
        raise ValueError(f"Unknown atom decode mode: {settings.atom_decode_mode}")
    bond_objects, graph_diagnostics = _correct_initial_bonds(
        points, roles, labels, bond_objects, settings=settings, runtime=runtime
    )
    assignment = _assignment_plan(
        points, roles, bond_objects, settings=settings, runtime=runtime
    )
    scene, assembly_diagnostics = _assemble(
        settings=settings,
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        bond_objects=bond_objects,
        assignment_mode=assignment.mode,
        junction_assignments=assignment.assignments,
    )
    atom_decoding = _decode_atom_states(
        scene,
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        settings=settings,
        runtime=runtime,
    )
    scene = atom_decoding.scene
    atom_diagnostics = atom_decoding.diagnostics
    if settings.graph_correction_mode == "invalid_only":
        recovered = _recover_invalid_scene(
            scene,
            assembly_diagnostics,
            atom_decoding,
            pixels=pixels,
            points=points,
            roles=roles,
            labels=labels,
            bond_objects=bond_objects,
            assignment=assignment,
            settings=settings,
            runtime=runtime,
        )
        scene = recovered.scene
        assembly_diagnostics = recovered.assembly_diagnostics
        atom_diagnostics = recovered.atom_diagnostics
        graph_diagnostics = recovered.graph_diagnostics
    diagnostics = _merge_diagnostics(
        assembly_diagnostics,
        assignment.diagnostics,
        graph_diagnostics,
        atom_diagnostics,
        component_diagnostics,
        labels=labels,
        settings=settings,
    )
    return HybridSceneCompilationResult(scene=scene, diagnostics=diagnostics)
