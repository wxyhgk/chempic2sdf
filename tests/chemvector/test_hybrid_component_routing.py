from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chemvector.dual_vertex_selector import IMPLICIT_ROLE, TEXT_ROLE
from chemvector.hybrid_component_routing import (
    HybridComponentRoutingRuntime,
    route_component_batches,
    route_single_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _junction(label: str = "N"):
    return (
        torch.tensor(((0.1, 0.1), (0.9, 0.9))),
        torch.tensor((TEXT_ROLE, IMPLICIT_ROLE)),
        [label, "C"],
    )


def _bond(begin=(0.1, 0.1), end=(0.9, 0.9), score=0.9):
    return {
        "endpoints": torch.tensor((begin, end)),
        "score": score,
        "display_type": "single",
    }


def _runtime(calls: list, *, batch_size: int = 1, veto=False):
    def decode_bonds(features, threshold):
        calls.append(("bonds", threshold, tuple(features.shape)))
        return [[_bond()] for _ in range(batch_size)], {"bond_route": threshold}

    def detect_junctions(pixels, features, real, bbox, include_ocr):
        calls.append(("junctions", real, bbox, include_ocr))
        return [_junction() for _ in range(batch_size)], {"junction_route": 1}

    def aware(pixels, bonds):
        calls.append(("aware", len(bonds), len(pixels)))
        return (
            SimpleNamespace(
                points=torch.tensor(((0.1, 0.1), (0.9, 0.9))),
                bond_objects=tuple(
                    {**bond, "junction_pair": (0, 1)} for bond in bonds
                ),
            ),
            {"aware_batch": len(pixels)},
        )

    def semantic(pixels, points):
        calls.append(("semantic", len(points)))
        return [{**_bond(), "junction_pair": (0, 1)}], {"semantic_route": 1}

    def veto_bonds(pixels, features, bonds):
        calls.append(("veto", len(bonds)))
        return [], 3, [{"bond_index": 0}]

    return HybridComponentRoutingRuntime(
        device=torch.device("cpu"),
        decode_bonds=decode_bonds,
        detect_junctions=detect_junctions,
        junction_aware_bonds=aware,
        semantic_relation_bonds=semantic,
        text_interior_bond_veto_enabled=veto,
        text_interior_bond_veto=veto_bonds if veto else None,
    )


def _inputs(batch_size=1):
    return (
        torch.zeros((batch_size, 3, 32, 32)),
        torch.zeros((batch_size, 8, 8, 8)),
    )


def test_single_vertex_routes_bonds_and_junctions_with_diagnostics() -> None:
    calls = []
    pixels, features = _inputs()
    routed = route_single_components(
        pixels,
        features,
        runtime=_runtime(calls),
        real_threshold=0.3,
        bond_threshold=0.4,
        bbox_expansion_pixels=5.0,
        include_ocr=False,
    )
    assert [call[0] for call in calls] == ["bonds", "junctions"]
    assert calls[0][1] == 0.4
    assert calls[1][1:] == (0.3, 5.0, False)
    assert routed.components[2] == ["N", "C"]
    assert routed.diagnostics == {
        "bond_route": 0.4,
        "text_interior_bond_veto_enabled": False,
        "text_interior_bond_veto_text_objects": 0,
        "text_interior_bond_veto_removed": 0,
        "text_interior_bond_veto_details": [],
        "junction_route": 1,
    }


def test_single_ocr_preserves_multiple_image_detection_error() -> None:
    calls = []
    pixels, features = _inputs()
    runtime = _runtime(calls, batch_size=2)
    with pytest.raises(
        ValueError,
        match="Single-image Junction inference received a batch",
    ):
        route_single_components(pixels, features, runtime=runtime)

    routed = route_single_components(
        pixels,
        features,
        runtime=runtime,
        include_ocr=False,
    )
    assert routed.components[2] == ["N", "C"]


def test_single_semantic_relations_skips_visual_bond_decoder() -> None:
    calls = []
    pixels, features = _inputs()
    routed = route_single_components(
        pixels,
        features,
        runtime=_runtime(calls),
        junction_source="semantic_relations",
    )
    assert [call[0] for call in calls] == ["junctions", "semantic"]
    assert routed.components[3][0]["junction_pair"] == (0, 1)
    assert "bond_route" not in routed.diagnostics
    assert routed.diagnostics["semantic_route"] == 1


def test_single_junction_semantic_relations_preserves_stage_order() -> None:
    calls = []
    pixels, features = _inputs()
    routed = route_single_components(
        pixels,
        features,
        runtime=_runtime(calls),
        junction_source="junction_semantic_relations",
    )
    assert [call[0] for call in calls] == [
        "bonds",
        "aware",
        "junctions",
        "semantic",
    ]
    assert calls[0][1] == 0.0
    assert routed.diagnostics["aware_batch"] == 1
    assert routed.diagnostics["semantic_route"] == 1


def test_single_bond_endpoints_without_ocr_does_not_detect_vertices() -> None:
    calls = []
    pixels, features = _inputs()
    routed = route_single_components(
        pixels,
        features,
        runtime=_runtime(calls),
        junction_source="bond_endpoints",
        include_ocr=False,
        endpoint_cluster_radius_pixels=2.0,
    )
    assert [call[0] for call in calls] == ["bonds"]
    assert routed.components[2] == ["C", "C"]
    assert routed.components[1].tolist() == [IMPLICIT_ROLE, IMPLICIT_ROLE]


def test_single_external_pairing_and_validation_are_preserved() -> None:
    calls = []
    pixels, features = _inputs()
    runtime = _runtime(calls)
    message = "External Junction source and external_junctions must be provided together"
    with pytest.raises(ValueError, match=message):
        route_single_components(
            pixels,
            features,
            runtime=runtime,
            junction_source="external",
        )
    assert calls == []

    routed = route_single_components(
        pixels,
        features,
        runtime=runtime,
        junction_source="external",
        external_junctions=torch.tensor(((0.2, 0.2), (0.8, 0.8))),
        include_ocr=False,
    )
    assert [call[0] for call in calls] == ["bonds"]
    assert routed.components[2] == ["C", "C"]


def test_single_text_veto_returns_legacy_diagnostic_shape() -> None:
    calls = []
    pixels, features = _inputs()
    with pytest.raises(
        ValueError,
        match="Bond endpoint clustering returned no Junctions",
    ):
        route_single_components(
            pixels,
            features,
            runtime=_runtime(calls, veto=True),
            junction_source="bond_endpoints",
        )
    assert [call[0] for call in calls] == ["bonds", "veto"]

    calls = []
    routed = route_single_components(
        pixels,
        features,
        runtime=_runtime(calls, veto=True),
        junction_source="vertex",
    )
    assert routed.diagnostics["text_interior_bond_veto_enabled"] is True
    assert routed.diagnostics["text_interior_bond_veto_text_objects"] == 3
    assert routed.diagnostics["text_interior_bond_veto_removed"] == 1


def test_single_unknown_source_raises_after_bond_decode_with_exact_text() -> None:
    calls = []
    pixels, features = _inputs()
    with pytest.raises(
        ValueError,
        match=r"Unknown Junction source: both",
    ):
        route_single_components(
            pixels,
            features,
            runtime=_runtime(calls),
            junction_source="both",
            include_ocr=False,
        )
    assert [call[0] for call in calls] == ["bonds"]


def test_single_external_with_ocr_attaches_only_text_within_radius() -> None:
    calls = []
    pixels, features = _inputs()
    routed = route_single_components(
        pixels,
        features,
        runtime=_runtime(calls),
        junction_source="external",
        external_junctions=torch.tensor(((0.2, 0.2), (0.8, 0.8))),
    )
    assert [call[0] for call in calls] == ["bonds", "junctions"]
    assert calls[1][3] is True
    assert routed.components[2] == ["N", "C"]
    assert routed.components[1].tolist() == [TEXT_ROLE, IMPLICIT_ROLE]
    assert len(routed.components[3]) == 1
    assert routed.components[3][0]["score"] == 0.9


def test_route_single_components_stays_within_line_budget() -> None:
    source = PROJECT_ROOT / "chemvector/inference/hybrid_component_routing.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "route_single_components"
    )
    assert function.end_lineno - function.lineno + 1 <= 70


def test_batch_vertex_preserves_zip_truncation_and_callback_order() -> None:
    calls = []
    pixels, features = _inputs(batch_size=2)
    runtime = _runtime(calls, batch_size=1)
    routed = route_component_batches(pixels, features, runtime=runtime)
    assert len(routed.components) == 1
    assert [call[0] for call in calls] == ["bonds", "junctions"]
    assert routed.diagnostics == {
        "bond_route": 0.25,
        "junction_route": 1,
    }


def test_batch_junction_aware_uses_zero_threshold_and_each_image_slice() -> None:
    calls = []
    pixels, features = _inputs(batch_size=2)
    routed = route_component_batches(
        pixels,
        features,
        runtime=_runtime(calls, batch_size=2),
        junction_source="junction_aware_bonds",
        include_ocr=False,
    )
    assert calls[0][0:2] == ("bonds", 0.0)
    assert [call[0] for call in calls] == [
        "bonds",
        "junctions",
        "aware",
        "aware",
    ]
    assert all(component[2] == ["C", "C"] for component in routed.components)
    assert routed.diagnostics["aware_batch"] == 1


def test_batch_unknown_source_runs_legacy_prerequisites_before_error() -> None:
    calls = []
    pixels, features = _inputs(batch_size=2)
    with pytest.raises(ValueError, match="Unknown Junction source: external"):
        route_component_batches(
            pixels,
            features,
            runtime=_runtime(calls, batch_size=2),
            junction_source="external",
        )
    assert [call[0] for call in calls] == ["bonds", "junctions"]


def test_batch_route_component_batches_stays_within_line_budget() -> None:
    source = PROJECT_ROOT / "chemvector/inference/hybrid_component_routing.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "route_component_batches"
    )
    assert function.end_lineno - function.lineno + 1 <= 55


def test_batch_bond_endpoints_with_ocr_attaches_detected_text() -> None:
    calls = []
    pixels, features = _inputs(batch_size=2)
    routed = route_component_batches(
        pixels,
        features,
        runtime=_runtime(calls, batch_size=2),
        junction_source="bond_endpoints",
        endpoint_cluster_radius_pixels=2.0,
    )
    assert [call[0] for call in calls] == ["bonds", "junctions"]
    assert calls[1][3] is True
    assert all(component[2] == ["N", "C"] for component in routed.components)
    assert routed.diagnostics == {
        "bond_route": 0.25,
        "junction_route": 1,
    }


def test_batch_endpoint_source_empty_points_raises_after_prerequisites() -> None:
    calls = []
    pixels, features = _inputs(batch_size=2)
    base = _runtime(calls, batch_size=2)

    def aware(pixels_slice, bonds):
        calls.append(("aware", len(bonds), len(pixels_slice)))
        return (
            SimpleNamespace(
                points=torch.empty((0, 2)),
                bond_objects=(),
            ),
            {"aware_batch": 1},
        )

    runtime = HybridComponentRoutingRuntime(
        device=torch.device("cpu"),
        decode_bonds=base.decode_bonds,
        detect_junctions=base.detect_junctions,
        junction_aware_bonds=aware,
        semantic_relation_bonds=base.semantic_relation_bonds,
    )
    with pytest.raises(
        ValueError,
        match="Bond endpoint clustering returned no Junctions",
    ):
        route_component_batches(
            pixels,
            features,
            runtime=runtime,
            junction_source="junction_aware_bonds",
        )
    assert [call[0] for call in calls] == ["bonds", "junctions", "aware"]


def test_batch_junction_aware_truncates_to_shortest_batch() -> None:
    calls = []
    pixels, features = _inputs(batch_size=2)
    routed = route_component_batches(
        pixels,
        features,
        runtime=_runtime(calls, batch_size=1),
        junction_source="junction_aware_bonds",
        include_ocr=False,
    )
    assert len(routed.components) == 1
    assert [call[0] for call in calls] == ["bonds", "junctions", "aware"]
    assert all(component[2] == ["C", "C"] for component in routed.components)


def test_component_router_has_no_pipeline_or_checkpoint_dependency() -> None:
    source = PROJECT_ROOT / "chemvector/inference/hybrid_component_routing.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "chemvector.infer_chemscene_hybrid" not in imports
    assert "chemvector.hybrid_checkpoint_loader" not in imports
