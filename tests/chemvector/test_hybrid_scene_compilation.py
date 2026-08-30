from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import chemvector.hybrid_scene_compilation as compilation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _components():
    pixels = torch.zeros((1, 3, 20, 24))
    points = torch.tensor(((0.2, 0.5), (0.8, 0.5)))
    roles = torch.zeros((2,), dtype=torch.long)
    labels = ["C", "C"]
    bonds = [
        {
            "score": 0.9,
            "endpoints": points.clone(),
            "junction_pair": (0, 1),
            "display": 0,
            "chemical": 0,
            "direction": 0,
        }
    ]
    return pixels, points, roles, labels, bonds


def _runtime(**overrides):
    def unchanged_bonds(points, roles, labels, bonds, **kwargs):
        return list(bonds), {"corrected": True}

    values = {
        "device": torch.device("cpu"),
        "assembler": None,
        "default_graph_keep_threshold": 0.3,
        "correct_bonds": unchanged_bonds,
        "apply_semantic_bond_types": lambda pixels, points, bonds: (
            list(bonds),
            {"semantic_changes": 0},
        ),
        "structured_atom_candidates": lambda *args: [[], []],
    }
    values.update(overrides)
    return compilation.HybridSceneCompilationRuntime(**values)


def _settings(**overrides):
    values = {"molecule_id": "sample"}
    values.update(overrides)
    return compilation.HybridSceneCompilationSettings(**values)


def test_module_does_not_depend_on_pipeline_or_checkpoint_loader() -> None:
    path = PROJECT_ROOT / "chemvector/inference/hybrid_scene_compilation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "chemvector.infer_chemscene_hybrid" not in imports
    assert "chemvector.hybrid_checkpoint_loader" not in imports


def test_compile_hybrid_scene_remains_a_short_stage_orchestrator() -> None:
    path = PROJECT_ROOT / "chemvector/inference/hybrid_scene_compilation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compile_hybrid_scene"
    )
    assert function.end_lineno - function.lineno + 1 <= 75


def test_unknown_atom_decode_mode_preserves_error_text() -> None:
    pixels, points, roles, labels, bonds = _components()
    with pytest.raises(ValueError, match="Unknown atom decode mode: future"):
        compilation.compile_hybrid_scene(
            pixels=pixels,
            points=points,
            roles=roles,
            labels=labels,
            bond_objects=bonds,
            settings=_settings(atom_decode_mode="future"),
            runtime=_runtime(),
        )


def test_explicit_source_uses_bond_pairs_without_assembler(monkeypatch) -> None:
    pixels, points, roles, labels, bonds = _components()
    received = {}

    def fake_assemble(*args, **kwargs):
        received.update(kwargs)
        return {"junctions": []}, {"assembled": 1}

    monkeypatch.setattr(compilation, "assemble_chemscene", fake_assemble)
    result = compilation.compile_hybrid_scene(
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        bond_objects=bonds,
        settings=_settings(junction_source="bond_endpoints"),
        runtime=_runtime(),
    )
    assert received["assignment_mode"] == "explicit"
    assert received["junction_assignments"] == [(0, 1)]
    assert result.diagnostics["assembled"] == 1


def test_learned_assignment_requires_assembler_checkpoint() -> None:
    pixels, points, roles, labels, bonds = _components()
    with pytest.raises(
        ValueError,
        match="Learned assignment requires an assembler checkpoint",
    ):
        compilation.compile_hybrid_scene(
            pixels=pixels,
            points=points,
            roles=roles,
            labels=labels,
            bond_objects=bonds,
            settings=_settings(assignment_mode="learned"),
            runtime=_runtime(),
        )


def test_learned_assignment_decodes_and_merges_diagnostics(monkeypatch) -> None:
    pixels, points, roles, labels, bonds = _components()
    received = {}

    class Assembler:
        def __call__(self, points, roles, endpoints, scores):
            return torch.tensor((((5.0, 0.0, -1.0), (0.0, 5.0, -1.0)),))

    def fake_decode(logits, scores):
        return [(0, 1)], {"assembler_decoded": 1}

    def fake_assemble(*args, **kwargs):
        received.update(kwargs)
        return {"junctions": []}, {"assembled": 1}

    monkeypatch.setattr(compilation, "decode_junction_assignments", fake_decode)
    monkeypatch.setattr(compilation, "assemble_chemscene", fake_assemble)
    result = compilation.compile_hybrid_scene(
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        bond_objects=bonds,
        settings=_settings(assignment_mode="learned", real_threshold=0.4),
        runtime=_runtime(assembler=Assembler()),
        component_diagnostics={"component": "kept"},
    )
    assert received["junction_assignments"] == [(0, 1)]
    assert result.diagnostics["assembler_decoded"] == 1
    assert result.diagnostics["component"] == "kept"
    assert result.diagnostics["real_threshold"] == 0.4
    assert result.diagnostics["ocr_labels"] == labels


def test_invalid_only_skips_correction_when_scene_compiles(monkeypatch) -> None:
    pixels, points, roles, labels, bonds = _components()
    calls = []

    monkeypatch.setattr(
        compilation,
        "compile_scene_with_aromatic_fallback",
        lambda scene: ("sdf", scene, False),
    )

    def unexpected_correction(*args, **kwargs):
        calls.append("correct")
        return list(bonds), {}

    result = compilation.compile_hybrid_scene(
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        bond_objects=bonds,
        settings=_settings(graph_correction_mode="invalid_only"),
        runtime=_runtime(correct_bonds=unexpected_correction),
    )
    assert calls == []
    assert result.diagnostics["graph_correction_triggered"] is False
    assert result.diagnostics["graph_keep_threshold"] == 0.3


def test_invalid_only_rebuilds_corrected_scene_and_prefixes_semantic_diagnostics(
    monkeypatch,
) -> None:
    pixels, points, roles, labels, bonds = _components()
    assembled_bonds = []
    candidates = [[object()], [object()]]
    events = []

    def fake_assemble(*args, **kwargs):
        events.append("assemble_base" if not assembled_bonds else "assemble_corrected")
        assembled_bonds.append(args[4])
        return {"junctions": [{}, {}], "pass": len(assembled_bonds)}, {
            "assembly_pass": len(assembled_bonds)
        }

    def fail_compile(scene):
        events.append("compile")
        raise ValueError("base invalid")

    corrected = [{**bonds[0], "chemical": 1}]

    def correct(*args, **kwargs):
        events.append("correct")
        assert kwargs["mode"] == "type_only"
        return corrected, {"graph_chemical_type_changes": 1}

    def semantic(pixels, points, values):
        events.append("semantic")
        assert values is corrected
        return values, {"semantic_chemical_type_changes": 2}

    rerank_calls = []

    def rerank(scene, atom_candidates):
        events.append(f"rerank_{scene['pass']}")
        rerank_calls.append((scene["pass"], atom_candidates))
        return {**scene, "reranked": True}, {"atom_decode_mode": "structured_invalid_only"}

    monkeypatch.setattr(compilation, "assemble_chemscene", fake_assemble)
    monkeypatch.setattr(compilation, "compile_scene_with_aromatic_fallback", fail_compile)
    monkeypatch.setattr(compilation, "rerank_invalid_scene_atom_states", rerank)
    result = compilation.compile_hybrid_scene(
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        bond_objects=bonds,
        settings=_settings(
            graph_correction_mode="invalid_only",
            atom_decode_mode="structured_invalid_only",
        ),
        runtime=_runtime(
            correct_bonds=correct,
            apply_semantic_bond_types=semantic,
            structured_atom_candidates=lambda *args: (
                events.append("candidates") or candidates
            ),
        ),
    )
    assert assembled_bonds == [bonds, corrected]
    assert rerank_calls == [(1, candidates), (2, candidates)]
    assert result.scene["pass"] == 2
    assert result.scene["reranked"] is True
    assert result.diagnostics["graph_correction_triggered"] is True
    assert result.diagnostics["graph_correction_base_error"] == "base invalid"
    assert result.diagnostics["post_graph_semantic_chemical_type_changes"] == 2
    assert events == [
        "assemble_base",
        "candidates",
        "rerank_1",
        "compile",
        "correct",
        "semantic",
        "assemble_corrected",
        "rerank_2",
    ]


def test_type_only_correction_happens_before_explicit_assignment(monkeypatch) -> None:
    pixels, points, roles, labels, bonds = _components()
    corrected = [{**bonds[0], "junction_pair": (1, 0)}]
    received = {}

    def correct(*args, **kwargs):
        return corrected, {"graph_correction_mode": "type_only"}

    def fake_assemble(*args, **kwargs):
        received["bond_objects"] = args[4]
        received.update(kwargs)
        return {"junctions": []}, {}

    monkeypatch.setattr(compilation, "assemble_chemscene", fake_assemble)
    result = compilation.compile_hybrid_scene(
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        bond_objects=bonds,
        settings=_settings(
            graph_correction_mode="type_only",
            junction_source="semantic_relations",
        ),
        runtime=_runtime(correct_bonds=correct),
    )
    assert received["bond_objects"] is corrected
    assert received["junction_assignments"] == [(1, 0)]
    assert result.diagnostics["graph_correction_mode"] == "type_only"


def test_diagnostics_preserve_original_override_order(monkeypatch) -> None:
    pixels, points, roles, labels, bonds = _components()

    def correct(*args, **kwargs):
        return list(bonds), {"shared": "graph"}

    monkeypatch.setattr(
        compilation,
        "assemble_chemscene",
        lambda *args, **kwargs: ({"junctions": []}, {"shared": "assembly"}),
    )
    result = compilation.compile_hybrid_scene(
        pixels=pixels,
        points=points,
        roles=roles,
        labels=labels,
        bond_objects=bonds,
        settings=_settings(
            graph_correction_mode="full",
            junction_source="vertex",
        ),
        runtime=_runtime(correct_bonds=correct),
        component_diagnostics={"shared": "component", "junction_source": "wrong"},
    )
    assert result.diagnostics["shared"] == "component"
    assert result.diagnostics["junction_source"] == "vertex"
