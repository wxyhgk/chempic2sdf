from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

import chemvector.chemscene_assembly as assembly
import chemvector.direct_scene_inference as direct_scene
import chemvector.graph_prediction_correction as correction
import chemvector.infer_chemscene_hybrid as hybrid
import chemvector.infer_direct_atom_pair_bond as direct_entry
import chemvector.hybrid_atom_decoding as atom_decoding
import chemvector.hybrid_bond_routing as bond_routing
import chemvector.hybrid_checkpoint_loader as checkpoint_loader
import chemvector.hybrid_junction_selection as junction_selection
import chemvector.hybrid_ocr_inference as ocr_inference
import chemvector.junction_text_attachment as attachment
import chemvector.text_interior_bond_veto as veto


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYBRID_SOURCE = PROJECT_ROOT / "chemvector/inference/infer_chemscene_hybrid.py"
DIRECT_SOURCE = PROJECT_ROOT / "chemvector/inference/infer_direct_atom_pair_bond.py"
COMPONENT_SOURCES = (
    PROJECT_ROOT / "chemvector/components/graph/joint/chemscene_assembly.py",
    PROJECT_ROOT / "chemvector/components/graph/joint/junction_text_attachment.py",
    PROJECT_ROOT / "chemvector/components/graph/correction/graph_prediction_correction.py",
    PROJECT_ROOT / "chemvector/components/bonds/selection/text_interior_bond_veto.py",
    PROJECT_ROOT / "chemvector/direct_scene_inference.py",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_legacy_hybrid_exports_are_the_component_functions() -> None:
    assert hybrid.assemble_chemscene is assembly.assemble_chemscene
    assert (
        hybrid.compile_scene_with_aromatic_fallback
        is assembly.compile_scene_with_aromatic_fallback
    )
    assert hybrid.repair_impossible_explicit_h is assembly.repair_impossible_explicit_h
    assert hybrid.apply_graph_correction is correction.apply_graph_correction
    assert hybrid.attach_text_to_bond_junctions is attachment.attach_text_to_bond_junctions
    assert (
        hybrid.components_from_external_junctions
        is attachment.components_from_external_junctions
    )
    assert (
        hybrid.suppress_text_interior_bond_objects
        is veto.suppress_text_interior_bond_objects
    )
    assert (
        hybrid.load_compatible_fallback_ctc_ocr
        is checkpoint_loader.load_compatible_fallback_ctc_ocr
    )


def test_hybrid_entrypoint_no_longer_defines_component_algorithms() -> None:
    tree = ast.parse(HYBRID_SOURCE.read_text(encoding="utf-8"))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert definitions.isdisjoint(
        {
            "apply_graph_correction",
            "assemble_chemscene",
            "attach_text_to_bond_junctions",
            "compile_scene_with_aromatic_fallback",
            "components_from_external_junctions",
            "repair_impossible_explicit_h",
            "suppress_text_interior_bond_objects",
            "load_compatible_fallback_ctc_ocr",
        }
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
        for node in ast.walk(tree)
    )


def test_direct_entry_exports_are_the_component_functions() -> None:
    assert direct_entry.load_direct_model is direct_scene.load_direct_model
    assert (
        direct_entry.load_action_ranking_model is direct_scene.load_action_ranking_model
    )
    assert direct_entry.apply_atom_ocr is direct_scene.apply_atom_ocr
    assert (
        direct_entry.apply_structural_twin_suppression
        is direct_scene.apply_structural_twin_suppression
    )
    assert direct_entry.direct_scene_tensors is direct_scene.direct_scene_tensors


def test_direct_entrypoint_no_longer_defines_component_algorithms() -> None:
    tree = ast.parse(DIRECT_SOURCE.read_text(encoding="utf-8"))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert definitions.isdisjoint(
        {
            "_q",
            "apply_atom_ocr",
            "apply_structural_twin_suppression",
            "direct_scene_tensors",
            "load_action_ranking_model",
            "load_direct_model",
        }
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
        for node in ast.walk(tree)
    )


def test_extracted_components_do_not_depend_on_entrypoint_layers() -> None:
    forbidden_prefixes = (
        "chemvector.inference",
        "chemvector.training",
        "chemvector.evaluation",
        "chemvector.tools",
        "chemvector.workflows",
    )
    for path in COMPONENT_SOURCES:
        imports = _imports(path)
        assert not {
            module
            for module in imports
            if module.startswith(forbidden_prefixes)
        }, path


def test_junction_selection_does_not_depend_on_ocr_or_pipeline() -> None:
    source = (
        PROJECT_ROOT / "chemvector/inference/hybrid_junction_selection.py"
    )
    imports = _imports(source)
    assert "chemvector.infer_chemscene_hybrid" not in imports
    assert not {
        module
        for module in imports
        if module in {"chemvector.ctc_ocr", "chemvector.ocr_utility_router"}
    }


def test_hybrid_ocr_does_not_depend_on_pipeline_or_checkpoint_loading() -> None:
    source = PROJECT_ROOT / "chemvector/inference/hybrid_ocr_inference.py"
    imports = _imports(source)
    assert "chemvector.infer_chemscene_hybrid" not in imports
    assert "chemvector.hybrid_checkpoint_loader" not in imports


def test_hybrid_bond_routing_does_not_depend_on_pipeline_or_checkpoints() -> None:
    source = PROJECT_ROOT / "chemvector/inference/hybrid_bond_routing.py"
    imports = _imports(source)
    assert "chemvector.infer_chemscene_hybrid" not in imports
    assert "chemvector.hybrid_checkpoint_loader" not in imports


def test_hybrid_atom_decoding_does_not_depend_on_pipeline_or_checkpoints() -> None:
    source = PROJECT_ROOT / "chemvector/inference/hybrid_atom_decoding.py"
    imports = _imports(source)
    assert "chemvector.infer_chemscene_hybrid" not in imports
    assert "chemvector.hybrid_checkpoint_loader" not in imports


def test_hybrid_graph_stages_do_not_depend_on_pipeline_or_checkpoints() -> None:
    for name in ("hybrid_graph_refinement.py", "hybrid_junction_bonds.py"):
        imports = _imports(PROJECT_ROOT / "chemvector/inference" / name)
        assert "chemvector.infer_chemscene_hybrid" not in imports
        assert "chemvector.hybrid_checkpoint_loader" not in imports


def test_high_level_inference_stages_do_not_depend_on_pipeline_or_checkpoints() -> None:
    names = (
        "hybrid_component_routing.py",
        "hybrid_scene_compilation.py",
        "hybrid_cli_execution.py",
    )
    for name in names:
        imports = _imports(PROJECT_ROOT / "chemvector/inference" / name)
        assert "chemvector.infer_chemscene_hybrid" not in imports
        assert "chemvector.hybrid_checkpoint_loader" not in imports


def test_pipeline_public_orchestration_methods_remain_thin() -> None:
    tree = ast.parse(HYBRID_SOURCE.read_text(encoding="utf-8"))
    pipeline = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChemSceneHybridPipeline"
    )
    methods = {
        node.name: node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef)
    }
    for name, maximum_lines in {
        "predict_components": 55,
        "predict_components_batch": 45,
        "predict": 75,
    }.items():
        node = methods[name]
        assert node.end_lineno - node.lineno + 1 <= maximum_lines


def test_pipeline_graph_stage_methods_remain_thin_delegates() -> None:
    tree = ast.parse(HYBRID_SOURCE.read_text(encoding="utf-8"))
    pipeline = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChemSceneHybridPipeline"
    )
    methods = {
        node.name: node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef)
    }
    limits = {
        "_correct_bond_objects": 32,
        "_apply_semantic_bond_types": 18,
        "_junction_aware_bonds": 38,
        "_semantic_relation_bonds": 18,
    }
    for name, maximum_lines in limits.items():
        node = methods[name]
        assert node.end_lineno - node.lineno + 1 <= maximum_lines


def test_pipeline_junction_method_delegates_structure_selection(
    monkeypatch,
) -> None:
    selected = junction_selection.JunctionBatchSelection(
        text_batches=[[]],
        point_batches=[torch.tensor(((0.2, 0.3), (0.8, 0.7)))],
        role_batches=[torch.tensor((0, 0))],
        padded_points=torch.tensor((((0.2, 0.3), (0.8, 0.7)),)),
        point_mask=torch.tensor(((True, True),)),
    )
    received = {}

    def fake_selection(**kwargs):
        received.update(kwargs)
        return selected

    monkeypatch.setattr(hybrid, "select_junction_batches", fake_selection)
    pipeline = object.__new__(hybrid.ChemSceneHybridPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.query_detector = object()
    pipeline.text_head = object()
    pipeline.implicit_head = object()
    pipeline.protocol = {}
    pipeline.selector_config = object()
    pipeline.selector = object()
    pipeline.grounded_selector = object()
    pipeline.graph = object()
    pipeline.base = object()
    pipeline.codec = object()
    pipeline.ctc_ocr = None
    pipeline.expert_ctc_ocr = None
    pipeline.ocr_utility_router = None
    pipeline.ocr_utility_feature_names = None
    pipeline.ocr_utility_feature_mean = None
    pipeline.ocr_utility_feature_std = None
    pipeline.ocr_utility_router_threshold = 0.95
    pipeline.ctc_anchor_fallback_margin = None
    pipeline.component_diagnostics = {}
    output = pipeline._junctions_batch(
        torch.zeros((1, 3, 32, 32)),
        torch.zeros((1, 8, 8, 8)),
        real_threshold=0.2,
        bbox_expansion_pixels=2.0,
        include_ocr=False,
    )
    assert output[0][2] == ["C", "C"]
    assert received["real_threshold"] == 0.2


def test_pipeline_junction_method_delegates_ocr_labeling(monkeypatch) -> None:
    selected = junction_selection.JunctionBatchSelection(
        text_batches=[[]],
        point_batches=[torch.tensor(((0.2, 0.3),))],
        role_batches=[torch.tensor((0,))],
        padded_points=torch.tensor((((0.2, 0.3),),)),
        point_mask=torch.tensor(((True,),)),
    )
    result = [(selected.point_batches[0], selected.role_batches[0], ["C"])]
    received = {}

    monkeypatch.setattr(hybrid, "select_junction_batches", lambda **_: selected)

    def fake_decode(**kwargs):
        received.update(kwargs)
        return result, {"ocr_test": 1}

    monkeypatch.setattr(hybrid, "decode_junction_labels", fake_decode)
    pipeline = object.__new__(hybrid.ChemSceneHybridPipeline)
    for name in (
        "query_detector",
        "text_head",
        "implicit_head",
        "protocol",
        "selector_config",
        "selector",
        "grounded_selector",
        "graph",
        "base",
        "codec",
        "ctc_ocr",
        "expert_ctc_ocr",
        "ocr_utility_router",
        "ocr_utility_feature_names",
        "ocr_utility_feature_mean",
        "ocr_utility_feature_std",
        "ctc_anchor_fallback_margin",
    ):
        setattr(pipeline, name, None)
    pipeline.device = torch.device("cpu")
    pipeline.ocr_utility_router_threshold = 0.95
    pipeline.component_diagnostics = {}
    output = pipeline._junctions_batch(
        torch.zeros((1, 3, 32, 32)),
        torch.zeros((1, 8, 8, 8)),
        real_threshold=0.2,
        bbox_expansion_pixels=2.0,
    )
    assert output is result
    assert isinstance(received["runtime"], ocr_inference.HybridOCRRuntime)
    assert received["selection"] is selected
    assert pipeline.component_diagnostics == {"ocr_test": 1}


def test_pipeline_bond_method_delegates_routing(monkeypatch) -> None:
    received = {}

    def fake_route(stride4, **kwargs):
        received["stride4"] = stride4
        received.update(kwargs)
        return [[{"score": 0.8}]], {"bond_test": 1}

    monkeypatch.setattr(hybrid, "decode_routed_bond_objects", fake_route)
    pipeline = object.__new__(hybrid.ChemSceneHybridPipeline)
    for name in (
        "base",
        "bond_detector",
        "expert_bond_detector",
        "style_gate",
        "bond_utility_router",
        "bond_utility_feature_names",
        "bond_utility_feature_mean",
        "bond_utility_feature_std",
        "bond_utility_threshold",
    ):
        setattr(pipeline, name, None)
    pipeline.device = torch.device("cpu")
    pipeline.style_gate_threshold = 0.9
    pipeline.component_diagnostics = {}
    stride4 = torch.zeros((1, 4, 8, 8))
    output = pipeline._decode_routed_bond_objects(stride4, threshold=0.25)
    assert output == [[{"score": 0.8}]]
    assert received["stride4"] is stride4
    assert isinstance(received["runtime"], bond_routing.HybridBondRoutingRuntime)
    assert pipeline.component_diagnostics == {"bond_test": 1}


def test_pipeline_structured_atom_method_delegates_decoding(monkeypatch) -> None:
    received = {}
    expected = [[object()]]

    def fake_candidates(pixels, points, roles, labels, **kwargs):
        received.update(
            pixels=pixels,
            points=points,
            roles=roles,
            labels=labels,
            **kwargs,
        )
        return expected

    monkeypatch.setattr(hybrid, "structured_atom_candidates", fake_candidates)
    pipeline = object.__new__(hybrid.ChemSceneHybridPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.base = object()
    pipeline.text_head = object()
    pipeline.protocol = object()
    pipeline.selector_config = SimpleNamespace(maximum_candidates=32)
    pipeline.ctc_ocr = object()
    pixels = torch.zeros((1, 3, 32, 32))
    points = torch.tensor(((0.5, 0.5),))
    roles = torch.tensor((0,))
    output = pipeline._structured_atom_candidates(pixels, points, roles, ["C"])
    assert output is expected
    assert isinstance(received["runtime"], atom_decoding.HybridAtomDecodingRuntime)
    assert received["runtime"].maximum_text_objects == 32
