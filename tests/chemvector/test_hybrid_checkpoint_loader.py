from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import chemvector.infer_chemscene_hybrid as hybrid
from chemvector.hybrid_checkpoint_loader import (
    HybridRuntimeComponents,
    validate_hybrid_checkpoint_topology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOADER_SOURCE = PROJECT_ROOT / "chemvector/inference/hybrid_checkpoint_loader.py"
HYBRID_SOURCE = PROJECT_ROOT / "chemvector/inference/infer_chemscene_hybrid.py"


def _topology(**overrides: Path | None) -> None:
    values = {
        "expert_bond_checkpoint": None,
        "style_gate_checkpoint": None,
        "bond_utility_router_checkpoint": None,
        "expert_ctc_ocr_checkpoint": None,
        "ocr_utility_router_checkpoint": None,
        "bond_set_selector_checkpoint": None,
        "endpoint_grouper_checkpoint": None,
        "line_band_checkpoint": None,
    }
    values.update(overrides)
    validate_hybrid_checkpoint_topology(**values)


def test_empty_and_complete_optional_checkpoint_groups_are_valid() -> None:
    _topology()
    _topology(
        expert_bond_checkpoint=Path("expert-bond.pt"),
        style_gate_checkpoint=Path("style-gate.pt"),
        bond_utility_router_checkpoint=Path("bond-router.pt"),
        expert_ctc_ocr_checkpoint=Path("expert-ocr.pt"),
        ocr_utility_router_checkpoint=Path("ocr-router.pt"),
        bond_set_selector_checkpoint=Path("selector.pt"),
        endpoint_grouper_checkpoint=Path("grouper.pt"),
        line_band_checkpoint=Path("line-band.pt"),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"expert_bond_checkpoint": Path("expert.pt")},
            "Expert BondObject checkpoint and style gate",
        ),
        (
            {"bond_utility_router_checkpoint": Path("router.pt")},
            "Bond utility router requires",
        ),
        (
            {"expert_ctc_ocr_checkpoint": Path("expert-ocr.pt")},
            "Expert CTC OCR and OCR utility router",
        ),
        (
            {"bond_set_selector_checkpoint": Path("selector.pt")},
            "Junction-aware inference requires",
        ),
    ),
)
def test_partial_checkpoint_groups_are_rejected_before_loading(
    overrides: dict[str, Path], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _topology(**overrides)


def test_runtime_schema_contains_every_pipeline_loaded_component() -> None:
    fields = set(HybridRuntimeComponents.__dataclass_fields__)
    assert {
        "base",
        "selector",
        "bond_detector",
        "semantic_bond_relation",
        "semantic_bond_existence",
        "assembler",
        "graph_corrector",
        "ctc_ocr",
        "expert_ctc_ocr",
        "bond_set_selector",
        "endpoint_grouper",
        "line_band_reader",
        "codec",
    }.issubset(fields)


def test_loader_does_not_import_the_pipeline_entrypoint() -> None:
    tree = ast.parse(LOADER_SOURCE.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "chemvector.infer_chemscene_hybrid" not in imports


def test_pipeline_constructor_is_only_configuration_and_binding() -> None:
    tree = ast.parse(HYBRID_SOURCE.read_text(encoding="utf-8"))
    pipeline = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChemSceneHybridPipeline"
    )
    constructor = next(
        node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert constructor.end_lineno - constructor.lineno + 1 < 120
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
        for node in ast.walk(constructor)
    )


def test_pipeline_binds_the_explicit_runtime_without_loading_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        name: object() for name in HybridRuntimeComponents.__dataclass_fields__
    }
    values["graph_corrector_keep_threshold"] = 0.3
    values["graph_corrector_uses_elements"] = False
    runtime = HybridRuntimeComponents(**values)
    received = {}

    def fake_loader(**kwargs):
        received.update(kwargs)
        return runtime

    monkeypatch.setattr(hybrid, "load_hybrid_runtime_components", fake_loader)
    pipeline = hybrid.ChemSceneHybridPipeline(
        Path("selector.pt"),
        Path("bond.pt"),
        torch.device("cpu"),
        text_interior_bond_veto=True,
    )
    assert pipeline.bond_detector is runtime.bond_detector
    assert pipeline.codec is runtime.codec
    assert pipeline.text_interior_bond_veto is True
    assert pipeline.component_diagnostics == {}
    assert received["selector_checkpoint"] == Path("selector.pt")
    assert received["bond_checkpoint"] == Path("bond.pt")
