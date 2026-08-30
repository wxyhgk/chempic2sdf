"""Load the checkpoint graph used by ChemScene hybrid inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chemvector.bond_utility_router import BondUtilityRouter
from chemvector.chemscene_graph_corrector import (
    ChemSceneGraphCorrector,
    ChemSceneGraphCorrectorConfig,
)
from chemvector.ctc_ocr import PretrainedPixelCTCConfig, PretrainedPixelCTCOCR
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
)
from chemvector.evaluate_group_selector_bond import (
    load_evaluation_selector,
    selector_candidate_protocol,
)
from chemvector.junction_bond_assembler import (
    JunctionBondAssembler,
    JunctionBondAssemblerConfig,
)
from chemvector.junction_aware_bond_selector import (
    load_bond_set_selector,
    load_endpoint_grouper,
)
from chemvector.ocr import OCRTextCodec
from chemvector.ocr_utility_router import OCRUtilityRouter
from chemvector.pretrain_bond_set_selector import load_line_band_reader
from chemvector.pretrain_dual_vertex_selector import load_dual_pipeline
from chemvector.semantic_bond_relation import (
    SemanticBondExistence,
    SemanticBondRelation,
    SemanticBondRelationConfig,
)
from chemvector.structure_gate import StructureStyleGate
from chemvector.text_object import TextObjectBBoxHead


@dataclass(frozen=True, slots=True)
class HybridRuntimeComponents:
    semantic_bond_relation: Any
    semantic_bond_existence: Any
    selector: Any
    selector_config: Any
    base: Any
    query_detector: Any
    grounded_selector: Any
    graph: Any
    text_head: Any
    implicit_head: Any
    protocol: Any
    bond_detector: Any
    expert_bond_detector: Any
    style_gate: Any
    bond_utility_router: Any
    bond_utility_feature_names: Any
    bond_utility_feature_mean: Any
    bond_utility_feature_std: Any
    bond_utility_threshold: Any
    assembler: Any
    graph_corrector: Any
    graph_corrector_keep_threshold: float
    graph_corrector_uses_elements: bool
    ctc_ocr: Any
    expert_ctc_ocr: Any
    ocr_utility_router: Any
    ocr_utility_feature_names: Any
    ocr_utility_feature_mean: Any
    ocr_utility_feature_std: Any
    codec: OCRTextCodec
    bond_set_selector: Any
    bond_set_selector_payload: Any
    endpoint_grouper: Any
    endpoint_grouper_payload: Any
    line_band_reader: Any


def validate_hybrid_checkpoint_topology(
    *,
    expert_bond_checkpoint: Path | None,
    style_gate_checkpoint: Path | None,
    bond_utility_router_checkpoint: Path | None,
    expert_ctc_ocr_checkpoint: Path | None,
    ocr_utility_router_checkpoint: Path | None,
    bond_set_selector_checkpoint: Path | None,
    endpoint_grouper_checkpoint: Path | None,
    line_band_checkpoint: Path | None,
) -> None:
    if bool(expert_bond_checkpoint) != bool(style_gate_checkpoint):
        raise ValueError(
            "Expert BondObject checkpoint and style gate must be used together"
        )
    if bond_utility_router_checkpoint is not None and expert_bond_checkpoint is None:
        raise ValueError("Bond utility router requires an expert BondObject checkpoint")
    if bool(expert_ctc_ocr_checkpoint) != bool(ocr_utility_router_checkpoint):
        raise ValueError("Expert CTC OCR and OCR utility router must be used together")
    junction_aware_paths = (
        bond_set_selector_checkpoint,
        endpoint_grouper_checkpoint,
        line_band_checkpoint,
    )
    if any(path is not None for path in junction_aware_paths) and not all(
        path is not None for path in junction_aware_paths
    ):
        raise ValueError(
            "Junction-aware inference requires selector, grouper, and line-band checkpoints"
        )


def _load_semantic_models(
    relation_checkpoint: Path | None,
    existence_checkpoint: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    existence = None
    if existence_checkpoint is not None:
        payload = torch.load(existence_checkpoint, map_location="cpu", weights_only=False)
        config = SemanticBondRelationConfig(**payload["config"])
        existence = SemanticBondExistence(config).to(device)
        existence.load_state_dict(payload["semantic_bond_existence"])
        existence.eval().requires_grad_(False)
    relation = None
    if relation_checkpoint is not None:
        payload = torch.load(relation_checkpoint, map_location="cpu", weights_only=False)
        config = SemanticBondRelationConfig(**payload["config"])
        relation = SemanticBondRelation(config).to(device)
        relation.load_state_dict(payload["semantic_bond_relation"])
        relation.eval().requires_grad_(False)
    return {
        "semantic_bond_existence": existence,
        "semantic_bond_relation": relation,
    }


def _load_selector_stack(
    selector_checkpoint: Path,
    text_bbox_checkpoint: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    selector, selector_config, payload, _ = load_evaluation_selector(
        str(selector_checkpoint), device
    )
    (
        base,
        query_detector,
        grounded_selector,
        graph,
        text_head,
        implicit_head,
    ) = load_dual_pipeline(
        payload["text_object_checkpoint"],
        payload["implicit_vertex_checkpoint"],
        device,
        maximum_candidates=selector_config.maximum_candidates,
    )
    if text_bbox_checkpoint is not None:
        text_bbox_payload = torch.load(
            text_bbox_checkpoint, map_location="cpu", weights_only=False
        )
        if text_bbox_payload.get("head_type") != "bbox":
            raise ValueError("Text bbox checkpoint does not contain a bbox head")
        text_head = TextObjectBBoxHead(text_bbox_payload["input_channels"]).to(device)
        text_head.load_state_dict(text_bbox_payload["text_object_head"])
        text_head = text_head.eval().requires_grad_(False)
    return {
        "base": base,
        "graph": graph,
        "grounded_selector": grounded_selector,
        "implicit_head": implicit_head,
        "protocol": selector_candidate_protocol(payload),
        "query_detector": query_detector,
        "selector": selector,
        "selector_config": selector_config,
        "text_head": text_head,
    }


def _bond_detector(payload: dict, device: torch.device):
    config = DenseBondObjectConfig(**payload["config"])
    state = payload["dense_bond_object"]
    detector = (
        MultiScaleDenseBondObjectDetector(config).to(device)
        if any(key.startswith("base.") for key in state)
        else DenseBondObjectDetector(config).to(device)
    )
    detector.load_state_dict(state)
    return detector.eval().requires_grad_(False)


def _load_bond_stack(
    *,
    bond_checkpoint: Path,
    expert_bond_checkpoint: Path | None,
    style_gate_checkpoint: Path | None,
    style_gate_threshold: float,
    bond_utility_router_checkpoint: Path | None,
    bond_utility_break_budget: str,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(bond_checkpoint, map_location="cpu", weights_only=False)
    detector = _bond_detector(payload, device)
    result = {
        "bond_detector": detector,
        "bond_utility_feature_mean": None,
        "bond_utility_feature_names": None,
        "bond_utility_feature_std": None,
        "bond_utility_router": None,
        "bond_utility_threshold": None,
        "expert_bond_detector": None,
        "style_gate": None,
    }
    if expert_bond_checkpoint is None:
        return result
    if not 0.0 <= style_gate_threshold <= 1.0:
        raise ValueError("Style gate threshold must be in [0, 1]")
    expert_payload = torch.load(
        expert_bond_checkpoint, map_location="cpu", weights_only=False
    )
    if Path(expert_payload["grounded_graph_checkpoint"]).resolve() != Path(
        payload["grounded_graph_checkpoint"]
    ).resolve():
        raise ValueError("Base and style expert use different grounded vision checkpoints")
    expert = _bond_detector(expert_payload, device)
    if type(expert) is not type(detector):
        raise ValueError("Base and style expert BondObject types differ")
    gate_payload = torch.load(style_gate_checkpoint, map_location="cpu", weights_only=False)
    if Path(gate_payload["base_bond_checkpoint"]).resolve() != bond_checkpoint.resolve():
        raise ValueError("Style gate base BondObject checkpoint does not match")
    if Path(gate_payload["expert_bond_checkpoint"]).resolve() != expert_bond_checkpoint.resolve():
        raise ValueError("Style gate expert BondObject checkpoint does not match")
    gate = StructureStyleGate(
        gate_payload["input_channels"], hidden_dim=gate_payload["hidden_dim"]
    ).to(device)
    gate.load_state_dict(gate_payload["gate"])
    result["expert_bond_detector"] = expert
    result["style_gate"] = gate.eval().requires_grad_(False)
    if bond_utility_router_checkpoint is None:
        return result
    utility = torch.load(
        bond_utility_router_checkpoint, map_location="cpu", weights_only=False
    )
    if bond_utility_break_budget not in utility["validation_frontier"]:
        raise ValueError("Unknown Bond utility break budget")
    router = BondUtilityRouter(utility["input_dim"], utility["hidden_dim"]).to(device)
    router.load_state_dict(utility["router"])
    result.update(
        {
            "bond_utility_feature_mean": utility["feature_mean"].to(device),
            "bond_utility_feature_names": utility["feature_names"],
            "bond_utility_feature_std": utility["feature_std"].to(device),
            "bond_utility_router": router.eval().requires_grad_(False),
            "bond_utility_threshold": float(
                utility["validation_frontier"][bond_utility_break_budget]["threshold"]
            ),
        }
    )
    return result


def _load_graph_stack(
    assembler_checkpoint: Path | None,
    graph_corrector_checkpoint: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    assembler = None
    if assembler_checkpoint is not None:
        payload = torch.load(assembler_checkpoint, map_location="cpu", weights_only=False)
        config = JunctionBondAssemblerConfig(**payload["config"])
        assembler = JunctionBondAssembler(config).to(device)
        assembler.load_state_dict(payload["junction_bond_assembler"])
        assembler.eval().requires_grad_(False)
    corrector = None
    keep_threshold = 0.3
    uses_elements = False
    if graph_corrector_checkpoint is not None:
        payload = torch.load(
            graph_corrector_checkpoint, map_location="cpu", weights_only=False
        )
        config = ChemSceneGraphCorrectorConfig(**payload["config"])
        corrector = ChemSceneGraphCorrector(config).to(device)
        corrector.load_state_dict(payload["chemscene_graph_corrector"])
        corrector.eval().requires_grad_(False)
        keep_threshold = float(payload.get("validation", {}).get("keep_threshold", 0.3))
        uses_elements = bool(
            payload.get("protocol", {}).get("include_ocr_elements", False)
        )
    return {
        "assembler": assembler,
        "graph_corrector": corrector,
        "graph_corrector_keep_threshold": keep_threshold,
        "graph_corrector_uses_elements": uses_elements,
    }


def _load_ctc_model(path: Path, device: torch.device) -> PretrainedPixelCTCOCR:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = PretrainedPixelCTCConfig(**payload["config"])
    model = PretrainedPixelCTCOCR(config).to(device)
    model.load_state_dict(payload["pretrained_ctc_ocr"])
    model.pixel_encoder_loaded = True
    return model.eval().requires_grad_(False)


def load_compatible_fallback_ctc_ocr(
    checkpoint: str | Path,
    primary: PretrainedPixelCTCOCR,
    device: torch.device,
) -> PretrainedPixelCTCOCR:
    """Load the invalid-only OCR fallback against the already-loaded primary model."""

    model = _load_ctc_model(Path(checkpoint), device)
    if model.config != primary.config:
        raise ValueError("Primary and fallback CTC OCR configs differ")
    return model


def _load_ocr_stack(
    *,
    base: Any,
    ocr_checkpoint: Path | None,
    ctc_ocr_checkpoint: Path | None,
    expert_ctc_ocr_checkpoint: Path | None,
    ocr_utility_router_checkpoint: Path | None,
    ocr_utility_router_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    if ocr_checkpoint is not None:
        payload = torch.load(ocr_checkpoint, map_location="cpu", weights_only=False)
        prefix = "ocr_head."
        state = {
            key[len(prefix) :]: value
            for key, value in payload["model"].items()
            if key.startswith(prefix)
        }
        base.ocr_head.load_state_dict(state, strict=True)
        base.ocr_head.eval().requires_grad_(False)
    primary = (
        _load_ctc_model(ctc_ocr_checkpoint, device)
        if ctc_ocr_checkpoint is not None
        else None
    )
    result = {
        "ctc_ocr": primary,
        "expert_ctc_ocr": None,
        "ocr_utility_feature_mean": None,
        "ocr_utility_feature_names": None,
        "ocr_utility_feature_std": None,
        "ocr_utility_router": None,
    }
    if expert_ctc_ocr_checkpoint is None:
        return result
    if primary is None:
        raise ValueError("Expert CTC OCR requires a primary CTC checkpoint")
    if not 0.0 <= ocr_utility_router_threshold <= 1.0:
        raise ValueError("OCR utility router threshold must lie in [0, 1]")
    expert = _load_ctc_model(expert_ctc_ocr_checkpoint, device)
    if expert.config != primary.config:
        raise ValueError("Primary and expert CTC OCR configs differ")
    payload = torch.load(
        ocr_utility_router_checkpoint, map_location="cpu", weights_only=False
    )
    if payload.get("checkpoint_schema_version") not in {
        "ocr-utility-router-0.1",
        "ocr-utility-router-0.2",
    }:
        raise ValueError("OCR utility router checkpoint schema differs")
    router = OCRUtilityRouter(payload["input_dim"], payload["hidden_dim"]).to(device)
    router.load_state_dict(payload["router"])
    result.update(
        {
            "expert_ctc_ocr": expert,
            "ocr_utility_feature_mean": payload["feature_mean"].to(device),
            "ocr_utility_feature_names": payload["feature_names"],
            "ocr_utility_feature_std": payload["feature_std"].to(device),
            "ocr_utility_router": router.eval().requires_grad_(False),
        }
    )
    return result


def _load_junction_aware_stack(
    selector_checkpoint: Path | None,
    grouper_checkpoint: Path | None,
    line_band_checkpoint: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    result = {
        "bond_set_selector": None,
        "bond_set_selector_payload": None,
        "endpoint_grouper": None,
        "endpoint_grouper_payload": None,
        "line_band_reader": None,
    }
    if selector_checkpoint is None:
        return result
    result["bond_set_selector"], result["bond_set_selector_payload"] = (
        load_bond_set_selector(selector_checkpoint, device)
    )
    result["endpoint_grouper"], result["endpoint_grouper_payload"] = (
        load_endpoint_grouper(grouper_checkpoint, device)
    )
    result["line_band_reader"] = load_line_band_reader(line_band_checkpoint, device)
    return result


def load_hybrid_runtime_components(
    *,
    selector_checkpoint: Path,
    bond_checkpoint: Path,
    device: torch.device,
    assembler_checkpoint: Path | None = None,
    graph_corrector_checkpoint: Path | None = None,
    ocr_checkpoint: Path | None = None,
    ctc_ocr_checkpoint: Path | None = None,
    text_bbox_checkpoint: Path | None = None,
    bond_set_selector_checkpoint: Path | None = None,
    endpoint_grouper_checkpoint: Path | None = None,
    line_band_checkpoint: Path | None = None,
    semantic_bond_relation_checkpoint: Path | None = None,
    semantic_bond_existence_checkpoint: Path | None = None,
    expert_bond_checkpoint: Path | None = None,
    style_gate_checkpoint: Path | None = None,
    style_gate_threshold: float = 0.90,
    bond_utility_router_checkpoint: Path | None = None,
    bond_utility_break_budget: str = "0",
    expert_ctc_ocr_checkpoint: Path | None = None,
    ocr_utility_router_checkpoint: Path | None = None,
    ocr_utility_router_threshold: float = 0.95,
) -> HybridRuntimeComponents:
    validate_hybrid_checkpoint_topology(
        expert_bond_checkpoint=expert_bond_checkpoint,
        style_gate_checkpoint=style_gate_checkpoint,
        bond_utility_router_checkpoint=bond_utility_router_checkpoint,
        expert_ctc_ocr_checkpoint=expert_ctc_ocr_checkpoint,
        ocr_utility_router_checkpoint=ocr_utility_router_checkpoint,
        bond_set_selector_checkpoint=bond_set_selector_checkpoint,
        endpoint_grouper_checkpoint=endpoint_grouper_checkpoint,
        line_band_checkpoint=line_band_checkpoint,
    )
    values: dict[str, Any] = {}
    values.update(
        _load_semantic_models(
            semantic_bond_relation_checkpoint,
            semantic_bond_existence_checkpoint,
            device,
        )
    )
    values.update(_load_selector_stack(selector_checkpoint, text_bbox_checkpoint, device))
    values.update(
        _load_bond_stack(
            bond_checkpoint=bond_checkpoint,
            expert_bond_checkpoint=expert_bond_checkpoint,
            style_gate_checkpoint=style_gate_checkpoint,
            style_gate_threshold=style_gate_threshold,
            bond_utility_router_checkpoint=bond_utility_router_checkpoint,
            bond_utility_break_budget=bond_utility_break_budget,
            device=device,
        )
    )
    values.update(_load_graph_stack(assembler_checkpoint, graph_corrector_checkpoint, device))
    values.update(
        _load_ocr_stack(
            base=values["base"],
            ocr_checkpoint=ocr_checkpoint,
            ctc_ocr_checkpoint=ctc_ocr_checkpoint,
            expert_ctc_ocr_checkpoint=expert_ctc_ocr_checkpoint,
            ocr_utility_router_checkpoint=ocr_utility_router_checkpoint,
            ocr_utility_router_threshold=ocr_utility_router_threshold,
            device=device,
        )
    )
    values["codec"] = OCRTextCodec(max_length=values["base"].config.ocr_max_length)
    values.update(
        _load_junction_aware_stack(
            bond_set_selector_checkpoint,
            endpoint_grouper_checkpoint,
            line_band_checkpoint,
            device,
        )
    )
    return HybridRuntimeComponents(**values)


def bind_hybrid_runtime(target: Any, runtime: HybridRuntimeComponents) -> None:
    """Bind the explicit runtime schema without copying torch modules."""

    for field_name in runtime.__dataclass_fields__:
        setattr(target, field_name, getattr(runtime, field_name))
