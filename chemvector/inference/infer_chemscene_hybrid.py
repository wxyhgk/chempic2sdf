#!/usr/bin/env python3
"""Run the v0.8 ChemScene pipeline with predicted junctions, OCR, and bonds."""

from __future__ import annotations

from pathlib import Path
import torch

from chemvector.chemscene_assembly import (
    assemble_chemscene,
    compile_scene_with_aromatic_fallback,
    repair_impossible_explicit_h,
)
from chemvector.dual_vertex_selector import IMPLICIT_ROLE, TEXT_ROLE
from chemvector.graph_prediction_correction import apply_graph_correction
from chemvector.hybrid_checkpoint_loader import (
    bind_hybrid_runtime,
    load_compatible_fallback_ctc_ocr,
    load_hybrid_runtime_components,
)
from chemvector.hybrid_bond_routing import (
    HybridBondRoutingRuntime,
    decode_routed_bond_objects,
)
from chemvector.hybrid_atom_decoding import (
    HybridAtomDecodingRuntime,
    structured_atom_candidates,
)
from chemvector.hybrid_cli import build_hybrid_parser
from chemvector.hybrid_cli_execution import (
    load_hybrid_cli_runtime,
    predict_hybrid_cli,
    prepare_hybrid_cli_input,
    write_hybrid_cli_prediction,
)
from chemvector.hybrid_component_routing import (
    HybridComponentRoutingRuntime,
    route_component_batches,
    route_single_components,
)
from chemvector.hybrid_graph_refinement import (
    GraphCorrectionRuntime,
    SemanticBondRuntime,
    apply_semantic_bond_types,
    correct_bond_objects,
    semantic_relation_bonds,
)
from chemvector.hybrid_junction_bonds import (
    HybridJunctionBondRuntime,
    junction_aware_bonds,
)
from chemvector.hybrid_junction_selection import select_junction_batches
from chemvector.hybrid_ocr_inference import (
    HybridOCRRuntime,
    decode_junction_labels,
)
from chemvector.hybrid_scene_compilation import (
    HybridSceneCompilationRuntime,
    HybridSceneCompilationSettings,
    compile_hybrid_scene,
)
from chemvector.junction_text_attachment import (
    attach_text_to_bond_junctions,
    components_from_external_junctions,
)
from chemvector.text_object import decode_text_objects
from chemvector.text_interior_bond_veto import suppress_text_interior_bond_objects


class ChemSceneHybridPipeline:
    def __init__(
        self,
        selector_checkpoint: Path,
        bond_checkpoint: Path,
        device: torch.device,
        assembler_checkpoint: Path | None = None,
        graph_corrector_checkpoint: Path | None = None,
        ocr_checkpoint: Path | None = None,
        ctc_ocr_checkpoint: Path | None = None,
        text_bbox_checkpoint: Path | None = None,
        ctc_anchor_fallback_margin: float | None = None,
        bond_set_selector_checkpoint: Path | None = None,
        endpoint_grouper_checkpoint: Path | None = None,
        line_band_checkpoint: Path | None = None,
        junction_projection_score_source: str = "detector",
        junction_bond_type_source: str = "detector",
        semantic_bond_relation_checkpoint: Path | None = None,
        semantic_bond_relation_mode: str = "chemical_all",
        semantic_aromatic_mean_threshold: float = 0.72,
        semantic_aromatic_minimum_threshold: float = 0.25,
        semantic_bond_existence_checkpoint: Path | None = None,
        semantic_bond_existence_threshold: float = 0.60,
        semantic_bond_existence_chunk_size: int = 256,
        expert_bond_checkpoint: Path | None = None,
        style_gate_checkpoint: Path | None = None,
        style_gate_threshold: float = 0.90,
        bond_utility_router_checkpoint: Path | None = None,
        bond_utility_break_budget: str = "0",
        text_interior_bond_veto: bool = False,
        expert_ctc_ocr_checkpoint: Path | None = None,
        ocr_utility_router_checkpoint: Path | None = None,
        ocr_utility_router_threshold: float = 0.95,
    ) -> None:
        if junction_projection_score_source not in {"detector", "selector"}:
            raise ValueError(
                "Junction projection score source must be detector or selector"
            )
        if junction_bond_type_source not in {"detector", "line_band"}:
            raise ValueError("Junction Bond type source must be detector or line_band")
        if semantic_bond_relation_mode not in {"chemical_all", "aromatic_ring_only"}:
            raise ValueError("Unknown Semantic Bond relation mode")
        if not 0.0 < semantic_bond_existence_threshold < 1.0:
            raise ValueError("Semantic Bond existence threshold must lie in (0, 1)")
        if semantic_bond_existence_chunk_size < 1:
            raise ValueError("Semantic Bond existence chunk size must be positive")

        self.device = device
        self.text_interior_bond_veto = text_interior_bond_veto
        self.junction_projection_score_source = junction_projection_score_source
        self.junction_bond_type_source = junction_bond_type_source
        self.semantic_bond_relation_checkpoint = semantic_bond_relation_checkpoint
        self.semantic_bond_relation_mode = semantic_bond_relation_mode
        self.semantic_aromatic_mean_threshold = semantic_aromatic_mean_threshold
        self.semantic_aromatic_minimum_threshold = semantic_aromatic_minimum_threshold
        self.semantic_bond_existence_checkpoint = semantic_bond_existence_checkpoint
        self.semantic_bond_existence_threshold = semantic_bond_existence_threshold
        self.semantic_bond_existence_chunk_size = semantic_bond_existence_chunk_size
        self.text_bbox_checkpoint = text_bbox_checkpoint
        self.style_gate_threshold = style_gate_threshold
        self.assembler_checkpoint = assembler_checkpoint
        self.graph_corrector_checkpoint = graph_corrector_checkpoint
        self.ocr_checkpoint = ocr_checkpoint
        self.ctc_ocr_checkpoint = ctc_ocr_checkpoint
        self.ctc_anchor_fallback_margin = ctc_anchor_fallback_margin
        self.expert_ctc_ocr_checkpoint = expert_ctc_ocr_checkpoint
        self.ocr_utility_router_threshold = ocr_utility_router_threshold

        runtime = load_hybrid_runtime_components(
            selector_checkpoint=selector_checkpoint,
            bond_checkpoint=bond_checkpoint,
            device=device,
            assembler_checkpoint=assembler_checkpoint,
            graph_corrector_checkpoint=graph_corrector_checkpoint,
            ocr_checkpoint=ocr_checkpoint,
            ctc_ocr_checkpoint=ctc_ocr_checkpoint,
            text_bbox_checkpoint=text_bbox_checkpoint,
            bond_set_selector_checkpoint=bond_set_selector_checkpoint,
            endpoint_grouper_checkpoint=endpoint_grouper_checkpoint,
            line_band_checkpoint=line_band_checkpoint,
            semantic_bond_relation_checkpoint=semantic_bond_relation_checkpoint,
            semantic_bond_existence_checkpoint=semantic_bond_existence_checkpoint,
            expert_bond_checkpoint=expert_bond_checkpoint,
            style_gate_checkpoint=style_gate_checkpoint,
            style_gate_threshold=style_gate_threshold,
            bond_utility_router_checkpoint=bond_utility_router_checkpoint,
            bond_utility_break_budget=bond_utility_break_budget,
            expert_ctc_ocr_checkpoint=expert_ctc_ocr_checkpoint,
            ocr_utility_router_checkpoint=ocr_utility_router_checkpoint,
            ocr_utility_router_threshold=ocr_utility_router_threshold,
        )
        bind_hybrid_runtime(self, runtime)
        self.component_diagnostics: dict = {}

    def _correct_bond_objects(
        self,
        points: torch.Tensor,
        roles: torch.Tensor,
        labels: list[str],
        bond_objects: list[dict],
        *,
        mode: str,
        graph_keep_threshold: float | None,
    ) -> tuple[list[dict], dict]:
        return correct_bond_objects(
            points,
            roles,
            labels,
            bond_objects,
            mode=mode,
            graph_keep_threshold=graph_keep_threshold,
            runtime=GraphCorrectionRuntime(
                device=self.device,
                corrector=self.graph_corrector,
                default_keep_threshold=self.graph_corrector_keep_threshold,
                uses_elements=self.graph_corrector_uses_elements,
            ),
        )

    def _semantic_bond_runtime(self) -> SemanticBondRuntime:
        return SemanticBondRuntime(
            device=self.device,
            relation=self.semantic_bond_relation,
            relation_mode=self.semantic_bond_relation_mode,
            aromatic_mean_threshold=self.semantic_aromatic_mean_threshold,
            aromatic_minimum_threshold=self.semantic_aromatic_minimum_threshold,
            existence=self.semantic_bond_existence,
            existence_threshold=self.semantic_bond_existence_threshold,
            existence_chunk_size=self.semantic_bond_existence_chunk_size,
        )

    def _apply_semantic_bond_types(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
        bond_objects: list[dict] | tuple[dict, ...],
    ) -> tuple[list[dict], dict]:
        return apply_semantic_bond_types(
            pixels,
            points,
            bond_objects,
            runtime=self._semantic_bond_runtime(),
        )

    def _junction_aware_bonds(
        self,
        pixels: torch.Tensor,
        bond_objects: list[dict],
    ):
        result, diagnostics = self._junction_aware_bonds_with_diagnostics(
            pixels, bond_objects
        )
        self.component_diagnostics.update(diagnostics)
        return result

    def _junction_aware_bonds_with_diagnostics(
        self,
        pixels: torch.Tensor,
        bond_objects: list[dict],
    ):
        result, diagnostics = junction_aware_bonds(
            pixels,
            bond_objects,
            runtime=HybridJunctionBondRuntime(
                bond_set_selector=self.bond_set_selector,
                bond_set_selector_payload=self.bond_set_selector_payload,
                endpoint_grouper=self.endpoint_grouper,
                endpoint_grouper_payload=self.endpoint_grouper_payload,
                line_band_reader=self.line_band_reader,
                projection_score_source=self.junction_projection_score_source,
                bond_type_source=self.junction_bond_type_source,
                semantic=self._semantic_bond_runtime(),
            )
        )
        return result, diagnostics

    @torch.inference_mode()
    def _semantic_relation_bonds(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
    ) -> list[dict]:
        bonds, diagnostics = self._semantic_relation_bonds_with_diagnostics(
            pixels, points
        )
        self.component_diagnostics.update(diagnostics)
        return bonds

    def _semantic_relation_bonds_with_diagnostics(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[list[dict], dict]:
        bonds, diagnostics = semantic_relation_bonds(
            pixels,
            points,
            runtime=self._semantic_bond_runtime(),
        )
        return bonds, diagnostics

    @torch.inference_mode()
    def _structured_atom_candidates(
        self,
        pixels: torch.Tensor,
        points: torch.Tensor,
        roles: torch.Tensor,
        labels: list[str],
    ):
        return structured_atom_candidates(
            pixels,
            points,
            roles,
            labels,
            runtime=HybridAtomDecodingRuntime(
                device=self.device,
                base=self.base,
                text_head=self.text_head,
                protocol=self.protocol,
                maximum_text_objects=self.selector_config.maximum_candidates,
                ctc_ocr=self.ctc_ocr,
            ),
        )

    def _junctions(
        self,
        pixels: torch.Tensor,
        features: torch.Tensor,
        *,
        real_threshold: float,
        bbox_expansion_pixels: float,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        results = self._junctions_batch(
            pixels,
            features,
            real_threshold=real_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=True,
        )
        if len(results) != 1:
            raise ValueError("Single-image Junction inference received a batch")
        return results[0]

    def _junctions_batch(
        self,
        pixels: torch.Tensor,
        features: torch.Tensor,
        *,
        real_threshold: float,
        bbox_expansion_pixels: float,
        include_ocr: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor, list[str]]]:
        results, diagnostics = self._junctions_batch_with_diagnostics(
            pixels,
            features,
            real_threshold=real_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=include_ocr,
        )
        self.component_diagnostics.update(diagnostics)
        return results

    def _junctions_batch_with_diagnostics(
        self,
        pixels: torch.Tensor,
        features: torch.Tensor,
        *,
        real_threshold: float,
        bbox_expansion_pixels: float,
        include_ocr: bool = True,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor, list[str]]], dict]:
        selection = select_junction_batches(
            pixels=pixels,
            features=features,
            device=self.device,
            query_detector=self.query_detector,
            text_head=self.text_head,
            implicit_head=self.implicit_head,
            protocol=self.protocol,
            selector_config=self.selector_config,
            selector=self.selector,
            grounded_selector=self.grounded_selector,
            graph=self.graph,
            real_threshold=real_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
        )
        results, ocr_diagnostics = decode_junction_labels(
            selection=selection,
            pixels=pixels,
            features=features,
            include_ocr=include_ocr,
            runtime=HybridOCRRuntime(
                device=self.device,
                base=self.base,
                codec=self.codec,
                primary_ctc=self.ctc_ocr,
                expert_ctc=self.expert_ctc_ocr,
                utility_router=self.ocr_utility_router,
                utility_feature_names=self.ocr_utility_feature_names,
                utility_feature_mean=self.ocr_utility_feature_mean,
                utility_feature_std=self.ocr_utility_feature_std,
                utility_threshold=self.ocr_utility_router_threshold,
                anchor_fallback_margin=self.ctc_anchor_fallback_margin,
            ),
        )
        return results, ocr_diagnostics

    def _decode_routed_bond_objects(
        self,
        stride4: torch.Tensor,
        *,
        threshold: float,
    ) -> list[list[dict]]:
        selected, diagnostics = self._decode_routed_bond_objects_with_diagnostics(
            stride4, threshold=threshold
        )
        self.component_diagnostics.update(diagnostics)
        return selected

    def _decode_routed_bond_objects_with_diagnostics(
        self,
        stride4: torch.Tensor,
        *,
        threshold: float,
    ) -> tuple[list[list[dict]], dict]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            return decode_routed_bond_objects(
                stride4,
                threshold=threshold,
                runtime=HybridBondRoutingRuntime(
                    device=self.device,
                    base=self.base,
                    detector=self.bond_detector,
                    expert_detector=self.expert_bond_detector,
                    style_gate=self.style_gate,
                    style_threshold=self.style_gate_threshold,
                    utility_router=self.bond_utility_router,
                    utility_feature_names=self.bond_utility_feature_names,
                    utility_feature_mean=self.bond_utility_feature_mean,
                    utility_feature_std=self.bond_utility_feature_std,
                    utility_threshold=self.bond_utility_threshold,
                ),
            )

    def _text_interior_bond_veto_callback(
        self,
        pixels: torch.Tensor,
        features: torch.Tensor,
        bond_objects: list[dict],
    ) -> tuple[list[dict], int, list[dict]]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            text_output = self.text_head(features)
        text_objects = decode_text_objects(
            text_output,
            threshold=float(self.protocol["text_threshold"]),
            maximum_objects=self.selector_config.maximum_candidates,
        )[0]
        selected, removed = suppress_text_interior_bond_objects(
            bond_objects,
            text_objects,
            image_size=pixels.size(-1),
        )
        return selected, len(text_objects), removed

    def _component_routing_runtime(self) -> HybridComponentRoutingRuntime:
        return HybridComponentRoutingRuntime(
            device=self.device,
            decode_bonds=lambda features, threshold: (
                self._decode_routed_bond_objects_with_diagnostics(
                    features, threshold=threshold
                )
            ),
            detect_junctions=lambda pixels, features, real, bbox, include: (
                self._junctions_batch_with_diagnostics(
                    pixels,
                    features,
                    real_threshold=real,
                    bbox_expansion_pixels=bbox,
                    include_ocr=include,
                )
            ),
            junction_aware_bonds=self._junction_aware_bonds_with_diagnostics,
            semantic_relation_bonds=(
                self._semantic_relation_bonds_with_diagnostics
            ),
            text_interior_bond_veto_enabled=self.text_interior_bond_veto,
            text_interior_bond_veto=self._text_interior_bond_veto_callback,
        )

    @torch.inference_mode()
    def predict_components(
        self,
        pixels: torch.Tensor,
        *,
        real_threshold: float = 0.2,
        bond_threshold: float = 0.25,
        bbox_expansion_pixels: float = 8.0,
        include_ocr: bool = True,
        junction_source: str = "vertex",
        endpoint_cluster_radius_pixels: float = 12.0,
        text_attach_radius_pixels: float = 24.0,
        external_junctions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
        if (junction_source == "external") != (external_junctions is not None):
            raise ValueError(
                "External Junction source and external_junctions must be provided together"
            )
        self.component_diagnostics = {}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            features = self.base.vision_encoder.forward_stroke_resolution(pixels)
        routed = route_single_components(
            pixels,
            features,
            runtime=self._component_routing_runtime(),
            real_threshold=real_threshold,
            bond_threshold=bond_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=include_ocr,
            junction_source=junction_source,
            endpoint_cluster_radius_pixels=endpoint_cluster_radius_pixels,
            text_attach_radius_pixels=text_attach_radius_pixels,
            external_junctions=external_junctions,
        )
        self.component_diagnostics.update(routed.diagnostics)
        return routed.components

    @torch.inference_mode()
    def predict_components_batch(
        self,
        pixels: torch.Tensor,
        *,
        real_threshold: float = 0.2,
        bond_threshold: float = 0.25,
        bbox_expansion_pixels: float = 8.0,
        include_ocr: bool = True,
        junction_source: str = "vertex",
        endpoint_cluster_radius_pixels: float = 12.0,
        text_attach_radius_pixels: float = 24.0,
    ) -> list[tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]]:
        self.component_diagnostics = {}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            features = self.base.vision_encoder.forward_stroke_resolution(pixels)
        routed = route_component_batches(
            pixels,
            features,
            runtime=self._component_routing_runtime(),
            real_threshold=real_threshold,
            bond_threshold=bond_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            include_ocr=include_ocr,
            junction_source=junction_source,
            endpoint_cluster_radius_pixels=endpoint_cluster_radius_pixels,
            text_attach_radius_pixels=text_attach_radius_pixels,
        )
        self.component_diagnostics.update(routed.diagnostics)
        return routed.components

    @torch.inference_mode()
    def predict(
        self,
        pixels: torch.Tensor,
        molecule_id: str,
        *,
        real_threshold: float = 0.2,
        bond_threshold: float = 0.25,
        bbox_expansion_pixels: float = 8.0,
        assignment_mode: str = "nearest",
        graph_correction_mode: str = "none",
        graph_keep_threshold: float | None = None,
        junction_source: str = "vertex",
        endpoint_cluster_radius_pixels: float = 12.0,
        text_attach_radius_pixels: float = 24.0,
        atom_decode_mode: str = "greedy",
        external_junctions: torch.Tensor | None = None,
    ) -> tuple[dict, dict]:
        if atom_decode_mode not in {"greedy", "structured_invalid_only"}:
            raise ValueError(f"Unknown atom decode mode: {atom_decode_mode}")
        points, roles, labels, bond_objects = self.predict_components(
            pixels,
            real_threshold=real_threshold,
            bond_threshold=bond_threshold,
            bbox_expansion_pixels=bbox_expansion_pixels,
            junction_source=junction_source,
            endpoint_cluster_radius_pixels=endpoint_cluster_radius_pixels,
            text_attach_radius_pixels=text_attach_radius_pixels,
            external_junctions=external_junctions,
        )
        result = compile_hybrid_scene(
            pixels=pixels,
            points=points,
            roles=roles,
            labels=labels,
            bond_objects=bond_objects,
            settings=HybridSceneCompilationSettings(
                molecule_id=molecule_id,
                assignment_mode=assignment_mode,
                graph_correction_mode=graph_correction_mode,
                graph_keep_threshold=graph_keep_threshold,
                junction_source=junction_source,
                atom_decode_mode=atom_decode_mode,
                real_threshold=real_threshold,
                bond_threshold=bond_threshold,
                endpoint_cluster_radius_pixels=endpoint_cluster_radius_pixels,
                text_attach_radius_pixels=text_attach_radius_pixels,
            ),
            runtime=HybridSceneCompilationRuntime(
                device=self.device,
                assembler=self.assembler,
                default_graph_keep_threshold=self.graph_corrector_keep_threshold,
                correct_bonds=self._correct_bond_objects,
                apply_semantic_bond_types=self._apply_semantic_bond_types,
                structured_atom_candidates=self._structured_atom_candidates,
            ),
            component_diagnostics=self.component_diagnostics,
        )
        return result.scene, result.diagnostics


def main() -> None:
    args = build_hybrid_parser().parse_args()
    runtime = load_hybrid_cli_runtime(
        args,
        pipeline_factory=ChemSceneHybridPipeline,
        fallback_loader=load_compatible_fallback_ctc_ocr,
    )
    prepared = prepare_hybrid_cli_input(args, runtime.device)
    prediction = predict_hybrid_cli(args, runtime=runtime, prepared=prepared)
    write_hybrid_cli_prediction(
        args,
        runtime=runtime,
        prepared=prepared,
        prediction=prediction,
    )


if __name__ == "__main__":
    main()
