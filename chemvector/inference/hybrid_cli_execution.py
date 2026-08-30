"""Reusable execution stages for the single-image hybrid inference CLI.

This module intentionally does not import ``infer_chemscene_hybrid``.  The
legacy entry point supplies its pipeline class and optional fallback loader as
factories, keeping checkpoint compatibility at the composition root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Protocol

from PIL import Image
import torch

from chemvector.chemscene_assembly import compile_scene_with_aromatic_fallback
from chemvector.hybrid_output import (
    HybridFallbackArtifacts,
    HybridOutputSettings,
    write_hybrid_outputs,
)
from chemvector.image_io import image_tensor, prepare_input_image


class HybridPipeline(Protocol):
    ctc_ocr: Any

    def predict(
        self, pixels: torch.Tensor, molecule_id: str, **kwargs: Any
    ) -> tuple[dict, dict]: ...


@dataclass(frozen=True, slots=True)
class HybridCLIInput:
    image: Image.Image
    pixels: torch.Tensor
    molecule_id: str


@dataclass(frozen=True, slots=True)
class HybridCLIRuntime:
    device: torch.device
    pipeline: HybridPipeline
    fallback_ctc_ocr: Any | None
    load_seconds: float


@dataclass(frozen=True, slots=True)
class HybridPrediction:
    scene: dict
    diagnostics: dict
    fallback: HybridFallbackArtifacts
    inference_seconds: float


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value) if value else None


def validate_hybrid_cli_configuration(args: Any) -> None:
    """Validate cross-component options which argparse cannot express."""

    if args.invalid_fallback_ctc_ocr_checkpoint and (
        args.expert_ctc_ocr_checkpoint or args.ocr_utility_router_checkpoint
    ):
        raise ValueError(
            "Invalid-only OCR fallback cannot be combined with utility OCR routing"
        )


def hybrid_pipeline_kwargs(args: Any, device: torch.device) -> dict[str, Any]:
    """Translate the CLI namespace into the pipeline construction contract."""

    return {
        "selector_checkpoint": Path(args.selector_checkpoint),
        "bond_checkpoint": Path(args.bond_checkpoint),
        "device": device,
        "assembler_checkpoint": _optional_path(args.assembler_checkpoint),
        "graph_corrector_checkpoint": _optional_path(
            args.graph_corrector_checkpoint
        ),
        "ocr_checkpoint": _optional_path(args.ocr_checkpoint),
        "ctc_ocr_checkpoint": _optional_path(args.ctc_ocr_checkpoint),
        "text_bbox_checkpoint": _optional_path(args.text_bbox_checkpoint),
        "ctc_anchor_fallback_margin": args.ctc_anchor_fallback_margin,
        "bond_set_selector_checkpoint": _optional_path(
            args.bond_set_selector_checkpoint
        ),
        "endpoint_grouper_checkpoint": _optional_path(
            args.endpoint_grouper_checkpoint
        ),
        "line_band_checkpoint": _optional_path(args.line_band_checkpoint),
        "junction_projection_score_source": args.junction_projection_score_source,
        "junction_bond_type_source": args.junction_bond_type_source,
        "semantic_bond_relation_checkpoint": _optional_path(
            args.semantic_bond_relation_checkpoint
        ),
        "semantic_bond_relation_mode": args.semantic_bond_relation_mode,
        "semantic_aromatic_mean_threshold": args.semantic_aromatic_mean_threshold,
        "semantic_aromatic_minimum_threshold": (
            args.semantic_aromatic_minimum_threshold
        ),
        "semantic_bond_existence_checkpoint": _optional_path(
            args.semantic_bond_existence_checkpoint
        ),
        "semantic_bond_existence_threshold": args.semantic_bond_existence_threshold,
        "semantic_bond_existence_chunk_size": (
            args.semantic_bond_existence_chunk_size
        ),
        "expert_bond_checkpoint": _optional_path(args.expert_bond_checkpoint),
        "style_gate_checkpoint": _optional_path(args.style_gate_checkpoint),
        "style_gate_threshold": args.style_gate_threshold,
        "bond_utility_router_checkpoint": _optional_path(
            args.bond_utility_router_checkpoint
        ),
        "bond_utility_break_budget": args.bond_utility_break_budget,
        "text_interior_bond_veto": args.text_interior_bond_veto,
        "expert_ctc_ocr_checkpoint": _optional_path(
            args.expert_ctc_ocr_checkpoint
        ),
        "ocr_utility_router_checkpoint": _optional_path(
            args.ocr_utility_router_checkpoint
        ),
        "ocr_utility_router_threshold": args.ocr_utility_router_threshold,
    }


def load_hybrid_cli_runtime(
    args: Any,
    *,
    pipeline_factory: Callable[..., HybridPipeline],
    fallback_loader: Callable[[str | Path, Any, torch.device], Any] | None = None,
) -> HybridCLIRuntime:
    """Construct models and the optional invalid-only OCR fallback."""

    validate_hybrid_cli_configuration(args)
    device = torch.device(args.device)
    started = time.perf_counter()
    pipeline = pipeline_factory(**hybrid_pipeline_kwargs(args, device))

    fallback_ctc_ocr = None
    checkpoint = args.invalid_fallback_ctc_ocr_checkpoint
    if checkpoint:
        if pipeline.ctc_ocr is None:
            raise ValueError("Invalid-only OCR fallback requires a primary CTC OCR")
        if fallback_loader is None:
            raise ValueError("Invalid-only OCR fallback requires a fallback loader")
        fallback_ctc_ocr = fallback_loader(checkpoint, pipeline.ctc_ocr, device)

    return HybridCLIRuntime(
        device=device,
        pipeline=pipeline,
        fallback_ctc_ocr=fallback_ctc_ocr,
        load_seconds=time.perf_counter() - started,
    )


def prepare_hybrid_cli_input(args: Any, device: torch.device) -> HybridCLIInput:
    """Load, normalize, size, and tensorize one CLI image."""

    with Image.open(args.image) as source:
        image = (
            source.convert("RGB")
            if args.image_is_prepared
            else prepare_input_image(
                source, image_size=args.image_size, content_size=args.content_size
            )
        )
    if image.size != (args.image_size, args.image_size):
        image = image.resize(
            (args.image_size, args.image_size), Image.Resampling.BICUBIC
        )
    pixels = image_tensor(image).unsqueeze(0).to(device)
    return HybridCLIInput(
        image=image,
        pixels=pixels,
        molecule_id=Path(args.image).stem,
    )


def hybrid_predict_kwargs(args: Any) -> dict[str, Any]:
    return {
        "real_threshold": args.real_threshold,
        "bond_threshold": args.bond_threshold,
        "assignment_mode": args.assignment_mode,
        "graph_correction_mode": args.graph_correction_mode,
        "graph_keep_threshold": args.graph_keep_threshold,
        "junction_source": args.junction_source,
        "endpoint_cluster_radius_pixels": args.endpoint_cluster_radius_pixels,
        "text_attach_radius_pixels": args.text_attach_radius_pixels,
        "atom_decode_mode": args.atom_decode_mode,
    }


def predict_hybrid_cli(
    args: Any,
    *,
    runtime: HybridCLIRuntime,
    prepared: HybridCLIInput,
    compile_scene: Callable[[dict], tuple[str, dict, bool]] = (
        compile_scene_with_aromatic_fallback
    ),
) -> HybridPrediction:
    """Predict once, retry invalid output with fallback OCR, and select a scene."""

    started = time.perf_counter()
    predict_kwargs = hybrid_predict_kwargs(args)
    pipeline = runtime.pipeline
    scene, diagnostics = pipeline.predict(
        prepared.pixels, prepared.molecule_id, **predict_kwargs
    )
    primary_scene = scene
    primary_compile = None
    fallback_scene = None
    fallback_diagnostics = None
    fallback_compile = None
    primary_error = None
    fallback_error = None
    fallback_triggered = False
    fallback_selected = False

    if runtime.fallback_ctc_ocr is not None:
        try:
            primary_compile = compile_scene(primary_scene)
        except Exception as error:
            primary_error = str(error)
            fallback_triggered = True
            primary_ctc_ocr = pipeline.ctc_ocr
            pipeline.ctc_ocr = runtime.fallback_ctc_ocr
            try:
                fallback_scene, fallback_diagnostics = pipeline.predict(
                    prepared.pixels, prepared.molecule_id, **predict_kwargs
                )
            finally:
                pipeline.ctc_ocr = primary_ctc_ocr
            try:
                fallback_compile = compile_scene(fallback_scene)
            except Exception as error:
                fallback_error = str(error)
            if fallback_compile is not None:
                scene = fallback_scene
                diagnostics = fallback_diagnostics
                fallback_selected = True

        diagnostics = {
            **diagnostics,
            "invalid_ocr_fallback": {
                "enabled": True,
                "triggered": fallback_triggered,
                "selected": fallback_selected,
                "reason": (
                    "primary_invalid_fallback_valid"
                    if fallback_selected
                    else "primary_valid"
                    if not fallback_triggered
                    else "both_invalid"
                ),
                "primary_valid": primary_compile is not None,
                "fallback_valid": fallback_compile is not None,
                "primary_error": primary_error,
                "fallback_error": fallback_error,
                "checkpoint": args.invalid_fallback_ctc_ocr_checkpoint,
            },
        }

    fallback = HybridFallbackArtifacts(
        enabled=runtime.fallback_ctc_ocr is not None,
        primary_scene=primary_scene,
        primary_compile=primary_compile,
        fallback_scene=fallback_scene,
        fallback_compile=fallback_compile,
    )
    return HybridPrediction(
        scene=scene,
        diagnostics=diagnostics,
        fallback=fallback,
        inference_seconds=time.perf_counter() - started,
    )


def hybrid_output_settings(args: Any) -> HybridOutputSettings:
    """Translate CLI values into the stable output writer contract."""

    return HybridOutputSettings(
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        model_version=args.model_version,
        device=args.device,
        atom_decode_mode=args.atom_decode_mode,
        junction_source=args.junction_source,
        text_bbox_checkpoint=args.text_bbox_checkpoint,
        ctc_ocr_checkpoint=args.ctc_ocr_checkpoint,
        ocr_checkpoint=args.ocr_checkpoint,
        graph_correction_mode=args.graph_correction_mode,
        assembler_checkpoint=args.assembler_checkpoint,
        graph_corrector_checkpoint=args.graph_corrector_checkpoint,
        expert_ctc_ocr_checkpoint=args.expert_ctc_ocr_checkpoint,
        invalid_fallback_ctc_ocr_checkpoint=(
            args.invalid_fallback_ctc_ocr_checkpoint
        ),
        ocr_utility_router_checkpoint=args.ocr_utility_router_checkpoint,
        ocr_utility_router_threshold=args.ocr_utility_router_threshold,
        bond_set_selector_checkpoint=args.bond_set_selector_checkpoint,
        endpoint_grouper_checkpoint=args.endpoint_grouper_checkpoint,
        line_band_checkpoint=args.line_band_checkpoint,
        semantic_bond_relation_checkpoint=args.semantic_bond_relation_checkpoint,
        semantic_bond_relation_mode=args.semantic_bond_relation_mode,
        semantic_aromatic_mean_threshold=args.semantic_aromatic_mean_threshold,
        semantic_aromatic_minimum_threshold=(
            args.semantic_aromatic_minimum_threshold
        ),
        ctc_anchor_fallback_margin=args.ctc_anchor_fallback_margin,
        junction_bond_type_source=args.junction_bond_type_source,
        endpoint_cluster_radius_pixels=args.endpoint_cluster_radius_pixels,
        text_attach_radius_pixels=args.text_attach_radius_pixels,
    )


def write_hybrid_cli_prediction(
    args: Any,
    *,
    runtime: HybridCLIRuntime,
    prepared: HybridCLIInput,
    prediction: HybridPrediction,
    output_writer: Callable[..., dict] = write_hybrid_outputs,
) -> dict:
    """Persist a CLI prediction through the stable hybrid output writer."""

    return output_writer(
        image=prepared.image,
        scene=prediction.scene,
        diagnostics=prediction.diagnostics,
        settings=hybrid_output_settings(args),
        fallback=prediction.fallback,
        load_seconds=runtime.load_seconds,
        inference_seconds=prediction.inference_seconds,
    )
