"""Persist ChemScene hybrid predictions and diagnostic artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from Data.sdf.chemscene_v0_1 import chemsvg_from_chemscene_v0_1
from chemvector.chemscene_assembly import compile_scene_with_aromatic_fallback
from chemvector.diagnostic_sdf import write_invalid_prediction_sdf


@dataclass(frozen=True, slots=True)
class HybridOutputSettings:
    output_dir: Path
    image_size: int
    model_version: str | None
    device: str
    atom_decode_mode: str
    junction_source: str
    text_bbox_checkpoint: str | None
    ctc_ocr_checkpoint: str | None
    ocr_checkpoint: str | None
    graph_correction_mode: str
    assembler_checkpoint: str | None
    graph_corrector_checkpoint: str | None
    expert_ctc_ocr_checkpoint: str | None
    invalid_fallback_ctc_ocr_checkpoint: str | None
    ocr_utility_router_checkpoint: str | None
    ocr_utility_router_threshold: float
    bond_set_selector_checkpoint: str | None
    endpoint_grouper_checkpoint: str | None
    line_band_checkpoint: str | None
    semantic_bond_relation_checkpoint: str | None
    semantic_bond_relation_mode: str
    semantic_aromatic_mean_threshold: float
    semantic_aromatic_minimum_threshold: float
    ctc_anchor_fallback_margin: float | None
    junction_bond_type_source: str
    endpoint_cluster_radius_pixels: float
    text_attach_radius_pixels: float


@dataclass(frozen=True, slots=True)
class HybridFallbackArtifacts:
    enabled: bool
    primary_scene: dict
    primary_compile: tuple[str, dict, bool] | None
    fallback_scene: dict | None
    fallback_compile: tuple[str, dict, bool] | None


def _json_text(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _model_version(settings: HybridOutputSettings, *, valid: bool) -> str:
    if settings.model_version:
        return settings.model_version
    if settings.atom_decode_mode == "structured_invalid_only":
        return "v0.8.23"
    if settings.junction_source == "junction_aware_bonds":
        return "v0.8.22"
    if settings.junction_source == "bond_endpoints":
        return "v0.8.16"
    if not valid:
        return "v0.8.x"
    if settings.text_bbox_checkpoint:
        return "v0.8.9"
    if settings.ctc_ocr_checkpoint:
        return "v0.8.8"
    if settings.ocr_checkpoint:
        return "v0.8.6"
    if settings.graph_correction_mode != "none":
        return "v0.8.5"
    return "v0.8.4" if settings.assembler_checkpoint else "v0.8.3"


def _protocol(settings: HybridOutputSettings) -> str:
    if settings.junction_source == "junction_aware_bonds":
        atom_decode = (
            "CTC top-k fixed-topology atom decode + "
            if settings.atom_decode_mode == "structured_invalid_only"
            else ""
        )
        return (
            "top-128 Bond selector -> visual endpoint grouping -> OCR label attachment + "
            f"{atom_decode}graph {settings.graph_correction_mode} + ChemScene"
        )
    if settings.junction_source == "bond_endpoints":
        return (
            "semantic BondObject endpoints -> Junction + OCR label attachment + "
            f"graph {settings.graph_correction_mode} + ChemScene"
        )
    if settings.text_bbox_checkpoint:
        return (
            "predicted Junction + asymmetric TextObject bbox + conservative "
            "pretrained CTC OCR + multiscale "
            f"BondObject + graph {settings.graph_correction_mode} + ChemScene"
        )
    if settings.ctc_ocr_checkpoint:
        return (
            "predicted Junction + bbox pretrained CTC OCR + multiscale "
            f"BondObject + graph {settings.graph_correction_mode} + ChemScene"
        )
    if settings.graph_correction_mode != "none":
        return (
            "predicted Junction + Pixel OCR + multiscale BondObject + "
            f"graph {settings.graph_correction_mode} + ChemScene"
        )
    return "predicted Junction + Pixel OCR + multiscale BondObject + ChemScene"


def _write_fallback_candidates(
    output: Path, artifacts: HybridFallbackArtifacts
) -> None:
    if not artifacts.enabled:
        return
    (output / "candidate-primary.chemscene.json").write_text(
        _json_text(artifacts.primary_scene), encoding="utf-8"
    )
    if artifacts.primary_compile is not None:
        (output / "candidate-primary.sdf").write_text(
            artifacts.primary_compile[0], encoding="utf-8"
        )
    if artifacts.fallback_scene is not None:
        (output / "candidate-fallback.chemscene.json").write_text(
            _json_text(artifacts.fallback_scene), encoding="utf-8"
        )
    if artifacts.fallback_compile is not None:
        (output / "candidate-fallback.sdf").write_text(
            artifacts.fallback_compile[0], encoding="utf-8"
        )


def _render_molecule(mol, path: Path, image_size: int) -> None:
    drawer = rdMolDraw2D.MolDraw2DCairo(image_size, image_size)
    drawer.drawOptions().padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


def _invalid_metrics(
    *,
    settings: HybridOutputSettings,
    diagnostics: dict,
    load_seconds: float,
    inference_seconds: float,
    atoms: int,
    bonds: int,
    diagnostic_rendered: bool,
    error: Exception,
) -> dict:
    protocol = "ChemScene diagnostic invalid SDF"
    if settings.invalid_fallback_ctc_ocr_checkpoint:
        protocol += " + invalid-only dual OCR fallback"
    return {
        "model_version": _model_version(settings, valid=False),
        "protocol": protocol,
        "device": settings.device,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "vertices": atoms,
        "candidate_strokes": diagnostics["decoded_bond_objects"],
        "repaired_strokes": bonds,
        "sdf_valid": False,
        "invalid_fallback_ctc_ocr_checkpoint": (
            settings.invalid_fallback_ctc_ocr_checkpoint
        ),
        "diagnostic_rendered": diagnostic_rendered,
        "error": str(error),
        "junction_source": settings.junction_source,
        "atom_decode_mode": settings.atom_decode_mode,
        "diagnostics": diagnostics,
    }


def _valid_metrics(
    *,
    settings: HybridOutputSettings,
    scene: dict,
    compiled_scene: dict,
    diagnostics: dict,
    mol,
    aromatic_fallback: bool,
    load_seconds: float,
    inference_seconds: float,
) -> dict:
    protocol = _protocol(settings)
    if settings.invalid_fallback_ctc_ocr_checkpoint:
        protocol += " + invalid-only dual OCR fallback"
    return {
        "model_version": _model_version(settings, valid=True),
        "protocol": protocol,
        "device": settings.device,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "vertices": mol.GetNumAtoms(),
        "candidate_strokes": diagnostics["decoded_bond_objects"],
        "repaired_strokes": mol.GetNumBonds(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "inchikey": Chem.MolToInchiKey(mol),
        "aromatic_fallback": aromatic_fallback,
        "explicit_h_valence_repairs": sum(
            int(raw["atom"].get("explicit_h_count", 0))
            != int(compiled["atom"].get("explicit_h_count", 0))
            for raw, compiled in zip(scene["junctions"], compiled_scene["junctions"])
        ),
        "assembler_checkpoint": settings.assembler_checkpoint,
        "graph_corrector_checkpoint": settings.graph_corrector_checkpoint,
        "ocr_checkpoint": settings.ocr_checkpoint,
        "ctc_ocr_checkpoint": settings.ctc_ocr_checkpoint,
        "expert_ctc_ocr_checkpoint": settings.expert_ctc_ocr_checkpoint,
        "invalid_fallback_ctc_ocr_checkpoint": (
            settings.invalid_fallback_ctc_ocr_checkpoint
        ),
        "ocr_utility_router_checkpoint": settings.ocr_utility_router_checkpoint,
        "ocr_utility_router_threshold": settings.ocr_utility_router_threshold,
        "text_bbox_checkpoint": settings.text_bbox_checkpoint,
        "atom_decode_mode": settings.atom_decode_mode,
        "bond_set_selector_checkpoint": settings.bond_set_selector_checkpoint,
        "endpoint_grouper_checkpoint": settings.endpoint_grouper_checkpoint,
        "line_band_checkpoint": settings.line_band_checkpoint,
        "semantic_bond_relation_checkpoint": (
            settings.semantic_bond_relation_checkpoint
        ),
        "semantic_bond_relation_mode": settings.semantic_bond_relation_mode,
        "semantic_aromatic_mean_threshold": (
            settings.semantic_aromatic_mean_threshold
        ),
        "semantic_aromatic_minimum_threshold": (
            settings.semantic_aromatic_minimum_threshold
        ),
        "ctc_anchor_fallback_margin": settings.ctc_anchor_fallback_margin,
        "graph_correction_mode": settings.graph_correction_mode,
        "junction_source": settings.junction_source,
        "junction_bond_type_source": settings.junction_bond_type_source,
        "endpoint_cluster_radius_pixels": settings.endpoint_cluster_radius_pixels,
        "text_attach_radius_pixels": settings.text_attach_radius_pixels,
        "diagnostics": diagnostics,
    }


def write_hybrid_outputs(
    *,
    image: Image.Image,
    scene: dict,
    diagnostics: dict,
    settings: HybridOutputSettings,
    fallback: HybridFallbackArtifacts,
    load_seconds: float,
    inference_seconds: float,
) -> dict:
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "input-prepared.png")
    _write_fallback_candidates(output, fallback)
    (output / "prediction-raw.chemscene.json").write_text(
        _json_text(scene), encoding="utf-8"
    )
    try:
        sdf, compiled_scene, aromatic_fallback = compile_scene_with_aromatic_fallback(
            scene
        )
    except Exception as error:
        (output / "prediction.chemscene.json").write_text(
            _json_text(scene), encoding="utf-8"
        )
        diagnostic_sdf = output / "prediction-repaired.sdf"
        atoms, bonds = write_invalid_prediction_sdf(
            chemsvg_from_chemscene_v0_1(scene), diagnostic_sdf
        )
        diagnostic_rendered = False
        diagnostic_mol = Chem.MolFromMolFile(
            str(diagnostic_sdf), removeHs=False, sanitize=False, strictParsing=False
        )
        if diagnostic_mol is not None:
            try:
                diagnostic_mol.UpdatePropertyCache(strict=False)
                _render_molecule(
                    diagnostic_mol,
                    output / "prediction-repaired.png",
                    settings.image_size,
                )
                diagnostic_rendered = True
            except Exception:
                diagnostic_rendered = False
        metrics = _invalid_metrics(
            settings=settings,
            diagnostics=diagnostics,
            load_seconds=load_seconds,
            inference_seconds=inference_seconds,
            atoms=atoms,
            bonds=bonds,
            diagnostic_rendered=diagnostic_rendered,
            error=error,
        )
        (output / "metrics.json").write_text(_json_text(metrics), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    (output / "prediction.chemscene.json").write_text(
        _json_text(compiled_scene), encoding="utf-8"
    )
    sdf_path = output / "prediction-repaired.sdf"
    sdf_path.write_text(sdf, encoding="utf-8")
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        raise RuntimeError("Cannot read compiled ChemScene SDF")
    _render_molecule(mol, output / "prediction-repaired.png", settings.image_size)
    metrics = _valid_metrics(
        settings=settings,
        scene=scene,
        compiled_scene=compiled_scene,
        diagnostics=diagnostics,
        mol=mol,
        aromatic_fallback=aromatic_fallback,
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
    )
    (output / "metrics.json").write_text(_json_text(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return metrics
