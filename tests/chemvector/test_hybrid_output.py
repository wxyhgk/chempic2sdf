from __future__ import annotations

import ast
import json
from pathlib import Path

from PIL import Image
import torch

import chemvector.benchmark_mr_tadf as legacy_benchmark
from chemvector.chemscene_assembly import (
    assemble_chemscene,
    compile_scene_with_aromatic_fallback,
)
from chemvector.diagnostic_sdf import write_invalid_prediction_sdf
from chemvector.hybrid_output import (
    HybridFallbackArtifacts,
    HybridOutputSettings,
    write_hybrid_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_SOURCE = PROJECT_ROOT / "chemvector/inference/hybrid_output.py"


def _settings(output: Path) -> HybridOutputSettings:
    return HybridOutputSettings(
        output_dir=output,
        image_size=128,
        model_version="test-model",
        device="cpu",
        atom_decode_mode="greedy",
        junction_source="bond_endpoints",
        text_bbox_checkpoint=None,
        ctc_ocr_checkpoint=None,
        ocr_checkpoint=None,
        graph_correction_mode="none",
        assembler_checkpoint=None,
        graph_corrector_checkpoint=None,
        expert_ctc_ocr_checkpoint=None,
        invalid_fallback_ctc_ocr_checkpoint=None,
        ocr_utility_router_checkpoint=None,
        ocr_utility_router_threshold=0.95,
        bond_set_selector_checkpoint=None,
        endpoint_grouper_checkpoint=None,
        line_band_checkpoint=None,
        semantic_bond_relation_checkpoint=None,
        semantic_bond_relation_mode="chemical_all",
        semantic_aromatic_mean_threshold=0.72,
        semantic_aromatic_minimum_threshold=0.25,
        ctc_anchor_fallback_margin=None,
        junction_bond_type_source="detector",
        endpoint_cluster_radius_pixels=12.0,
        text_attach_radius_pixels=24.0,
    )


def _scene() -> dict:
    scene, _ = assemble_chemscene(
        "output-test",
        torch.tensor(((0.2, 0.5), (0.8, 0.5))),
        torch.tensor((0, 0)),
        ["C", "C"],
        [
            {
                "score": 0.9,
                "endpoints": torch.tensor(((0.2, 0.5), (0.8, 0.5))),
                "display": 0,
                "chemical": 0,
                "direction": 0,
            }
        ],
    )
    return scene


def test_valid_output_writer_owns_all_prediction_artifacts(
    tmp_path: Path, capsys
) -> None:
    scene = _scene()
    metrics = write_hybrid_outputs(
        image=Image.new("RGB", (128, 128), "white"),
        scene=scene,
        diagnostics={"decoded_bond_objects": 1},
        settings=_settings(tmp_path),
        fallback=HybridFallbackArtifacts(
            enabled=False,
            primary_scene=scene,
            primary_compile=None,
            fallback_scene=None,
            fallback_compile=None,
        ),
        load_seconds=0.1,
        inference_seconds=0.2,
    )
    expected = {
        "input-prepared.png",
        "metrics.json",
        "prediction-raw.chemscene.json",
        "prediction.chemscene.json",
        "prediction-repaired.png",
        "prediction-repaired.sdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert metrics["model_version"] == "test-model"
    assert metrics["vertices"] == 2
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert capsys.readouterr().out


def test_fallback_candidates_are_written_only_when_enabled(tmp_path: Path) -> None:
    scene = _scene()
    compiled = compile_scene_with_aromatic_fallback(scene)
    write_hybrid_outputs(
        image=Image.new("RGB", (128, 128), "white"),
        scene=scene,
        diagnostics={"decoded_bond_objects": 1},
        settings=_settings(tmp_path),
        fallback=HybridFallbackArtifacts(
            enabled=True,
            primary_scene=scene,
            primary_compile=compiled,
            fallback_scene=None,
            fallback_compile=None,
        ),
        load_seconds=0.1,
        inference_seconds=0.2,
    )
    assert (tmp_path / "candidate-primary.chemscene.json").is_file()
    assert (tmp_path / "candidate-primary.sdf").is_file()
    assert not (tmp_path / "candidate-fallback.sdf").exists()


def test_output_layer_does_not_depend_on_benchmark_or_pipeline_modules() -> None:
    tree = ast.parse(OUTPUT_SOURCE.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "chemvector.benchmark_mr_tadf" not in imports
    assert "chemvector.infer_chemscene_hybrid" not in imports


def test_legacy_benchmark_reexports_core_diagnostic_writer() -> None:
    assert legacy_benchmark.write_invalid_prediction_sdf is write_invalid_prediction_sdf
