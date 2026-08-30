from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from chemvector.hybrid_cli import build_hybrid_parser
from chemvector.hybrid_cli_execution import (
    HybridCLIInput,
    HybridCLIRuntime,
    hybrid_output_settings,
    hybrid_pipeline_kwargs,
    load_hybrid_cli_runtime,
    predict_hybrid_cli,
    prepare_hybrid_cli_input,
    write_hybrid_cli_prediction,
)


def _args(*extra: str):
    return build_hybrid_parser().parse_args(
        ["--image", "input.png", "--output-dir", "out", *extra]
    )


def test_pipeline_kwargs_convert_only_checkpoint_paths() -> None:
    args = _args(
        "--assembler-checkpoint",
        "assembler.pt",
        "--style-gate-checkpoint",
        "style.pt",
        "--text-interior-bond-veto",
    )
    kwargs = hybrid_pipeline_kwargs(args, torch.device("cpu"))

    assert kwargs["selector_checkpoint"] == Path(args.selector_checkpoint)
    assert kwargs["assembler_checkpoint"] == Path("assembler.pt")
    assert kwargs["style_gate_checkpoint"] == Path("style.pt")
    assert kwargs["ocr_checkpoint"] is None
    assert kwargs["text_interior_bond_veto"] is True
    assert kwargs["semantic_bond_existence_threshold"] == pytest.approx(0.60)


def test_load_runtime_injects_pipeline_and_fallback_factories() -> None:
    args = _args(
        "--ctc-ocr-checkpoint",
        "primary.pt",
        "--invalid-fallback-ctc-ocr-checkpoint",
        "fallback.pt",
        "--device",
        "cpu",
    )
    primary = object()
    pipeline = SimpleNamespace(ctc_ocr=primary)
    captured = {}

    def pipeline_factory(**kwargs):
        captured["pipeline"] = kwargs
        return pipeline

    def fallback_loader(checkpoint, received_primary, device):
        captured["fallback"] = (checkpoint, received_primary, device)
        return "fallback-model"

    runtime = load_hybrid_cli_runtime(
        args,
        pipeline_factory=pipeline_factory,
        fallback_loader=fallback_loader,
    )

    assert runtime.pipeline is pipeline
    assert runtime.fallback_ctc_ocr == "fallback-model"
    assert captured["fallback"] == (
        "fallback.pt",
        primary,
        torch.device("cpu"),
    )
    assert captured["pipeline"]["ctc_ocr_checkpoint"] == Path("primary.pt")


def test_load_runtime_rejects_incompatible_fallback_routing() -> None:
    args = _args(
        "--invalid-fallback-ctc-ocr-checkpoint",
        "fallback.pt",
        "--expert-ctc-ocr-checkpoint",
        "expert.pt",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        load_hybrid_cli_runtime(args, pipeline_factory=lambda **_: None)


def test_prepare_input_closes_source_and_enforces_rgb_size(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("L", (21, 13), 127).save(path)
    args = _args(
        "--image",
        str(path),
        "--image-size",
        "32",
        "--image-is-prepared",
        "--device",
        "cpu",
    )

    prepared = prepare_hybrid_cli_input(args, torch.device("cpu"))

    assert prepared.image.mode == "RGB"
    assert prepared.image.size == (32, 32)
    assert prepared.pixels.shape == (1, 3, 32, 32)
    assert prepared.molecule_id == "sample"


class _FallbackPipeline:
    def __init__(self) -> None:
        self.ctc_ocr = "primary"
        self.seen_ocr = []

    def predict(self, pixels, molecule_id, **kwargs):
        self.seen_ocr.append(self.ctc_ocr)
        if self.ctc_ocr == "primary":
            return {"candidate": "primary"}, {"decoded_bond_objects": 1}
        return {"candidate": "fallback"}, {"decoded_bond_objects": 2}


def test_prediction_selects_valid_fallback_and_restores_primary_ocr() -> None:
    args = _args(
        "--invalid-fallback-ctc-ocr-checkpoint",
        "fallback.pt",
        "--device",
        "cpu",
    )
    pipeline = _FallbackPipeline()
    runtime = HybridCLIRuntime(
        device=torch.device("cpu"),
        pipeline=pipeline,
        fallback_ctc_ocr="fallback",
        load_seconds=0.1,
    )
    prepared = HybridCLIInput(
        image=Image.new("RGB", (4, 4)),
        pixels=torch.zeros(1, 3, 4, 4),
        molecule_id="molecule",
    )

    def compile_scene(scene):
        if scene["candidate"] == "primary":
            raise ValueError("invalid primary")
        return "SDF", {"compiled": True}, False

    prediction = predict_hybrid_cli(
        args,
        runtime=runtime,
        prepared=prepared,
        compile_scene=compile_scene,
    )

    assert prediction.scene == {"candidate": "fallback"}
    assert pipeline.seen_ocr == ["primary", "fallback"]
    assert pipeline.ctc_ocr == "primary"
    assert prediction.fallback.primary_compile is None
    assert prediction.fallback.fallback_compile[0] == "SDF"
    assert prediction.diagnostics["invalid_ocr_fallback"] == {
        "enabled": True,
        "triggered": True,
        "selected": True,
        "reason": "primary_invalid_fallback_valid",
        "primary_valid": False,
        "fallback_valid": True,
        "primary_error": "invalid primary",
        "fallback_error": None,
        "checkpoint": "fallback.pt",
    }


def test_output_settings_preserve_cli_values() -> None:
    args = _args(
        "--model-version",
        "v-test",
        "--junction-source",
        "bond_endpoints",
        "--semantic-bond-relation-mode",
        "aromatic_ring_only",
    )
    settings = hybrid_output_settings(args)

    assert settings.output_dir == Path("out")
    assert settings.model_version == "v-test"
    assert settings.junction_source == "bond_endpoints"
    assert settings.semantic_bond_relation_mode == "aromatic_ring_only"


def test_write_prediction_delegates_with_runtime_timings() -> None:
    args = _args("--device", "cpu")
    pipeline = SimpleNamespace(ctc_ocr=None)
    runtime = HybridCLIRuntime(
        device=torch.device("cpu"),
        pipeline=pipeline,
        fallback_ctc_ocr=None,
        load_seconds=1.25,
    )
    prepared = HybridCLIInput(
        image=Image.new("RGB", (4, 4)),
        pixels=torch.zeros(1, 3, 4, 4),
        molecule_id="molecule",
    )
    fallback = SimpleNamespace(enabled=False)
    prediction = SimpleNamespace(
        scene={"scene": 1},
        diagnostics={"diagnostics": 1},
        fallback=fallback,
        inference_seconds=2.5,
    )
    received = {}

    def writer(**kwargs):
        received.update(kwargs)
        return {"written": True}

    result = write_hybrid_cli_prediction(
        args,
        runtime=runtime,
        prepared=prepared,
        prediction=prediction,
        output_writer=writer,
    )

    assert result == {"written": True}
    assert received["image"] is prepared.image
    assert received["scene"] == {"scene": 1}
    assert received["fallback"] is fallback
    assert received["load_seconds"] == pytest.approx(1.25)
    assert received["inference_seconds"] == pytest.approx(2.5)
