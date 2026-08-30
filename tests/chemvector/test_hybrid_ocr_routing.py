from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import chemvector.hybrid_ocr_routing as routing


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RecordingRouter:
    def __init__(self, outputs: tuple[float, ...]) -> None:
        self.outputs = outputs
        self.features = None

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        self.features = features.detach().clone()
        return torch.tensor(self.outputs, device=features.device)


def _runtime(**overrides):
    values = {
        "device": torch.device("cpu"),
        "utility_router": RecordingRouter((0.0,)),
        "utility_feature_names": ("old_score", "new_score"),
        "utility_feature_mean": torch.zeros(2),
        "utility_feature_std": torch.ones(2),
        "utility_threshold": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_legacy_module_reexports_routing_function() -> None:
    import chemvector.hybrid_ocr_inference as inference

    assert inference.route_ocr_predictions is routing.route_ocr_predictions


def test_routing_module_does_not_depend_on_pipeline_or_junction_layers() -> None:
    source = PROJECT_ROOT / "chemvector/inference/hybrid_ocr_routing.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "chemvector.infer_chemscene_hybrid" not in imports
    assert "chemvector.hybrid_checkpoint_loader" not in imports
    assert "chemvector.hybrid_junction_selection" not in imports


def test_missing_router_and_mismatched_lengths_keep_existing_error() -> None:
    with pytest.raises(ValueError, match="^OCR utility routing inputs differ$"):
        routing.route_ocr_predictions(
            [("N", -0.1)], [("O", -0.2)], runtime=_runtime(utility_router=None)
        )
    with pytest.raises(ValueError, match="^OCR utility routing inputs differ$"):
        routing.route_ocr_predictions(
            [("N", -0.1)], [], runtime=_runtime()
        )


def test_no_disagreement_preserves_primary_without_calling_router() -> None:
    class FailingRouter:
        def __call__(self, features):
            raise AssertionError("router must not run")

    primary = [("N", -0.1), ("Cl", -0.4)]
    selected, disagreements, routes = routing.route_ocr_predictions(
        primary,
        [("N", -0.2), ("Cl", -0.3)],
        runtime=_runtime(utility_router=FailingRouter()),
    )
    assert selected == primary
    assert selected is not primary
    assert disagreements == 0
    assert routes == 0


def test_visual_schema_requires_paired_descriptors() -> None:
    runtime = _runtime(
        utility_feature_names=("old_visual_00",),
        utility_feature_mean=torch.zeros(1),
        utility_feature_std=torch.ones(1),
    )
    with pytest.raises(
        ValueError, match="^Visual OCR router requires paired descriptors$"
    ):
        routing.route_ocr_predictions(
            [("N", -0.1)], [("O", -0.2)], runtime=runtime
        )


def test_cross_score_schema_requires_paired_logits() -> None:
    runtime = _runtime(
        utility_feature_names=("old_model_new_text_score",),
        utility_feature_mean=torch.zeros(1),
        utility_feature_std=torch.ones(1),
    )
    with pytest.raises(
        ValueError, match="^Cross-score OCR router requires paired logits$"
    ):
        routing.route_ocr_predictions(
            [("N", -0.1)], [("O", -0.2)], runtime=runtime
        )


def test_dynamic_features_are_standardized_in_declared_order() -> None:
    router = RecordingRouter((10.0,))
    runtime = _runtime(
        utility_router=router,
        utility_feature_names=(
            "new_score",
            "old_score",
            "score_delta_new_minus_old",
        ),
        utility_feature_mean=torch.tensor((2.0, 1.0, 3.0)),
        utility_feature_std=torch.tensor((4.0, 2.0, 5.0)),
    )
    selected, disagreements, routes = routing.route_ocr_predictions(
        [("N", 3.0)], [("O", 10.0)], runtime=runtime
    )
    assert router.features.shape == (1, 3)
    assert torch.allclose(router.features, torch.tensor(((2.0, 1.0, 0.8),)))
    assert selected == [("O", 10.0)]
    assert disagreements == 1
    assert routes == 1


def test_visual_and_cross_features_follow_dynamic_schema(monkeypatch) -> None:
    router = RecordingRouter((-10.0,))
    runtime = _runtime(
        utility_router=router,
        utility_feature_names=(
            "old_visual_00",
            "new_visual_00",
            "visual_delta_00",
            "old_model_new_text_score",
            "new_model_old_text_score",
        ),
        utility_feature_mean=torch.zeros(5),
        utility_feature_std=torch.ones(5),
    )
    calls = []

    def score(logits, text, *, normalize_by_bytes):
        calls.append((logits, text, normalize_by_bytes))
        return -3.0 if text == "O" else -4.0

    monkeypatch.setattr(routing, "ctc_text_log_score", score)
    primary_logits = torch.zeros((1, 2, 3))
    expert_logits = torch.ones((1, 2, 3))
    primary_visual = torch.zeros((1, 48))
    expert_visual = torch.ones((1, 48))
    selected, disagreements, routes = routing.route_ocr_predictions(
        [("N", -0.1)],
        [("O", -0.2)],
        runtime=runtime,
        primary_visual=primary_visual,
        expert_visual=expert_visual,
        primary_logits=primary_logits,
        expert_logits=expert_logits,
    )
    assert torch.equal(
        router.features, torch.tensor(((0.0, 1.0, 1.0, -3.0, -4.0),))
    )
    assert len(calls) == 2
    assert torch.equal(calls[0][0], primary_logits[0])
    assert calls[0][1:] == ("O", True)
    assert torch.equal(calls[1][0], expert_logits[0])
    assert calls[1][1:] == ("N", True)
    assert selected == [("N", -0.1)]
    assert (disagreements, routes) == (1, 0)


def test_mixed_routes_preserve_order_and_disagreement_shape_contract() -> None:
    router = RecordingRouter((10.0, -10.0))
    runtime = _runtime(utility_router=router)
    primary = [("N", -0.1), ("Cl", -0.2), ("B", -0.3)]
    expert = [("O", -0.4), ("Cl", -0.5), ("F", -0.6)]

    selected, disagreements, routes = routing.route_ocr_predictions(
        primary, expert, runtime=runtime
    )

    assert router.features.shape == (2, 2)
    assert selected == [("O", -0.4), ("Cl", -0.2), ("B", -0.3)]
    assert disagreements == 2
    assert routes == 1
