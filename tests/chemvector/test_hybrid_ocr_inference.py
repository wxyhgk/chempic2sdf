from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import chemvector.hybrid_ocr_inference as ocr
from chemvector.dual_vertex_selector import IMPLICIT_ROLE, TEXT_ROLE
from chemvector.hybrid_junction_selection import JunctionBatchSelection


def _selection(roles: tuple[int, ...]) -> JunctionBatchSelection:
    points = torch.tensor(
        [(0.2 + index * 0.2, 0.4) for index in range(len(roles))]
    )
    role_tensor = torch.tensor(roles)
    return JunctionBatchSelection(
        text_batches=[[]],
        point_batches=[points],
        role_batches=[role_tensor],
        padded_points=points.unsqueeze(0),
        point_mask=torch.ones((1, len(roles)), dtype=torch.bool),
    )


def _runtime(**overrides) -> ocr.HybridOCRRuntime:
    values = {
        "device": torch.device("cpu"),
        "base": None,
        "codec": None,
        "primary_ctc": None,
        "expert_ctc": None,
        "utility_router": None,
        "utility_feature_names": None,
        "utility_feature_mean": None,
        "utility_feature_std": None,
        "utility_threshold": 0.95,
        "anchor_fallback_margin": None,
    }
    values.update(overrides)
    return ocr.HybridOCRRuntime(**values)


def test_disabled_ocr_preserves_structure_and_uses_carbon_labels() -> None:
    selection = _selection((TEXT_ROLE, IMPLICIT_ROLE))
    results, diagnostics = ocr.decode_junction_labels(
        selection=selection,
        pixels=torch.zeros((1, 3, 32, 32)),
        features=torch.zeros((1, 4, 8, 8)),
        include_ocr=False,
        runtime=_runtime(),
    )
    points, roles, labels = results[0]
    assert points is selection.point_batches[0]
    assert roles is selection.role_batches[0]
    assert labels == ["C", "C"]
    assert diagnostics == {}


def test_legacy_ocr_only_labels_text_roles() -> None:
    class Head:
        def generate(self, *args, **kwargs):
            return torch.tensor(((1,), (2,)))

    class Codec:
        def decode(self, values):
            return {1: "N", 2: "O"}[values[0]]

    base = SimpleNamespace(
        ocr_source=lambda pixels, features: features,
        ocr_head=Head(),
    )
    results, diagnostics = ocr.decode_junction_labels(
        selection=_selection((TEXT_ROLE, IMPLICIT_ROLE)),
        pixels=torch.zeros((1, 3, 32, 32)),
        features=torch.zeros((1, 4, 8, 8)),
        include_ocr=True,
        runtime=_runtime(base=base, codec=Codec()),
    )
    assert results[0][2] == ["N", "C"]
    assert diagnostics == {}


def test_ocr_output_count_must_match_selected_junction_count() -> None:
    class Head:
        def generate(self, *args, **kwargs):
            return torch.tensor(((1,),))

    base = SimpleNamespace(
        ocr_source=lambda pixels, features: features,
        ocr_head=Head(),
    )
    codec = SimpleNamespace(decode=lambda values: "N")
    with pytest.raises(ValueError, match="OCR output count differs"):
        ocr.decode_junction_labels(
            selection=_selection((TEXT_ROLE, IMPLICIT_ROLE)),
            pixels=torch.zeros((1, 3, 32, 32)),
            features=torch.zeros((1, 4, 8, 8)),
            include_ocr=True,
            runtime=_runtime(base=base, codec=codec),
        )


def test_identical_primary_and_expert_predictions_do_not_call_router() -> None:
    runtime = _runtime(
        utility_router=SimpleNamespace(),
        utility_feature_names=("old_score",),
    )
    selected, disagreements, routes = ocr.route_ocr_predictions(
        [("N", -0.1)], [("N", -0.2)], runtime=runtime
    )
    assert selected == [("N", -0.1)]
    assert disagreements == 0
    assert routes == 0
