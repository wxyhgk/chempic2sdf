from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import chemvector.hybrid_junction_selection as selection
from chemvector.dual_vertex_selector import IMPLICIT_ROLE, TEXT_ROLE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "chemvector/inference/hybrid_junction_selection.py"


def _text(x: float, score: float) -> dict:
    return {
        "x": x,
        "y": 0.5,
        "score": score,
        "half_width": 0.02,
        "half_height": 0.02,
    }


def _implicit(x: float, score: float) -> dict:
    return {"x": x, "y": 0.5, "score": score, "aromatic": False}


def _install_candidate_decoders(monkeypatch, calls: list) -> None:
    main_text = [_text(0.1, 0.8), _text(0.2, 0.9)]
    filter_text = [_text(0.5, 0.7)]

    def decode_text(output, *, threshold, maximum_objects):
        calls.append(("decode_text", threshold, maximum_objects))
        return [main_text if threshold == 0.6 else filter_text]

    def decode_implicit(output, *, threshold, maximum_vertices, nms_kernel_size):
        calls.append(
            ("decode_implicit", threshold, maximum_vertices, nms_kernel_size)
        )
        return [[_implicit(0.5, 0.95), _implicit(0.9, 0.7)]]

    def fuse(query_output, merged_batches, *, maximum_candidates, image_size):
        calls.append(
            (
                "fuse",
                [item["x"] for item in merged_batches[0]],
                maximum_candidates,
                image_size,
            )
        )
        centers = torch.tensor(
            [[(item["x"], item["y"]) for item in merged_batches[0]]],
            dtype=torch.float32,
        )
        mask = torch.ones(centers.shape[:2], dtype=torch.bool)
        return {
            "grounded_centers": centers,
            "candidate_mask": mask,
            "candidate_auxiliary": torch.zeros((*mask.shape, 30)),
            "query_features": torch.ones((*mask.shape, 4)),
            "scalar_features": torch.ones((*mask.shape, 3)),
            "base_logits": torch.zeros(mask.shape),
        }

    monkeypatch.setattr(selection, "decode_text_objects", decode_text)
    monkeypatch.setattr(selection, "decode_vertices", decode_implicit)
    monkeypatch.setattr(selection, "fuse_dense_candidates_with_queries", fuse)


@pytest.mark.parametrize(
    ("patch_mode", "expects_patch"),
    (("none", False), ("feature", False), ("pixel", True), ("semantic_pixel", True)),
)
def test_prepare_candidates_preserves_threshold_roles_and_patch_modes(
    monkeypatch,
    patch_mode: str,
    expects_patch: bool,
) -> None:
    calls = []
    _install_candidate_decoders(monkeypatch, calls)
    prepared = selection.prepare_junction_candidates(
        pixels=torch.zeros((1, 3, 768, 768)),
        features=torch.zeros((1, 2, 8, 8)),
        query_output={},
        text_output=object(),
        implicit_output=object(),
        protocol={
            "text_threshold": 0.6,
            "text_filter_threshold": 0.3,
            "implicit_threshold": 0.4,
        },
        selector_config=SimpleNamespace(maximum_candidates=3),
        scoring_config=SimpleNamespace(patch_mode=patch_mode, patch_size=4),
        bbox_expansion_pixels=0.0,
    )
    assert calls[:3] == [
        ("decode_text", 0.6, 3),
        ("decode_text", 0.3, 3),
        ("decode_implicit", 0.4, 3, 1),
    ]
    assert calls[3] == ("fuse", [0.2, 0.1, 0.9], 3, 768)
    assert prepared.text_batches[0] == [_text(0.1, 0.8), _text(0.2, 0.9)]
    assert prepared.candidate_roles.tolist() == [[TEXT_ROLE, TEXT_ROLE, IMPLICIT_ROLE]]
    assert (prepared.candidate_pixel_patches is not None) is expects_patch


def test_finalize_preserves_role_order_suppression_and_padding() -> None:
    centers = torch.tensor(
        [
            ((0.10, 0.10), (0.11, 0.10), (0.10, 0.10), (0.80, 0.80), (0.0, 0.0)),
            ((0.20, 0.20), (0.70, 0.70), (0.50, 0.50), (0.51, 0.50), (0.0, 0.0)),
        ],
        dtype=torch.float64,
    )
    probabilities = torch.tensor(
        ((0.90, 0.80, 0.85, 0.70, 0.99), (0.90, 0.50, 0.90, 0.80, 0.99)),
        dtype=torch.float64,
    )
    mask = torch.tensor(
        ((True, True, True, True, False), (True, True, True, True, False))
    )
    roles = torch.tensor(
        (
            (IMPLICIT_ROLE, IMPLICIT_ROLE, TEXT_ROLE, TEXT_ROLE, IMPLICIT_ROLE),
            (IMPLICIT_ROLE, IMPLICIT_ROLE, TEXT_ROLE, TEXT_ROLE, IMPLICIT_ROLE),
        )
    )
    finalized = selection.finalize_junction_batches(
        text_batches=[[{"id": 0}], [{"id": 1}]],
        logits=torch.logit(probabilities),
        grounded_centers=centers,
        candidate_mask=mask,
        candidate_roles=roles,
        batch_size=2,
        real_threshold=0.6,
        suppression_radius_pixels=2.0,
        image_size=100,
        device=torch.device("cpu"),
    )
    assert finalized.role_batches[0].tolist() == [
        IMPLICIT_ROLE,
        TEXT_ROLE,
        TEXT_ROLE,
    ]
    assert finalized.role_batches[1].tolist() == [IMPLICIT_ROLE, TEXT_ROLE]
    torch.testing.assert_close(
        finalized.point_batches[0], centers[0, [0, 2, 3]]
    )
    torch.testing.assert_close(finalized.point_batches[1], centers[1, [0, 2]])
    assert finalized.padded_points.dtype == torch.float64
    assert finalized.padded_points.shape == (2, 3, 2)
    assert finalized.point_mask.tolist() == [[True, True, True], [True, True, False]]
    assert finalized.padded_points[1, 2].tolist() == [0.0, 0.0]


def test_finalize_preserves_indexed_empty_set_error() -> None:
    with pytest.raises(
        ValueError,
        match="Junction selector returned an empty set for batch item 1",
    ):
        selection.finalize_junction_batches(
            text_batches=[[], []],
            logits=torch.tensor(((10.0,), (-10.0,))),
            grounded_centers=torch.tensor((((0.2, 0.2),), ((0.8, 0.8),))),
            candidate_mask=torch.ones((2, 1), dtype=torch.bool),
            candidate_roles=torch.zeros((2, 1), dtype=torch.long),
            batch_size=2,
            real_threshold=0.5,
            suppression_radius_pixels=2.0,
            image_size=768,
            device=torch.device("cpu"),
        )


def test_select_junction_batches_matches_frozen_legacy_fixture(
    monkeypatch,
) -> None:
    calls = []
    _install_candidate_decoders(monkeypatch, calls)

    class Head:
        def __init__(self, name):
            self.name = name

        def __call__(self, features):
            calls.append((self.name, tuple(features.shape)))
            return {"head": self.name}

    class GroundedSelector:
        def encode(self, query, scalar, centers, mask):
            calls.append(("encode", tuple(centers.shape)))
            return torch.ones((1, 3, 4))

    class Graph:
        def __call__(self, hidden, pixels, centers, mask, base_logits):
            calls.append(("graph", tuple(hidden.shape)))
            return {
                "edge_logits": torch.ones((1, 1), requires_grad=True),
                "pair_indices": torch.tensor(((0, 1),)),
            }

    class Selector:
        config = SimpleNamespace(
            patch_mode="semantic_pixel",
            patch_size=4,
            edge_auxiliary_dim=4,
        )

        def __call__(self, *arguments, **optional):
            edge_logits = arguments[4]
            calls.append(
                (
                    "selector",
                    edge_logits.requires_grad,
                    optional["edge_auxiliary"].tolist(),
                    tuple(optional["candidate_pixel_patches"].shape),
                )
            )
            return torch.logit(torch.tensor(((0.9, 0.8, 0.85),)))

    def endpoint_features(pixels, centers, mask, pair_indices):
        calls.append(("edge_auxiliary", pair_indices.tolist()))
        return torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    monkeypatch.setattr(selection, "endpoint_ray_features", endpoint_features)
    pixels = torch.zeros((1, 3, 768, 768))
    result = selection.select_junction_batches(
        pixels=pixels,
        features=torch.zeros((1, 2, 8, 8)),
        device=torch.device("cpu"),
        query_detector=Head("query"),
        text_head=Head("text"),
        implicit_head=Head("implicit"),
        protocol={
            "text_threshold": 0.6,
            "text_filter_threshold": 0.3,
            "implicit_threshold": 0.4,
            "suppression_radius_pixels": 6.0,
        },
        selector_config=SimpleNamespace(maximum_candidates=3),
        selector=Selector(),
        grounded_selector=GroundedSelector(),
        graph=Graph(),
        real_threshold=0.5,
        bbox_expansion_pixels=0.0,
    )
    assert calls[:3] == [
        ("query", (1, 2, 8, 8)),
        ("text", (1, 2, 8, 8)),
        ("implicit", (1, 2, 8, 8)),
    ]
    assert calls[-2] == ("edge_auxiliary", [[0, 1]])
    assert calls[-1][0:2] == ("selector", False)
    assert result.text_batches[0] == [_text(0.1, 0.8), _text(0.2, 0.9)]
    assert result.role_batches[0].tolist() == [
        IMPLICIT_ROLE,
        TEXT_ROLE,
        TEXT_ROLE,
    ]
    torch.testing.assert_close(
        result.point_batches[0],
        torch.tensor(((0.9, 0.5), (0.2, 0.5), (0.1, 0.5))),
    )
    assert result.point_mask.tolist() == [[True, True, True]]


def test_select_junction_batches_is_thin_orchestration() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "select_junction_batches"
    )
    assert function.end_lineno - function.lineno + 1 <= 100
    definitions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "prepare_junction_candidates",
        "finalize_junction_batches",
    } <= definitions
