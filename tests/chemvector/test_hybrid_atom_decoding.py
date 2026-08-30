from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import chemvector.hybrid_atom_decoding as decoding
from chemvector.dual_vertex_selector import IMPLICIT_ROLE, TEXT_ROLE


def _runtime(**overrides) -> decoding.HybridAtomDecodingRuntime:
    values = {
        "device": torch.device("cpu"),
        "base": None,
        "text_head": None,
        "protocol": {"text_threshold": 0.2},
        "maximum_text_objects": 16,
        "ctc_ocr": None,
    }
    values.update(overrides)
    return decoding.HybridAtomDecodingRuntime(**values)


def test_structured_candidates_require_ctc_and_single_image() -> None:
    points = torch.tensor(((0.5, 0.5),))
    roles = torch.tensor((TEXT_ROLE,))
    with pytest.raises(ValueError, match="requires a CTC"):
        decoding.structured_atom_candidates(
            torch.zeros((1, 3, 32, 32)),
            points,
            roles,
            ["N"],
            runtime=_runtime(),
        )
    with pytest.raises(ValueError, match="expects one image"):
        decoding.structured_atom_candidates(
            torch.zeros((2, 3, 32, 32)),
            points,
            roles,
            ["N"],
            runtime=_runtime(ctc_ocr=object()),
        )


def test_unmatched_text_role_keeps_current_label_as_candidate(monkeypatch) -> None:
    class Vision:
        def forward_stroke_resolution(self, pixels):
            return torch.zeros((1, 4, 8, 8))

    monkeypatch.setattr(decoding, "decode_text_objects", lambda *a, **k: [[]])
    monkeypatch.setattr(
        decoding,
        "detected_text_crops",
        lambda pixels, points, roles, objects, **kwargs: (
            torch.empty((0, 3, 48, 192)),
            [],
        ),
    )
    runtime = _runtime(
        base=SimpleNamespace(vision_encoder=Vision()),
        text_head=lambda features: features,
        ctc_ocr=object(),
    )
    candidates = decoding.structured_atom_candidates(
        torch.zeros((1, 3, 32, 32)),
        torch.tensor(((0.3, 0.5), (0.7, 0.5))),
        torch.tensor((TEXT_ROLE, IMPLICIT_ROLE)),
        ["N", "C"],
        runtime=runtime,
    )
    assert candidates[0][0].text == "N"
    assert candidates[0][0].source == "current"
    assert candidates[1] == []
