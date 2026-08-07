from __future__ import annotations

import pytest
import torch

from chemvector.evaluation.audits.atoms.audit_g3_p3e29_bond_incident_signal import (
    binary_auc,
    choose_oracle_fusion,
    continuous_incident_features,
    incident_feature_names,
)


def test_continuous_incident_features_favor_near_endpoint() -> None:
    candidates = torch.tensor(((0.1, 0.1), (0.5, 0.5), (0.9, 0.9)))
    scores = torch.tensor((0.8, 0.4))
    endpoints = torch.tensor(
        (
            ((0.1, 0.1), (0.5, 0.5)),
            ((0.2, 0.2), (0.3, 0.3)),
        )
    )
    types = torch.eye(2)
    mask = torch.tensor((True, False))
    result = continuous_incident_features(
        candidates, scores, endpoints, types, mask, image_size=100.0
    )
    names = incident_feature_names(type_count=2)
    support = result[:, names.index("support_max_sigma12")]
    assert result.shape == (3, len(names))
    assert support[0] == pytest.approx(0.8)
    assert support[1] == pytest.approx(0.8)
    assert support[2] < 0.01


def test_continuous_incident_features_ignore_masked_bonds() -> None:
    result = continuous_incident_features(
        torch.tensor(((0.2, 0.2),)),
        torch.tensor((1.0,)),
        torch.tensor((((0.2, 0.2), (0.3, 0.3)),)),
        torch.ones((1, 8)),
        torch.tensor((False,)),
    )
    assert result[0, 0] == pytest.approx(2**0.5)
    assert torch.count_nonzero(result[0, 1:]) == 0


def test_binary_auc_handles_ties() -> None:
    score = torch.tensor((0.0, 0.5, 0.5, 1.0))
    labels = torch.tensor((False, False, True, True))
    assert binary_auc(score, labels) == pytest.approx(0.875)


def test_choose_oracle_fusion_prefers_tail_then_global() -> None:
    results = {
        "bond_only": {"tail_exact_images": 8, "exact_images": 8, "matched_atoms": 8},
        "alpha=0": {"tail_exact_images": 0, "exact_images": 10, "matched_atoms": 90},
        "alpha=1": {"tail_exact_images": 5, "exact_images": 11, "matched_atoms": 91},
        "alpha=2": {"tail_exact_images": 5, "exact_images": 12, "matched_atoms": 90},
    }
    name, _ = choose_oracle_fusion(results)
    assert name == "alpha=2"


def test_multiscale_source_uses_explicit_three_feature_forward(monkeypatch) -> None:
    from chemvector.components.bonds.objects.dense_bond_object import (
        DenseBondObjectConfig,
        MultiScaleDenseBondObjectDetector,
    )
    from chemvector.evaluation.audits.atoms.audit_g3_p3e29_bond_incident_signal import (
        forward_bond_source,
    )

    config = DenseBondObjectConfig(input_channels=4, hidden_channels=4, slots=2)
    bond = MultiScaleDenseBondObjectDetector(config)
    observed = {}

    def fake_forward(stroke, stride8, stride16):
        observed["shapes"] = (stroke.shape, stride8.shape, stride16.shape)
        return {"ok": stroke}

    monkeypatch.setattr(bond, "forward", fake_forward)

    class Vision:
        stages = (None, None, None, torch.nn.Identity(), torch.nn.Identity())

        @staticmethod
        def forward_high_resolution_from_stroke(value):
            return value + 1

    stroke = torch.zeros((1, 4, 3, 3))
    output = forward_bond_source(Vision(), bond, stroke)
    assert output["ok"] is stroke
    assert observed["shapes"] == (stroke.shape, stroke.shape, stroke.shape)
