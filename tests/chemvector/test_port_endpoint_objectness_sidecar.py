from __future__ import annotations

import torch
import pytest

from tools.train_port_endpoint_objectness_sidecar import (
    PortEndpointObjectness,
    _average_precision,
    _evaluate,
    _select_threshold,
    _target_ids_for_ports,
    build_port_features,
)


def test_port_features_keep_only_unoccupied_masked_ports() -> None:
    bonds = {
        "bond_mask": torch.tensor([[True, False]]),
        "bond_endpoints": torch.tensor([[[[0.1, 0.1], [0.8, 0.8]], [[0.2, 0.2], [0.3, 0.3]]]]),
        "bond_features": torch.zeros((1, 2, 3)),
        "bond_types": torch.zeros((1, 2, 2)),
        "bond_scores": torch.tensor([[0.9, 0.1]]),
    }
    features, mask = build_port_features(bonds, torch.tensor([[0.1, 0.1]]))
    assert features.shape == (4, 11)
    assert mask.tolist() == [False, True, False, False]


def test_target_assignment_is_residual_and_allows_junction_ports() -> None:
    endpoints = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.9, 0.9]])
    labels = _target_ids_for_ports(
        endpoints,
        torch.tensor([True, True, True]),
        torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
        {0},
    )
    assert labels.tolist() == [-1, 1, -1]


def test_sidecar_parameter_budget_and_metrics_are_deterministic() -> None:
    model = PortEndpointObjectness(11)
    assert sum(parameter.numel() for parameter in model.parameters()) < 50_000
    scores = torch.tensor([0.9, 0.8, 0.1])
    labels = torch.tensor([True, False, True])
    assert _average_precision(scores, labels) == pytest.approx(5.0 / 6.0)
    threshold = _select_threshold(scores, labels)
    assert threshold["selected"] is not None
    result = _evaluate(scores, labels, torch.tensor([0, 0, 1]), 0.8)
    assert result["selected"] == 2
    assert result["matched"] == 1
