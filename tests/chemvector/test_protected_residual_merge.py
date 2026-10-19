from __future__ import annotations

import pytest
import torch

from chemvector.components.graph.proposals.protected_residual_merge import (
    CandidateSet,
    baseline_prefix_unchanged,
    merge_candidates,
    select_residual_candidates,
)
from chemvector.components.graph.proposals.residual_bond_attachment import (
    attach_residual_candidates,
)


def test_residual_selection_is_independent_and_deterministic() -> None:
    points = torch.tensor([[.4, .4], [.1, .1], [.2, .2], [.8, .8]])
    scores = torch.tensor([.7, .9, .8, .2])
    selected = select_residual_candidates(points, scores, threshold=.5, maximum=2)
    assert selected.source == "residual"
    assert torch.equal(selected.points, points[[1, 2]])
    assert torch.equal(selected.scores, scores[[1, 2]])


def test_merge_does_not_change_baseline_prefix() -> None:
    baseline = CandidateSet(
        torch.tensor([[.1, .1], [.5, .5]]),
        torch.tensor([.8, .6]),
        "baseline",
    )
    residual = CandidateSet(
        torch.tensor([[.1001, .1], [.9, .9]]),
        torch.tensor([.99, .7]),
        "residual",
    )
    merged = merge_candidates(baseline, residual, duplicate_radius=.01)
    assert merged.points.shape == (3, 2)
    assert torch.equal(merged.points[:2], baseline.points)
    assert torch.equal(merged.scores[:2], baseline.scores)
    assert baseline_prefix_unchanged(baseline, merged)


def test_empty_residual_and_invalid_sources() -> None:
    baseline = CandidateSet(torch.zeros((1, 2)), torch.ones(1), "baseline")
    residual = CandidateSet(torch.zeros((0, 2)), torch.zeros(0), "residual")
    merged = merge_candidates(baseline, residual)
    assert baseline_prefix_unchanged(baseline, merged)
    with pytest.raises(ValueError):
        merge_candidates(residual, baseline)


def test_residual_bond_attachment_respects_existing_endpoint_ownership() -> None:
    points = torch.tensor([[.101, .1], [.9, .9]])
    endpoints = torch.tensor([
        [[.1, .1], [.5, .5]],
        [[.9, .9], [.2, .2]],
    ])
    mask = torch.tensor([True, True])
    occupied = torch.tensor([[True, False], [False, False]])
    result = attach_residual_candidates(
        points, endpoints, mask, radius_px=2.0, occupied_endpoints=occupied
    )
    assert result["matched"].tolist() == [False, True]
    assert result["bond_indices"].tolist() == [-1, 1]
    assert result["endpoint_indices"].tolist() == [-1, 0]
