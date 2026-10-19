from __future__ import annotations

import torch

from tools.audit_bond_port_candidate_reachability import (
    _locked_target_ids,
    _match_bond_ports,
    _summarize,
)
from chemvector.components.graph.proposals.protected_residual_merge import CandidateSet


def test_match_bond_ports_records_reversal_without_target_selection() -> None:
    predicted = torch.tensor([[[0.8, 0.2], [0.2, 0.2]]])
    target = torch.tensor([[[0.2, 0.2], [0.8, 0.2]]])
    mapping, target_ids = _match_bond_ports(
        predicted,
        torch.tensor([True]),
        target,
        torch.tensor([True]),
    )
    assert target_ids.tolist() == [0]
    assert mapping == {0: (0, True)}


def test_locked_target_ids_are_one_to_one_and_unmatched_safe() -> None:
    residual = CandidateSet(
        torch.tensor([[0.2, 0.2], [0.8, 0.8], [0.8, 0.8]]),
        torch.tensor([0.9, 0.8, 0.7]),
        "residual",
    )
    target = torch.tensor([[0.2, 0.2], [0.8, 0.8]])
    labels = _locked_target_ids({0}, residual, target)
    assert labels.tolist() == [-1, 1, -1]


def test_summary_uses_boolean_masks_for_single_candidate() -> None:
    result = _summarize([
        {
            "sample_key": "one",
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([-1]),
            "attached": torch.tensor([True]),
            "oracle_attached": torch.tensor([False]),
            "port_distance": torch.tensor([1.0]),
        }
    ])
    assert result["candidates"] == 1
    assert result["positives"] == 0
