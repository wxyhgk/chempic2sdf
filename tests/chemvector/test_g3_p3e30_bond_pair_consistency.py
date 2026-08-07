from __future__ import annotations

import pytest
import torch

from chemvector.evaluation.audits.atoms.audit_g3_p3e30_bond_pair_consistency import (
    adaptive_pair_affinity,
    choose_oracle_result,
    greedy_pair_selection,
)


def test_pair_affinity_requires_opposite_bond_endpoints() -> None:
    candidates = torch.tensor(((0.1, 0.1), (0.9, 0.9), (0.1, 0.9)))
    affinity, scale = adaptive_pair_affinity(
        candidates,
        torch.tensor((1.0,)),
        torch.tensor((((0.1, 0.1), (0.9, 0.9)),)),
        torch.tensor((True,)),
        scale_ratio=0.16,
    )
    assert scale > 0.0
    assert affinity[0, 1] == pytest.approx(1.0)
    assert affinity[1, 0] == pytest.approx(1.0)
    assert affinity[0, 0] == 0.0
    assert affinity[0, 2] < 0.01


def test_pair_affinity_scale_tracks_bond_length() -> None:
    candidates = torch.tensor(((0.1, 0.1), (0.9, 0.9)))
    _, short = adaptive_pair_affinity(
        candidates,
        torch.tensor((1.0,)),
        torch.tensor((((0.1, 0.1), (0.3, 0.1)),)),
        torch.tensor((True,)),
        scale_ratio=0.1,
    )
    _, long = adaptive_pair_affinity(
        candidates,
        torch.tensor((1.0,)),
        torch.tensor((((0.1, 0.1), (0.7, 0.1)),)),
        torch.tensor((True,)),
        scale_ratio=0.1,
    )
    assert long == pytest.approx(short * 3.0)


def test_greedy_pair_selection_can_complete_low_raw_partner() -> None:
    raw = torch.tensor((3.0, 2.0, 0.0))
    affinity = torch.zeros((3, 3))
    affinity[0, 2] = affinity[2, 0] = 1.0
    baseline = greedy_pair_selection(raw, affinity, 2, alpha=0.0)
    structured = greedy_pair_selection(raw, affinity, 2, alpha=4.0)
    assert baseline.tolist() == [0, 1]
    assert structured.tolist() == [0, 2]


def test_choose_oracle_result_prioritizes_tail_exact() -> None:
    results = {
        "ratio=0.1,alpha=0": {
            "tail_exact_images": 0,
            "exact_images": 20,
            "matched_atoms": 100,
        },
        "ratio=0.1,alpha=1": {
            "tail_exact_images": 3,
            "exact_images": 10,
            "matched_atoms": 90,
        },
    }
    name, _ = choose_oracle_result(results)
    assert name == "ratio=0.1,alpha=1"
