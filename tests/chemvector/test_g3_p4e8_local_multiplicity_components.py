from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p4e8_local_multiplicity_components import (
    choose_decision,
    deficient_bipartite_components,
)


def test_components_find_two_gt_one_candidate_deficiency() -> None:
    expected = torch.tensor([[0.0, 0.0], [4.0 / 768.0, 0.0]])
    predicted = torch.tensor([[2.0 / 768.0, 0.0]])

    components = deficient_bipartite_components(
        predicted, expected, radius_pixels=6.0
    )

    assert len(components) == 1
    assert components[0]["gt_count"] == 2
    assert components[0]["candidate_count"] == 1
    assert components[0]["deficiency"] == 1
    assert components[0]["kind"] == "shared_deficient"


def test_components_keep_gt_without_any_candidate() -> None:
    expected = torch.tensor([[0.0, 0.0], [100.0 / 768.0, 0.0]])
    predicted = torch.tensor([[1.0 / 768.0, 0.0]])

    components = deficient_bipartite_components(
        predicted, expected, radius_pixels=6.0
    )

    assert len(components) == 1
    assert components[0]["gt_indices"] == [1]
    assert components[0]["kind"] == "no_candidate"


def test_bounded_local_multiplicity_gate() -> None:
    components = [
        {"kind": "shared_deficient", "gt_count": 2, "deficiency": 1},
        {"kind": "shared_deficient", "gt_count": 4, "deficiency": 2},
        {"kind": "no_candidate", "gt_count": 1, "deficiency": 1},
    ]

    decision, checks = choose_decision(components)

    assert decision == "bounded_local_multiplicity_go"
    assert all(checks.values())
