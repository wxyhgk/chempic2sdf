import torch

from chemvector.evaluation.audits.bonds.audit_v1p2d0_local_partner_oracle import (
    _selected_result,
)


def test_selected_result_reports_sparse_pair_oracle() -> None:
    costs = torch.tensor(
        (
            (0.0, 20.0),
            (20.0, 0.0),
            (20.0, 20.0),
        )
    )
    result = _selected_result(costs, torch.tensor((0, 1)), expected_bonds=2)
    assert result == {"selected": 2, "recovered": 2, "exact": True}


def test_selected_result_handles_empty_candidate_union() -> None:
    costs = torch.empty((0, 1))
    result = _selected_result(
        costs, torch.empty(0, dtype=torch.long), expected_bonds=1
    )
    assert result == {"selected": 0, "recovered": 0, "exact": False}
