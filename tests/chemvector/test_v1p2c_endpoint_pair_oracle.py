import torch

from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    _pair_costs,
    _route_result,
)


def test_pair_costs_match_unordered_bond_ports() -> None:
    points = torch.tensor(((0.1, 0.2), (0.8, 0.7), (0.5, 0.5)))
    pairs = torch.tensor(((0, 1), (0, 2)))
    expected = torch.tensor([[[0.8, 0.7], [0.1, 0.2]]])
    costs = _pair_costs(points, pairs, expected)
    assert costs.shape == (2, 1)
    assert float(costs[0, 0]) < 1e-4
    assert float(costs[1, 0]) > 6.0


def test_route_budget_counts_recovered_bonds_and_exact_images() -> None:
    costs = torch.tensor(
        (
            (0.0, 20.0),
            (20.0, 0.0),
            (20.0, 20.0),
            (20.0, 20.0),
        )
    )
    scores = torch.tensor((0.9, 0.8, 0.7, 0.6))
    result = _route_result(costs, scores, expected_bonds=2, multiplier=1)
    assert result == {"selected": 2, "recovered": 2, "exact": True}


def test_route_result_excludes_invalid_short_pairs() -> None:
    costs = torch.tensor(((0.0,), (20.0,)))
    scores = torch.tensor((-1e6, 0.2))
    result = _route_result(costs, scores, expected_bonds=1, multiplier=None)
    assert result == {"selected": 1, "recovered": 0, "exact": False}
