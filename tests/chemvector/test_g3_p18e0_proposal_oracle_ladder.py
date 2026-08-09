import torch

from chemvector.evaluation.audits.graph.audit_g3_p18e0_proposal_oracle_ladder import (
    _maximum_bond_assignment,
    _maximum_radius_assignment,
    _record_oracle_metrics,
    select_fresh640,
)


def test_select_fresh640_is_group_complete_and_deterministic(monkeypatch):
    monkeypatch.setattr(
        "chemvector.evaluation.audits.graph.audit_g3_p18e0_proposal_oracle_ladder.MOLECULES",
        2,
    )
    rows = []
    for molecule in range(4):
        for variant in ("coordgen", "original", "rdkit", "stress"):
            rows.append(
                {
                    "molecule_id": f"mol-{molecule}",
                    "sample_key": f"mol-{molecule}-{variant}",
                    "variant": variant,
                    "scaffold": f"scaffold-{molecule}",
                }
            )
    first, first_audit = select_fresh640(rows)
    second, second_audit = select_fresh640(list(reversed(rows)))
    assert [row["sample_key"] for row in first] == [row["sample_key"] for row in second]
    assert first_audit == second_audit
    assert len(first) == 8
    assert all(count == 2 for count in first_audit["variants"].values())


def test_radius_assignment_maximizes_feasible_cardinality():
    expected = torch.tensor([[0.0, 0.0], [10.0 / 768.0, 0.0]])
    predicted = torch.tensor([[5.0 / 768.0, 0.0], [0.0, 0.0]])
    assignment = _maximum_radius_assignment(predicted, expected, 6.0)
    assert assignment.tolist() == [1, 0]


def test_bond_assignment_is_endpoint_order_invariant():
    expected = torch.tensor([[[0.1, 0.2], [0.8, 0.9]]])
    predicted = expected.flip(1)
    assignment, reversed_assignment = _maximum_bond_assignment(
        predicted, expected, 1.0
    )
    assert assignment.tolist() == [0]
    assert reversed_assignment.tolist() == [True]


def test_oracle_metrics_compile_exact_graph():
    atom_anchors = torch.tensor([[0.1, 0.1], [0.5, 0.1], [0.9, 0.1]])
    atom_padding = torch.zeros(5, 2)
    bond_endpoints = torch.tensor(
        [
            [[0.1, 0.1], [0.5, 0.1]],
            [[0.9, 0.1], [0.5, 0.1]],
        ]
    )
    bond_type = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 2
    )
    record = {
        "target_atom_anchors": torch.cat((atom_anchors, atom_padding)),
        "target_atom_mask": torch.tensor([True, True, True, False, False, False, False, False]),
        "target_bond_atom_refs": torch.tensor([[0, 1], [1, 2], [0, 0]]),
        "target_bond_types": torch.tensor([0, 0, 0]),
        "target_bond_mask": torch.tensor([True, True, False]),
        "atom_candidate_anchors": atom_anchors[[2, 0, 1]],
        "atom_candidate_mask": torch.ones(3, dtype=torch.bool),
        "bond_candidate_endpoints": bond_endpoints,
        "bond_candidate_types": bond_type,
        "bond_candidate_mask": torch.ones(2, dtype=torch.bool),
    }
    metrics = _record_oracle_metrics(record)
    assert metrics["atom_exact"]
    assert metrics["bond_exact"]
    assert metrics["typed_bond_exact"]
    assert metrics["incidence_feasible"]
    assert metrics["compiled_exact"]
