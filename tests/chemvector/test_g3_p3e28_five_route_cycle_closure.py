from __future__ import annotations

from chemvector.evaluation.audits.atoms.audit_g3_p3e28_five_route_cycle_closure import (
    closure_metrics,
)


def test_closure_stops_when_only_bond_can_still_pass() -> None:
    metrics = {
        "hard": {"known_atom_exact": 47, "predicted_atom_exact": 36},
        "soft": {"known_atom_exact": 49, "predicted_atom_exact": 38},
        "query": {"known_atom_exact": 11, "predicted_atom_exact": 11},
    }
    result = closure_metrics(metrics)
    assert result["best_known_atom_exact_upper_bound"] == 49
    assert result["hybrid_structured_count_can_pass"] is False
    assert result["maximum_possible_passes"] == 1
    assert result["cycle_can_reach_micro_gate"] is False


def test_closure_would_not_stop_with_one_executed_pass() -> None:
    metrics = {
        "hard": {"known_atom_exact": 152, "predicted_atom_exact": 152},
        "soft": {"known_atom_exact": 49, "predicted_atom_exact": 38},
    }
    result = closure_metrics(metrics)
    assert result["maximum_possible_passes"] >= 2
    assert result["cycle_can_reach_micro_gate"] is True
