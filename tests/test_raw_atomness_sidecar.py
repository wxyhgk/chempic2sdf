import pytest
import torch

from chemvector.components.graph.proposals.raw_atomness_sidecar import (
    apply_raw_atomness_sidecar, audit_raw_atomness_decision,
)


def test_protected_prefix_is_immutable_and_low_scores_abstain():
    logits = torch.tensor([-100.0, -1.0, 0.0, 1.0])
    protected = torch.tensor([True, False, True, False])
    decision = apply_raw_atomness_sidecar(logits, protected)
    assert decision.keep_mask.tolist() == [True, False, True, True]
    assert decision.protected_mask.tolist() == protected.tolist()
    assert decision.sidecar_selected_mask.tolist() == [False, False, False, True]
    assert decision.abstained_mask.tolist() == [False, True, False, False]


def test_shape_dtype_and_finite_contracts():
    with pytest.raises(ValueError):
        apply_raw_atomness_sidecar(torch.ones(2, 1), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        apply_raw_atomness_sidecar(torch.ones(2), torch.zeros(2, dtype=torch.float32))
    with pytest.raises(ValueError):
        apply_raw_atomness_sidecar(torch.tensor([float("nan")]), torch.zeros(1, dtype=torch.bool))


def test_threshold_is_fixed_gate_not_a_sweep():
    logits = torch.tensor([-0.2, 0.2])
    protected = torch.zeros(2, dtype=torch.bool)
    decision = apply_raw_atomness_sidecar(logits, protected, threshold=0.1)
    assert decision.keep_mask.tolist() == [False, True]


def test_audit_reports_unknown_without_treating_it_as_negative():
    decision = apply_raw_atomness_sidecar(
        torch.tensor([1.0, 1.0, 1.0]), torch.tensor([True, False, False])
    )
    metrics = audit_raw_atomness_decision(
        ["protected", "strict_positive", "near_unknown"], decision
    )
    assert metrics["protected_retained"] == 1
    assert metrics["unknown_selected"] == 1
    assert metrics["hard_negative_selected"] == 0
