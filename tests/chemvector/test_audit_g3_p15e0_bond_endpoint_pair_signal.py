import torch

from chemvector.evaluation.audits.atoms.audit_g3_p15e0_bond_endpoint_pair_signal import (
    INCIDENT_DIM,
    MINIMUM_AUC,
    MINIMUM_AUC_LIFT,
    MINIMUM_RESCUE_ENDPOINT_COVERAGE,
    POSITIONS,
    gate,
    pair_incident_features,
    scaffold_fold,
)


def test_pair_incident_features_are_source_alternative_and_delta() -> None:
    values = torch.randn(5, POSITIONS, INCIDENT_DIM)
    source_rows = torch.tensor([1, 3])
    alternatives = torch.tensor([2, 6])
    result = pair_incident_features(values, source_rows, alternatives)
    assert result.shape == (2, INCIDENT_DIM * 3)
    torch.testing.assert_close(result[:, :INCIDENT_DIM], values[source_rows, 0])
    expected = torch.stack((values[1, 3], values[3, 7]))
    torch.testing.assert_close(result[:, INCIDENT_DIM : 2 * INCIDENT_DIM], expected)
    torch.testing.assert_close(
        result[:, 2 * INCIDENT_DIM :], expected - values[source_rows, 0]
    )


def test_scaffold_fold_keeps_group_together() -> None:
    assert scaffold_fold("same") == scaffold_fold("same")
    assert 0 <= scaffold_fold("other") < 3


def test_signal_gate_is_conjunctive() -> None:
    decision, checks = gate(
        MINIMUM_AUC,
        MINIMUM_AUC - MINIMUM_AUC_LIFT,
        MINIMUM_RESCUE_ENDPOINT_COVERAGE,
    )
    assert decision == "bond_endpoint_pair_signal_go"
    assert all(checks.values())
    assert gate(MINIMUM_AUC, MINIMUM_AUC, 1.0)[0].endswith("no_go")
