from chemvector.audit_proposal_delta_actions import outcome_name, summarize_actions


def test_outcome_name_covers_rescue_and_break() -> None:
    assert outcome_name(False, True) == "rescue"
    assert outcome_name(True, False) == "break"
    assert outcome_name(True, True) == "preserved_correct"
    assert outcome_name(False, False) == "preserved_wrong"


def test_summary_reports_domain_net_exact() -> None:
    rows = [
        {"domain": "a", "outcome": "rescue", "features": {"bond_additions": 1, "bond_deletions": 0, "incidence_endpoint_changes": 0}},
        {"domain": "a", "outcome": "break", "features": {"bond_additions": 0, "bond_deletions": 1, "incidence_endpoint_changes": 0}},
        {"domain": "b", "outcome": "rescue", "features": {"bond_additions": 0, "bond_deletions": 0, "incidence_endpoint_changes": 1}},
    ]

    summary = summarize_actions(rows)

    assert summary["net_full_exact"] == 1
    assert summary["domains"]["a"]["net_full_exact"] == 0
    assert summary["domains"]["b"]["action_samples"] == 1
