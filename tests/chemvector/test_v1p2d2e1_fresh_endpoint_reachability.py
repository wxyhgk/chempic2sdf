from chemvector.evaluation.audits.bonds.audit_v1p2d2e1_fresh_endpoint_reachability import (
    route_decision,
)


def _summary(port_recall: float, pair_recall: float) -> dict:
    return {
        "candidate_unique_port_recall": port_recall,
        "routes": {
            "endpoint_score": {"full": {"pair_recall": pair_recall}},
            "line_score": {"full": {"pair_recall": pair_recall}},
            "line_blocker_score": {"full": {"pair_recall": pair_recall}},
            "line_gated_blocker_score": {"full": {"pair_recall": pair_recall}},
        },
    }


def test_reachability_routes_to_endpoint_when_ports_are_missing() -> None:
    assert route_decision(_summary(0.96, 0.98)) == "endpoint_reachability_blocker"


def test_reachability_routes_to_partner_when_exhaustive_candidates_pass() -> None:
    assert route_decision(_summary(0.98, 0.98)) == "partner_selection_blocker"
