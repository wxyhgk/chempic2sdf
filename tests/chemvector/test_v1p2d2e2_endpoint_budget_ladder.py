from chemvector.evaluation.audits.bonds.audit_v1p2d2e2_endpoint_budget_ladder import (
    MAXIMUM_POINTS,
    THRESHOLDS,
    config_name,
    route_decision,
)


def _configs(default: float = 0.95) -> dict:
    return {
        config_name(threshold, budget): {
            "unique_port_recall": default,
            "pairable_bond_recall": default,
        }
        for threshold in THRESHOLDS
        for budget in MAXIMUM_POINTS
    }


def test_endpoint_budget_route_prefers_fixed_threshold_budget() -> None:
    configs = _configs()
    configs[config_name(0.05, 384)] = {
        "unique_port_recall": 0.98,
        "pairable_bond_recall": 0.97,
    }
    assert route_decision(configs) == "maximum_points_ceiling"


def test_endpoint_budget_route_uses_threshold_only_when_budget_fails() -> None:
    configs = _configs()
    configs[config_name(0.02, 512)] = {
        "unique_port_recall": 0.98,
        "pairable_bond_recall": 0.98,
    }
    assert route_decision(configs) == "endpoint_threshold_ceiling"


def test_endpoint_budget_route_retrains_when_all_configs_fail() -> None:
    assert route_decision(_configs()) == "endpoint_model_retrain"
