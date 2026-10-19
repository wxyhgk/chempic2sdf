import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_next_route_keeps_selector_and_initial_frozen():
    path = ROOT / "outputs/audits/chemvector-oa13-oa33-coverage-next-route-20261017-r1/protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["status"] == "conditional_protocol_not_authorization"
    assert all(value is False for value in protocol["production_changes"].values())
    assert protocol["pass_branch"]["max_steps"] == 100
    assert protocol["pass_branch"]["max_wall_seconds"] == 300
    assert protocol["pass_branch"]["threshold_search"] is False
    assert protocol["fail_branch"]["gpu_training"] is False
