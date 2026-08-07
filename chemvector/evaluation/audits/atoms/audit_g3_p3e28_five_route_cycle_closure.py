#!/usr/bin/env python3
"""Close the P3-E21 five-route cycle from fixed Micro results and upper bounds."""

from __future__ import annotations

import json
from pathlib import Path

from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e28-five-route-cycle-closure"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
MICRO_EXACT_GATE = 152
MINIMUM_PASSING_ROUTES = 2
ROUTES = {
    "hard_latent_occupancy": {
        "path": Path(
            "outputs/checkpoints/chemvector-g3-p3e25-hard-latent-occupancy-micro160/"
            "final_metrics.json"
        ),
        "sha256": "fff2096e55b23e856af204683318f45cb7f18ba98a41a0e3c9da24f0b57a04e9",
        "decision": "hard_latent_occupancy_micro160_no_go",
    },
    "soft_coverage_occupancy": {
        "path": Path(
            "outputs/checkpoints/chemvector-g3-p3e26-soft-coverage-occupancy-micro160/"
            "final_metrics.json"
        ),
        "sha256": "b5fd8110ce4c0ac3d70d8dce5fc20c8bb9dd7a162c53800f123f19c863a04425",
        "decision": "soft_coverage_occupancy_micro160_no_go",
    },
    "query_slot_occupancy": {
        "path": Path(
            "outputs/checkpoints/chemvector-g3-p3e27-query-slot-occupancy-micro160/"
            "final_metrics.json"
        ),
        "sha256": "760414e2dbf4711bccf5cf818ecad63de1633ef4d6a0045076d72f4e6199c623",
        "decision": "query_slot_occupancy_micro160_no_go",
    },
}
BOND_SOURCE = Path(
    "outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt"
)
BOND_SOURCE_SHA256 = "e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac"


def closure_metrics(route_metrics: dict[str, dict]) -> dict:
    passed = [
        name for name, value in route_metrics.items()
        if value["known_atom_exact"] >= MICRO_EXACT_GATE
        and value["predicted_atom_exact"] >= MICRO_EXACT_GATE
    ]
    best_known_upper_bound = max(
        value["known_atom_exact"] for value in route_metrics.values()
    )
    hybrid_can_pass = best_known_upper_bound >= MICRO_EXACT_GATE
    remaining_bond_maximum = 1
    maximum_possible_passes = (
        len(passed) + int(hybrid_can_pass) + remaining_bond_maximum
    )
    return {
        "passing_executed_routes": passed,
        "passing_executed_route_count": len(passed),
        "best_known_atom_exact_upper_bound": best_known_upper_bound,
        "hybrid_structured_count_can_pass": hybrid_can_pass,
        "remaining_bond_route_maximum": remaining_bond_maximum,
        "maximum_possible_passes": maximum_possible_passes,
        "minimum_required_passes": MINIMUM_PASSING_ROUTES,
        "cycle_can_reach_micro_gate": (
            maximum_possible_passes >= MINIMUM_PASSING_ROUTES
        ),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E28 output is not empty: {OUTPUT}")
    if file_sha256(BOND_SOURCE) != BOND_SOURCE_SHA256:
        raise ValueError("P3-E28 frozen Bond source SHA256 differs")
    metrics = {}
    input_audit = {}
    for name, config in ROUTES.items():
        if file_sha256(config["path"]) != config["sha256"]:
            raise ValueError(f"P3-E28 route metrics SHA256 differs: {name}")
        payload = json.loads(config["path"].read_text())
        if payload["decision"] != config["decision"]:
            raise ValueError(f"P3-E28 route decision differs: {name}")
        values = payload["metrics"]
        metrics[name] = {
            "decision": payload["decision"],
            "count_exact": int(values["count_exact"]),
            "known_atom_exact": int(
                values["known_count"]["whole_set_exact_images"]
            ),
            "predicted_atom_exact": int(
                values["predicted"]["whole_set_exact_images"]
            ),
            "newly_reachable_exact": int(values["newly_reachable_exact"]),
        }
        input_audit[name] = {
            "path": str(config["path"].resolve()),
            "sha256": config["sha256"],
        }
    closure = closure_metrics(metrics)
    checks = {
        "three_executed_routes_no_go": all(
            value["decision"].endswith("_no_go") for value in metrics.values()
        ),
        "three_executed_routes_below_atom_gate": all(
            value["known_atom_exact"] < MICRO_EXACT_GATE
            or value["predicted_atom_exact"] < MICRO_EXACT_GATE
            for value in metrics.values()
        ),
        "hybrid_known_count_upper_bound_below_gate": (
            closure["best_known_atom_exact_upper_bound"] < MICRO_EXACT_GATE
        ),
        "maximum_possible_passes_below_required": (
            closure["maximum_possible_passes"] < MINIMUM_PASSING_ROUTES
        ),
        "bond_source_available_for_future_cycle": True,
    }
    decision = (
        "five_route_micro_gate_no_go"
        if all(checks.values())
        else "five_route_cycle_closure_protocol_failed"
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    audit = {
        "schema_version": "g3-p3e28-five-route-cycle-closure-audit-0.1",
        "decision": decision,
        "checks": checks,
        "route_metrics": metrics,
        "closure": closure,
        "hybrid_proof": (
            "predicted_exact = count_exact AND known_selector_exact; "
            "therefore predicted_exact <= known_selector_exact"
        ),
        "bond_conditioned": {
            "status": "feasible_for_future_cycle_but_cannot_unlock_current_cycle",
            "source": str(BOND_SOURCE.resolve()),
            "source_sha256": BOND_SOURCE_SHA256,
            "evidence": "GT-free frozen BondObject endpoints -> continuous incident kernel",
        },
        "inputs": input_audit,
        "data_access": {
            "dev_read": False,
            "old_confirm_read": False,
            "confirm_v2_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
        "downstream_sdf_run": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
