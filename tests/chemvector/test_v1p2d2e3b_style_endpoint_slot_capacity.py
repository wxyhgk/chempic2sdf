from chemvector.evaluation.audits.bonds.audit_v1p2d2e3b_style_endpoint_slot_capacity import (
    endpoint_cell_occupancy,
    route_decision,
)


def test_endpoint_cell_occupancy_deduplicates_shared_ports() -> None:
    scene = {
        "bond_objects": [
            {"ports": [[100, 100], [200, 200]]},
            {"ports": [[100, 100], [202, 202]]},
        ]
    }
    occupancy = endpoint_cell_occupancy(scene)
    assert sum(occupancy.values()) == 3
    assert max(occupancy.values()) == 2


def test_slot_capacity_route_is_explicit() -> None:
    assert route_decision(3) == "current_slots_sufficient"
    assert route_decision(4) == "increase_slots_to_4"
    assert route_decision(5) == "dynamic_cell_set_required"
