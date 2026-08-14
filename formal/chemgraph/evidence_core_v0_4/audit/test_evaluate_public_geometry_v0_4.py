from formal.chemgraph.evidence_core_v0_4.audit.evaluate_public_geometry_v0_4 import (
    maximum_matching,
    unordered_endpoint_cost,
)


def test_unordered_endpoint_cost_ignores_direction() -> None:
    segment = ((16, 32), (160, 320))
    assert unordered_endpoint_cost(segment, tuple(reversed(segment))) == (0.0, 0.0)


def test_matching_is_one_to_one() -> None:
    truth = [((0, 0), (160, 0)), ((0, 160), (160, 160))]
    predicted = [((0, 0), (160, 0)), ((0, 0), (160, 0)), ((0, 160), (160, 160))]
    pairs, cost = maximum_matching(truth, predicted)
    assert len(pairs) == 2
    assert len({predicted_id for _, predicted_id in pairs}) == 2
    assert cost == 0.0


def test_matching_rejects_large_endpoint_error() -> None:
    pairs, _ = maximum_matching([((0, 0), (160, 0))], [((0, 400), (160, 400))])
    assert pairs == []
