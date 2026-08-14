import torch

from chemvector.evaluation.audits.bonds.build_v1p2d2e6_candidate_graph_capacity_cache import (
    candidate_graph_set_targets,
)


def test_candidate_graph_targets_group_equivalent_nodes_and_edges() -> None:
    expected = torch.tensor(
        (
            ((0.1, 0.2), (0.5, 0.5)),
            ((0.5, 0.5), (0.9, 0.2)),
        )
    )
    points = torch.tensor(
        ((0.1, 0.2), (0.5, 0.5), (0.9, 0.2), (0.502, 0.501))
    )
    pairs = torch.tensor(((0, 1), (1, 2), (0, 3), (3, 2)))
    targets = candidate_graph_set_targets(
        points, pairs, expected, match_radius_pixels=6.0, image_size=768.0
    )
    assert targets["node_port_ids"].tolist() == [0, 1, 2, 1]
    assert targets["node_target_degrees"].tolist() == [1, 2, 1, 2]
    assert targets["edge_target_bond_ids"].tolist() == [0, 1, 0, 1]
    assert targets["reachable_bond_count"] == 2


def test_candidate_graph_targets_report_unreachable_bond() -> None:
    targets = candidate_graph_set_targets(
        torch.tensor(((0.1, 0.2), (0.5, 0.5), (0.9, 0.2))),
        torch.tensor(((0, 1),)),
        torch.tensor(
            (
                ((0.1, 0.2), (0.5, 0.5)),
                ((0.5, 0.5), (0.9, 0.2)),
            )
        ),
    )
    assert targets["reachable_bond_count"] == 1
