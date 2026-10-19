import torch

from chemvector.components.graph.proposals.vertex_bond_integration import (
    attach_vertex_candidates,
    attachment_geometry_metrics,
    deduplicate_vertex_candidates,
)


def test_dedup_is_score_ranked_and_target_free():
    points = torch.tensor([[0.1, 0.1], [0.102, 0.1], [0.5, 0.5]])
    kept, suppressed = deduplicate_vertex_candidates(
        points, torch.tensor([0.2, 0.9, 0.1]), radius_px=3.0
    )
    assert kept.tolist() == [1, 2]
    assert suppressed.tolist() == [True, False, False]


def test_junction_reuse_allowed_but_same_bond_reuse_vetoed():
    # Two bonds share a junction at (100,100); the first bond's opposite end
    # is at (200,100), the second at (100,200).
    endpoints = torch.tensor([
        [[100 / 768, 100 / 768], [200 / 768, 100 / 768]],
        [[100 / 768, 100 / 768], [100 / 768, 200 / 768]],
    ])
    points = torch.tensor([
        [100 / 768, 100 / 768],
        [200 / 768, 100 / 768],
        [100 / 768, 200 / 768],
    ])
    result = attach_vertex_candidates(
        points, endpoints, torch.tensor([True, True]), scores=torch.tensor([1., .8, .7])
    )
    assert result.endpoint_candidate.tolist() == [[0, 1], [0, 2]]
    metrics = attachment_geometry_metrics(result, torch.tensor([True, True]))
    assert metrics["two_endpoint_bonds"] == 2
    assert metrics["candidate_reuse_endpoints"] == 1
    assert metrics["candidate_reuse_conflicts"] == 0


def test_same_candidate_cannot_claim_both_ends_of_one_bond():
    endpoints = torch.tensor([[[100 / 768, 100 / 768], [104 / 768, 100 / 768]]])
    points = torch.tensor([[102 / 768, 100 / 768], [300 / 768, 300 / 768]])
    result = attach_vertex_candidates(
        points, endpoints, torch.tensor([True]), radius_px=4.0, duplicate_radius_px=1.0
    )
    assert result.endpoint_candidate[0].tolist().count(0) == 1
    assert int((result.endpoint_candidate >= 0).sum()) == 1
