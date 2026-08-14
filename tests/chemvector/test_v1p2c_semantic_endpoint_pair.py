import torch

from chemvector.components.bonds.objects.semantic_endpoint_pair import (
    SemanticEndpointPairConfig,
    enumerate_endpoint_pairs,
    score_endpoint_pairs,
)


def test_endpoint_pairs_have_no_maximum_length_cutoff() -> None:
    points = torch.tensor(((0.01, 0.5), (0.99, 0.5), (0.5, 0.5)))
    pairs = enumerate_endpoint_pairs(points)
    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2]]
    probability = torch.zeros((1, 1, 64, 64))
    probability[:, :, 31:34, :] = 1.0
    output = score_endpoint_pairs(
        probability,
        points,
        torch.full((3,), 0.9),
        SemanticEndpointPairConfig(image_size=64.0),
    )
    assert float(output["length_pixels"][0]) > 60.0
    assert float(output["line_score"][0]) > -1e5


def test_line_support_prefers_ink_and_blocker_penalizes_skipping_endpoint() -> None:
    probability = torch.zeros((1, 1, 64, 64))
    probability[:, :, 31:34, 6:59] = 1.0
    points = torch.tensor(
        ((0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.1, 0.8))
    )
    output = score_endpoint_pairs(
        probability,
        points,
        torch.tensor((0.9, 0.95, 0.9, 0.9)),
        SemanticEndpointPairConfig(image_size=64.0),
    )
    pair_to_row = {
        tuple(pair): index for index, pair in enumerate(output["pairs"].tolist())
    }
    adjacent = pair_to_row[(0, 1)]
    skipping = pair_to_row[(0, 2)]
    off_ink = pair_to_row[(0, 3)]
    assert float(output["line_score"][skipping]) > float(output["line_score"][off_ink])
    assert float(output["blocker"][skipping]) > float(output["blocker"][adjacent])
    assert float(output["line_blocker_score"][adjacent]) > float(
        output["line_blocker_score"][skipping]
    )
    assert float(output["line_gated_blocker_score"][adjacent]) > float(
        output["line_gated_blocker_score"][skipping]
    )


def test_near_duplicate_endpoint_pair_is_invalidated() -> None:
    probability = torch.ones((1, 1, 64, 64))
    points = torch.tensor(((0.5, 0.5), (0.51, 0.5)))
    output = score_endpoint_pairs(
        probability,
        points,
        torch.ones(2),
        SemanticEndpointPairConfig(
            image_size=64.0, minimum_pair_distance_pixels=2.0
        ),
    )
    assert float(output["line_score"][0]) < -1e5
