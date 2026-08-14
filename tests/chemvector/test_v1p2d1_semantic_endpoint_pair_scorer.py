import torch

from chemvector.components.bonds.objects.semantic_endpoint_pair import (
    SemanticEndpointPairConfig,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair_scorer import (
    PAIR_FEATURE_DIM,
    SemanticEndpointPairScorer,
    SemanticEndpointPairScorerConfig,
    deduplicate_scored_pair_segments,
    extract_endpoint_pair_features,
    semantic_endpoint_pair_scorer_loss,
)


def test_pair_scorer_features_preserve_ribbon_and_geometry() -> None:
    probability = torch.zeros((1, 1, 64, 64))
    probability[:, :, 31:34, 5:59] = 1.0
    points = torch.tensor(((0.1, 0.5), (0.9, 0.5), (0.1, 0.8)))
    pairs = torch.tensor(((0, 1), (0, 2)))
    features = extract_endpoint_pair_features(
        probability,
        points,
        torch.tensor((0.9, 0.8, 0.7)),
        pairs,
        SemanticEndpointPairConfig(image_size=64.0),
    )
    assert features.shape == (2, PAIR_FEATURE_DIM)
    assert float(features[0, :48].mean()) > float(features[1, :48].mean())


def test_pair_scorer_and_listwise_loss_backpropagate() -> None:
    model = SemanticEndpointPairScorer(
        SemanticEndpointPairScorerConfig(hidden_dim=32)
    )
    features = torch.rand((6, PAIR_FEATURE_DIM))
    logits = model(features)
    losses = semantic_endpoint_pair_scorer_loss(
        logits,
        torch.tensor((0, -1, 1, -1, -1, -1)),
        bond_count=2,
    )
    assert logits.shape == (6,)
    assert torch.isfinite(losses["loss"])
    assert int(losses["positive_pairs"]) == 2
    assert int(losses["reachable_bonds"]) == 2
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pair_scorer_rejects_wrong_feature_width() -> None:
    model = SemanticEndpointPairScorer()
    try:
        model(torch.rand((2, PAIR_FEATURE_DIM - 1)))
    except ValueError as error:
        assert "invalid shape" in str(error)
    else:
        raise AssertionError("Pair scorer should reject an invalid feature width")


def test_pair_segment_nms_is_unordered_and_score_stable() -> None:
    points = torch.tensor(
        (
            (0.1, 0.2),
            (0.8, 0.7),
            (0.102, 0.201),
            (0.802, 0.701),
            (0.5, 0.5),
        )
    )
    pairs = torch.tensor(((0, 1), (3, 2), (0, 4)))
    selected = deduplicate_scored_pair_segments(
        points,
        pairs,
        torch.tensor((0.8, 0.9, 0.7)),
        image_size=768.0,
        radius_pixels=4.0,
    )
    assert selected.tolist() == [1, 2]
