import torch

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
    build_semantic_endpoint_targets,
    decode_semantic_endpoints,
    semantic_endpoint_loss,
)


def _small_model() -> SemanticBondEndpointDetector:
    return SemanticBondEndpointDetector(
        SemanticBondEndpointConfig(
            stride4_channels=16,
            stride8_channels=24,
            stride16_channels=32,
            hidden_channels=16,
            slots=3,
        )
    )


def test_semantic_endpoint_detector_outputs_local_points() -> None:
    model = _small_model()
    output = model(torch.rand(2, 1, 64, 64))
    assert output["heatmap_logits"].shape == (2, 3, 16, 16)
    assert output["offsets"].shape == (2, 3, 2, 16, 16)
    assert output["points"].shape == (2, 3, 2, 16, 16)
    assert bool(((output["points"] >= 0.0) & (output["points"] <= 1.0)).all())


def test_semantic_endpoint_targets_merge_shared_ports_and_keep_collisions() -> None:
    endpoints = torch.tensor(
        [
            [
                [[0.1000, 0.2000], [0.5000, 0.5000]],
                [[0.5000, 0.5000], [0.1040, 0.2040]],
            ]
        ]
    )
    targets = build_semantic_endpoint_targets(
        endpoints,
        torch.ones((1, 2), dtype=torch.bool),
        height=16,
        width=16,
        slots=3,
    )
    assert int(targets["unique_counts"][0]) == 3
    assert int(targets["positive"].sum()) == 3
    assert int(targets["occupancy"].max()) == 2


def test_semantic_endpoint_loss_is_finite_and_backpropagates() -> None:
    model = _small_model()
    output = model(torch.rand(1, 1, 64, 64))
    endpoints = torch.tensor(
        [[[[0.15, 0.25], [0.75, 0.25]], [[0.75, 0.25], [0.75, 0.75]]]]
    )
    losses = semantic_endpoint_loss(
        output, endpoints, torch.ones((1, 2), dtype=torch.bool)
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_decode_semantic_endpoints_uses_scores_and_offsets() -> None:
    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, 2, 1] = 10.0
    points = torch.zeros((1, 1, 2, 4, 4))
    points[0, 0, :, 2, 1] = torch.tensor((0.3, 0.6))
    decoded = decode_semantic_endpoints(
        {"heatmap_logits": logits, "points": points},
        threshold=0.5,
        maximum_points=4,
    )
    assert len(decoded[0]) == 1
    torch.testing.assert_close(decoded[0][0]["point"], torch.tensor((0.3, 0.6)))


def test_semantic_endpoint_detector_rejects_rgb() -> None:
    model = _small_model()
    try:
        model(torch.rand(1, 3, 64, 64))
    except ValueError as error:
        assert "[batch, 1, H, W]" in str(error)
    else:
        raise AssertionError("RGB endpoint input should be rejected")
