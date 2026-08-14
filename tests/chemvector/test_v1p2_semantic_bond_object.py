import torch

from chemvector.components.bonds.objects.semantic_bond_object import (
    SemanticBondObjectConfig,
    SemanticBondObjectDetector,
)


def test_semantic_bond_object_outputs_stride4_objects():
    config = SemanticBondObjectConfig(
        stride4_channels=16,
        stride8_channels=24,
        stride16_channels=32,
        hidden_channels=16,
        slots=3,
    )
    model = SemanticBondObjectDetector(config)
    output = model(torch.rand(2, 1, 64, 64))
    assert output["heatmap_logits"].shape == (2, 3, 16, 16)
    assert output["endpoints"].shape == (2, 3, 2, 2, 16, 16)
    assert torch.isfinite(output["endpoints"]).all()


def test_semantic_bond_object_rejects_rgb_input():
    model = SemanticBondObjectDetector(
        SemanticBondObjectConfig(
            stride4_channels=16,
            stride8_channels=24,
            stride16_channels=32,
            hidden_channels=16,
        )
    )
    try:
        model(torch.rand(1, 3, 64, 64))
    except ValueError as error:
        assert "[batch, 1, H, W]" in str(error)
    else:
        raise AssertionError("RGB input must be rejected")
