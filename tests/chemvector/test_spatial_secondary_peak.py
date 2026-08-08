import torch

from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
    SpatialSecondaryPeakHead,
    decode_secondary_peak_offsets,
    dihedral_transform,
    secondary_peak_loss,
    secondary_peak_targets,
)


def test_dense_target_and_decode_preserve_direction() -> None:
    offset = torch.tensor([[8.0, -3.0]]) / 768.0
    target = secondary_peak_targets(offset)
    logits = target.clamp_min(1e-8).log()
    decoded = decode_secondary_peak_offsets(logits)
    assert target.shape == (1, 32, 32)
    assert torch.linalg.vector_norm((decoded - offset) * 768.0) < 1.0
    assert torch.isfinite(secondary_peak_loss(logits, offset))


def test_dihedral_transform_rotates_and_flips_vector_targets() -> None:
    crop = torch.arange(32 * 32).reshape(1, 1, 32, 32).float()
    offsets = torch.tensor([[8.0, 3.0]]) / 768.0
    _, _, rotated = dihedral_transform(
        crop, crop, offsets, torch.tensor([1])
    )
    torch.testing.assert_close(rotated * 768.0, torch.tensor([[3.0, -8.0]]))
    _, _, flipped = dihedral_transform(
        crop, crop, offsets, torch.tensor([4])
    )
    torch.testing.assert_close(flipped * 768.0, torch.tensor([[-8.0, 3.0]]))


def test_spatial_head_retains_dense_output_and_backpropagates() -> None:
    model = SpatialSecondaryPeakHead()
    logits = model(
        torch.rand(2, 1, 32, 32),
        torch.rand(2, 1, 32, 32),
        torch.rand(2),
        torch.rand(2),
    )
    assert logits.shape == (2, 32, 32)
    assert model.parameter_count < 50_000
    loss = secondary_peak_loss(
        logits, torch.tensor([[7.0, 2.0], [-8.0, 3.0]]) / 768.0
    )
    loss.backward()
    assert all(value.grad is not None for value in model.parameters())


def test_contextual_head_has_full_crop_path_and_relative_coordinates() -> None:
    model = ContextualSecondaryPeakHead()
    logits = model(
        torch.rand(2, 1, 32, 32),
        torch.rand(2, 1, 32, 32),
        torch.rand(2),
        torch.rand(2),
    )
    assert logits.shape == (2, 32, 32)
    assert model.bottleneck.layers[0].kernel_size == (3, 3)
    assert model.relative_coordinates.shape == (2, 32, 32)
    assert model.parameter_count < 100_000
    loss = secondary_peak_loss(
        logits, torch.tensor([[7.0, 2.0], [-8.0, 3.0]]) / 768.0
    )
    loss.backward()
    assert all(value.grad is not None for value in model.parameters())
