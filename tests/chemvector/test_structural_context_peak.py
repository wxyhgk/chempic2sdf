import torch

from chemvector.components.recovery.structural_context_peak import (
    LOCAL_END,
    LOCAL_START,
    StructuralContextPeakHead,
    anchor_centered_context_crops,
)
from chemvector.components.recovery.spatial_secondary_peak import secondary_peak_loss
from chemvector.training.atoms.g3_p4_local_recovery_data import anchor_centered_crops


def test_p6_context_crop_center_matches_p5_crop() -> None:
    values = torch.rand(1, 1, 128, 128)
    anchors = torch.tensor([[[0.5, 0.5], [0.6, 0.4]]])
    context = anchor_centered_context_crops(values, anchors)
    local = anchor_centered_crops(values, anchors)
    torch.testing.assert_close(
        context[:, :, :, LOCAL_START:LOCAL_END, LOCAL_START:LOCAL_END], local
    )


def test_p6_context_head_outputs_local_heatmap_and_backpropagates() -> None:
    model = StructuralContextPeakHead()
    logits = model(
        torch.rand(2, 1, 96, 96),
        torch.rand(2, 1, 96, 96),
        torch.rand(2),
        torch.rand(2),
    )
    assert logits.shape == (2, 32, 32)
    assert model.parameter_count < 200_000
    loss = secondary_peak_loss(
        logits, torch.tensor([[7.0, 2.0], [-8.0, 3.0]]) / 768.0
    )
    loss.backward()
    assert all(value.grad is not None for value in model.parameters())
