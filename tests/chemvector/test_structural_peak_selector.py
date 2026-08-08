import torch

from chemvector.components.recovery.structural_peak_selector import (
    StructuralPeakSelector,
    compatible_peak_selection_loss,
)


def test_structural_peak_selector_scores_eight_candidates() -> None:
    model = StructuralPeakSelector()
    mask = torch.ones(2, 8, dtype=torch.bool)
    scores = model(
        torch.rand(2, 1, 96, 96),
        torch.rand(2, 1, 96, 96),
        (torch.rand(2, 8, 2) - 0.5) * (24.0 / 768.0),
        torch.rand(2, 8),
        mask,
        torch.rand(2),
        torch.rand(2),
    )
    assert scores.shape == (2, 8)
    assert model.parameter_count < 250_000
    loss = compatible_peak_selection_loss(
        scores,
        torch.zeros(2, 8, 2),
        mask,
        torch.zeros(2, 2),
    )
    loss.backward()
    assert all(value.grad is not None for value in model.parameters())


def test_structural_peak_selector_loss_accepts_equivalent_candidates() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    offsets = torch.tensor([[[0.0, 0.0], [5.0 / 768.0, 0.0], [12.0 / 768.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss = compatible_peak_selection_loss(scores, offsets, mask, torch.zeros(1, 2))
    loss.backward()
    expected = -torch.log((scores[0, :2].exp().sum()) / scores[0].exp().sum())
    torch.testing.assert_close(loss.detach(), expected.detach())
    fixed_identity_loss = -torch.log(scores[0, 0].exp() / scores[0].exp().sum())
    assert loss.item() < fixed_identity_loss.item()
    assert scores.grad is not None
