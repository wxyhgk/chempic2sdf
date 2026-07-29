import torch

from chemvector.direct_atom_text_bbox import (
    DirectAtomTextBBox,
    DirectAtomTextBBoxConfig,
    direct_atom_text_bbox_loss,
)


def test_direct_atom_text_bbox_initializes_at_anchor_with_finite_loss() -> None:
    config = DirectAtomTextBBoxConfig()
    model = DirectAtomTextBBox(config)
    pixels = torch.zeros(1, 3, 768, 768)
    anchors = torch.tensor([[[0.5, 0.4]]])
    output = model(pixels, anchors, torch.tensor([0]), torch.tensor([0]))
    assert torch.allclose(output["centers"], anchors[0])
    assert torch.allclose(
        output["half_extents"], torch.full((1, 2), 12.0 / 768.0)
    )
    target = torch.tensor([[0.52, 0.4, 30.0 / 768.0, 20.0 / 768.0]])
    losses = direct_atom_text_bbox_loss(output, anchors[0], target, config)
    assert losses["loss"].isfinite()
    assert losses["loss"] > 0


def test_direct_atom_text_bbox_supports_empty_selection() -> None:
    model = DirectAtomTextBBox()
    output = model(
        torch.zeros(1, 3, 32, 32),
        torch.zeros(1, 2, 2),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    assert output["centers"].shape == (0, 2)
    assert output["half_extents"].shape == (0, 2)
