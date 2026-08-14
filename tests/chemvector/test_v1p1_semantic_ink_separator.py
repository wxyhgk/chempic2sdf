import torch

from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
    semantic_ink_loss,
)


def test_semantic_ink_separator_preserves_native_shape():
    model = SemanticInkSeparator(SemanticInkSeparatorConfig(base_channels=8))
    output = model(torch.rand(2, 3, 64, 64))
    assert output["logits"].shape == (2, 2, 64, 64)
    assert output["features"].shape[-2:] == (64, 64)


def test_semantic_ink_loss_prefers_correct_channels():
    config = SemanticInkSeparatorConfig(base_channels=8)
    targets = torch.zeros(1, 2, 16, 16)
    targets[:, 0, :, :8] = 1.0
    targets[:, 1, :, 8:] = 1.0
    correct = torch.where(targets.bool(), torch.tensor(8.0), torch.tensor(-8.0))
    swapped = correct.flip(1)
    correct_loss = semantic_ink_loss({"logits": correct}, targets, config)["loss"]
    swapped_loss = semantic_ink_loss({"logits": swapped}, targets, config)["loss"]
    assert correct_loss < swapped_loss


def test_semantic_ink_channels_allow_true_overlap():
    config = SemanticInkSeparatorConfig(base_channels=8)
    targets = torch.zeros(1, 2, 8, 8)
    targets[:, :, 3:5, 3:5] = 1.0
    logits = torch.where(targets.bool(), torch.tensor(8.0), torch.tensor(-8.0))
    loss = semantic_ink_loss({"logits": logits}, targets, config)["loss"]
    assert torch.isfinite(loss)
    assert float(loss) < 0.01
