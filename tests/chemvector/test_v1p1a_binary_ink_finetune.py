import torch

from chemvector.training.vision.train_v1p1a_binary_ink_finetune import (
    ALPHA_FOREGROUND_THRESHOLD,
    binary_ink_targets,
)


def test_binary_ink_targets_match_the_frozen_evaluation_boundary():
    values = torch.tensor(
        [0.0, ALPHA_FOREGROUND_THRESHOLD - 1e-4, ALPHA_FOREGROUND_THRESHOLD, 1.0]
    ).reshape(1, 2, 1, 2)
    result = binary_ink_targets(values)
    assert result.dtype == torch.float32
    assert result.flatten().tolist() == [0.0, 0.0, 1.0, 1.0]
