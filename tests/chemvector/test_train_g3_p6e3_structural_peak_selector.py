import torch

from chemvector.components.recovery.structural_peak_selector import StructuralPeakSelector
from chemvector.training.atoms.train_g3_p6e3_structural_peak_selector import (
    GENERIC_ACTIONS,
    MIXED_STEPS,
    MODEL_TYPE,
    PRETRAIN_STEPS,
    REAL_ACTIONS,
    candidate_tensors,
)


def test_p6e3_training_protocol_is_frozen() -> None:
    assert MODEL_TYPE == "chemvector-g3-p6e3-structural-peak-selector50k-fit1378"
    assert GENERIC_ACTIONS == 50_000
    assert REAL_ACTIONS == 1_378
    assert PRETRAIN_STEPS == 10_000
    assert MIXED_STEPS == 5_000
    assert StructuralPeakSelector().parameter_count == 115_581


def test_p6e3_candidate_tensors_preserve_peak_scores() -> None:
    logits = torch.full((1, 32, 32), -20.0)
    logits[0, 10, 15] = 7.0
    offsets, peak_logits, mask = candidate_tensors(logits)
    assert offsets.shape == (1, 8, 2)
    assert mask[0, 0]
    torch.testing.assert_close(peak_logits[0, 0], torch.tensor(7.0))
