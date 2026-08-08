import torch

from chemvector.evaluation.audits.atoms.audit_g3_p6e0_context_capacity import (
    in_crop,
    select_context_size,
)


def test_p6e0_context_geometry_and_selection_are_frozen() -> None:
    source = torch.tensor([0.5, 0.5])
    assert in_crop(source + torch.tensor([15.0 / 768, 0.0]), source, 32)
    assert not in_crop(source + torch.tensor([17.0 / 768, 0.0]), source, 32)
    rates = {
        64: {"calibration": 0.5, "holdout": 0.4},
        96: {"calibration": 0.9, "holdout": 0.9},
    }
    assert select_context_size(rates) == 64
    rates[64]["holdout"] = 0.39
    assert select_context_size(rates) == 96
