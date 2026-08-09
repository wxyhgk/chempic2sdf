import torch

from chemvector.training.atoms.train_g3_p17_groupwise_linear_keep_move import (
    EXPECTED_TRAIN_GROUPS,
    MINIMUM_CALIBRATION_RESCUES,
    STEPS,
    groupwise_max_loss,
)


def test_p17_protocol_is_fixed() -> None:
    assert STEPS == 4000
    assert EXPECTED_TRAIN_GROUPS == {
        "rescue": 191,
        "keep": 1584,
        "unsupported": 34,
    }
    assert MINIMUM_CALIBRATION_RESCUES == 20


def test_groupwise_loss_penalizes_deployed_maxima() -> None:
    scores = torch.tensor(
        [[2.0, -2.0, -3.0], [-2.0, -1.0, -4.0], [-3.0, -2.0, -1.0]],
        requires_grad=True,
    )
    valid = torch.ones_like(scores, dtype=torch.bool)
    good = torch.tensor(
        [[True, False, False], [False, False, False], [False, False, False]]
    )
    kind = torch.tensor([0, 1, 2])
    losses = groupwise_max_loss(scores, valid, good, kind)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert scores.grad[0, 0] < 0
    assert scores.grad[1, 1] > 0
    assert scores.grad[2, 2] > 0
