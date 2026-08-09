import math

import pytest
import torch

from chemvector.training.atoms.train_g3_p8_pairwise_consensus import (
    FOLDS,
    MINIMUM_CROSS_CERTIFICATE_RESCUES,
    MINIMUM_FULL_RESCUES,
    certificate_fold,
    select_unanimous_candidate,
)


def test_p8_protocol_is_fixed() -> None:
    assert FOLDS == 5
    assert MINIMUM_FULL_RESCUES == 80
    assert MINIMUM_CROSS_CERTIFICATE_RESCUES == 60
    assert certificate_fold("x") == certificate_fold("x")


def test_p8_consensus_uses_worst_member_score() -> None:
    winner, score = select_unanimous_candidate(
        torch.tensor([[1.0, 2.0], [0.5, 1.5], [0.2, 3.0]])
    )
    assert winner == 1
    assert score == 1.5


def test_p8_disagreement_means_no_action() -> None:
    winner, score = select_unanimous_candidate(
        torch.tensor([[2.0, 1.0], [0.5, 1.5], [0.2, 3.0]])
    )
    assert winner is None
    assert score == -math.inf


def test_p8_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        select_unanimous_candidate(torch.tensor([[1.0, float("nan")], [1.0, 2.0]]))
