import pytest
import torch

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.training.curriculum.aromatic_ring_objective import (
    AromaticRingObjectiveConfig,
    aromatic_ring_safety_loss,
)


AROMATIC = BOND_TYPE_TO_ID["AROMATIC"]
SINGLE = BOND_TYPE_TO_ID["SINGLE"]


def ring_inputs(*, probability=0.20):
    pairs = torch.tensor([[i, (i + 1) % 6] for i in range(6)], dtype=torch.long)
    samples = torch.zeros(6, dtype=torch.long)
    known = torch.ones(6, dtype=torch.bool)
    presence = torch.ones(6, dtype=torch.long)
    targets = torch.full((6,), AROMATIC, dtype=torch.long)
    # Four equal non-aromatic logits make the aromatic probability explicit.
    # Three non-aromatic classes have zero logits, so compensate for their
    # combined softmax mass when constructing an exact aromatic probability.
    aromatic_logit = torch.log(torch.tensor(3.0 * probability / (1.0 - probability)))
    logits = torch.zeros(6, 4)
    logits[:, AROMATIC] = aromatic_logit
    return logits, pairs, samples, known, presence, targets


def test_complete_ring_has_positive_loss_and_gradients_on_all_edges():
    inputs = ring_inputs(probability=0.20)
    logits = inputs[0].clone().requires_grad_()
    loss, info = aromatic_ring_safety_loss(logits, *inputs[1:])
    assert info["rings"] == 1
    assert info["reachable_aromatic_edges"] == 6
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad[:, AROMATIC] != 0).all())


def test_ring_above_safety_floors_has_zero_loss():
    inputs = ring_inputs(probability=0.70)
    loss, info = aromatic_ring_safety_loss(inputs[0], *inputs[1:])
    assert float(loss) == 0.0
    assert info["rings"] == 1
    assert info["mean_loss"] == 0.0 and info["minimum_loss"] == 0.0


def test_no_complete_ring_returns_differentiable_zero():
    inputs = ring_inputs()
    inputs = tuple(value[:5] if index == 5 else value for index, value in enumerate(inputs))
    # Remove one edge while keeping the target tensors aligned.
    pairs, samples, known, presence, targets = (value[:5] for value in inputs[1:])
    logits = inputs[0][:5].clone().requires_grad_()
    loss, info = aromatic_ring_safety_loss(logits, pairs, samples, known, presence, targets)
    assert float(loss.detach()) == 0.0 and info["rings"] == 0
    assert loss.requires_grad


def test_unknown_positive_and_duplicate_pairs_fail_closed():
    inputs = ring_inputs()
    unknown = inputs[3].clone()
    unknown[0] = False
    with pytest.raises(ValueError, match="Unknown pairs"):
        aromatic_ring_safety_loss(inputs[0], inputs[1], inputs[2], unknown, inputs[4], inputs[5])
    duplicate = inputs[1].clone()
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match="Duplicate"):
        aromatic_ring_safety_loss(inputs[0], duplicate, inputs[2], inputs[3], inputs[4], inputs[5])


def test_config_requires_headroom_and_nonzero_weight():
    with pytest.raises(ValueError):
        AromaticRingObjectiveConfig(mean_floor=0.04, minimum_floor=0.05)
    with pytest.raises(ValueError):
        AromaticRingObjectiveConfig(mean_weight=0.0, minimum_weight=0.0)
