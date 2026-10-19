"""Pure tensor contract tests; no real samples, models, optimizer or GPU."""

import pytest
import torch
from torch.nn import functional as F

from tools.selector_retention_prototype import positive_retention_loss, proposed_objective


def test_threshold_softplus_reused_and_original_objective_preserved():
    z = torch.tensor([[-.6, -.8, -.09, 9.]], requires_grad=True)
    positive = torch.tensor([[True, True, False, False]])
    mask = torch.tensor([[True, True, True, False]])
    snapshot = z.detach().clone()
    original = (z ** 2).mean()
    guard = positive_retention_loss(z, positive, mask, threshold=-.25)
    assert float(guard.detach()) == pytest.approx(float(F.softplus(-.25 - z[0, :2]).mean().detach()))
    total = proposed_objective(original, z, positive, mask, threshold=-.25)
    assert torch.equal(total.detach(), (original + guard).detach())
    derivative = torch.autograd.grad(guard, z)[0]
    assert bool((derivative[0, :2] < 0).all()) and derivative[0, 1] < derivative[0, 0]
    assert torch.equal(derivative[0, 2:], torch.zeros(2))
    assert z.grad is None and torch.equal(z, snapshot)


def test_normalization_per_image_positive_count_not_candidate_count():
    z = torch.zeros(2, 4, requires_grad=True)
    positive = torch.tensor([[True, False, False, False], [True, True, True, False]])
    mask = torch.ones_like(positive)
    loss = positive_retention_loss(z, positive, mask, threshold=-.25)
    derivative = torch.autograd.grad(loss, z)[0]
    assert derivative[0].sum() == pytest.approx(float(derivative[1].sum()))
    assert float(loss.detach()) == pytest.approx(float(F.softplus(torch.tensor(-.25))))


def test_empty_positive_image_has_zero_gradient():
    z = torch.randn(1, 3, requires_grad=True)
    positive = torch.zeros_like(z, dtype=torch.bool)
    loss = positive_retention_loss(z, positive, ~positive, threshold=-.25)
    assert float(loss.detach()) == 0
    assert torch.equal(torch.autograd.grad(loss, z)[0], torch.zeros_like(z))


def test_shared_parameters_can_raise_negative_despite_zero_direct_gradient():
    p = torch.nn.Parameter(torch.zeros(1))
    z = (torch.tensor([-.6, -.1]) + p).unsqueeze(0)
    positive = torch.tensor([[True, False]])
    loss = positive_retention_loss(z, positive, torch.ones_like(positive), threshold=-.25)
    gradient = torch.autograd.grad(loss, p, retain_graph=True)[0]
    negative_jacobian = torch.autograd.grad(z[0, 1], p)[0]
    assert float(-negative_jacobian @ gradient) > 0
    assert p.grad is None and float(p.detach()) == 0


@pytest.mark.parametrize("fault", ["dtype", "outside_mask", "nonfinite", "threshold"])
def test_malformed_inputs_fail_closed(fault):
    z = torch.zeros(1, 2)
    positive = torch.tensor([[True, False]])
    mask = torch.ones_like(positive)
    threshold = -.25
    if fault == "dtype":
        positive = positive.float()
    elif fault == "outside_mask":
        mask[0, 0] = False
    elif fault == "nonfinite":
        z[0, 1] = float("nan")
    else:
        threshold = float("inf")
    with pytest.raises(ValueError):
        positive_retention_loss(z, positive, mask, threshold=threshold)
