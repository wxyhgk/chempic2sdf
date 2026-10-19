"""Synthetic contract tests for group existence/ranking/abstention only."""

import pytest
import torch

from tools.selector_group_abstention import (
    AbstainGroupSpec,
    AbstainPolicy,
    evaluate_group_abstain_objective,
    select_group_with_abstention,
)


def test_confirmed_negative_group_has_existence_loss_but_no_forced_winner():
    candidate = torch.tensor([4.0, 3.0], dtype=torch.float64, requires_grad=True)
    existence = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
    result = evaluate_group_abstain_objective(
        candidate,
        existence,
        [AbstainGroupSpec("negative", (0, 1), existence_target=False)],
    )
    gc, ge = torch.autograd.grad(result.total, (candidate, existence))
    assert result.existence_targets == (("negative", False),)
    assert result.ranking_loss.item() == 0.0
    assert torch.equal(gc, torch.zeros_like(candidate))
    assert ge.item() > 0.0  # pushes existence down, never a candidate winner
    decision = select_group_with_abstention(candidate.detach(), existence.detach())
    assert decision.status == "abstain_existence"
    assert decision.winner_slot is None


def test_positive_group_ranks_only_reviewed_negative_and_unknown_is_zero_gradient():
    candidate = torch.tensor([-1.0, 2.0, 7.0], dtype=torch.float64, requires_grad=True)
    existence = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    result = evaluate_group_abstain_objective(
        candidate,
        existence,
        [AbstainGroupSpec("positive", (0, 1, 2), (1,), (0,))],
    )
    gc, _ = torch.autograd.grad(result.total, (candidate, existence))
    assert gc[0] > 0.0 and gc[1] < 0.0
    assert gc[2].item() == 0.0  # unknown alternative never made a negative
    assert result.unknown_slots == (2,)


def test_abstention_checks_existence_before_argmax_and_margin():
    low = select_group_with_abstention(
        torch.tensor([99.0, 1.0]), -0.1,
        policy=AbstainPolicy(existence_threshold=0.0),
    )
    assert low.status == "abstain_existence" and low.winner_slot is None
    tie = select_group_with_abstention(
        torch.tensor([2.0, 1.9]), 2.0,
        policy=AbstainPolicy(margin_threshold=0.2),
    )
    assert tie.status == "abstain_margin" and tie.winner_slot is None
    selected = select_group_with_abstention(
        torch.tensor([2.0, 1.0]), 2.0,
        policy=AbstainPolicy(margin_threshold=0.2),
    )
    assert selected.status == "selected" and selected.winner_slot == 0


def test_protected_prefix_is_hard_locked_and_has_no_supervision():
    candidate = torch.tensor([3.0, 2.0], dtype=torch.float64, requires_grad=True)
    existence = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    result = evaluate_group_abstain_objective(
        candidate,
        existence,
        [AbstainGroupSpec("protected", (0, 1), protected=True)],
    )
    gc, ge = torch.autograd.grad(result.total, (candidate, existence))
    assert result.protected_node_ids == ("protected",)
    assert result.total.item() == 0.0
    assert torch.equal(gc, torch.zeros_like(candidate)) and torch.equal(ge, torch.zeros_like(existence))
    assert select_group_with_abstention(candidate.detach(), existence.detach(), protected=True).status == "protected"


@pytest.mark.parametrize(
    "spec",
    [
        AbstainGroupSpec("bad", (0,), ()),
        AbstainGroupSpec("bad", (0,), (), existence_target=True),
        AbstainGroupSpec("bad", (0,), (0,), (0,)),
    ],
)
def test_missing_negative_target_and_overlap_fail_closed(spec):
    with pytest.raises(ValueError):
        evaluate_group_abstain_objective(torch.zeros(1), torch.zeros(1), [spec])


def test_no_files_models_or_optimization_in_module():
    import inspect
    import tools.selector_group_abstention as module

    source = inspect.getsource(module)
    assert "torch.load" not in source
    assert "torch.optim" not in source
    assert "read_text(" not in source and "open(" not in source
    assert ".backward(" not in source and ".cuda(" not in source

