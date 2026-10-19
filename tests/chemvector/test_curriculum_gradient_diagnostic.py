"""Synthetic-only checks of gradient geometry and the unapplied Adam reference."""

import pytest
import torch

from chemvector.training.curriculum.gradient_diagnostic import (
    TASKS, attribute_losses, cold_adamw_direction, cosine, measure,
)
from chemvector.training.curriculum.gradient_preflight import state_digest
from tests.chemvector.test_curriculum_continuation import tiny


def test_known_gradient_conflict_and_total_reconstruction():
    p = torch.nn.Parameter(torch.tensor([2., 3.], dtype=torch.float64))
    # Pair wants +x under descent, while the stronger auxiliary loss wants -x.
    losses = dict(atom=2 * p[0], bond=p[1], set_head=0 * p.sum(), pair=-p[0], bbox=0 * p.sum())
    losses["loss"] = sum(losses.values())
    result = attribute_losses(losses, [{"name": "shared", "params": [p], "lr": 1e-3}])
    group = result["groups"]["shared"]
    assert group["pair_vs_task_cosine"]["atom"] == -1
    assert group["pair_vs_task_cosine"]["set_head"] is None
    assert group["pair_vs_joint_cosine"] == pytest.approx(-2 ** -.5)
    assert group["cold_adamw_linearized_task_changes"]["pair"] > 0
    assert result["gradient_sum_relative_error"] == 0
    assert p.grad is None and p.tolist() == [2., 3.]


@pytest.mark.parametrize("clip", [1., .01])
def test_cold_direction_matches_first_adamw_step_on_synthetic_parameter(clip):
    p = torch.nn.Parameter(torch.tensor([.5, -.3, .8, .1], dtype=torch.float64))
    gradient = torch.tensor([2., -3., 0., 1e-10], dtype=torch.float64)
    before = p.detach().clone()
    expected = cold_adamw_direction(gradient, 1e-3, clip)
    # Optimizer is allowed ONLY in this synthetic formula test, never the audit.
    optimizer = torch.optim.AdamW([p], lr=1e-3, weight_decay=0)
    p.grad = gradient * clip
    optimizer.step()
    torch.testing.assert_close(p.detach() - before, expected, rtol=1e-9, atol=1e-15)
    torch.testing.assert_close(gradient, torch.tensor([2., -3., 0., 1e-10], dtype=torch.float64))


def test_zero_gradient_has_no_direction():
    assert cosine(0., 0., 1.) is None
    assert cosine(0., 1., 0.) is None


def test_real_tiny_loss_routes_preserve_parameters_modes_rng_and_empty_grads(tiny):
    model, config, batch = tiny
    model.configure_update_recipe("freeze-role-v1")
    state, modes, rng = state_digest(model), [m.training for m in model.modules()], torch.get_rng_state().clone()
    result = measure(model, batch, config, {"atom_threshold": -1e6, "bond_threshold": -.5})
    assert result["gradient_sum_relative_error"] < 2e-5
    groups = result["groups"]
    for name in ("pair", "bbox", "set_head"):
        assert groups[name]["task_gradient_l2"][name] > 0
        assert all(groups[name]["task_gradient_l2"][task] == 0 for task in TASKS if task != name)
    assert groups["vision"]["task_gradient_l2"]["bbox"] == 0
    assert all(p.grad is None for p in model.parameters())
    assert state_digest(model) == state and modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())


def test_empty_pair_branch_is_not_reported_as_orthogonal_or_given_gradients(tiny):
    model, config, batch = tiny
    model.configure_update_recipe("freeze-role-v1")
    result = measure(model, batch, config, {"atom_threshold": 1e6, "bond_threshold": -.5})
    assert result["coverage"]["pair_rows"] == 0
    assert all(g["task_gradient_l2"]["pair"] == 0 and g["pair_vs_joint_cosine"] is None
               for g in result["groups"].values())
    assert result["cold_adamw_total_linearized_task_changes"]["pair"] == 0


def test_dirty_or_duplicate_parameter_groups_are_rejected():
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="unique"):
        attribute_losses({}, [{"name": "bad", "params": [p, p], "lr": .1}])
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="empty"):
        attribute_losses({}, [{"name": "bad", "params": [p], "lr": .1}])


def test_missing_loss_contribution_is_not_hidden_by_float32_tolerance():
    p = torch.nn.Parameter(torch.ones(1))
    losses = {task: p.sum() for task in TASKS}
    losses["loss"] = sum(losses.values()) + p.sum()
    with pytest.raises(ValueError, match="do not reconstruct"):
        attribute_losses(losses, [{"name": "shared", "params": [p], "lr": .1}])
