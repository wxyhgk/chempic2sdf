"""Actual existing heads/losses with controlled synthetic candidates; no updates."""

from copy import deepcopy
from itertools import combinations

import pytest
import torch

from chemvector.training.curriculum import continuation as legacy
from chemvector.training.curriculum import continuation_pair_supervision as wired
from chemvector.training.curriculum import pair_supervision as supervision
from chemvector.training.curriculum.gradient_preflight import state_digest
from tests.chemvector.test_curriculum_continuation import tiny
from tests.chemvector.test_curriculum_selected_matching import candidates


@pytest.mark.parametrize('policy', [None, supervision.LEGACY])
@pytest.mark.parametrize('recipe', ['legacy-all-v1', 'freeze-role-v1'])
def test_default_delegates_verbatim_with_identical_rng_and_all_losses(tiny, policy, recipe):
    model, config, batch = tiny
    model.configure_update_recipe(recipe)
    model.eval()
    before = state_digest(model)
    rng = torch.get_rng_state().clone()
    expected_losses, expected_counts = legacy.continuation_losses(model, batch, config, atom_threshold=-1e6)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    kw = {} if policy is None else {'pair_supervision': policy}
    actual_losses, actual_counts = wired.continuation_losses(model, batch, config, atom_threshold=-1e6, **kw)
    assert actual_counts == expected_counts
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name in expected_losses:
        torch.testing.assert_close(actual_losses[name], expected_losses[name], rtol=0, atol=0)
    assert state_digest(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_invalid_policy_fails_before_forward(monkeypatch):
    monkeypatch.setattr(legacy, 'structure_forward', lambda *a: pytest.fail('Invalid policy forwarded'))
    with pytest.raises(ValueError, match='Unknown'):
        wired.continuation_losses(None, None, None, pair_supervision='conditional_ownership')


@pytest.mark.parametrize('recipe', ['legacy-all-v1', 'freeze-role-v1'])
def test_opt_in_all_matched_preserves_every_loss_labels_and_text_rows(candidates, recipe):
    model, config, batch, raw, selected, captured = candidates
    model.configure_update_recipe(recipe)
    before = state_digest(model)
    rng = torch.get_rng_state().clone()
    old_loss, old_counts = legacy.continuation_losses(model, batch, config)
    old_labels = deepcopy(captured['build_pair_targets'][1])
    old_mapping = captured['build_pair_targets'][0][2].clone()
    torch.set_rng_state(rng)
    new_loss, new_counts = wired.continuation_losses(model, batch, config, pair_supervision=supervision.MATCHED_ONLY)
    for name in old_loss:
        torch.testing.assert_close(new_loss[name], old_loss[name], rtol=0, atol=0)
    assert all(new_counts[k] == v for k, v in old_counts.items())
    for field in old_labels:
        assert torch.equal(old_labels[field], captured['build_pair_targets'][1][field])
    assert torch.equal(old_mapping, captured['build_pair_targets'][0][2])
    assert new_counts['pair_ignored_rows'] == 0
    assert new_counts['pair_sampled_positive_rows'] == 2
    assert new_counts['pair_sampled_negative_rows'] == 1
    assert state_digest(model) == before


@pytest.mark.parametrize('distance,kept_pairs,ignored,positives', [
    (6., [[0, 1], [0, 2], [1, 2]], 0, 2),
    (6.25, [[1, 2]], 2, 1),
    (7., [[1, 2]], 2, 1),
])
def test_uncertain_endpoints_never_reach_actual_pair_head(candidates, distance, kept_pairs, ignored, positives):
    model, config, batch, raw, selected, captured = candidates
    selected['atom_geometry'][0, 0, 0] = batch['atom_anchors'][0, 0, 0] + distance / config.image_size
    head_calls = []
    handle = model.pair.register_forward_pre_hook(lambda m, args: head_calls.append(args[4].detach().clone()))
    before = state_digest(model)
    try:
        losses, coverage = wired.continuation_losses(model, batch, config, pair_supervision=supervision.MATCHED_ONLY)
    finally:
        handle.remove()
    assert len(head_calls) == 1
    assert sorted(head_calls[0].tolist()) == kept_pairs
    assert coverage['pair_ignored_rows'] == ignored
    assert coverage['pair_sampled_positive_rows'] == positives
    assert all(torch.isfinite(v) for v in losses.values())
    # Raw supervision still sees rejected exact GT candidates; no raw relabeling.
    assert captured['associate_predicted_proposals'][1]['atom_target_indices'][0, 3:6].tolist() == [0, 1, 2]
    assert state_digest(model) == before
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('selected_ids', [[0, 1], [], [0]])
def test_empty_supervised_branch_skips_real_pair_head_with_finite_zero(candidates, selected_ids):
    model, config, batch, raw, selected, captured = candidates
    selected['atom_mask'].zero_()
    selected['atom_mask'][0, selected_ids] = True
    selected['atom_geometry'][0, 0, 0] = batch['atom_anchors'][0, 0, 0] + 7 / config.image_size
    before = state_digest(model)
    handle = model.pair.register_forward_pre_hook(lambda *a: pytest.fail('Empty pair branch forwarded'))
    try:
        losses, coverage = wired.continuation_losses(model, batch, config, pair_supervision=supervision.MATCHED_ONLY)
        losses['loss'].backward()
    finally:
        handle.remove()
    assert losses['pair'].item() == 0
    assert coverage['pair_rows'] == 0 and coverage['pair_branch_skipped']
    assert all(torch.isfinite(v) for v in losses.values())
    assert all(p.grad is None for p in model.pair.parameters())
    assert any(p.grad is not None for p in model.atom.parameters())
    assert state_digest(model) == before


def test_opt_in_backward_preserves_role_freeze_and_model_state(candidates):
    model, config, batch, raw, selected, captured = candidates
    model.configure_update_recipe('freeze-role-v1')
    selected['atom_geometry'][0, 0, 0] = batch['atom_anchors'][0, 0, 0] + 7 / config.image_size
    before = state_digest(model)
    losses, coverage = wired.continuation_losses(model, batch, config, pair_supervision=supervision.MATCHED_ONLY)
    output, presence, chemical = captured['direct_atom_pair_loss'][0]
    output['presence_logits'].retain_grad()
    assert presence.tolist() == [True]
    losses['loss'].backward()
    assert bool((output['presence_logits'].grad < 0).all())
    assert all(p.grad is None for p in model.role.parameters())
    assert not model.role.training
    assert any(p.grad is not None for p in model.pair.parameters())
    assert coverage['role_update_rows'] == 0
    assert state_digest(model) == before


def test_opt_in_keeps_non_pair_losses_and_all_prediction_tensors(candidates):
    model, config, batch, raw, selected, captured = candidates
    selected['atom_geometry'][0, 0, 0] = batch['atom_anchors'][0, 0, 0] + 7 / config.image_size
    before = {k: v.detach().clone() for k, v in selected.items()}
    old_loss, old_counts = legacy.continuation_losses(model, batch, config)
    old_labels = deepcopy(captured['build_pair_targets'][1])
    new_loss, new_counts = wired.continuation_losses(model, batch, config, pair_supervision=supervision.MATCHED_ONLY)
    for name in ('atom', 'bond', 'set_head', 'role', 'bbox'):
        torch.testing.assert_close(new_loss[name], old_loss[name], rtol=0, atol=0)
    for key in ('matched_selected_atoms', 'reachable_positive_pairs', 'role_rows', 'bbox_rows', 'selected_atoms'):
        assert new_counts[key] == old_counts[key]
    for field in old_labels:
        assert torch.equal(old_labels[field], captured['build_pair_targets'][1][field])
    assert all(torch.equal(v, selected[k]) for k, v in before.items())
    assert old_counts['pair_rows'] == 3 and new_counts['pair_rows'] == 1


def test_multi_sample_mask_uses_own_mapping_not_another_image():
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]] * 2)
    samples = torch.tensor([0, 0, 0, 1, 1, 1])
    mapping = torch.tensor([[-1, 1, 2], [0, 1, -1]])
    presence = torch.tensor([False, False, True, True, False, False])
    actual = supervision.supervision_mask(pairs, samples, mapping, presence, policy=supervision.MATCHED_ONLY)
    assert actual.tolist() == [False, False, True, True, False, False]


@pytest.mark.parametrize('kind', ['negative_index', 'unknown_sample', 'self_pair', 'masked_positive'])
def test_corrupt_pair_supervision_fails_closed(kind):
    pairs, samples = torch.tensor([[0, 1]]), torch.tensor([0])
    mapping, presence = torch.tensor([[0, -1]]), torch.tensor([False])
    if kind == 'negative_index':
        pairs[0, 0] = -1
    elif kind == 'unknown_sample':
        samples[0] = 1
    elif kind == 'self_pair':
        pairs[0, 1] = 0
    else:
        presence[0] = True
    with pytest.raises(ValueError):
        supervision.supervision_mask(pairs, samples, mapping, presence, policy=supervision.MATCHED_ONLY)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_runtime_sampler_filters_before_budget_and_returns_original_row_ids(batch_size):
    pairs = torch.tensor(list(combinations(range(4), 2))).repeat(batch_size, 1)
    samples = torch.arange(batch_size).repeat_interleave(6)
    mapping = torch.tensor([[-1, 1, 2, 3]]).repeat(batch_size, 1)
    targets = dict(presence=torch.tensor([False, False, False, True, False, False]).repeat(batch_size),
        chemical=torch.zeros(6 * batch_size, dtype=torch.long),
        structured_negative=torch.zeros(6 * batch_size, dtype=torch.bool))
    atoms = dict(atom_geometry=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2).repeat(batch_size, 1, 1))
    chosen, mask = supervision.sample_supervised_pairs(pairs, samples, mapping, atoms, targets,
        policy=supervision.MATCHED_ONLY, negative_ratio=1, minimum_negatives=2)
    assert len(chosen) == 3 * batch_size
    assert bool(mask[chosen].all())
    for sample in range(batch_size):
        local = chosen[samples[chosen] == sample]
        assert set(local.tolist()) == {sample * 6 + v for v in (3, 4, 5)}


def test_actual_loss_head_receives_correct_pairs_for_each_batch_sample(candidates, monkeypatch):
    model, config, batch, raw, selected, captured = candidates
    forward = legacy.structure_forward(model, batch['pixel_values'], config, 0)

    def duplicate(value):
        if isinstance(value, dict):
            return {k: duplicate(v) for k, v in value.items()}
        return torch.cat((value, value), dim=0)

    batch = duplicate(batch)
    forward = [duplicate(v) for v in forward]
    selected = forward[-1]
    selected['atom_geometry'][0, 0, 0] = batch['atom_anchors'][0, 0, 0] + 7 / config.image_size
    selected['atom_geometry'][1, 2, 0] = batch['atom_anchors'][1, 2, 0] + 7 / config.image_size
    monkeypatch.setattr(legacy, 'structure_forward', lambda *a, **k: tuple(forward))
    calls = []
    handle = model.pair.register_forward_pre_hook(lambda m, args: calls.append((args[4].tolist(), args[5].tolist())))
    try:
        losses, counts = wired.continuation_losses(model, batch, config, pair_supervision=supervision.MATCHED_ONLY)
    finally:
        handle.remove()
    assert calls == [([[1, 2], [0, 1]], [0, 1])]
    assert captured['direct_atom_pair_loss'][0][1].tolist() == [True, True]
    assert counts['pair_ignored_rows'] == 4 and counts['pair_sampled_positive_rows'] == 2
    assert all(torch.isfinite(v) for v in losses.values())
