"""Future opt-in observer on synthetic frozen caches, with zero real updates."""

from copy import deepcopy

import pytest
import torch

from chemvector.training.curriculum import pair_negative_observer as observer
from tests.chemvector.test_pair_only_trial import fixture


def bound(monkeypatch):
    stage, _, cache, _ = fixture(monkeypatch)
    cache['key'] = 'train:synthetic'
    binding = observer.bind_negative_probe(cache, (0, 2))
    return stage, cache, binding


def test_probe_resolves_current_matching_and_observes_without_grad_or_parameter_changes(monkeypatch):
    stage, cache, binding = bound(monkeypatch)
    state = {k: v.clone() for k, v in stage.state_dict().items()}
    result = observer.observe(stage.components.pair, cache, binding)
    assert binding['candidate_pair'] == [0, 2] and binding['label'] is False
    assert result['presence_logit'] == 0.
    assert all(torch.equal(v, stage.state_dict()[k]) for k, v in state.items())
    assert all(p.grad is None for p in stage.parameters())


@pytest.mark.parametrize('pair', [(0, 1), (0, 3), (2, 0), (1, 1)])
def test_positive_unknown_unordered_or_self_pairs_rejected(monkeypatch, pair):
    _, cache, _ = bound(monkeypatch)
    with pytest.raises(ValueError): observer.bind_negative_probe(cache, pair)


@pytest.mark.parametrize('change', ['row', 'cache', 'train', 'key'])
def test_stale_or_modified_probe_rejected(monkeypatch, change):
    stage, cache, binding = bound(monkeypatch)
    if change == 'row': binding['row_id'] += 1
    elif change == 'cache': cache['tensors']['features'].add_(1.)
    elif change == 'train': stage.train()
    else: binding['key'] = 'other'
    with pytest.raises(ValueError): observer.observe(stage.components.pair, cache, binding)


def test_update_report_distinguishes_own_and_other_sample_without_claiming_causality(monkeypatch):
    stage, cache, binding = bound(monkeypatch)
    before = observer.observe(stage.components.pair, cache, binding)
    after = deepcopy(before); after['presence_logit'] += .3  # Synthetic receipt, no model edit.
    own = observer.paired_update_record(before, after, step=1, updated_sample_key=cache['key'])
    other = observer.paired_update_record(before, after, step=2, updated_sample_key='train:other')
    assert own['own_sample_update'] and not other['own_sample_update']
    assert own['negative_score_increased'] and own['diagnostic_only']
    assert own['identifies_observed_update_not_loss_component_causality']
