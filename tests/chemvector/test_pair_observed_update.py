"""Tiny synthetic CPU optimization only; no checkpoint, real images or GPU."""

from copy import deepcopy

import pytest
import torch

from chemvector.training.curriculum import pair_cached_trial as cached
from chemvector.training.curriculum.pair_observed_update import ObservedPairUpdates
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_pair_only_trial import fixture


def setup(monkeypatch, *, maximum=50, maximum_attempts=58, emit=None):
    stage, _, cache, optimizer = fixture(monkeypatch)
    cache['key'] = 'train:a'
    other = deepcopy(cache); other['key'] = 'train:b'
    events, limit = [], StepLimit(maximum)
    adapter = ObservedPairUpdates(stage, optimizer, limit, [(cache, (0, 2)), (other, (0, 2))],
        emit=emit or (lambda event, **kw: events.append((event, kw))),
        maximum_attempts=maximum_attempts)
    return stage, cache, other, optimizer, adapter, events, limit


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_same(x, y)
    else:
        assert a == b


def test_observed_and_original_updates_match_params_adam_sampler_losses_and_rng(monkeypatch):
    stage, cache, other, optimizer, adapter, events, limit = setup(monkeypatch)
    baseline = deepcopy(stage)
    baseline_optimizer = torch.optim.AdamW(baseline.pair_parameters(), lr=2e-5, weight_decay=0.)
    baseline_limit = StepLimit(50)
    for attempt, sample in enumerate((cache, other)):
        rng = torch.random.get_rng_state().clone()
        expected = cached.update_once(baseline, sample, baseline_optimizer, baseline_limit,
                                     attempt=attempt, emit=lambda *a, **k: None)
        assert torch.equal(rng, torch.random.get_rng_state())
        actual = adapter.update(sample, attempt=attempt)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert_same(actual['update'], expected)
        assert_same(stage.state_dict(), baseline.state_dict())
        assert_same(optimizer.state_dict(), baseline_optimizer.state_dict())
        assert [p['own_sample_update'] for p in actual['probes']] == [attempt == 0, attempt == 1]
        assert [p['probe_row_sampled_in_update'] for p in actual['probes']] == [attempt == 0, attempt == 1]
        assert all(p['step'] == attempt + 1 and p['attempt'] == attempt for p in actual['probes'])
        assert all(p['identifies_observed_update_not_loss_component_causality'] for p in actual['probes'])
        cached.verify_cache(cache); cached.verify_cache(other)
    assert [e[0] for e in events] == ['observed_attempt_before', 'step_reserved', 'observed_attempt_complete'] * 2
    assert limit.consumed == 2
    assert all(p.grad is None for p in stage.parameters())


def test_empty_branch_has_no_probe_forward_optimizer_step_or_fake_step(monkeypatch):
    stage, _, _, optimizer, adapter, events, limit = setup(monkeypatch)
    _, _, empty, _ = fixture(monkeypatch, unknown=True); empty['key'] = 'train:empty'
    monkeypatch.setattr(stage.components.pair, 'forward', lambda *a, **k: pytest.fail('Empty forward'))
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Empty optimizer'))
    result = adapter.update(empty, attempt=0)
    assert not result['update']['updated'] and result['probes'] == [] and limit.consumed == 0
    assert [e[0] for e in events] == ['observed_attempt_before', 'observed_attempt_complete']


@pytest.mark.parametrize('change', ['stale_probe', 'positive', 'unknown', 'duplicates', 'train', 'count', 'budget', 'attempt_budget'])
def test_invalid_probe_or_construction_rejected(monkeypatch, change):
    stage, cache, other, optimizer, _, _, _ = setup(monkeypatch)
    probes = [(cache, (0, 2)), (other, (0, 2))]
    limit, attempts = StepLimit(50), 58
    if change == 'stale_probe': cache['tensors']['features'].add_(1.)
    elif change == 'positive': probes[0] = (cache, (0, 1))
    elif change == 'unknown': probes[0] = (cache, (0, 3))
    elif change == 'duplicates': other['key'] = cache['key']
    elif change == 'train': stage.train()
    elif change == 'count': probes.pop()
    elif change == 'budget': limit = StepLimit(51)
    else: attempts = 59
    with pytest.raises(ValueError):
        ObservedPairUpdates(stage, optimizer, limit, probes, emit=lambda *a, **k: None,
                            maximum_attempts=attempts)


@pytest.mark.parametrize('change', ['attempt', 'limit', 'weights', 'cache', 'probe_key', 'own_cache', 'train'])
def test_modified_sequence_fails_before_optimizer_and_disallows_retry(monkeypatch, change):
    stage, cache, other, optimizer, adapter, _, limit = setup(monkeypatch)
    attempt = 0
    if change == 'attempt': attempt = 1
    elif change == 'limit': limit.consume()
    elif change == 'weights':
        with torch.no_grad(): next(stage.components.pair.parameters()).add_(1.)
    elif change == 'cache': other['tensors']['features'].add_(1.)
    elif change == 'probe_key': other['key'] = 'train:changed'
    elif change == 'own_cache':
        cache = deepcopy(cache); cache['tensors']['features'].add_(1.)
        cache['cache_sha256'] = cached.tensor_tree_digest(cache['tensors'])
    else: stage.train()
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Invalid optimizer call'))
    with pytest.raises(ValueError): adapter.update(cache, attempt=attempt)
    with pytest.raises(RuntimeError, match='no retry'): adapter.update(cache, attempt=0)


@pytest.mark.parametrize('failure', ['before', 'reserved', 'after', 'optimizer'])
def test_incomplete_receipt_or_update_never_retries(monkeypatch, failure):
    events = []
    def emit(event, **kw):
        events.append(event)
        if event == dict(before='observed_attempt_before', reserved='step_reserved',
                         after='observed_attempt_complete').get(failure):
            raise OSError('Synthetic persistence failure')
    _, cache, _, optimizer, adapter, _, limit = setup(monkeypatch, emit=emit)
    if failure == 'optimizer':
        def fail(): raise OSError('Synthetic optimizer failure')
        monkeypatch.setattr(optimizer, 'step', fail)
    with pytest.raises(OSError): adapter.update(cache, attempt=0)
    with pytest.raises(RuntimeError, match='no retry'): adapter.update(cache, attempt=0)
    assert limit.consumed == (0 if failure == 'before' else 1)


def test_budget_exhaustion_never_calls_an_extra_update(monkeypatch):
    _, cache, _, optimizer, adapter, _, limit = setup(monkeypatch, maximum=1)
    adapter.update(cache, attempt=0)
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Exceeded budget'))
    with pytest.raises(ValueError): adapter.update(cache, attempt=1)
    assert limit.consumed == 1


def test_probe_rng_isolation_and_no_upstream_calls(monkeypatch):
    stage, cache, _, _, adapter, _, _ = setup(monkeypatch)
    original = stage.components.pair.forward
    def stochastic(*args, **kwargs):
        torch.rand(7)
        return original(*args, **kwargs)
    monkeypatch.setattr(stage.components.pair, 'forward', stochastic)
    for name, module in stage.components.named_children():
        if name != 'pair':
            monkeypatch.setattr(module, 'forward', lambda *a, **k: pytest.fail('Upstream called'))
    rng = torch.random.get_rng_state().clone()
    adapter.update(cache, attempt=0)
    observed_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng); torch.rand(7)  # Only the real synthetic update's draw remains.
    assert torch.equal(observed_rng, torch.random.get_rng_state())
