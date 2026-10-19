"""Synthetic CPU tests only. No real images, weights, training job or CUDA."""

from copy import deepcopy
import json

import pytest
import torch

from chemvector.training.curriculum import pair_cached_trial as cached
from chemvector.training.curriculum import pair_observed_trial as launch
from chemvector.training.curriculum.pair_event_analysis import analyze_one_event, check_adam_transition, choose_analysis_event
from chemvector.training.curriculum.pair_negative_observer import bind_negative_probe, observe
from chemvector.training.curriculum.pair_observed_capture import (CapturedUpdates, DurableEvents, check_probe_parity,
    cpu_copy, mixed_digest, summarize_trajectories, termination_status)
from chemvector.training.curriculum.runtime import StepLimit, ResourceBudget, run_budgeted
from tests.chemvector.test_pair_only_trial import fixture
from tests.chemvector.test_pair_observed_update import assert_same


def setup(monkeypatch, *, emit=None, maximum=50, extra=False):
    stage, _, a, optimizer = fixture(monkeypatch); a['key'] = 'train:a'
    b = deepcopy(a); b['key'] = 'train:b'
    caches = {a['key']: a, b['key']: b}
    if extra:
        for i in range(6):
            c = deepcopy(a); c['key'] = 'train:extra' + str(i); caches[c['key']] = c
    events = []
    captured = CapturedUpdates(stage, optimizer, StepLimit(maximum), [(a, (0, 2)), (b, (0, 2))],
        list(caches), emit=emit or (lambda e, **kw: events.append((e, kw))))
    return stage, caches, optimizer, captured, events


def test_capture_preserves_exact_training_state_sampler_and_rng_and_replays_snapshot(monkeypatch):
    stage, caches, optimizer, captured, _ = setup(monkeypatch)
    original = deepcopy(stage)
    old_optimizer = torch.optim.AdamW(original.pair_parameters(), lr=2e-5, weight_decay=0.)
    old_limit = StepLimit(50)
    records = []
    for attempt, cache in enumerate(caches.values()):
        rng = torch.random.get_rng_state().clone()
        expected = cached.update_once(original, cache, old_optimizer, old_limit, attempt=attempt, emit=lambda *a, **k: None)
        actual = captured.update(cache, attempt=attempt); records.append(actual)
        assert_same(actual['update'], expected)
        assert_same(stage.state_dict(), original.state_dict())
        assert_same(optimizer.state_dict(), old_optimizer.state_dict())
        assert torch.equal(rng, torch.random.get_rng_state())
    assert len(captured.events) == 2
    for step, event in captured.events.items():
        check_adam_transition(event)
        # Additional optimizer invocation is only on a tiny synthetic copy.
        replay = deepcopy(stage); replay.components.pair.load_state_dict(event['pair_before'])
        opt = torch.optim.AdamW(replay.pair_parameters(), lr=2e-5, weight_decay=0.)
        opt.load_state_dict(event['optimizer_before'])
        limit = StepLimit(50); limit.consumed = step - 1
        result = cached.update_once(replay, event['cache'], opt, limit, attempt=event['attempt'], emit=lambda *a, **k: None)
        assert_same(result, event['record']['update'])
        assert_same(replay.components.pair.state_dict(), event['pair_after'])
    summaries = summarize_trajectories(records)
    assert summaries['train:a']['optimizer_updates'] == 2
    assert not summaries['train:a']['loss_component_causality_proven']


def test_snapshot_analysis_has_no_optimizer_step_or_mutation(monkeypatch):
    stage, caches, optimizer, captured, _ = setup(monkeypatch)
    captured.update(caches['train:a'], attempt=0)
    event = captured.events[1]
    before = mixed_digest((stage.state_dict(), optimizer.state_dict(), caches))
    rng = torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *a, **k: pytest.fail('Analysis attempted an update'))
    report = analyze_one_event(stage.components.pair, event, caches['train:a'],
                               bind_negative_probe(caches['train:a'], (0, 2)))
    assert report['optimizer_steps'] == 0 and report['pair_forwards'] == 2
    assert not report['loss_component_causality_proven']
    assert mixed_digest((stage.state_dict(), optimizer.state_dict(), caches)) == before
    assert torch.equal(rng, torch.random.get_rng_state())
    assert report['clipped_gradient_replay_max_absolute_error'] < 1e-6


def test_real_head_architecture_on_random_synthetic_inputs_supports_capture_and_analysis(monkeypatch):
    from chemvector.direct_atom_pair_bond import DirectAtomPairBond, DirectAtomPairBondConfig
    stage, caches, _, _, _ = setup(monkeypatch)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260922)
        stage.components.pair = DirectAtomPairBond(DirectAtomPairBondConfig(
            feature_channels=2, atom_feature_dim=2, longitudinal_samples=8,
            transverse_samples=3, hidden_dim=8, pair_chunk_size=2))
        stage.eval()
        optimizer = torch.optim.AdamW(stage.pair_parameters(), lr=2e-5, weight_decay=0.)
        captured = CapturedUpdates(stage, optimizer, StepLimit(50),
            [(caches['train:a'], (0, 2)), (caches['train:b'], (0, 2))], list(caches), emit=lambda *a, **k: None)
        for attempt, cache in enumerate(caches.values()): captured.update(cache, attempt=attempt)
        event = captured.events[2]
        assert check_adam_transition(event)['max_parameter_absolute_error'] < 1e-6
        report = analyze_one_event(stage.components.pair, event, caches['train:b'],
                                   bind_negative_probe(caches['train:b'], (0, 2)))
        assert report['clipped_gradient_replay_max_absolute_error'] < 1e-4
        assert report['optimizer_steps'] == 0


@pytest.mark.parametrize('change', ['order', 'lr', 'beta', 'optimizer_state', 'attempt'])
def test_sequence_or_optimizer_drift_fails_before_update_and_poisoned(monkeypatch, change):
    _, caches, optimizer, captured, _ = setup(monkeypatch)
    cache, attempt = caches['train:a'], 0
    if change == 'order': cache = caches['train:b']
    elif change == 'lr': optimizer.param_groups[0]['lr'] *= 2
    elif change == 'beta': optimizer.param_groups[0]['betas'] = (.5, .9)
    elif change == 'optimizer_state': optimizer.state[next(iter(optimizer.param_groups[0]['params']))]['step'] = torch.tensor(7.)
    else: attempt = 1
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Invalid state updated'))
    with pytest.raises(ValueError): captured.update(cache, attempt=attempt)
    with pytest.raises(RuntimeError): captured.update(cache, attempt=0)


def test_empty_supervision_skips_snapshots_and_optimizer(monkeypatch):
    _, caches, optimizer, captured, _ = setup(monkeypatch)
    _, _, empty, _ = fixture(monkeypatch, unknown=True); empty['key'] = 'train:a'
    # A probe's own changed cache must be rejected, even for an empty branch.
    with pytest.raises(ValueError): captured.update(empty, attempt=0)
    _, caches, optimizer, captured, _ = setup(monkeypatch, extra=True)
    captured.update(caches['train:a'], attempt=0); captured.update(caches['train:b'], attempt=1)
    empty['key'] = 'train:extra0'
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Empty branch updated'))
    result = captured.update(empty, attempt=2)
    assert not result['update']['updated'] and captured.limit.consumed == 2 and captured.pending is None


def test_snapshot_selection_keeps_own_and_largest_rise_only(monkeypatch):
    stage, caches, optimizer, captured, _ = setup(monkeypatch, extra=True)
    # Increase only the synthetic fixture LR before constructing a new wrapper.
    optimizer.param_groups[0]['lr'] = .01
    captured = CapturedUpdates(stage, optimizer, StepLimit(50),
        [(caches['train:a'], (0, 2)), (caches['train:b'], (0, 2))], list(caches), emit=lambda *a, **k: None)
    records = []
    for attempt, cache in enumerate(caches.values()): records.append(captured.update(cache, attempt=attempt))
    assert len(captured.events) <= 4
    assert captured.slots['own:train:a'] == 1 and captured.slots['own:train:b'] == 2
    for key in ('train:a', 'train:b'):
        track = [p for r in records for p in r['probes'] if p['probe']['key'] == key]
        best = max(track, key=lambda p: (p['presence_logit_delta'], -p['step']))
        assert captured.slots['maxrise:' + key] == best['step']
    assert set(captured.events) == set(captured.slots.values())
    assert choose_analysis_event(captured.events) is not None


def test_step_cap_never_performs_extra_step(monkeypatch):
    _, caches, optimizer, captured, _ = setup(monkeypatch, maximum=1)
    captured.update(caches['train:a'], attempt=0)
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Exceeded step cap'))
    with pytest.raises(ValueError): captured.update(caches['train:b'], attempt=1)
    assert captured.limit.consumed == 1


@pytest.mark.parametrize('event', ['observed_attempt_before', 'step_reserved', 'observed_attempt_complete'])
def test_durable_log_failure_blocks_retry(monkeypatch, event):
    def emit(name, **kw):
        if name == event: raise OSError('synthetic log failure')
    _, caches, _, captured, _ = setup(monkeypatch, emit=emit)
    with pytest.raises(OSError): captured.update(caches['train:a'], attempt=0)
    with pytest.raises(RuntimeError): captured.update(caches['train:a'], attempt=0)


def test_fsync_failure_poisoned_logger(tmp_path, monkeypatch):
    with (tmp_path / 'events.jsonl').open('x') as f:
        sink = DurableEvents(f, lambda: 0.)
        def fail(_): raise OSError('fsync failure')
        monkeypatch.setattr('os.fsync', fail)
        with pytest.raises(OSError): sink('before', value=1)
        with pytest.raises(RuntimeError): sink('after', value=2)
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 1


def test_probe_full_parity_and_tolerance(monkeypatch):
    stage, caches, _, _, _ = setup(monkeypatch)
    cache = caches['train:a']; binding = bind_negative_probe(cache, (0, 2))
    single = observe(stage.components.pair, cache, binding)
    full = dict(presence_logits=torch.tensor([single['presence_logit']]), chemical_logits=torch.tensor([single['chemical_logits']]))
    assert check_probe_parity(single, full, 0)['max_absolute_difference'] == 0.
    full['presence_logits'].add_(.01)
    with pytest.raises(ValueError): check_probe_parity(single, full, 0)


@pytest.mark.parametrize('steps', [0, 1, 49, 50])
def test_actual_terminal_step_not_mislabeled_as_step50(steps):
    result = termination_status(steps)
    assert result['actual_final_step'] == steps
    assert result['full_50_step_diagnostic'] == (steps == 50)
    assert result['needs_actual_final_evaluation'] and not result['deployment_authorized']


@pytest.mark.parametrize('change', ['budget', 'deployment', 'old_schema', 'implementation', 'statement'])
def test_exact_new_authorization_rejects_expansion(change):
    auth = launch.authorization({p: 'a' * 64 for p in launch.IMPLEMENTATIONS})
    launch.validate_authorization(auth)
    if change == 'budget': auth['limits']['maximum_steps'] = 51
    elif change == 'deployment': auth['deployment_authorized'] = True
    elif change == 'old_schema': auth['schema_version'] = 'old'
    elif change == 'implementation': auth['implementation_sha256'].pop(launch.IMPLEMENTATIONS[0])
    else: auth['user_statement'] = 'historical permission'
    with pytest.raises(ValueError): launch.validate_authorization(auth)


def test_no_reuse_of_observed_reservation(tmp_path):
    path = tmp_path / 'attempt.json'
    launch.old.reserve_once(path, 'a' * 64)
    before = path.read_bytes()
    with pytest.raises(FileExistsError): launch.old.reserve_once(path, 'a' * 64)
    assert path.read_bytes() == before


def test_parent_timeout_stops_only_synthetic_child(tmp_path):
    import sys
    receipt = run_budgeted([sys.executable, '-c', 'import time; time.sleep(5)'],
                           output=tmp_path / 'run', budget=ResourceBudget(.1))
    assert receipt['status'] == 'timeout' and not receipt['training']
    assert receipt['elapsed_seconds'] < 2
