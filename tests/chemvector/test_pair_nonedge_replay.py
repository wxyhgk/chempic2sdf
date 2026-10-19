"""Synthetic CPU repair contracts: no corpus, checkpoint loading or GPU."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from chemvector.training.curriculum import pair_cached_trial as cached
from chemvector.training.curriculum import pair_nonedge_replay as replay
from chemvector.training.curriculum import pair_replay_trial as launch
from chemvector.training.curriculum.pair_event_analysis import check_adam_transition
from chemvector.training.curriculum.pair_replay_worker import require_control_equal
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_curriculum_pair_stage import setup as stage_setup
from tests.chemvector.test_pair_only_trial import fixture, authorization as old_authorization
from tests.chemvector.test_pair_observed_update import assert_same


@pytest.fixture(autouse=True)
def no_model_loading_or_cuda(monkeypatch):
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('No checkpoint loading permitted'))
    monkeypatch.setattr(torch.cuda, 'init', lambda *a, **k: pytest.fail('No GPU permitted'))


def row(name, molecule=None):
    return dict(source='train', sample_key=name, molecule_id=molecule or name,
                split='train', fit_domain='old_full_molecule', meta_sha256='a' * 64)


def setup(monkeypatch, *, empty_first=False):
    stage, _, a, optimizer = fixture(monkeypatch); a['key'] = 'train:a'
    b = deepcopy(a); b['key'] = 'train:b'
    caches = {a['key']: a, b['key']: b}
    rows = [row('a'), row('b')]
    logits = {k: torch.zeros(len(c['tensors']['known'])) for k, c in caches.items()}
    bank = replay.ReplayBank(caches, logits, rows)
    if empty_first:
        _, _, empty, _ = fixture(monkeypatch, unknown=True); empty['key'] = 'train:empty'
        caches[empty['key']] = empty; rows.insert(0, row('empty'))
    limit, events = StepLimit(50), []
    captured = replay.CapturedReplayUpdates(stage, optimizer, limit, [(a, (0, 2)), (b, (0, 2))],
        rows, bank=bank, emit=lambda e, **kw: events.append((e, kw)))
    return stage, caches, optimizer, rows, bank, captured, events


def large_cache():
    count = 13
    xy = torch.stack((torch.linspace(.05, .95, count), torch.full((count,), .3)), dim=-1)[None]
    atoms = dict(atom_mask=torch.ones(1, count, dtype=torch.bool), atom_indices=torch.arange(count)[None],
        atom_geometry=xy.clone(), atom_features=torch.ones(1, count, 2), atom_scores=torch.ones(1, count))
    batch = dict(atom_mask=atoms['atom_mask'].clone(), atom_anchors=xy.clone(),
        bond_mask=torch.ones(1, count - 1, dtype=torch.bool),
        bond_atom_refs=torch.stack((torch.arange(count - 1), torch.arange(1, count)), dim=-1)[None],
        bond_chemical=torch.zeros(1, count - 1, dtype=torch.long))
    atoms['atom_geometry'][0, -1, 1] = .95  # One genuinely unmatched prediction.
    cache = cached.make_cache((torch.ones(1, 2, 2, 2), None, None, atoms, {}, {}, atoms),
        batch, SimpleNamespace(image_size=768, atom_match_radius_px=6))
    cache['key'] = 'train:large'
    return cache


def test_fixed_top32_ranking_excludes_positive_unknown_and_breaks_ties_by_row():
    cache = large_cache(); t = cache['tensors']
    eligible = torch.where(t['known'] & ~t['labels']['presence'])[0].tolist()
    assert len(eligible) == 55
    logits = torch.full(t['known'].shape, 1000.)
    logits[eligible] = 1.
    logits[eligible[-3:]] = 2.
    rng = torch.random.get_rng_state().clone()
    binding = replay.bind_replay(cache, logits)
    assert binding['row_ids'] == eligible[-3:] + eligible[:29]
    assert len(binding['row_ids']) == 32 and binding['presence_target'] is False
    assert binding['positive_rows'] == binding['unknown_rows'] == 0
    assert torch.equal(rng, torch.random.get_rng_state())
    assert binding == replay.bind_replay(cache, logits)


@pytest.mark.parametrize('change', ['shape', 'dtype', 'nan', 'grad', 'cache', 'key', 'no_negatives'])
def test_invalid_binding_rejected(monkeypatch, change):
    _, _, cache, _ = fixture(monkeypatch); cache['key'] = 'train:a'
    logits = torch.zeros(len(cache['tensors']['known']))
    if change == 'shape': logits = logits[:, None]
    elif change == 'dtype': logits = logits.double()
    elif change == 'nan': logits[0] = float('nan')
    elif change == 'grad': logits.requires_grad_()
    elif change == 'cache': cache['tensors']['features'].add_(1)
    elif change == 'key': cache['key'] = ''
    else:
        _, _, cache, _ = fixture(monkeypatch, unknown=True); cache['key'] = 'train:a'
    with pytest.raises(ValueError): replay.bind_replay(cache, logits)


def test_rotation_is_update_indexed_skips_same_molecule_and_copies_inputs(monkeypatch):
    _, caches, _, rows, bank, _, _ = setup(monkeypatch)
    assert bank.select(row('elsewhere'), 0)['cache']['key'] == 'train:a'
    assert bank.select(row('elsewhere'), 1)['cache']['key'] == 'train:b'
    assert bank.select(row('style_variant', 'a'), 0)['cache']['key'] == 'train:b'
    assert bank.select(rows[1], 1)['cache']['key'] == 'train:a'
    rows[0]['molecule_id'] = 'changed_outside_bank'
    assert bank.select(row('style_variant', 'a'), 0)['cache']['key'] == 'train:b'
    assert cached.verify_cache(caches['train:a']) is None


@pytest.mark.parametrize('change', ['split', 'domain', 'duplicate', 'molecule', 'cache_key', 'count'])
def test_donor_pool_rejects_wrong_or_duplicate_cohort(monkeypatch, change):
    _, caches, _, rows, _, _, _ = setup(monkeypatch)
    logits = {k: torch.zeros(len(c['tensors']['known'])) for k, c in caches.items()}
    if change == 'split': rows[0]['split'] = 'validation'
    elif change == 'domain': rows[0]['fit_domain'] = 'witness'
    elif change == 'duplicate': rows.append(deepcopy(rows[0]))
    elif change == 'molecule': rows[1]['molecule_id'] = rows[0]['molecule_id']
    elif change == 'cache_key': caches['train:a']['key'] = 'train:wrong'
    else: rows.pop()
    with pytest.raises(ValueError): replay.ReplayBank(caches, logits, rows)


@pytest.mark.parametrize('change', ['features', 'logits', 'binding', 'rows'])
def test_bank_mutation_fails_closed(monkeypatch, change):
    _, _, _, _, bank, _, _ = setup(monkeypatch)
    entry = bank.entries['train:a']
    if change == 'features': entry['cache']['tensors']['features'].add_(1)
    elif change == 'logits': entry['initial_logits'].add_(1)
    elif change == 'binding': entry['binding']['row_ids'].clear()
    else: bank.rows.reverse()
    with pytest.raises(ValueError): bank.select(row('elsewhere'), 0)


@pytest.mark.parametrize('index', [-1, 50, True, 1.0])
def test_invalid_schedule_index_rejected(index):
    with pytest.raises(ValueError): replay.donor_key([row('a'), row('b')], row('elsewhere'), index)


def test_no_eligible_different_molecule_donor_is_an_error():
    with pytest.raises(ValueError): replay.donor_key([row('a')], row('style_variant', 'a'), 0)


def test_zero_replay_has_exact_original_losses_sampling_weights_adam_and_rng(monkeypatch):
    stage, caches, optimizer, _, _, _, _ = setup(monkeypatch)
    baseline = deepcopy(stage)
    opt = torch.optim.AdamW(baseline.pair_parameters(), lr=2e-5, weight_decay=0.)
    limit, old_limit = StepLimit(50), StepLimit(50)
    for attempt in range(4):
        cache = caches['train:a' if attempt % 2 == 0 else 'train:b']
        rng = torch.random.get_rng_state().clone()
        expected = cached.update_once(baseline, cache, opt, old_limit, attempt=attempt, emit=lambda *a, **k: None)
        actual = replay.update_once(stage, cache, optimizer, limit, attempt=attempt,
                                    emit=lambda *a, **k: None, weight=0.)
        assert actual.pop('replay') is None
        assert actual.pop('objective_loss') == expected['losses']['loss']
        assert_same(actual, expected)
        assert_same(stage.state_dict(), baseline.state_dict())
        assert_same(optimizer.state_dict(), opt.state_dict())
        assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('main_positives,main_negatives', [(1, 0), (1, 20), (20, 1), (0, 20)])
def test_independent_replay_gradient_does_not_scale_with_main_counts(main_positives, main_negatives):
    target = torch.tensor([True] * main_positives + [False] * main_negatives)
    main = torch.zeros(len(target), requires_grad=True)
    chemical = torch.zeros(len(target), 4, requires_grad=True)
    donor = torch.tensor([-.3, .7, 1.2], requires_grad=True)
    loss = cached.direct_atom_pair_loss(dict(presence_logits=main, chemical_logits=chemical),
                                       target, torch.zeros(len(target), dtype=torch.long))['loss']
    (loss + replay.replay_loss(donor)).backward()
    assert torch.allclose(donor.grad, donor.detach().sigmoid() / len(donor))


@pytest.mark.parametrize('values', [[], [0.] * 33, [float('nan')], [[0.]]])
def test_invalid_replay_loss_rejected(values):
    with pytest.raises(ValueError): replay.replay_loss(torch.tensor(values))


def test_combined_loss_one_clip_one_step_no_replay_chemical_loss(monkeypatch):
    stage, caches, optimizer, rows, bank, _, _ = setup(monkeypatch)
    calls, clipped = [], []
    original_loss = cached.direct_atom_pair_loss
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = optimizer.step
    def loss(*args, **kwargs):
        calls.append('main_loss'); return original_loss(*args, **kwargs)
    def clip(parameters, *args, **kwargs):
        params = list(parameters)
        clipped.append([p.grad.clone() if p.grad is not None else None for p in params])
        return original_clip(params, *args, **kwargs)
    def step(*args, **kwargs):
        calls.append('step'); return original_step(*args, **kwargs)
    monkeypatch.setattr(cached, 'direct_atom_pair_loss', loss)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    monkeypatch.setattr(optimizer, 'step', step)
    result = replay.update_once(stage, caches['train:a'], optimizer, StepLimit(50), attempt=0,
        emit=lambda *a, **k: None, replay_entry=bank.select(rows[0], 0))
    assert calls == ['main_loss', 'step'] and len(clipped) == 1
    assert len(stage.components.pair.calls) == 2  # Main and replay, no upstream.
    assert torch.allclose(clipped[0][0], torch.tensor([1. / 3.]))  # -1/6 main + 1/2 replay.
    assert torch.allclose(clipped[0][1], torch.tensor([-.25, -.25, .25, .25]))
    assert result['replay']['chemical_loss_added'] is False
    assert result['objective_loss'] == pytest.approx(result['losses']['loss'] + result['replay']['negative_bce'])
    assert all(p.grad is None for p in stage.parameters())


def test_empty_main_does_not_forward_replay_select_or_consume_update(monkeypatch):
    stage, caches, optimizer, _, bank, captured, events = setup(monkeypatch, empty_first=True)
    monkeypatch.setattr(stage.components.pair, 'forward', lambda *a, **k: pytest.fail('Empty forward'))
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Empty optimizer step'))
    monkeypatch.setattr(bank, 'select', lambda *a, **k: pytest.fail('Empty replay selection'))
    actual = captured.update(caches['train:empty'], attempt=0)
    assert not actual['update']['updated'] and actual['probes'] == []
    assert captured.limit.consumed == 0 and not captured.events
    assert [e[0] for e in events] == ['observed_attempt_before', 'observed_attempt_complete']


def test_negative_only_main_still_updates(monkeypatch):
    stage, batch, atoms, features, _ = stage_setup(monkeypatch); stage.eval()
    batch['bond_mask'].zero_()
    main = cached.make_cache((features, None, None, atoms, {}, {}, atoms), batch, stage.config)
    main['key'] = 'train:main'; donor = deepcopy(main); donor['key'] = 'train:donor'
    logits = torch.zeros(len(donor['tensors']['known']))
    entry = dict(cache=donor, initial_logits=logits, binding=replay.bind_replay(donor, logits))
    optimizer = torch.optim.AdamW(stage.pair_parameters(), lr=2e-5, weight_decay=0.)
    result = replay.update_once(stage, main, optimizer, StepLimit(50), attempt=0,
                               emit=lambda *a, **k: None, replay_entry=entry)
    assert result['updated'] and result['positive_rows'] == 0 and result['sampled_rows'] == 3


@pytest.mark.parametrize('real_head', [False, True])
def test_observed_combined_update_has_reproducible_snapshots_and_replay_attribution(monkeypatch, real_head):
    stage, caches, optimizer, rows, bank, captured, _ = setup(monkeypatch)
    if real_head:
        from chemvector.direct_atom_pair_bond import DirectAtomPairBond, DirectAtomPairBondConfig
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260922)
            stage.components.pair = DirectAtomPairBond(DirectAtomPairBondConfig(feature_channels=2,
                atom_feature_dim=2, longitudinal_samples=8, transverse_samples=3, hidden_dim=8, pair_chunk_size=2))
        stage.eval(); optimizer = torch.optim.AdamW(stage.pair_parameters(), lr=2e-5, weight_decay=0.)
        captured = replay.CapturedReplayUpdates(stage, optimizer, StepLimit(50),
            [(caches['train:a'], (0, 2)), (caches['train:b'], (0, 2))], rows, bank=bank, emit=lambda *a, **k: None)
    for attempt, cache in enumerate(caches.values()):
        result = captured.update(cache, attempt=attempt)
        own, other = result['probes'][attempt], result['probes'][1 - attempt]
        assert own['own_sample_update'] and own['probe_row_sampled_in_update']
        assert not own['replay_image_is_probe'] and not own['probe_row_in_replay']
        assert not other['own_sample_update'] and other['probe_row_in_replay']
    assert len(captured.events) == 2
    for step, event in captured.events.items():
        assert check_adam_transition(event)['max_parameter_absolute_error'] < 1e-6
        cloned = deepcopy(stage); cloned.components.pair.load_state_dict(event['pair_before'])
        opt = torch.optim.AdamW(cloned.pair_parameters(), lr=2e-5, weight_decay=0.)
        opt.load_state_dict(event['optimizer_before'])
        limit = StepLimit(50); limit.consumed = step - 1
        entry = dict(cache=event['replay_cache'], binding=event['replay_binding'],
                     initial_logits=event['replay_initial_logits'])
        actual = replay.update_once(cloned, event['cache'], opt, limit, attempt=event['attempt'],
                                    emit=lambda *a, **k: None, replay_entry=entry)
        assert_same(actual, event['record']['update'])
        assert_same(cloned.components.pair.state_dict(), event['pair_after'])


@pytest.mark.parametrize('change', ['attempt', 'lr', 'replay_binding', 'weights', 'limit', 'main_order'])
def test_invalid_update_poisoned_without_retry(monkeypatch, change):
    stage, caches, optimizer, _, bank, captured, _ = setup(monkeypatch)
    attempt, cache = 0, caches['train:a']
    if change == 'attempt': attempt = 1
    elif change == 'lr': optimizer.param_groups[0]['lr'] *= 2
    elif change == 'replay_binding': bank.entries['train:b']['binding']['row_ids'].clear()
    elif change == 'weights':
        with torch.no_grad(): next(stage.components.pair.parameters()).add_(1)
    elif change == 'limit': captured.limit.maximum = 51
    else: cache = caches['train:b']
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Invalid update'))
    with pytest.raises(ValueError): captured.update(cache, attempt=attempt)
    with pytest.raises(RuntimeError, match='no retry'): captured.update(caches['train:a'], attempt=0)


def authorization():
    return dict(schema_version='pair-nonedge-replay-authorization-0.1', user_statement='SYNTHETIC TEST ONLY',
        protocol_sha256=cached.json_digest(launch.protocol()), manifest_sha256=launch.rules.MANIFEST_SHA,
        initial_sha256=launch.rules.INITIAL_SHA, original_admission_sha256=launch.rules.ADMISSION_SHA,
        historical_control_report_sha256=launch.CONTROL_SHA, training_authorized=True,
        deployment_authorized=False, maximum_invocations=1, limits=deepcopy(launch.protocol()['limits']),
        use='local_pair_nonedge_replay58_once', output=str(launch.OUTPUT.relative_to(launch.ROOT)),
        implementation_sha256={p: 'a' * 64 for p in launch.IMPLEMENTATIONS})


def test_proposal_is_not_authorization_and_old_grant_rejected():
    assert launch.protocol()['training_authorized'] is False
    launch.validate_authorization(authorization())
    for value in (old_authorization(), launch.protocol(), {}):
        with pytest.raises(ValueError): launch.validate_authorization(value)


@pytest.mark.parametrize('change', ['steps', 'attempts', 'gpu', 'deploy', 'boolean', 'statement', 'implementation'])
def test_expanded_or_missing_repair_authorization_rejected(change):
    value = authorization()
    if change == 'steps': value['limits']['maximum_steps'] = 51
    elif change == 'attempts': value['limits']['maximum_batch_attempts'] = 59
    elif change == 'gpu': value['limits']['gpu_devices'] = ['0', '1']
    elif change == 'deploy': value['deployment_authorized'] = True
    elif change == 'boolean': value['maximum_invocations'] = True
    elif change == 'statement': value['user_statement'] = ' '
    else: value['implementation_sha256'].pop(launch.IMPLEMENTATIONS[-1])
    with pytest.raises(ValueError): launch.validate_authorization(value)


def test_exact_control_match_rejects_dtype_boolean_or_prediction_change():
    require_control_equal({'logits': [1.]}, {'logits': [1.]}, 'synthetic')
    for value in ({'logits': [1]}, {'logits': [True]}, {'logits': [1.01]}):
        with pytest.raises(ValueError): require_control_equal(value, {'logits': [1.]}, 'synthetic')


def synthetic_control():
    rows = [dict(row('w' + str(i)), fit_domain='witness') for i in range(26)]
    rows += [row('d' + str(i)) for i in range(32)]
    receipts, attempts = [], []
    for i, r in enumerate(rows):
        key = r['source'] + ':' + r['sample_key']
        receipts.append(dict(key=key, coverage=dict(known_negative_pairs=0 if i < 4 else 53,
                                                    reachable_positive_pairs=0)))
        if i < 54:
            event = dict(key=key, attempt=i, updated=i >= 4)
            if i >= 4: event['step'] = i - 3
            attempts.append(event)
    return rows, receipts, attempts


def test_metadata_schedule_uses_completed_steps_not_attempts_and_never_pair_slots():
    from tools.prepare_pair_replay_trial import metadata_plan
    plan = metadata_plan(*synthetic_control())
    assert plan['assignments'][0] == dict(attempt=4, step=1, main_key='train:w4', donor_key='train:d0', maximum_pairs=32)
    assert plan['planned_replay_pair_rows'] == 1600 and plan['empty_attempts'] == 4
    assert len(plan['assignments']) == 50 and len(plan['donor_visit_counts']) == 32
    assert plan['historical_pair_slots_reused'] is False


@pytest.mark.parametrize('change', ['order', 'donors', 'negatives', 'steps', 'empty', 'fewer_steps'])
def test_invalid_metadata_preparation_rejected(change):
    from tools.prepare_pair_replay_trial import metadata_plan
    rows, receipts, attempts = synthetic_control()
    if change == 'order': receipts.reverse()
    elif change == 'donors': rows[-1]['molecule_id'] = rows[-2]['molecule_id']
    elif change == 'negatives': receipts[-1]['coverage']['known_negative_pairs'] = 31
    elif change == 'steps': attempts[5]['step'] = 7
    elif change == 'empty': attempts[0]['step'] = 1
    else: attempts.pop()
    with pytest.raises(ValueError): metadata_plan(rows, receipts, attempts)


def test_default_launcher_without_new_authority_cannot_reserve_or_run(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, 'AUTHORIZATION', tmp_path / 'absent.json')
    monkeypatch.setattr(launch, 'run_budgeted', lambda *a, **k: pytest.fail('Unapproved training invocation'))
    monkeypatch.setattr(launch.previous.old, 'reserve_once', lambda *a, **k: pytest.fail('Unapproved reservation'))
    with pytest.raises(FileNotFoundError): launch.main([])
    assert not list(tmp_path.iterdir())
