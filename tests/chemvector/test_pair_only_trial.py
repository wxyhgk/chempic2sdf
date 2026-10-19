"""Synthetic CPU cache/update/authorization tests; no real checkpoint or GPU."""

from copy import deepcopy

import pytest
import torch

from chemvector.training.curriculum import pair_cached_trial as cached, pair_only_trial as launch
from chemvector.training.curriculum import pair_trial_contract as rules
from chemvector.training.curriculum.pair_only_worker import save_delta
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_curriculum_pair_stage import setup


def fixture(monkeypatch, *, unknown=False):
    stage, batch, atoms, features, calls = setup(monkeypatch, all_unknown=unknown)
    stage.eval()
    forward = (features, None, None, atoms, {}, {}, atoms)
    cache = cached.make_cache(forward, batch, stage.config)
    optimizer = torch.optim.AdamW(stage.pair_parameters(), lr=2e-5, weight_decay=0.)
    return stage, batch, cache, optimizer


def test_cached_objective_matches_existing_stage_before_update(monkeypatch):
    stage, batch, cache, optimizer = fixture(monkeypatch)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rules.RECIPE['seed'])
        expected = {k: float(v.detach()) for k, v in stage(batch)['losses'].items()}
    before = {k: v.clone() for k, v in stage.components.state_dict().items()}
    limit, events = StepLimit(200), []
    result = cached.update_once(stage, cache, optimizer, limit, attempt=0,
                               emit=lambda event, **values: events.append((event, values)))
    assert result['losses'] == expected
    assert result['positive_rows'] == 2 and result['sampled_unknown_pairs'] == 0
    assert result['updated'] and limit.consumed == 1
    changed = [k for k, v in stage.components.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all(k.startswith('pair.') for k in changed)
    assert events[0][0] == 'step_reserved'
    cached.verify_cache(cache)
    assert all(p.grad is None for p in stage.parameters())


def test_empty_branch_does_not_call_optimizer_or_consume_step(monkeypatch):
    stage, _, cache, optimizer = fixture(monkeypatch, unknown=True)
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Empty branch called optimizer'))
    limit = StepLimit(200)
    result = cached.update_once(stage, cache, optimizer, limit, attempt=0, emit=lambda *a, **k: None)
    assert not result['updated'] and limit.consumed == 0
    assert cache['coverage']['target_bonds'] == cache['coverage']['unreachable_bonds'] == 3


def test_known_negatives_without_positives_are_not_an_empty_branch(monkeypatch):
    stage, batch, atoms, features, _ = setup(monkeypatch)
    stage.eval()
    batch['bond_mask'].zero_()
    cache = cached.make_cache((features, None, None, atoms, {}, {}, atoms), batch, stage.config)
    optimizer = torch.optim.AdamW(stage.pair_parameters(), lr=2e-5, weight_decay=0.)
    result = cached.update_once(stage, cache, optimizer, StepLimit(200), attempt=0, emit=lambda *a, **k: None)
    assert result['updated'] and result['positive_rows'] == 0 and result['sampled_rows'] == 3


@pytest.mark.parametrize('change', ['cache', 'train_mode', 'unfreeze', 'optimizer', 'step_cap', 'attempt_cap'])
def test_invalid_update_is_rejected_before_optimizer_step(monkeypatch, change):
    stage, _, cache, optimizer = fixture(monkeypatch)
    limit, attempt = StepLimit(200), 0
    if change == 'cache': cache['tensors']['features'].add_(1)
    elif change == 'train_mode': stage.train()
    elif change == 'unfreeze': stage.components.vision.requires_grad_(True)
    elif change == 'optimizer': optimizer.add_param_group({'params': list(stage.components.vision.parameters())})
    elif change == 'step_cap': limit.consumed = 200
    else: attempt = 232
    monkeypatch.setattr(optimizer, 'step', lambda: pytest.fail('Invalid update called optimizer'))
    with pytest.raises((ValueError, RuntimeError)):
        cached.update_once(stage, cache, optimizer, limit, attempt=attempt, emit=lambda *a, **k: None)


def test_cache_has_no_alias_or_pair_hidden_and_no_retained_autograd(monkeypatch):
    stage, _, cache, _ = fixture(monkeypatch)
    t = cache['tensors']
    assert set(t) == {'features', 'raw', 'bonds', 'context', 'selected', 'mapping',
                      'pair_indices', 'sample_indices', 'known', 'labels', 'bond_slots'}
    assert not t['features'].requires_grad and t['features'].device.type == 'cpu'
    assert t['raw']['atom_features'].data_ptr() != t['selected']['atom_features'].data_ptr()


def test_full_pair_metrics_include_unknown_and_keep_unreachable_denominator(monkeypatch):
    _, _, cache, _ = fixture(monkeypatch)
    t = cache['tensors']; count = len(t['pair_indices'])
    chemistry = torch.full((count, 4), -20.)
    chemistry[torch.arange(count), t['labels']['chemical']] = 20.
    structure = dict(pairs=t['pair_indices'], output=dict(presence_logits=torch.full((count,), 20.),
                                                        chemical_logits=chemistry))
    metric = cached.score_case(cache, structure, dict(molblock=None, fallback=False),
                              dict(roles=[], ocr=[]), key='synthetic', expected='CCCC')
    assert metric['correct_typed_bond_ids'] == [0, 1]
    assert metric['target_bonds'] == 3 and metric['reachable_positive_pairs'] == 2
    assert metric['confirmed_false_positives'] == 1 and metric['unknown_present'] == 3
    assert not metric['sdf_valid'] and not metric['sdf_exact']


def authorization():
    return dict(schema_version='chemvector-pair-only-training-authorization-0.1',
        user_statement='持续推进，开始第一轮的训练', reaffirmation='继续',
        protocol_sha256=launch.PROPOSAL_SHA, manifest_sha256=rules.MANIFEST_SHA,
        original_admission_sha256=rules.ADMISSION_SHA, initial_sha256=rules.INITIAL_SHA,
        training_authorized=True, deployment_authorized=False, maximum_invocations=1,
        output=str(launch.OUTPUT.relative_to(launch.ROOT)), recipe=deepcopy(rules.RECIPE),
        implementation_sha256={p: 'a' * 64 for p in launch.IMPLEMENTATIONS},
        data_use_extension=dict(previous_use='local_selector_retention_fit_only', approved_use=rules.USE,
            training_images=58, new_data=False, validation_sample_artifacts=False, identity_queries=0))


def test_authorization_requires_exact_new_budget_scope_and_implementation():
    launch.validate_authorization(authorization())


@pytest.mark.parametrize('change', ['old_schema', 'missing', 'extra_data', 'budget', 'deployment', 'boolean'])
def test_old_missing_or_expanded_authorization_rejected(change):
    value = authorization()
    if change == 'old_schema': value['schema_version'] = 'chemvector-selector-retention-training-authorization-0.1'
    elif change == 'missing': value['implementation_sha256'].pop(launch.IMPLEMENTATIONS[0])
    elif change == 'extra_data': value['data_use_extension']['training_images'] = 122
    elif change == 'budget': value['recipe']['maximum_steps'] = 201
    elif change == 'deployment': value['deployment_authorized'] = True
    else: value['maximum_invocations'] = True
    with pytest.raises(ValueError): launch.validate_authorization(value)


def test_consumed_reservation_is_not_reusable(tmp_path):
    path = tmp_path / 'attempt.json'
    launch.reserve_once(path, 'a' * 64)
    before = path.read_bytes()
    with pytest.raises(FileExistsError): launch.reserve_once(path, 'a' * 64)
    assert path.read_bytes() == before


def test_delta_publication_is_exclusive_and_loadable(tmp_path):
    path = tmp_path / 'delta.pt'
    payload = {'pair_supervision': rules.RECIPE['pair_supervision'], 'pair_state': {'weight': torch.ones(2)}}
    digest = save_delta(path, payload)
    assert len(digest) == 64
    restored = torch.load(path, weights_only=True)
    assert torch.equal(restored['pair_state']['weight'], payload['pair_state']['weight'])
    with pytest.raises(FileExistsError): save_delta(path, payload)
    assert not list(tmp_path.glob('.pair-delta-*'))
