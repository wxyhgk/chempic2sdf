"""Synthetic CPU checks only: no production weights, data fitting, or GPU."""

from copy import deepcopy
import json

import pytest
import torch

from chemvector.predicted_incidence import associate_predicted_proposals
from chemvector.training.curriculum import selector_retention_stage as stage
from chemvector.training.curriculum import selector_retention_worker as worker
from chemvector.training.curriculum import selector_retention_trial as trial
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_curriculum_continuation import tiny


@pytest.fixture
def cached(tiny):
    model, config, batch = tiny
    stage.configure(model)
    with torch.no_grad():
        forward = structure_forward(model, batch['pixel_values'], config, stage.THRESHOLD)
    raw, bonds = forward[3:5]
    labels = associate_predicted_proposals(raw, bonds, batch, config)
    cache = dict(key='old:x', sample_key='x', purpose='training', training_authorized=True,
        manifest_sha256='test', fit_domain='witness', raw=raw, bonds=bonds, labels=labels,
        batch={k: v for k, v in batch.items() if k != 'pixel_values'})
    return model, config, cache, forward


def optimizer(model):
    return torch.optim.AdamW(stage.parameters(model), lr=5e-6, eps=1e-8, weight_decay=0, foreach=False)


def test_cached_forward_and_finite_selector_only_update(cached):
    model, config, cache, forward = cached
    out = stage.call_selector(model.set_head, cache['raw'], cache['bonds'])
    assert all(torch.equal(out[k], v) for k, v in forward[5].items())
    before, frozen = state_digest(model), stage.frozen_digest(model)
    cached_before = worker.cache_digest([cache])
    reserved, limit = [], StepLimit(1)
    opt = optimizer(model)
    result = stage.update(model, cache, opt, limit, 'test', 'cpu', reserved.append)
    assert result['loss'] == pytest.approx(result['original_full_loss'] + result['retention_loss'])
    assert reserved == [1] and limit.consumed == 1
    assert before != state_digest(model) and frozen == stage.frozen_digest(model)
    assert cached_before == worker.cache_digest([cache])
    after = state_digest(model)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        stage.update(model, cache, opt, limit, 'test', 'cpu', reserved.append)
    assert after == state_digest(model)
    record = worker.assess(model, config, None, cache, device='cpu')
    assert record['prediction'] is None and record['metrics']['target_atoms'] == 3


@pytest.mark.parametrize('field,value', [('purpose', 'diagnostic'), ('training_authorized', False),
    ('manifest_sha256', 'wrong'), ('fit_domain', 'new_development')])
def test_unadmitted_cache_stops_before_optimizer(cached, field, value):
    model, _, cache, _ = cached
    cache = {**cache, field: value}
    before = state_digest(model)
    with pytest.raises(ValueError, match='cannot train'):
        stage.update(model, cache, optimizer(model), StepLimit(1), 'test', 'cpu', lambda _: None)
    assert before == state_digest(model)


def test_failed_reservation_is_before_update(cached):
    model, _, cache, _ = cached
    before = state_digest(model)
    def fail(_):
        raise OSError('journal unavailable')
    with pytest.raises(OSError):
        stage.update(model, cache, optimizer(model), StepLimit(1), 'test', 'cpu', fail)
    assert before == state_digest(model)


def test_optimizer_and_scope_drift_refused(cached):
    model, _, cache, _ = cached
    opt = optimizer(model)
    opt.param_groups[0]['lr'] *= 2
    with pytest.raises(ValueError, match='fixed AdamW'):
        stage.update(model, cache, opt, StepLimit(1), 'test', 'cpu', lambda _: None)
    model.pair.requires_grad_(True)
    with pytest.raises(ValueError, match='scope changed'):
        stage.parameters(model)


def test_repeatable_two_old_one_witness_cycle(cached):
    _, _, cache, _ = cached
    caches = [cache, {**cache, 'key': 'old:y', 'fit_domain': 'old_full_molecule'}]
    a, b = [stage.training_order(caches, 'test') for _ in range(2)]
    assert [next(a)['key'] for _ in range(12)] == [next(b)['key'] for _ in range(12)]
    order = stage.training_order(caches, 'test')
    assert [next(order)['fit_domain'] for _ in range(6)] == ['old_full_molecule', 'old_full_molecule', 'witness'] * 2


def panels():
    def row(key, group, positive):
        return dict(key=key, sample_key=key, group=group, prediction=dict(exact=True) if group == 'old_full_molecule' else None,
            metrics=dict(complete_endpoints=True, matched_target_ids=[0, 1], reachable_bond_ids=[0],
                retained_positive_slots=positive, selected_negative_slots=[], positive_slots=positive))
    return [row('old1', 'old_full_molecule', [0, 1]), row('old2', 'old_full_molecule', [0, 1])] + [
        row(k, 'witness', p) for k, p in zip(stage.CORE_KEYS, ([1, 2], [2, 4], [0, 1, 5], [0, 1, 2, 3]))]


def test_no_netting_away_broken_old_image_or_bond():
    before = panels()
    after = deepcopy(before)
    after[0]['prediction']['exact'] = False
    after[0]['metrics']['reachable_bond_ids'] = []
    after[1]['metrics']['reachable_bond_ids'] += [1, 2]
    gate = stage.local_gate(before, after, 100)
    assert gate['stop_reasons'] == ['old_strict_graph_regression', 'old_reachable_bond_regression']
    assert not gate['local_pass']


def test_200_step_nonimprovement_stops():
    rows = panels()
    assert not stage.local_gate(rows, rows, 100)['stop_reasons']
    assert stage.local_gate(rows, rows, 200)['stop_reasons'] == ['no_fitting_complete_endpoint_improvement_by_step200']


@pytest.mark.parametrize('fault', ['positive_lost', 'false_added', 'ethyne_wrong'])
def test_fixed_core_gate(fault):
    before, after = panels(), panels()
    if fault == 'positive_lost':
        after[3]['metrics']['retained_positive_slots'] = [2]
    elif fault == 'false_added':
        after[3]['metrics']['selected_negative_slots'] = [5]
    else:
        after[2]['metrics']['selected_negative_slots'] = [0]
    assert not stage.local_gate(before, after, 100)['local_pass']


@pytest.mark.parametrize('fault', ['atoms', 'bonds', 'false', 'complete'])
def test_new_development_regression_refuses_full_run(fault):
    before = [panels()[2]]
    after = deepcopy(before)
    m = after[0]['metrics']
    if fault == 'atoms':
        m['matched_target_ids'] = [0]
    elif fault == 'bonds':
        m['reachable_bond_ids'] = []
    elif fault == 'false':
        m['selected_negative_slots'] = [0]
    else:
        m['complete_endpoints'] = False
    result = stage.final_gate(dict(local_pass=True), before, after)
    assert not result['full_regression_allowed']


def test_pairing_and_one_shot_reservation(tmp_path):
    before = panels()
    with pytest.raises(ValueError, match='panel changed'):
        stage.compare(before, before[:-1])
    auth, attempt = tmp_path / 'auth.json', tmp_path / 'attempt.json'
    auth.write_text('{}')
    trial.reserve_once(attempt, auth)
    assert json.loads(attempt.read_text())['automatic_retries'] == 0
    with pytest.raises(FileExistsError):
        trial.reserve_once(attempt, auth)


def test_worker_requires_supervisor(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='supervised single GPU'):
        worker.worker()
