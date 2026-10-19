"""Synthetic CPU control/retention parity and scope tests; no real fitting."""

import ast
from copy import deepcopy
import inspect
import json

import pytest
import torch

from chemvector.training.curriculum import selector_full_loss_stage as control
from chemvector.training.curriculum import selector_full_loss_trial as trial
from chemvector.training.curriculum import selector_full_loss_evidence as evidence
from chemvector.training.curriculum import selector_full_loss_worker as worker
from chemvector.training.curriculum import selector_retention_stage as previous
from chemvector.training.curriculum.selector_retention_worker import assess, cache_digest
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_curriculum_continuation import tiny
from tests.chemvector.test_selector_retention_trial import cached, optimizer


def test_only_objective_differs_not_update_guards_or_sampler():
    assert ast.dump(ast.parse(inspect.getsource(control.update))) == ast.dump(ast.parse(inspect.getsource(previous.update)))
    for name in ('configure', 'parameters', 'training_order', 'local_gate', 'final_gate'):
        assert getattr(control, name) is getattr(previous, name)
    assert {k for k in trial.PROTOCOL if trial.PROTOCOL[k] != trial.previous.PROTOCOL[k]} == {
        'retention_weight', 'conditional_regression_cpu_seconds'}


def test_control_objective_and_gradients_match_original_full_exactly(cached):
    model, _, cache, _ = cached
    output = control.call_selector(model.set_head, cache['raw'], cache['bonds'])
    total, full, retention = control.objective(output, cache['raw'], cache['bonds'], cache['labels'])
    old_total, old_full, old_keep = previous.objective(output, cache['raw'], cache['bonds'], cache['labels'])
    assert torch.equal(total, full) and torch.equal(total, old_full) and retention == 0
    assert torch.allclose(old_total - total, old_keep)
    ps = control.parameters(model)
    a = torch.autograd.grad(total, ps, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(old_full, ps, allow_unused=True)
    assert all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(a, b))


def test_control_update_is_exact_manual_full_update_and_preserves_frozen(cached):
    model, _, cache, _ = cached
    direct = deepcopy(model)
    before, frozen, inputs = state_digest(model), control.frozen_digest(model), cache_digest([cache])
    reserved = []
    result = control.update(model, cache, optimizer(model), StepLimit(1), 'test', 'cpu', reserved.append)
    ps, opt = control.parameters(direct), optimizer(direct)
    out = control.call_selector(direct.set_head, cache['raw'], cache['bonds'])
    previous.objective(out, cache['raw'], cache['bonds'], cache['labels'])[1].backward()
    torch.nn.utils.clip_grad_norm_(ps, 1., error_if_nonfinite=True)
    opt.step()
    assert state_digest(model) == state_digest(direct) != before
    assert control.frozen_digest(model) == frozen and cache_digest([cache]) == inputs
    assert reserved == [1] and result['retention_loss'] == 0 and result['loss'] == result['original_full_loss']


@pytest.mark.parametrize('field,value', [('purpose', 'diagnostic'), ('training_authorized', False), ('manifest_sha256', 'wrong')])
def test_control_rejects_unadmitted_cache_before_update(cached, field, value):
    model, _, cache, _ = cached
    before = state_digest(model)
    with pytest.raises(ValueError, match='cannot train'):
        control.update(model, {**cache, field: value}, optimizer(model), StepLimit(1), 'test', 'cpu', lambda _: None)
    assert before == state_digest(model)


def test_geometry_and_matching_are_observational(cached):
    model, config, cache, _ = cached
    before = cache_digest([cache])
    record = assess(model, config, None, cache, device='cpu')
    geo = evidence.geometry(cache, config)
    matched = evidence.matching(cache, record, config)
    assert len(geo['target_anchors']) == 3 and len(geo['candidates']) == int(cache['raw']['atom_mask'].sum())
    assert all(c['nearest_distance_px'] >= 0 for c in geo['candidates'])
    assert len(matched['selected_target_by_slot']) == config.maximum_atoms
    assert before == cache_digest([cache])
    evidence.baseline_parity(record, deepcopy(record))
    changed = deepcopy(record)
    changed['metrics']['logits']['0'] += .001
    with pytest.raises(ValueError, match='baseline differs'):
        evidence.baseline_parity(record, changed)


@pytest.mark.parametrize('step', [0, 100, 200, 300])
def test_unequal_step_development_comparison_forbidden(tmp_path, step):
    with pytest.raises(ValueError, match='equal-step'):
        evidence.paired_reference(tmp_path, [], step=step, development=True)


def test_equal_step_pairing_and_summary(cached, tmp_path):
    model, config, cache, _ = cached
    record = assess(model, config, None, cache, device='cpu')
    for prefix in ('check-000100', 'final-development'):
        (tmp_path / (prefix + '--x.json')).write_text(json.dumps(record))
    a = evidence.paired_reference(tmp_path, [record], step=100)
    b = evidence.paired_reference(tmp_path, [record], step=400, development=True)
    assert a['reference'] == a['control'] == b['control']
    assert a['control']['strict_sdf_exact'] is None


def test_reference_journal_must_be_complete(tmp_path):
    events = [dict(event=e, step=i, key='x') for i in range(1, 401) for e in ('step_reserved', 'update')]
    path = tmp_path / 'steps.jsonl'
    path.write_text('\n'.join(json.dumps(e) for e in events))
    assert evidence.reference_updates(tmp_path) == dict.fromkeys(range(1, 401), 'x')
    events[-1]['key'] = 'wrong'
    path.write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match='incomplete'):
        evidence.reference_updates(tmp_path)


def test_worker_requires_supervisor(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='supervised single GPU'):
        worker.worker()
