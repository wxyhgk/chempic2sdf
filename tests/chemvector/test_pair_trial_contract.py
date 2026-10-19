"""Synthetic CPU protocol/mask/freeze/acceptance checks. No real training."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from chemvector.training.curriculum import pair_trial_contract as trial
from tests.chemvector.test_curriculum_supervision_contract import pair_fixture


def rows():
    return [dict(source='synthetic', sample_key=str(i), meta_sha256='a' * 64,
                 fit_domain='witness' if i < 26 else 'old_full_molecule', split='train')
            for i in range(58)]


def test_proposal_is_separate_default_deny_and_does_not_mutate_callers():
    records = rows()
    before = deepcopy(records)
    proposal = trial.make_proposal(records)
    trial.validate_proposal(proposal, records)
    assert records == before
    assert proposal['training_authorized'] is proposal['data_use_approved'] is False
    assert proposal['cache']['cache_pair_hidden'] is False
    assert proposal['recipe']['maximum_steps'] == 200
    proposal['recipe']['maximum_steps'] = 201
    assert trial.RECIPE['maximum_steps'] == 200


@pytest.mark.parametrize('field,value', [
    ('training_authorized', True), ('data_use_approved', True), ('deployment_authorized', True),
    ('training_authorized', 0), ('initial_sha256', 'b' * 64), ('use', 'local_selector_retention_fit_only'),
])
def test_old_or_fabricated_grants_cannot_turn_the_proposal_into_authorization(field, value):
    proposal = trial.make_proposal(rows())
    proposal[field] = value
    with pytest.raises(ValueError):
        trial.validate_proposal(proposal, rows())


@pytest.mark.parametrize('change', ['validation', 'duplicate', 'missing', 'wrong_group'])
def test_wrong_data_scope_rejected(change):
    records = rows()
    if change == 'validation': records[0]['split'] = 'validation'
    elif change == 'duplicate': records[1] = deepcopy(records[0])
    elif change == 'missing': records.pop()
    else: records[0]['fit_domain'] = 'other'
    with pytest.raises(ValueError): trial.make_proposal(records)


@pytest.mark.parametrize('kind', ['normal', 'unknown', 'negative_only'])
def test_old_stage_and_new_mask_equivalent_including_empty_and_no_positive(kind):
    atoms, mapping, targets = pair_fixture()
    if kind == 'unknown': mapping.fill_(-1)
    elif kind == 'negative_only': targets['bond_mask'].zero_()
    rng = torch.get_rng_state().clone()
    result = trial.compare_supervision(atoms, mapping, targets, seed=20260921)
    assert torch.equal(torch.get_rng_state(), rng)
    assert result['sampled_unknown_pairs'] == 0
    assert result['mask_equal'] and result['sampled_row_ids_equal']
    if kind == 'unknown':
        assert result['empty_branch'] and result['sampled_rows'] == 0
        assert result['unreachable_bonds'] == 3
    elif kind == 'negative_only':
        assert not result['empty_branch']
        assert result['sampled_positive_rows'] == 0 and result['sampled_negative_rows'] == 3
    else:
        assert result['sampled_positive_rows'] == 2 and result['sampled_negative_rows'] == 1


def test_reuse_stage_all_eval_and_only_pair_gradients_allowed():
    components = nn.Module()
    for name in ('vision', 'atom', 'bond', 'set_head', 'role', 'bbox', 'pair'):
        components.add_module(name, nn.Sequential(nn.Linear(2, 2), nn.Dropout(.5)))
    state = {k: v.clone() for k, v in components.state_dict().items()}
    stage = trial.configure_pair_stage(components, SimpleNamespace(image_size=768, atom_match_radius_px=6))
    assert all(not m.training for m in stage.modules())
    assert all(p.requires_grad == name.startswith('pair.') for name, p in components.named_parameters())
    assert all(torch.equal(state[k], v) for k, v in components.state_dict().items())
    with torch.enable_grad(): components.pair(torch.ones(1, 2)).sum().backward()
    assert all(p.grad is None for name, p in components.named_parameters() if not name.startswith('pair.'))


def metrics():
    return [dict(key='known', target_bonds=3, reachable_positive_pairs=2,
                 correct_typed_bond_ids=[0], confirmed_false_positives=1, unknown_present=3,
                 sdf_exact=False, sdf_valid=True, fallback=False,
                 candidate_label_sha256='a' * 64, text_sha256='b' * 64),
            dict(key='empty', target_bonds=1, reachable_positive_pairs=0,
                 correct_typed_bond_ids=[], confirmed_false_positives=0, unknown_present=1,
                 sdf_exact=False, sdf_valid=False, fallback=False,
                 candidate_label_sha256='c' * 64, text_sha256='d' * 64)]


@pytest.mark.parametrize('step,action', [(50, 'continue_within_original_budget'),
                                       (100, 'stop_no_improvement'), (200, 'stop_no_improvement')])
def test_no_connection_or_sdf_improvement_stops_without_using_loss(step, action):
    before = metrics()
    current = deepcopy(before)
    current[0]['training_loss'] = -1000
    assert trial.checkpoint_decision(before, current, step=step)['action'] == action


def test_real_typed_gain_passes_only_as_train_fit():
    before, current = metrics(), metrics()
    current[0]['correct_typed_bond_ids'] = [0, 1]
    result = trial.checkpoint_decision(before, current, step=200)
    assert result['action'] == 'complete_fit_probe_not_deployment'
    assert result['progress'] and result['typed_bond_gains'] == 1
    assert result['independent_accuracy_measured'] is result['deployment_authorized'] is False


def test_aggregate_gain_cannot_hide_one_broken_baseline_bond():
    before, current = metrics(), metrics()
    current[0]['correct_typed_bond_ids'] = [1, 2]
    result = trial.checkpoint_decision(before, current, step=100)
    assert result['action'] == 'stop_regression' and not result['progress']
    assert result['protection_failures'][0]['lost_correct_typed_bond_ids'] == [0]


@pytest.mark.parametrize('kind', ['exact', 'valid', 'fallback', 'false_positive'])
def test_each_individual_regression_triggers_stop(kind):
    before, current = metrics(), metrics()
    current[0]['correct_typed_bond_ids'] = [0, 1]
    if kind == 'exact': before[0]['sdf_exact'] = True
    elif kind == 'valid': current[0]['sdf_valid'] = False
    elif kind == 'fallback': current[0]['fallback'] = True
    else: current[0]['confirmed_false_positives'] += 1
    assert trial.checkpoint_decision(before, current, step=50)['action'] == 'stop_regression'


@pytest.mark.parametrize('kind', ['drop_empty', 'denominator', 'signature', 'text', 'duplicate_ids', 'negative_count', 'bool_count'])
def test_invalid_evaluation_inputs_fail_closed(kind):
    before, current = metrics(), metrics()
    if kind == 'drop_empty': current.pop()
    elif kind == 'denominator': current[0]['target_bonds'] += 1
    elif kind == 'signature': current[0]['candidate_label_sha256'] = 'e' * 64
    elif kind == 'text': current[0]['text_sha256'] = 'e' * 64
    elif kind == 'duplicate_ids': current[0]['correct_typed_bond_ids'] = [0, 0]
    elif kind == 'negative_count': current[0]['unknown_present'] = -1
    else: current[0]['confirmed_false_positives'] = True
    with pytest.raises(ValueError): trial.checkpoint_decision(before, current, step=100)


def test_unknown_predictions_not_mislabeled_as_false_edges_or_hidden():
    before, current = metrics(), metrics()
    current[0]['unknown_present'] += 4
    result = trial.checkpoint_decision(before, current, step=100)
    assert not result['protection_failures']
    assert result['unknown_present_after'] - result['unknown_present_before'] == 4
    assert result['action'] == 'stop_no_improvement'
