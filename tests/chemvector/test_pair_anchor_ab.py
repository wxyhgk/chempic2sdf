"""Pure paired gates and authorization tests. No corpus or GPU execution."""

from copy import deepcopy

import pytest

from chemvector.training.curriculum import pair_anchor_ab_contract as rules, pair_anchor_ab_trial as launch
from chemvector.training.curriculum import pair_anchor_ab_worker as worker
from chemvector.training.curriculum.pair_cached_trial import json_digest


def records():
    return [dict(key='one', target_bonds=3, reachable_positive_pairs=3, correct_typed_bond_ids=[0],
        promoted_correct_typed_bond_ids=[0], pair_count=6, confirmed_false_positives=1,
        confirmed_false_pair_row_ids=[4], unknown_present=0, unknown_present_row_ids=[],
        sdf_exact=False, sdf_valid=True, fallback=False, candidate_label_sha256='a'*64, text_sha256='b'*64)]


def test_protocol_is_default_deny_and_independent_copy():
    p = rules.protocol(); rules.validate_protocol(p)
    assert p['training_authorized'] is p['deployment_authorized'] is False
    assert p['limits']['total_updates'] == 108 and p['limits']['wall_seconds'] == 600
    p['limits']['total_updates'] = 109
    with pytest.raises(ValueError): rules.validate_protocol(p)
    assert rules.protocol()['limits']['total_updates'] == 108


def test_same_count_but_new_false_pair_is_a_regression():
    a, b = records(), records()
    b[0]['confirmed_false_pair_row_ids'] = [5]
    result = rules.protections(a, b)
    assert not result['passed']
    assert result['failures'][0]['new_confirmed_false_pair_row_ids'] == [5]


@pytest.mark.parametrize('change', ['raw', 'promoted', 'valid', 'fallback', 'exact'])
def test_every_initial_protection_applies_to_both_arms(change):
    a, b = records(), records()
    if change == 'raw': b[0]['correct_typed_bond_ids'] = [1, 2]
    elif change == 'promoted': b[0]['promoted_correct_typed_bond_ids'] = [1, 2]
    elif change == 'valid': b[0]['sdf_valid'] = False
    elif change == 'fallback': b[0]['fallback'] = True
    else: a[0]['sdf_exact'] = True
    for control, augmented in ((b, a), (a, b)):
        assert rules.paired_decision(a, control, augmented, attempt=29)['action'] == 'stop_regression'


def test_no_improvement_continues_only_to_the_predeclared_end():
    a = records()
    assert rules.paired_decision(a, a, a, attempt=29)['action'] == 'continue_within_fixed_budget'
    assert rules.paired_decision(a, a, a, attempt=58)['action'] == 'stop_no_safe_control_improvement'
    with pytest.raises(ValueError): rules.paired_decision(a, a, a, attempt=59)


def test_safe_strict_improvement_is_only_training_fit_evidence():
    a, b = records(), records()
    b[0]['correct_typed_bond_ids'] = b[0]['promoted_correct_typed_bond_ids'] = [0, 1]
    result = rules.paired_decision(a, a, b, attempt=58)
    assert result['success'] and result['training_accuracy_only']
    assert result['action'] == 'complete_fit_probe_not_deployment'


def test_augmented_cannot_trade_away_control_gains_for_a_larger_net_gain():
    initial, control, aug = records(), records(), records()
    control[0]['correct_typed_bond_ids'] = [0, 1]
    aug[0]['correct_typed_bond_ids'] = [0, 2]
    aug[0]['sdf_exact'] = True
    result = rules.paired_decision(initial, control, aug, attempt=58)
    assert not result['success'] and not result['control_comparison']['passed']


def test_unknown_increase_is_separate_not_falsely_labelled_as_nonedge():
    a, b = records(), records()
    b[0]['unknown_present'] = 1; b[0]['unknown_present_row_ids'] = [5]
    result = rules.protections(a, b)
    assert result['passed']
    assert result['unknown_changes'][0]['new_unknown_present_row_ids'] == [5]


@pytest.mark.parametrize('change', ['duplicate', 'out_of_range', 'bool_id', 'overlap', 'count', 'cohort', 'signature'])
def test_malformed_metric_or_changed_lineage_rejected(change):
    a, b = records(), records()
    if change == 'duplicate': b[0]['promoted_correct_typed_bond_ids'] = [0, 0]
    elif change == 'out_of_range': b[0]['confirmed_false_pair_row_ids'] = [6]
    elif change == 'bool_id': b[0]['confirmed_false_pair_row_ids'] = [True]
    elif change == 'overlap': b[0]['unknown_present_row_ids'] = [4]; b[0]['unknown_present'] = 1
    elif change == 'count': b[0]['confirmed_false_positives'] = 2
    elif change == 'cohort': b[0]['key'] = 'other'
    else: b[0]['candidate_label_sha256'] = 'c'*64
    with pytest.raises(ValueError): rules.protections(a, b)


def auth():
    return dict(schema_version='pair-anchor-ab-authorization-0.1', use=rules.USE,
        protocol_sha256=json_digest(rules.protocol()), manifest_sha256=rules.original.MANIFEST_SHA,
        initial_sha256=rules.original.INITIAL_SHA, original_admission_sha256=rules.original.ADMISSION_SHA,
        training_authorized=True, deployment_authorized=False, maximum_invocations=1,
        output=str(launch.OUTPUT.relative_to(launch.ROOT)), limits=deepcopy(rules.LIMITS),
        data_use_extension=dict(previous_use='local_selector_retention_fit_only', approved_use=rules.USE,
            training_images=58, new_data=False, validation_sample_artifacts=False, identity_queries=0),
        user_statement='synthetic unit test only; not stored as a real grant',
        preparation_report_sha256='a'*64, implementation_sha256={p: 'b'*64 for p in launch.IMPLEMENTATIONS})


def test_exact_auth_validator_only_accepts_new_bound_scope():
    launch.validate_authorization(auth())


@pytest.mark.parametrize('change', ['old_schema', 'old_use', 'old_protocol', 'missing_code', 'more_updates',
                                   'more_time', 'no_data_extension', 'deployment', 'empty_user', 'bad_hash'])
def test_old_or_broadened_grants_rejected(change):
    a = auth()
    if change == 'old_schema': a['schema_version'] = 'pair-nonedge-replay-authorization-0.1'
    elif change == 'old_use': a['use'] = 'local_selector_retention_fit_only'
    elif change == 'old_protocol': a['protocol_sha256'] = 'f'*64
    elif change == 'missing_code': a['implementation_sha256'].pop(next(iter(a['implementation_sha256'])))
    elif change == 'more_updates': a['limits']['total_updates'] += 1
    elif change == 'more_time': a['limits']['wall_seconds'] += 1
    elif change == 'no_data_extension': a.pop('data_use_extension')
    elif change == 'deployment': a['deployment_authorized'] = True
    elif change == 'empty_user': a['user_statement'] = ''
    else: a['preparation_report_sha256'] = 'not a digest'
    with pytest.raises(ValueError): launch.validate_authorization(a)


def test_unauthorized_worker_cannot_reach_cuda(monkeypatch):
    def deny(): raise ValueError('No new authority')
    monkeypatch.setattr(launch, 'authorize', deny)
    monkeypatch.setattr(worker.torch.cuda, 'is_available', lambda: pytest.fail('Touched CUDA before authorization'))
    with pytest.raises(ValueError, match='No new authority'): worker.worker()
