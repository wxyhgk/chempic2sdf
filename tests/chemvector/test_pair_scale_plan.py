"""Synthetic scale-plan tests. No real release/model/GPU or validation media."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from chemvector.training.curriculum.pair_scale_plan import (
    bond_bucket, coverage, deduplicate, draft_protocol, exposure_plan, sample_id, size_bucket, stratified_order)
from chemvector.training.curriculum.pair_scale_policy import development_decision
from tools.prepare_pair_scale_training import bound_bytes, check_partition, read_train_record


def rows(full=40, foundation=5):
    result = []
    for i in range(full + foundation):
        result.append(dict(source='test', sample_key=f's{i}', split='train',
            domain='full_molecule' if i < full else 'curriculum',
            canonical_smiles=f'identity-{i}', group_key=f'g{i}', molecule_id=f'm{i}',
            pixel_sha256=sha256(str(i).encode()).hexdigest(), heavy_atoms=10+i % 45,
            bond_counts={('SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC')[i % 4]: 1}, profile_id=f'p{i%3}'))
    return result


def test_finite_pass_and_explicit_replay_no_hidden_repeat():
    data = rows()
    schedule = exposure_plan(data)
    counts = Counter(s['index'] for s in schedule)
    assert len(schedule) == 50 and set(counts) == set(range(45))
    assert all(counts[i] == 1 for i in range(40))
    assert all(counts[i] == 2 for i in range(40, 45))
    assert all([data[s['index']]['domain'] for s in schedule[i:i+5]] == ['full_molecule']*4+['curriculum']
               for i in range(0, 50, 5))
    assert Counter(s['role'] for s in schedule) == {'coverage': 45, 'foundation_replay': 5}
    assert coverage(data, schedule)['distinct_images'] == 45


def test_sampler_deterministic_without_input_mutation():
    data = rows(); original = deepcopy(data)
    assert exposure_plan(data) == exposure_plan(data)
    assert data == original
    assert [sample_id(data[s['index']]) for s in exposure_plan(data)] == [
        sample_id(data[::-1][s['index']]) for s in exposure_plan(data[::-1])]


def test_non_multiple_four_full_tail_still_complete():
    data = rows(full=41)
    schedule = exposure_plan(data)
    assert len(schedule) == 52
    assert len({s['index'] for s in schedule}) == 46
    assert schedule[-1]['role'] == 'foundation_replay'


@pytest.mark.parametrize('mode', ['validation', 'quarantine', 'duplicate', 'missing_full', 'too_many_foundation'])
def test_schedule_rejects_invalid_population(mode):
    data = rows()
    if mode in ('validation','quarantine'): data[0]['split'] = mode
    elif mode == 'duplicate': data.append(data[0])
    elif mode == 'missing_full': data = data[40:]
    else: data = rows(full=4, foundation=5)
    with pytest.raises(ValueError): exposure_plan(data)


def test_molecule_styles_spread_before_second_visit_in_same_stratum():
    data = rows(full=4, foundation=0)
    for i,r in enumerate(data):
        r.update(canonical_smiles=f'm{i//2}', heavy_atoms=10, bond_counts={'SINGLE':1})
    ordered = stratified_order(data, salt='test')
    assert len({r['canonical_smiles'] for r in ordered[:2]}) == 2
    assert len({sample_id(r) for r in ordered}) == 4


def test_dedup_keeps_original_reference_and_records_alias():
    data=rows(); duplicate=deepcopy(data[0]); duplicate['sample_key']='duplicate'
    unique, aliases=deduplicate([*data,duplicate])
    assert len(unique)==45 and len(aliases)==1
    assert all(r in [*data,duplicate] for r in unique)
    assert len({r['pixel_sha256'] for r in unique})==45


@pytest.mark.parametrize('field', ['canonical_smiles','domain','group_key','split'])
def test_duplicate_conflicts_cannot_be_silently_removed(field):
    data=rows(); r=deepcopy(data[0]); r['sample_key']='alias'
    r[field]='curriculum' if field=='domain' else 'changed'
    with pytest.raises(ValueError): deduplicate([*data,r])


def test_preparation_cannot_open_validation_source():
    with pytest.raises(ValueError, match='Training-only'):
        read_train_record({},dict(split='validation',domain='full_molecule'))


def test_bound_bytes_rejects_changed_and_aliased_inputs(tmp_path):
    path=tmp_path/'one'; path.write_bytes(b'fixed')
    h=sha256(b'fixed').hexdigest()
    assert bound_bytes(path,h)==b'fixed'
    with pytest.raises(ValueError): bound_bytes(path,'0'*64)
    alias=tmp_path/'two'; alias.symlink_to(path)
    with pytest.raises(ValueError): bound_bytes(alias,h)


def test_partition_checks_manifest_only(monkeypatch):
    # No source/media path is needed or supplied.
    import tools.prepare_pair_scale_training as prepare
    monkeypatch.setattr(prepare,'identities',lambda s:dict(group_key=s,connectivity_prefix=s))
    data=[dict(source='a',sample_key='one',split='train',canonical_smiles='C',group_key='C',molecule_id='m1'),
          dict(source='b',sample_key='two',split='validation',canonical_smiles='N',group_key='N',molecule_id='m2')]
    assert check_partition(data)['train_development_overlap']==0
    data[1].update(canonical_smiles='C',group_key='C')
    with pytest.raises(ValueError,match='overlap'): check_partition(data)


@pytest.mark.parametrize('n,bucket',[(0,'00-07'),(7,'00-07'),(8,'08-20'),(20,'08-20'),(21,'21-40'),(40,'21-40'),(41,'41+')])
def test_size_strata(n,bucket): assert size_bucket(n)==bucket


def test_no_bond_still_in_schedule_and_unsupported_bond_rejected():
    data=rows(); data[-1]['bond_counts']={}
    assert coverage(data,exposure_plan(data))['zero_bond_images']==1
    assert bond_bucket({})=='NO_BOND'
    with pytest.raises(ValueError): bond_bucket({'DATIVE':1})


def metrics():
    return [dict(key=str(i),domain='full_molecule' if i<100 else 'curriculum',
                 exact=i%2==0,valid=True,fallback=False) for i in range(200)]


def test_one_old_error_no_longer_automatically_stops_training():
    a=metrics(); b=deepcopy(a); b[0]['exact']=False
    d=development_decision(a,b,complete_coverage=False)
    assert d['action']=='continue_to_next_fixed_checkpoint'
    assert d['groups']['full_molecule']['broken']==1
    assert not d['candidate_eligible']


@pytest.mark.parametrize('field', ['exact','valid'])
def test_five_percentage_point_drop_stops(field):
    a=metrics(); b=deepcopy(a)
    for i in range(0,10,2):
        b[i][field]=False
        if field=='valid': b[i]['exact']=False
    assert development_decision(a,b,complete_coverage=False)['action']=='stop_severe_regression'


def test_final_requires_gain_and_complete_coverage_not_loss():
    a=metrics(); b=deepcopy(a); b[1]['exact']=True
    assert not development_decision(a,b,complete_coverage=False)['candidate_eligible']
    assert development_decision(a,b,complete_coverage=True)['candidate_eligible']
    assert development_decision(a,a,complete_coverage=True)['action']=='complete_keep_initial'


@pytest.mark.parametrize('field',['valid','fallback','curriculum'])
def test_final_gain_cannot_hide_other_guard_failures(field):
    a=metrics(); b=deepcopy(a); b[1]['exact']=True
    if field=='valid': b[3]['valid']=False
    elif field=='fallback': b[3]['fallback']=True
    else:
        for i in (100,102,104): b[i]['exact']=False
    assert not development_decision(a,b,complete_coverage=True)['candidate_eligible']


def test_fixed_development_cohort_cannot_change():
    a=metrics(); b=deepcopy(a); b.pop()
    with pytest.raises(ValueError): development_decision(a,b,complete_coverage=True)


def test_draft_is_not_execution_and_has_full_pass_budget():
    p=draft_protocol(16000)
    assert not p['training_authorized'] and not p['deployment_authorized']
    assert p['proposed_budget']['maximum_attempts']==16000
    assert p['checkpoint_attempts']==[4000,8000,12000,16000]
    assert p['proposed_budget']['wall_seconds']==3600
    with pytest.raises(ValueError): draft_protocol(58)
