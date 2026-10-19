"""Synthetic CPU-only supervision proposal checks; no model is constructed."""

from copy import deepcopy

import pytest

from tools import selector_node_groups as g
from tools import selector_node_groups_review as review


def example(points=((100,100),(200,100),(103,100),(113,100)),
            targets=((100,100),(200,100)), raw=(0,1,-1,-1)):
    geometry = dict(image_size=768, match_radius_px=6,
        target_anchors=[[x/768,y/768] for x,y in targets],
        candidates=[dict(slot=i, geometry=[x/768,y/768,0,0,0,0,0,0], raw_target=raw[i])
                    for i,(x,y) in enumerate(points)])
    layers = dict(atoms=[dict(atom_id=i, element='C', text='', visible_text=i==0)
                         for i in range(len(targets))], bonds=[])
    return geometry, layers


def glyph(atom=0, bbox=(110,95,120,105)):
    return dict(atom_id=atom, text='r', bbox=bbox)


def bond_fixture():
    geo, layers = example(points=((100,100),(200,100),(100,110)), raw=(0,1,-1))
    bond = dict(bond_id=0, begin=[100,100], end=[200,100], begin_atom=0, end_atom=1)
    path = dict(bond_id=0, node_index=1, points=[(100,110),(200,110)], drawing_type='TRIPLE')
    layers['bonds'] = [bond]
    return geo, layers, path


def record(geo, layers, selected):
    cov = g.coverage(geo, selected, layers)
    return dict(metrics={**{k:v for k,v in cov.items() if k!='mapping'},
        'logits':{str(c['slot']): 1. if c['slot'] in selected else -1. for c in geo['candidates']}})


def test_replacement_is_protected_and_group_not_single_raw_label():
    geo, layers = example()
    p = g.propose(geo,layers,[glyph()],[],[])
    assert p['nodes'][0]['unique_slots'] == [0,2]
    assert p['nodes'][0]['alternative_slots'] == [2]
    assert p['candidates'][2]['status'] == 'protected_node_candidate'
    assert p['proposed_pairs'] == [dict(fragment_slot=3,owner_target=0,kinds=['glyph_fragment'],
        representative_slots=[0],acceptable_unique_slots=[0,2],requires_review=True,training_authorized=False)]
    assert p['training_authorized'] is False


def test_true_glyph_node_is_not_negative():
    geo,layers=example()
    p=g.propose(geo,layers,[glyph(bbox=(95,95,106,105))],[],[])
    assert p['candidates'][0]['evidence']['glyph_hits']
    assert p['candidates'][0]['status']=='protected_node_candidate'
    assert not p['proposed_pairs']


def test_multiple_bond_tip_has_explicit_source_bond_owner():
    geo,layers,path=bond_fixture()
    p=g.propose(geo,layers,[],[path],[])
    assert p['proposed_pairs'][0]['fragment_slot']==2
    assert p['proposed_pairs'][0]['owner_target']==0
    assert p['proposed_pairs'][0]['kinds']==['multiple_bond_off_axis_tip']


def test_unsupported_related_path_excludes_pair():
    geo,layers,path=bond_fixture()
    p=g.propose(geo,layers,[],[path],[dict(bond_id=0,node_index=2)])
    assert not p['proposed_pairs']
    assert p['candidates'][2]['status']=='related_bond_has_unparsed_paths'


def test_conflicting_glyph_and_bond_owners_are_not_resolved_by_nearest():
    geo,layers,path=bond_fixture()
    p=g.propose(geo,layers,[glyph(atom=1,bbox=(95,105,105,115))],[path],[])
    assert not p['proposed_pairs']
    assert p['candidates'][2]['status']=='owner_conflict_or_ambiguous_tip'


def test_overlapping_different_glyph_owners_remain_ambiguous():
    geo,layers=example()
    p=g.propose(geo,layers,[glyph(0),glyph(1)],[],[])
    assert not p['proposed_pairs']
    assert p['candidates'][3]['evidence']['possible_owners']==[0,1]


def test_multiple_glyphs_same_owner_are_one_pair():
    geo,layers=example()
    p=g.propose(geo,layers,[glyph(),glyph()],[],[])
    assert len(p['proposed_pairs'])==1


def test_multitarget_candidate_protects_both_groups_from_pairing():
    geo,layers=example(points=((104,100),(118,100)),targets=((100,100),(108,100)),raw=(0,-1))
    p=g.propose(geo,layers,[glyph(bbox=(116,95,120,105))],[],[])
    assert p['candidates'][0]['status']=='ambiguous_node_membership'
    assert all(n['status']=='ambiguous_membership' for n in p['nodes'])
    assert not p['proposed_pairs']


@pytest.mark.parametrize('points,raw,status', [(((103,100),(113,100)),(-1,-1),'no_raw_representative'),
                                             (((113,100),),(-1,),'no_candidate')])
def test_missing_raw_representative_not_invented(points,raw,status):
    geo,layers=example(points=points,targets=((100,100),),raw=raw)
    p=g.propose(geo,layers,[glyph()],[],[])
    assert p['nodes'][0]['status']==status
    assert not p['proposed_pairs']


def test_endpoint_tie_is_unknown_and_normal_line_tip_not_new_negative_class():
    geo,layers,path=bond_fixture()
    assert g.endpoint_owner((150,110),layers['bonds'][0],6) is None
    path['drawing_type']='SINGLE'
    assert not g.propose(geo,layers,[],[path],[])['proposed_pairs']


def test_noncontiguous_target_identity_rejected():
    geo,layers=example()
    layers['atoms'][1]['atom_id']=5
    with pytest.raises(ValueError,match='identity'):
        g.propose(geo,layers,[],[],[])


def test_raw_positive_outside_radius_fails_closed():
    geo,layers=example(raw=(0,1,-1,0))
    with pytest.raises(ValueError,match='Raw supervision'):
        g.propose(geo,layers,[],[],[])


@pytest.mark.parametrize('mutation', ['overlap','diagnostic','validation_split','duplicate'])
def test_training_scope_cannot_import_validation(mutation):
    row=dict(sample_key='train1',split='train',fit_domain='witness')
    manifest=dict(training_samples=[row],validation_only_samples=[dict(sample_key='dev1')])
    assert g.training_rows(manifest)==[row]
    if mutation=='overlap': row['sample_key']='dev1'
    elif mutation=='diagnostic': row['purpose']='diagnostic'
    elif mutation=='validation_split': row['split']='validation'
    else: manifest['training_samples'].append(deepcopy(row))
    with pytest.raises(ValueError,match='training/replay'):
        g.training_rows(manifest)


def test_counterfactual_preserves_true_text_and_replacements_without_mutation():
    geo,layers=example()
    p=g.propose(geo,layers,[glyph(),glyph(bbox=(99,99,105,105))],[],[])
    rec=record(geo,layers,{1,2,3})
    before=deepcopy((geo,p,rec,layers))
    check=g.removal_check(geo,p,rec,layers)
    assert check['removed_fragment_slots']==[3]
    assert check['original_valid_replacement_slots']==[2]
    assert check['original_glyph_matched_slots']==[2]
    assert not check['removed_correct_glyph_slots'] and not check['removed_valid_replacement_slots']
    assert check['original_matcher_nonregression']
    assert (geo,p,rec,layers)==before


def test_missing_and_duplicate_are_separate_constraints():
    geo,layers=example()
    p=g.propose(geo,layers,[glyph()],[],[])
    check=g.removal_check(geo,p,record(geo,layers,{0,2}),layers)
    assert [c['state'] for c in check['node_constraints']]==['duplicate','missing']


def test_tampered_baseline_and_exclusion_cannot_pass():
    geo,layers=example()
    p=g.propose(geo,layers,[glyph()],[],[])
    rec=record(geo,layers,{0,1,3})
    wrong=deepcopy(rec)
    wrong['metrics']['unmatched_selected']+=1
    with pytest.raises(ValueError,match='baseline differs'):
        g.removal_check(geo,p,wrong,layers)
    p['proposed_pairs'].append(dict(fragment_slot=0))
    with pytest.raises(ValueError,match='real/alternative'):
        g.removal_check(geo,p,rec,layers)


def test_unchanged_hungarian_can_lose_a_target_after_removing_an_outside_point():
    geo,layers=example(points=((104,100),(118,100),(91,100)),targets=((100,100),(110,100)),raw=(0,-1,-1))
    relations=g.memberships(geo)
    # Deliberately broad handcrafted proposal demonstrates why the check must
    # report matcher regression even when no spatially eligible point is removed.
    p=dict(nodes=[],proposed_pairs=[dict(fragment_slot=1)],candidates=[dict(slot=c['slot'],
        raw_target=c['raw_target'],memberships=relations[c['slot']],evidence=dict(glyph_hits=[])) for c in geo['candidates']])
    check=g.removal_check(geo,p,record(geo,layers,{0,1,2}),layers)
    assert check['spatial_eligibility_preserved']
    assert not check['original_matcher_nonregression']
    assert check['lost_target_ids']==[0]


def test_review_is_not_source_image_edit(tmp_path):
    from PIL import Image
    geo,layers=example()
    p=g.propose(geo,layers,[glyph()],[],[])
    p.update(sample_key=review.FOCUS[0],canonical_smiles='C&N',group='witness',phase_checks={})
    source=tmp_path/'source.png'
    Image.new('RGB',(768,768),'white').save(source)
    before=source.read_bytes()
    name=review.overlay(tmp_path,p,source)
    review.page(tmp_path,[p])
    assert source.read_bytes()==before and (tmp_path/name).exists()
    assert 'C&amp;N' in (tmp_path/'index.html').read_text()
