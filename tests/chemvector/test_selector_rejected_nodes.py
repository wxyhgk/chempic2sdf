"""Synthetic score/ownership joins; no real artifacts or forward passes."""

from copy import deepcopy

import pytest

from tools import selector_rejected_nodes as audit


@pytest.mark.parametrize('valid,bad,kind,separable',[
    (-.7,.2,'fragment_ties_or_outranks_valid',False),
    (-.7,-.7,'fragment_ties_or_outranks_valid',False),
    (-.7,-.9,'valid_over_fragment',True),
    (-.7,None,'no_stable_fragment_context',True)])
def test_absolute_rejection_and_wrong_order_are_distinct(valid,bad,kind,separable):
    scores={0:valid}
    if bad is not None: scores[1]=bad
    row=audit.score_state([0],[] if bad is None else [1],scores)
    assert row['node_state']=='missing' and row['comparison']==kind
    assert row['valid_fragment_threshold_separable']==separable
    assert row['best_valid_minus_threshold']==pytest.approx(-.45)


def test_spatial_alternative_not_forced_below_the_raw_representative():
    state=audit.score_state([0,1],[2],{0:-1.,1:.5,2:.1})
    assert state['best_valid_slots']==[1] and state['selected_valid_slots']==[1]
    assert not state['fixed_threshold_contract_pass']  # Bad fragment still present.
    assert state['valid_minus_fragment_margin']==pytest.approx(.4)


def test_valid_ties_do_not_invent_an_exactly_one_threshold():
    state=audit.score_state([0,1],[2],{0:.5,1:.5,2:-1.})
    assert state['valid_fragment_threshold_separable']
    assert not state['exactly_one_threshold_separable']
    assert state['node_state']=='duplicate' and state['best_valid_slots']==[0,1]


def test_selection_threshold_is_inclusive():
    state=audit.score_state([0],[1],{0:-.25,1:-.25})
    assert state['node_state']=='single' and state['selected_fragment_slots']==[1]
    assert state['exactly_one_threshold_interval']==dict(lower_exclusive=-.25,upper_inclusive=-.25)


@pytest.mark.parametrize('valid,fragments,scores',[
    ([],[],{0:0.}),([0,0],[],{0:0.}),([0],[0],{0:0.}),([0],[1,1],{0:0.,1:1.}),
    ([0],[],{0:float('nan')}),([0],[],{0:float('inf')}),([0],[1],{0:0.}),([False],[],{False:0.})])
def test_score_contract_rejects_ambiguous_or_nonfinite_inputs(valid,fragments,scores):
    with pytest.raises(ValueError): audit.score_state(valid,fragments,scores)


def fixture():
    geometry=dict(target_anchors=[[0.,0.],[.5,.5],[1.,1.]], candidates=[dict(slot=i) for i in range(6)])
    nodes=[dict(target_id=i,element='C',visible_text=False,label_text='',possible_slots=slots,
        raw_positive_slots=slots[:1],alternative_slots=slots[1:],status='reviewable' if slots else 'no_candidate')
        for i,slots in enumerate(([0],[1,3],[]))]
    subset=dict(training_authorized=False,requires_review=True,
        stable_proposal=dict(nodes=nodes,proposed_pairs=[dict(fragment_slot=2,owner_target=0,
            training_authorized=False,requires_review=True,kinds=['glyph_fragment'])]),
        excluded_pairs=[dict(fragment_slot=4,owner_target=0)])
    records={p:dict(metrics=dict(logits={'0':-.7,'1':.8,'2':.6,'3':-.9,'4':.9,'5':.1})) for p in audit.PHASES}
    radius=dict(training_authorized=False,raw=dict(radius_reference=dict(
        eligible_edges=[dict(slot=s,target_id=t) for s,t in ((0,0),(1,1),(3,1))],
        no_eligible_candidate_target_ids=[2])),phases={p:dict(selected_slots=[1,2,4,5],
        original=dict(before=dict(missing_target_ids=[0,1,2])),
        radius_reference=dict(before=dict(missing_target_ids=[0,2],no_eligible_candidate_target_ids=[0,2])))
        for p in audit.PHASES})
    return geometry,subset,radius,records


def test_focus_excludes_raw_missing_and_original_matcher_artifact():
    args=fixture();before=deepcopy(args)
    result=audit.inspect_case(*args)
    assert args==before
    assert result['focused_target_ids']==[0] and result['raw_missing_target_ids']==[2]
    assert result['original_matcher_only_missing_ids']==[1]
    assert result['nodes'][0]['stable_fragment_slots']==[2]
    assert result['nodes'][0]['held_sensitive_fragment_slots']==[4]
    assert result['nodes'][1]['protected_alternative_slots']==[3]
    assert not result['training_authorized'] and not result['model_predictions_changed']


@pytest.mark.parametrize('change',['training','phase','scores','selected','membership','fragment_positive','duplicate_fragment','sensitive','overlap'])
def test_joins_fail_closed(change):
    geo,subset,radius,records=fixture()
    if change=='training': subset['training_authorized']=True
    elif change=='phase': records.pop('control-200')
    elif change=='scores': records['initial']['metrics']['logits'].pop('5')
    elif change=='selected': radius['phases']['initial']['selected_slots'].remove(5)
    elif change=='membership': subset['stable_proposal']['nodes'][0]['possible_slots']=[1]
    elif change=='fragment_positive': subset['stable_proposal']['proposed_pairs'][0]['fragment_slot']=0
    elif change=='duplicate_fragment': subset['stable_proposal']['proposed_pairs']*=2
    elif change=='sensitive': subset['excluded_pairs'][0]['fragment_slot']=2
    else: radius['raw']['radius_reference']['eligible_edges'].append(dict(slot=1,target_id=0))
    with pytest.raises(ValueError): audit.inspect_case(geo,subset,radius,records)
