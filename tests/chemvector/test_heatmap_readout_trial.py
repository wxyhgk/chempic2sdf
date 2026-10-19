"""Synthetic trial guards and cache invalidation; never GPU or real data."""

from copy import deepcopy
from itertools import islice

import pytest
import torch
from torch import nn

from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.training.curriculum import heatmap_readout_stage as stage
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_selector_localization_contract import manual_output


def model_fixture():
    model = nn.Module()
    model.atom = nn.Module()
    model.atom.heatmap = nn.Conv2d(128, 1, 1)
    model.other = nn.Linear(1, 1)
    stage.configure(model)
    cache = dict(purpose='training',training_authorized=True,manifest_sha256='bound',
        atom_static=dict(object_features=torch.ones(1,128,3,3)),target_heatmap=torch.zeros(1,1,3,3))
    cache['target_heatmap'][0,0,1,1]=1
    return model,cache


def test_only_readout_updates_and_reservation_precedes_step():
    model,cache=model_fixture()
    frozen=stage.frozen_digest(model)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.SGD(stage.parameters(model),lr=.001)
    reservations=[]
    def reserve(step):
        assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
        reservations.append(step)
    result=stage.update(model,cache,optimizer,StepLimit(1),'cpu',reserve,'bound')
    assert result['loss'] > 0 and reservations == [1]
    assert stage.frozen_digest(model)==frozen
    changed={k for k,v in model.state_dict().items() if not torch.equal(initial[k],v)}
    assert changed == stage.PARAMETER_NAMES
    assert cache['atom_static']['object_features'].grad is None


@pytest.mark.parametrize('change',['scope','lr','optimizer','diagnostic','manifest','feature_grad','target_grad','nan','budget'])
def test_fail_before_update(change):
    model,cache=model_fixture()
    optimizer=torch.optim.SGD(stage.parameters(model),lr=.001)
    limit=StepLimit(1)
    if change=='scope': model.other.requires_grad_(True)
    elif change=='lr': optimizer.param_groups[0]['lr']=.002
    elif change=='optimizer': optimizer=torch.optim.AdamW(stage.parameters(model),lr=.001)
    elif change=='diagnostic': cache['purpose']='diagnostic'
    elif change=='manifest': cache['manifest_sha256']='wrong'
    elif change=='feature_grad': cache['atom_static']['object_features'].requires_grad_()
    elif change=='target_grad': cache['target_heatmap'].requires_grad_()
    elif change=='nan': cache['target_heatmap'][0,0,0,0]=float('nan')
    else: limit.consume()
    before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises((ValueError,RuntimeError)):
        stage.update(model,cache,optimizer,limit,'cpu',lambda step: None,'bound')
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls=0
    def forward(self,features,geometry,scores,mask,*bonds):
        self.calls+=1
        return dict(atom_presence_logits=scores)


def test_redecode_and_selector_refresh_when_heatmap_changes():
    model,_=model_fixture()
    model.set_head=Selector()
    model.eval()
    output,_=manual_output(torch.full((8,8),.5))
    output={k:v.detach() for k,v in output.items()}
    output['object_features']=torch.zeros(1,128,8,8)
    output.pop('heatmap_logits')
    cache=dict(atom_static=output,bonds={k:torch.zeros(1) for k in
        ('bond_features','bond_endpoints','bond_types','bond_scores','bond_mask')})
    with torch.no_grad(): model.atom.heatmap.bias.fill_(0)
    before,_,_=stage.refresh_proposals(model,cache,PredictedProposalConfig(maximum_atoms=4),'cpu')
    with torch.no_grad(): model.atom.heatmap.bias.fill_(-20)
    after,_,_=stage.refresh_proposals(model,cache,PredictedProposalConfig(maximum_atoms=4),'cpu')
    assert int(before['atom_mask'].sum())==4 and not after['atom_mask'].any()
    assert model.set_head.calls==2 and 'heatmap_logits' not in cache['atom_static']


def record(matched,missing,*,exact=False,extra=0):
    coverage=dict(matched_target_ids=list(matched),reachable_bond_ids=[0] if 0 in matched else [],
                  no_eligible_candidate_target_ids=list(missing),unmatched_selected=extra)
    return dict(sample_key='synthetic',raw={p:deepcopy(coverage) for p in ('original','radius_reference')},
        selected={p:deepcopy(coverage) for p in ('original','radius_reference')},
        prediction=dict(exact=exact),correct_typed_bond_ids=[0] if 0 in matched else [],false_or_wrong_typed_pairs=0)


def test_zero_recovery_early_stop_is_at200_not100():
    baseline=[record([0],[1,2,3])]
    assert not stage.gate(baseline,baseline,{'synthetic':[1,2,3]},100)['stop_reasons']
    assert stage.gate(baseline,baseline,{'synthetic':[1,2,3]},200)['stop_reasons']==['no_actual_raw_recovery_by_step200']


def test_net_gain_does_not_hide_loss_of_specific_baseline_node_or_graph():
    baseline=[record([0],[1,2,3],exact=True)]
    result=stage.gate(baseline,[record([1,2,3],[0])],{'synthetic':[1,2,3]},100)
    assert result['recovered_count']==3 and result['stop_reasons']
    assert not result['local_pass'] and result['lost_baseline_relations']


@pytest.mark.parametrize('extra,false_pairs,passed',[(0,0,True),(1,0,False),(0,1,False)])
def test_three_recoveries_need_no_extra_false_atoms_or_bonds(extra,false_pairs,passed):
    baseline=[record([0],[1,2,3])]
    after=record([0,1,2,3],[],extra=extra)
    after['false_or_wrong_typed_pairs']=false_pairs
    result=stage.gate(baseline,[after],{'synthetic':[1,2,3]},400)
    assert result['local_pass']==passed
    assert not result['deployment_allowed'] and not result['generalization_established']


def test_order_is_fixed_balanced_and_refuses_validation():
    caches=[dict(key=str(i),group='old_full_molecule' if i<32 else 'witness',
        purpose='training',training_authorized=True,manifest_sha256='bound') for i in range(58)]
    one=list(islice(stage.order(caches,'bound'),100))
    two=list(islice(stage.order(caches,'bound'),100))
    assert one==two and len({c['key'] for c in one})==58
    assert [c['group'] for c in one[:3]]==['old_full_molecule','old_full_molecule','witness']
    caches[0]['purpose']='diagnostic'
    with pytest.raises(ValueError): next(stage.order(caches,'bound'))
