"""Synthetic rank contracts only; no files, real models or optimizers."""

import pytest
import torch

from tools.selector_singleton_rank import SingleNodeContrast as Contrast, singleton_rank_contract as evaluate


def test_inverted_scores_raise_valid_and_lower_fragments_with_zero_common_shift():
    z=torch.tensor([-3.,2.,1.,7.],dtype=torch.float64,requires_grad=True)
    r=evaluate(z,[Contrast('node',(0,),(1,2))])
    g,=torch.autograd.grad(r['loss'],z)
    assert g[0]<0 and bool((g[1:3]>0).all()) and g[3]==0
    assert float(g.sum())==pytest.approx(0.,abs=1e-15)
    assert not r['training_authorized'] and not r['shared_parameter_direction_checked']


def test_absolute_coverage_is_not_guaranteed_or_hidden_by_ranking():
    groups=[Contrast('node',(0,),(1,))]
    a=evaluate(torch.tensor([-3.,-5.],dtype=torch.float64),groups)['loss']
    b=evaluate(torch.tensor([2.,0.],dtype=torch.float64),groups)['loss']
    assert torch.equal(a,b)  # Common shift changes -0.25 selection, not rank loss.
    assert float(a)>0 and not evaluate(torch.tensor([-3.,-5.]),groups)['threshold_coverage_guaranteed']


def test_nodes_are_equally_weighted_despite_fragment_count():
    z=torch.zeros(5,dtype=torch.float64,requires_grad=True)
    r=evaluate(z,[Contrast('a',(0,),(1,)),Contrast('b',(2,),(3,4))])
    g,=torch.autograd.grad(r['loss'],z)
    torch.testing.assert_close(g,torch.tensor([-.25,.25,-.25,.125,.125],dtype=torch.float64))


def test_gradcheck_and_no_input_mutation():
    z=torch.tensor([.3,-.2,1.],dtype=torch.float64,requires_grad=True)
    before=z.detach().clone();groups=[Contrast('node',(0,),(1,2))]
    assert torch.autograd.gradcheck(lambda x:evaluate(x,groups)['loss'],(z,))
    assert torch.equal(z.detach(),before)


@pytest.mark.parametrize('values',[[-1000.,1000.],[1000.,-1000.],[0.,0.]])
def test_extreme_and_tied_scores_are_finite(values):
    z=torch.tensor(values,dtype=torch.float64,requires_grad=True)
    loss=evaluate(z,[Contrast('node',(0,),(1,))])['loss']
    gradient,=torch.autograd.grad(loss,z)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert gradient[0]<=0 and gradient[1]>=0


@pytest.mark.parametrize('groups',[
    [],[Contrast('node',(0,1),(2,))],[Contrast('node',(),(1,))],
    [Contrast('node',(0,),())],[Contrast('node',(0,),(0,))],
    [Contrast('node',(0,),(1,1))],[Contrast('node',(0,),(99,))],
    [Contrast('same',(0,),(1,)),Contrast('same',(2,),(3,))],
    [Contrast('a',(0,),(1,)),Contrast('b',(1,),(2,))],
    [Contrast('',(0,),(1,))],[Contrast('node',(False,),(1,))]])
def test_invalid_or_multicandidate_groups_are_rejected(groups):
    with pytest.raises(ValueError): evaluate(torch.zeros(4),groups)


@pytest.mark.parametrize('z',[torch.tensor([float('nan'),0.]),torch.tensor([float('inf'),0.]),
                              torch.zeros(1,2),torch.tensor([0,1])])
def test_invalid_logits_rejected(z):
    with pytest.raises(ValueError): evaluate(z,[Contrast('node',(0,),(1,))])


def test_slot_and_node_order_equivariance():
    z=torch.tensor([.1,.8,-.5,.3],dtype=torch.float64,requires_grad=True)
    groups=[Contrast('a',(0,),(1,)),Contrast('b',(2,),(3,))]
    a=evaluate(z,groups)['loss'];ga,=torch.autograd.grad(a,z)
    perm=torch.tensor([2,0,3,1]);reordered=z.detach()[perm].requires_grad_()
    b=evaluate(reordered,[Contrast('b',(0,),(2,)),Contrast('a',(1,),(3,))])['loss']
    gb,=torch.autograd.grad(b,reordered)
    torch.testing.assert_close(a,b);torch.testing.assert_close(ga[perm],gb)
