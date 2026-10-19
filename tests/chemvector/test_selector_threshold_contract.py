"""Synthetic contracts only; no real images, checkpoints, or optimizers."""

from copy import deepcopy

import pytest
import torch

from tools.selector_singleton_rank import SingleNodeContrast as Contrast
from tools.selector_threshold_contract import (
    THRESHOLD, affine_gradient, feasible_ray, review_contrast, threshold_contract,
)


def test_absolute_derivatives_and_unreferenced_scores():
    z = torch.tensor([-3., 2., -1., 8.], dtype=torch.float64, requires_grad=True)
    r = threshold_contract(z, [Contrast('a', (0,), (1, 2))])
    g, = torch.autograd.grad(r['loss'], z)
    expected = torch.tensor([-.5 * torch.sigmoid(THRESHOLD-z[0]).item(),
        .25 * torch.sigmoid(z[1]-THRESHOLD).item(),
        .25 * torch.sigmoid(z[2]-THRESHOLD).item(), 0.], dtype=z.dtype)
    torch.testing.assert_close(g, expected)
    assert not r['training_authorized'] and not r['graph_training_objective_ready']


def test_common_shift_is_not_invisible_to_threshold_loss():
    group = [Contrast('a', (0,), (1,))]
    low = threshold_contract(torch.tensor([-3., -5.]), group)
    high = threshold_contract(torch.tensor([2., 0.]), group)
    assert not torch.equal(low['loss'], high['loss'])
    assert not low['decisions'][0]['valid_selected']
    assert high['decisions'][0]['valid_selected']
    assert high['decisions'][0]['selected_fragments'] == 1
    assert not any(r['decisions'][0]['both_decisions_correct'] for r in (low, high))


def test_threshold_equality_is_positive_selected_but_negative_failure():
    r = threshold_contract(torch.full((2,), THRESHOLD), [Contrast('a', (0,), (1,))])
    assert r['decisions'][0] == dict(node_id='a', valid_selected=True,
        selected_fragments=1, both_decisions_correct=False)


def test_equal_node_and_equal_class_weight_despite_fragment_count():
    z = torch.full((5,), THRESHOLD, dtype=torch.float64, requires_grad=True)
    loss = threshold_contract(z, [Contrast('a', (0,), (1,)), Contrast('b', (2,), (3, 4))])['loss']
    g, = torch.autograd.grad(loss, z)
    torch.testing.assert_close(g, torch.tensor([-.125, .125, -.125, .0625, .0625], dtype=z.dtype))


def test_identical_features_make_shared_direction_zero_despite_right_logit_signs():
    z = torch.full((2,), THRESHOLD, dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(threshold_contract(z, [Contrast('a', (0,), (1,))])['loss'], z)
    assert g[0] < 0 < g[1]
    parameter_gradient = affine_gradient(torch.ones(2, 3), g.detach())
    assert torch.count_nonzero(parameter_gradient) == 0
    assert not feasible_ray([
        dict(id='valid', logit=THRESHOLD, slope=1., selected=True),
        dict(id='fragment', logit=THRESHOLD, slope=1., selected=False),
    ])['feasible']


def test_chain_rule_matches_real_linear_autograd_with_frozen_pre_score():
    torch.manual_seed(11)
    hidden = torch.randn(4, 3, dtype=torch.float64)
    weight = torch.randn(3, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    z = hidden @ weight + bias + torch.tensor([1., -2., 3., -4.])
    loss = threshold_contract(z, [Contrast('a', (0,), (1, 2))])['loss']
    dz, dw, db = torch.autograd.grad(loss, (z, weight, bias))
    torch.testing.assert_close(affine_gradient(hidden, dz.detach()), torch.cat((dw, db[None])))
    assert weight.grad is None and bias.grad is None


def test_gradcheck_extreme_scores_and_permutation():
    group = [Contrast('a', (0,), (1, 2))]
    z = torch.tensor([.3, -.7, 1.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: threshold_contract(x, group)['loss'], (z,))
    before = z.detach().clone()
    expected = threshold_contract(z, group)['loss']
    actual = threshold_contract(z[[2, 0, 1]], [Contrast('a', (1,), (2, 0))])['loss']
    torch.testing.assert_close(actual, expected)
    assert torch.equal(before, z)
    extreme = torch.tensor([-1000., 1000., -1000.], requires_grad=True)
    loss = threshold_contract(extreme, group)['loss']
    assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss, extreme)[0]).all()


@pytest.mark.parametrize('groups', [[], [Contrast('a', (0, 1), (2,))],
    [Contrast('a', (0,), ())], [Contrast('a', (0,), (0,))],
    [Contrast('a', (0,), (1, 1))], [Contrast('a', (0,), (3,))],
    [Contrast('a', (False,), (1,))], [Contrast('', (0,), (1,))],
    [Contrast('a', (0,), (1,)), Contrast('a', (2,), (1,))],
    [Contrast('a', (0,), (1,)), Contrast('b', (2,), (1,))]])
def test_bad_or_multivalid_groups_rejected(groups):
    with pytest.raises(ValueError):
        threshold_contract(torch.zeros(3), groups)


@pytest.mark.parametrize('z', [torch.tensor([float('nan'), 0.]), torch.tensor([float('inf'), 0.]),
                              torch.ones(1, 2), torch.ones(2, dtype=torch.int64)])
def test_bad_scores_rejected(z):
    with pytest.raises(ValueError):
        threshold_contract(z, [Contrast('a', (0,), (1,))])


def test_ray_old_point_protection_can_make_target_repair_infeasible():
    constraints = [dict(id='valid', logit=-2., slope=1., selected=True),
                   dict(id='bad', logit=1., slope=-1., selected=False)]
    assert feasible_ray(constraints)['feasible']
    result = feasible_ray(constraints + [dict(id='old', logit=0., slope=-1., selected=True)])
    assert not result['feasible'] and result['lower'] == 1.75 and result['upper'] == .25
    assert result['upper_bound_ids'] == ['old']


def test_ray_strict_boundary_and_stationary_bad_point():
    lower = dict(id='lower', logit=-1.25, slope=1., selected=True)
    upper = dict(id='upper', logit=.75, slope=-1., selected=True)
    r = feasible_ray([lower, upper])
    assert r['feasible'] and r['lower'] == r['upper'] == 1
    upper = dict(id='upper', logit=-1.25, slope=1., selected=False)
    r = feasible_ray([lower, upper])
    assert not r['feasible'] and r['upper_exclusive']
    r = feasible_ray([dict(id='bad', logit=THRESHOLD, slope=0., selected=False)])
    assert not r['feasible'] and r['impossible_constant_ids'] == ['bad']
    assert feasible_ray([dict(id='ok', logit=THRESHOLD, slope=0., selected=True)])['feasible']


@pytest.mark.parametrize('rows', [[], [dict(id='a', logit=0., slope=float('nan'), selected=True)],
    [dict(id='a', logit=0., slope=1., selected=1)],
    [dict(id='a', logit=0., slope=1., selected=True)] * 2])
def test_invalid_ray_rejected(rows):
    with pytest.raises(ValueError): feasible_ray(rows)


def evidence():
    node = dict(target_id=0, valid_slots=[0], stable_fragment_slots=[1])
    original = dict(target_id=0, status='reviewable', possible_slots=[0], raw_positive_slots=[0],
                    alternative_slots=[], visible_text=True)
    pair = dict(fragment_slot=1, owner_target=0, kinds=['glyph_fragment'],
                training_authorized=False, requires_review=True)
    review = dict(fragment_slot=1, owner_target=0, coordinate_stable=True, review_reasons=[],
        unquantized_members=[], source_xy=[400., 383.5], training_authorized=False, requires_review=True,
        source_evidence=dict(possible_owners=[0], kinds=['glyph_fragment'], nearby_tips=[],
            glyph_hits=[dict(atom_id=0, bbox=[395., 380., 405., 390.])]))
    subset = dict(training_authorized=False, requires_review=True,
        stable_proposal=dict(nodes=[original], proposed_pairs=[pair]),
        excluded_pairs=[], stable_pair_reviews=[review])
    geometry = dict(image_size=768, match_radius_px=6, candidates=[
        dict(slot=0, raw_target=0, geometry=[.5, .5]),
        dict(slot=1, raw_target=-1, geometry=[400/767, .5])])
    radius = dict(training_authorized=False, raw=dict(radius_reference=dict(
        eligible_edges=[dict(slot=0, target_id=0)])))
    return node, subset, geometry, radius


def test_consistent_evidence_is_still_not_training_or_human_review():
    r = review_contrast(*evidence())
    assert r['diagnostic_evidence_consistent'] and not r['training_authorized']
    assert not r['human_semantic_review_performed'] and not r['label_release_authorized']


@pytest.mark.parametrize('problem', ['held', 'member', 'wrong_owner', 'new_label', 'sensitive', 'projection', 'no_glyph', 'multi'])
def test_review_rejects_conflicting_evidence(problem):
    node, subset, geometry, radius = deepcopy(evidence())
    review = subset['stable_pair_reviews'][0]
    if problem == 'held': subset['excluded_pairs'] = [dict(fragment_slot=1)]
    if problem == 'member': radius['raw']['radius_reference']['eligible_edges'].append(dict(slot=1, target_id=1))
    if problem == 'wrong_owner': review['source_evidence']['possible_owners'] = [1]
    if problem == 'new_label': subset['training_authorized'] = True
    if problem == 'sensitive': review['unquantized_members'] = [0]
    if problem == 'projection': review['source_xy'][0] += 1
    if problem == 'no_glyph': review['source_evidence']['glyph_hits'] = []
    if problem == 'multi': node['valid_slots'] = [0, 2]
    with pytest.raises(ValueError): review_contrast(node, subset, geometry, radius)
