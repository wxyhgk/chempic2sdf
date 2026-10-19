"""Synthetic frozen-feature chain rules; no real checkpoint/data or fitting."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.vertex import focal_heatmap_loss
from tests.chemvector.test_selector_dense_observation import TinyAtom, TinyVision, fixture
from tools import selector_heatmap_direction as direction


def small():
    generator = torch.Generator().manual_seed(731)
    features = torch.randn(1, 3, 8, 8, generator=generator)
    weight = torch.randn(1, 3, 1, 1, generator=generator)
    bias = torch.tensor([-1.])
    logits = F.conv2d(features, weight, bias)
    anchors = torch.tensor([[[.25, .25], [.75, .75]]])
    return features, logits, anchors, weight, bias


def test_projected_derivative_equals_parameter_autograd_without_mutation():
    hidden, logits, anchors, weight, bias = small()
    before = [v.clone() for v in (hidden, logits, anchors, weight, bias)]
    probe = direction.parameter_gradient(hidden, logits, anchors)
    w, b = weight.clone().requires_grad_(), bias.clone().requires_grad_()
    gw, gb = torch.autograd.grad(focal_heatmap_loss(F.conv2d(hidden, w, b), probe['heatmap']), (w, b))
    expected = torch.cat((gw.flatten(), gb))
    torch.testing.assert_close(probe['gradient'].float(), expected, rtol=2e-6, atol=1e-6)
    assert probe['positive_cells'] == 2
    for original, copy in zip((hidden, logits, anchors, weight, bias), before):
        assert torch.equal(original, copy) and original.grad is None


def test_descent_field_matches_linear_readout_difference_and_loss_derivative():
    hidden, logits, anchors, weight, bias = small()
    probe = direction.parameter_gradient(hidden, logits, anchors)
    gradient = probe['gradient']
    field = direction.descent_field(hidden, gradient)
    epsilon = .0001
    # Synthetic tensors only, no model or optimizer update.
    h, w, b = hidden.double(), weight.double(), bias.double()
    z0 = F.conv2d(h, w, b)
    z1 = F.conv2d(h, w-epsilon*gradient[:-1].reshape_as(w), b-epsilon*gradient[-1:])
    torch.testing.assert_close((z1-z0)[0, 0]/epsilon, field, rtol=1e-9, atol=1e-9)
    predicted = float((probe['logit_gradient'][0, 0].double()*field).sum())
    assert predicted == pytest.approx(-float(gradient.square().sum()), rel=1e-10)


def test_correct_independent_logit_signs_do_not_imply_shared_margin_progress():
    hidden = torch.ones(1, 1, 3, 3)
    logits = torch.zeros(1, 1, 3, 3)
    probe = direction.parameter_gradient(hidden, logits, torch.tensor([[[.5, .5]]]))
    assert probe['logit_gradient'][0, 0, 1, 1] < 0
    assert probe['logit_gradient'][0, 0, 1, 2] > 0
    field = direction.descent_field(hidden, probe['gradient'])
    assert direction.local_margin(logits[0, 0], field, 1, 1)['descent_margin_derivative'] == 0


def test_bias_only_shift_never_improves_peak_margin():
    hidden, logits, _, _, _ = small()
    field = direction.descent_field(hidden, torch.tensor([0., 0., 0., -2.], dtype=torch.float64))
    record = direction.local_margin(logits[0, 0], field, 3, 3)
    assert record['descent_logit_derivative'] == 2
    assert record['descent_margin_derivative'] == 0


def test_tied_neighbors_use_right_derivative_of_max_not_favorable_tie():
    logits = torch.zeros(3, 3)
    logits[1, 0] = logits[1, 2] = 2
    field = torch.zeros(3, 3, dtype=torch.float64)
    field[1, 0], field[1, 1], field[1, 2] = -4, 1, 3
    result = direction.local_margin(logits, field, 1, 1)
    assert result['neighbor_grid_ids'] == [3, 5]
    assert result['margin'] == -2 and result['descent_margin_derivative'] == -2


def test_all_atoms_and_actual_non_gt_candidate_peak_are_reported():
    hidden, logits, anchors, _, _ = small()
    probe = direction.parameter_gradient(hidden, logits, anchors)
    geo = dict(target_anchors=anchors[0].tolist(), candidates=[dict(slot=4, grid_id=7*8+7, raw_target=1)])
    result = direction.describe_image(hidden, logits, probe['heatmap'], geo, [0],
                                      probe['gradient'], probe['gradient'])
    assert len(result['atoms']) == 2
    assert [r['focused'] for r in result['atoms']] == [True, False]
    assert result['atoms'][1]['grid_id'] == 6*8+6
    assert result['original_raw_positive_peaks'][0]['grid_id'] == 7*8+7
    assert not result['candidate_recovery_measured'] and not result['parameters_changed']


def gate_fixture(slopes):
    cases = [dict(atoms=[]) for _ in range(7)]
    for i, slope in enumerate(slopes):
        cases[i % 7]['atoms'].append(dict(focused=True, descent_margin_derivative=slope,
                                         neighbor_positive_target_ids=[]))
    return cases


@pytest.mark.parametrize('slopes,expected', [([1.]*10, True), ([1.]*6+[-.1]*4, True),
    ([1.]*5+[-.1]*5, False), ([.1]*6+[-1.]*4, False), ([0.]*10, False)])
def test_fixed_gate_majority_and_total_both_required(slopes, expected):
    cases = gate_fixture(slopes)
    original = deepcopy(cases)
    result = direction.diagnostic_gate(cases)
    assert result['supports_bounded_trial_preparation'] == expected
    assert result['model_updated'] is False and cases == original


def test_positive_neighbor_does_not_receive_automatic_suppression_admission():
    cases = gate_fixture([1.]*10)
    cases[0]['atoms'][0]['neighbor_positive_target_ids'] = [7]
    assert direction.diagnostic_gate(cases)['status'] == 'NO_GO'


@pytest.mark.parametrize('change', ['features_nan', 'features_grad', 'dtype', 'logit_shape',
                                    'anchor_range', 'collision', 'empty', 'anchor_nan'])
def test_gradient_input_contract(change):
    hidden, logits, anchors, _, _ = small()
    if change == 'features_nan': hidden[0, 0, 0, 0] = float('nan')
    elif change == 'features_grad': hidden.requires_grad_()
    elif change == 'dtype': logits = logits.double()
    elif change == 'logit_shape': logits = logits[:, :, :7]
    elif change == 'anchor_range': anchors[0, 0, 0] = 2
    elif change == 'collision': anchors[0, 1] = anchors[0, 0]
    elif change == 'empty': anchors = anchors[:, :0]
    else: anchors[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        direction.parameter_gradient(hidden, logits, anchors)


def test_image_capture_adds_no_forward_and_removes_hook():
    snapshot, _, config, output = fixture()
    model = nn.Module()
    model.vision, model.atom = TinyVision(), TinyAtom(output)
    model.eval().requires_grad_(False)
    result, hidden = direction.image_only_features(model, torch.zeros(1, 3, 768, 768), config)
    assert model.atom.calls == 1 and not model.atom._forward_hooks
    assert all(torch.equal(v, result[k]) for k, v in snapshot.items())
    assert torch.equal(hidden, output['object_features']) and not hidden.requires_grad


def test_image_capture_removes_hook_even_on_invalid_input():
    _, _, config, output = fixture()
    model = nn.Module()
    model.vision, model.atom = TinyVision(), TinyAtom(output)
    model.eval().requires_grad_(False)
    with pytest.raises(ValueError):
        direction.image_only_features(model, torch.zeros(1, 3, 16, 16), config)
    assert model.atom.calls == 0 and not model.atom._forward_hooks


@pytest.mark.parametrize('cases', [gate_fixture([1.]*9), gate_fixture([float('nan')]*10), []])
def test_gate_rejects_changed_or_nonfinite_scope(cases):
    with pytest.raises(ValueError):
        direction.diagnostic_gate(cases)
