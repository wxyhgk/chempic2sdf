"""Synthetic additive loss attribution, no real model/data or optimizer."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from chemvector.predicted_incidence import associate_predicted_proposals
from chemvector.proposal_object_set import _focal_presence_loss, _set_ranking_loss
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.gradient_preflight import state_digest
from tests.chemvector.test_curriculum_continuation import tiny
from tools import diagnose_curriculum_v02_selector_loss_terms as audit


def scalar_case(values=(-.6, -.8, -.09), labels=(True, True, False)):
    parameter = torch.nn.Parameter(torch.zeros(2))
    jacobian = torch.tensor([[1., .1], [1., 0.], [2., .1]])
    logits = (torch.tensor(values) + jacobian @ parameter).unsqueeze(0)
    positive = torch.tensor([labels])
    mask = torch.ones_like(positive)
    zero = logits.sum() * 0
    losses = {name: zero for name in audit.WEIGHTS}
    losses.update(atom_presence_loss=_focal_presence_loss(logits, positive, mask),
                  atom_ranking_loss=_set_ranking_loss(logits, positive, mask),
                  atom_count_loss=F.smooth_l1_loss(logits.sigmoid().sum(-1), positive.sum(-1).float()))
    losses['loss'] = sum(audit.WEIGHTS[k] * v for k, v in losses.items())
    return parameter, jacobian, logits, positive, mask, losses


@pytest.mark.parametrize('values,labels', [((-.6, -.8, -.09), (True, True, False)),
    ((-.8, -.8, -.09), (True, True, False)), ((2., 2., -2.), (True, True, False)),
    ((0., 0., 0.), (True, True, True)), ((0., 0., 0.), (False, False, False))])
def test_sources_reconstruct_shared_gradient_including_ranking_ties(values, labels):
    p, jacobian, z, positive, mask, losses = scalar_case(values, labels)
    before = p.detach().clone()
    weighted, gradients, sources, full, error = audit.split_gradient_sources(losses, z, positive, mask, (p,))
    assert error < 1e-3 and len(gradients) == 10
    assert torch.allclose(sum(weighted.values()), losses['loss'])
    assert p.grad is None and torch.equal(p, before)
    for name, value in sources.items():
        expected = jacobian.double().T @ value[0].double()
        assert torch.allclose(gradients[name], expected, atol=1e-6)
    for row in jacobian:
        result = audit.contributions(row.double(), gradients, full)
        assert result['sum_components'] == pytest.approx(result['total'], abs=1e-5)


def test_ranking_can_improve_margin_while_lowering_both_correct_endpoints():
    p, jacobian, z, positive, mask, losses = scalar_case()
    _, gradients, _, _, _ = audit.split_gradient_sources(losses, z, positive, mask, (p,))
    ranking = gradients['atom_ranking_positive'] + gradients['atom_ranking_negative']
    for i in (0, 1):
        assert -float(jacobian[i].double() @ ranking) < 0
        assert -float((jacobian[i] - jacobian[2]).double() @ ranking) > 0


def test_common_normalizer_not_separately_normalized_terms():
    gradients = {'one': torch.tensor([3., 0.]), 'two': torch.tensor([-1., 4.])}
    full = sum(gradients.values())
    result = audit.contributions(torch.tensor([1., 0.]), gradients, full)
    assert result['total'] == pytest.approx(-2 / 20**.5)
    assert result['components']['one'] == pytest.approx(-3 / 20**.5)
    assert result['components']['two'] == pytest.approx(1 / 20**.5)
    with pytest.raises(ValueError, match='nonzero'):
        audit.contributions(torch.ones(2), {}, torch.zeros(2))


def test_wrong_loss_and_mask_fail_closed():
    p, _, z, positive, mask, losses = scalar_case()
    losses['loss'] = losses['loss'] + 1
    with pytest.raises(ValueError, match='reconstruct original'):
        audit.split_gradient_sources(losses, z, positive, mask, (p,))
    with pytest.raises(ValueError, match='Malformed'):
        audit.split_gradient_sources(losses, z, positive, mask.float(), (p,))


def test_fixed_selection_and_hard_negative_tie_break_are_saved_only():
    samples, saved, labels = [], {}, {}
    for i, smiles in enumerate(audit.WITNESSES):
        key = f'key{i}'
        samples.append(dict(canonical_smiles=smiles, sample_key=key))
        labels[key] = dict(target_details=[dict(raw_assigned_candidate=3)])
        saved[key] = dict(candidates=dict(candidates=[dict(candidate=3, selection_logit=5.),
            dict(candidate=2, selection_logit=1.), dict(candidate=1, selection_logit=1.)]))
    snapshot = deepcopy((samples, saved, labels))
    result = audit.select_witnesses({'samples': list(reversed(samples))}, saved, labels)
    assert [r['canonical_smiles'] for r in result] == list(audit.WITNESSES)
    assert all(r['hardest_negative'] == 1 and r['hardest_negative_ties'] == [1, 2] for r in result)
    assert (samples, saved, labels) == snapshot
    with pytest.raises(ValueError, match='Missing fixed'):
        audit.select_witnesses({'samples': samples[1:]}, saved, labels)


def test_actual_tiny_selector_decomposition_keeps_state_and_additivity(tiny):
    model, config, batch = tiny
    model.eval().requires_grad_(False)
    model.set_head.requires_grad_(True)
    before = state_digest(model)
    with torch.no_grad():
        front = structure_forward(model, batch['pixel_values'], config, -.25)
    raw, bonds = front[3:5]
    output = audit.previous.selector_call(model.set_head, raw, bonds)
    labels = associate_predicted_proposals(raw, bonds, batch, config)
    positives = labels['atom_presence_targets'][0].nonzero().flatten().tolist()
    negative = (~labels['atom_presence_targets'][0]).nonzero().flatten().tolist()[0]
    witness = dict(sample_key='synthetic', canonical_smiles='CC', positive_slots=positives,
                   hardest_negative=negative, probe_slots=sorted(positives + [negative]))
    result = audit.decompose(output, raw, bonds, labels, tuple(model.set_head.parameters()), witness,
                            {'candidates': []}, {'atom_threshold': -.25, 'bond_threshold': -.5})
    assert result['gradient_sum_relative_error'] < 1e-3 and state_digest(model) == before
    assert all(p.grad is None for p in model.parameters())
    by_id = {row['candidate']: row for row in result['rows']}
    for margin in result['margins']:
        pos, neg = by_id[margin['positive']]['retention'], by_id[margin['negative']]['retention']
        assert margin['attribution']['total'] == pytest.approx(pos['total'] - neg['total'], abs=1e-5)
    assert torch.equal(output['atom_presence_logits'].detach(), front[5]['atom_presence_logits'])


def test_budgeted_cpu_directory_guards(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU-only'):
        audit.audit(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='Fresh budgeted'):
        audit.audit(tmp_path / 'missing')
    (tmp_path / 'report.json').write_text('{}')
    with pytest.raises(ValueError, match='Fresh budgeted'):
        audit.audit(tmp_path)
