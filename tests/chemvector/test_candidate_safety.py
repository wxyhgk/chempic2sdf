from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from chemvector.experiments.candidate_safety import (
    intervention_decision, score_first_candidates, score_prediction, structural_prefilter,
)
from chemvector.experiments.selector_recheck import review_requests


def trace(slots, edges):
    return dict(compiled_candidate_indices=slots, scene_before_fallback=dict(
        atom_objects=[dict(id=i, anchor=[s, s]) for i, s in enumerate(slots)],
        bond_objects=[dict(id=i, atom_refs=[slots.index(a), slots.index(b)], chemical_type=t)
                      for i, (a, b, t) in enumerate(edges)]))


def test_no_orphan_prefilter_never_requests_even_with_rejected_points():
    edges = [(0, 1, 'SINGLE'), (1, 2, 'SINGLE'), (2, 3, 'SINGLE')]
    assert structural_prefilter(trace([0, 1, 2, 3], edges))['status'] == 'no_orphan'
    predictions = [dict(slot=i, xy=[70.*i, 100.], keep_logit=1., selected=i < 4) for i in range(6)]
    assert review_requests(predictions, [[a, b] for a, b, _ in edges])['requests'] == []


def test_insufficient_reference_is_coverage_limit_not_proven_safety():
    gate = structural_prefilter(trace([0, 1, 2], [(0, 1, 'SINGLE')]))
    assert gate['status'] == 'insufficient_predicted_length_reference'
    assert gate['orphan_slots'] == [2]


def test_orphan_with_reference_requires_raw_not_quantized_guesses():
    gate = structural_prefilter(trace([0, 1, 2, 3, 4], [(1, 2, 'SINGLE'), (2, 3, 'SINGLE'), (3, 4, 'SINGLE')]))
    assert gate['status'] == 'needs_raw_replay' and gate['orphan_slots'] == [0]


def request():
    return dict(action='review_only', graph_edits=0, status='requests_ready', selected_slots=[0, 1],
                requests=[dict(first_review_slot=2, candidates=[dict(slot=2, keep_logit=-1.)],
                               orphan_slot=0, partner_slot=1)])


def test_one_candidate_only_and_input_unchanged():
    q = request(); before = deepcopy(q)
    assert intervention_decision(q)['slot'] == 2
    assert before == q


@pytest.mark.parametrize('kind', ['multiple', 'tie', 'budget', 'none'])
def test_ambiguous_requests_abstain_without_selecting_after_scoring(kind):
    q = request()
    if kind == 'multiple': q['requests'] *= 2
    if kind == 'tie': q['requests'][0]['first_review_slot'] = None
    if kind == 'budget': q.update(status='review_budget_exceeded', requests=[])
    if kind == 'none': q.update(status='no_geometric_request', requests=[])
    assert intervention_decision(q)['slot'] is None


def fixtures():
    atoms = dict(atom_mask=torch.ones(1, 3, dtype=torch.bool), atom_indices=torch.arange(3)[None],
        atom_geometry=torch.tensor([[[0., 0.], [.1, 0.], [.2, 0.]]]),
        atom_scores=torch.ones(1, 3), atom_features=torch.ones(1, 3, 2))
    targets = dict(atom_mask=torch.tensor([[True, True, False]]),
        atom_anchors=atoms['atom_geometry'].clone(), bond_mask=torch.tensor([[True]]),
        bond_atom_refs=torch.tensor([[[0, 1]]]), bond_chemical=torch.tensor([[1]]))
    record = dict(pairs=[[0, 1], [0, 2], [1, 2]], presence_logits=[4., 4., -4.],
        chemical_logits=[[3., 0., 0., 0.]] * 3, prediction=dict(molblock=None, fallback=False),
        trace=trace([0, 1, 2], [(0, 1, 'DOUBLE'), (0, 2, 'SINGLE')]))
    return atoms, targets, record


def test_unknown_not_false_and_raw_not_scene_type():
    atoms, targets, record = fixtures()
    r = score_prediction(record, atoms, targets, SimpleNamespace(image_size=768, atom_match_radius_px=6), 'CC')
    assert r['unknown_present'] == 1 and r['confirmed_false_positives'] == 0
    assert r['correct_raw_typed_bonds'] == []
    assert r['scene_counts'] == {'correct_typed': 1, 'unknown_correspondence': 1}


def test_no_compiled_scene_is_unknown_not_zero_wrong_edges():
    atoms, targets, record = fixtures(); record['trace'] = {}
    r = score_prediction(record, atoms, targets, SimpleNamespace(image_size=768, atom_match_radius_px=6), 'CC')
    assert r['scene_edges'] is None and not r['sdf_valid']


def test_first_candidate_position_does_not_claim_unmatched_is_false():
    atoms, targets, _ = fixtures()
    q = request()
    r = score_first_candidates(q, atoms, targets, dict(matched_targets=[0, 1]))
    assert r[0]['category'] == 'outside_target_radius'
    atoms['atom_geometry'][0, 2] = torch.tensor([.1, 0.])
    r = score_first_candidates(q, atoms, targets, dict(matched_targets=[0, 1]))
    assert r[0]['category'] == 'near_already_matched_target'
    r = score_first_candidates(q, atoms, targets, dict(matched_targets=[0]))
    assert r[0]['category'] == 'near_unmatched_target'
