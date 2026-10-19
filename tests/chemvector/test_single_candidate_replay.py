from copy import deepcopy

import pytest
import torch

from chemvector.experiments.single_candidate_replay import admit_one, compare_scenes, first_review_candidate


def queue():
    return dict(action='review_only', graph_edits=0, status='requests_ready', selected_slots=[0, 2],
        requests=[dict(first_review_slot=1, candidates=[dict(slot=1, keep_logit=-1.), dict(slot=3, keep_logit=-2.)])])


def test_only_mask_cloned_and_one_bit_added():
    raw = dict(atom_mask=torch.tensor([[True, True, True, False]]))
    selected = dict(atom_mask=torch.tensor([[True, False, True, False]]), atom_features=torch.randn(1, 4, 8))
    original = selected['atom_mask'].clone()
    result = admit_one(raw, selected, queue())
    assert torch.equal(selected['atom_mask'], original)
    assert torch.equal(result['atom_mask'], torch.tensor([[True, True, True, False]]))
    assert result['atom_features'] is selected['atom_features']
    assert result['atom_mask'] is not selected['atom_mask']


@pytest.mark.parametrize('change', ['empty', 'multiple', 'tie', 'not_first', 'unsorted', 'selected', 'edited', 'nonfinite'])
def test_invalid_or_ambiguous_request_rejected(change):
    q = queue()
    if change == 'empty': q['requests'] = []
    if change == 'multiple': q['requests'] *= 2
    if change == 'tie': q['requests'][0]['candidates'][1]['keep_logit'] = -1.
    if change == 'not_first': q['requests'][0]['first_review_slot'] = 3
    if change == 'unsorted': q['requests'][0]['candidates'].reverse()
    if change == 'selected': q['selected_slots'].append(1)
    if change == 'edited': q['graph_edits'] = 1
    if change == 'nonfinite': q['requests'][0]['candidates'][0]['keep_logit'] = float('nan')
    with pytest.raises(ValueError): first_review_candidate(q)


@pytest.mark.parametrize('raw_mask,selected_mask', [
    ([True, False, True, False], [True, False, True, False]),
    ([True, True, True, True], [True, False, False, False]),
    ([False, True, True, True], [True, False, True, False]),
])
def test_stale_queue_or_inactive_candidate_rejected(raw_mask, selected_mask):
    with pytest.raises(ValueError):
        admit_one(dict(atom_mask=torch.tensor([raw_mask])), dict(atom_mask=torch.tensor([selected_mask])), queue())


def trace(slots, edges):
    return dict(compiled_candidate_indices=slots, scene_before_fallback=dict(
        atom_objects=[dict(id=i, role='IMPLICIT', anchor=[s, s], incident_ports=[]) for i, s in enumerate(slots)],
        bond_objects=[dict(id=i, atom_refs=[slots.index(a), slots.index(b)], chemical_type=t,
                           display_type=t, direction='NONE', stereo='STEREONONE')
                      for i, (a, b, t) in enumerate(edges)]))


def test_compiler_reindex_does_not_look_like_old_edge_change():
    before = trace([0, 2], [(0, 2, 'SINGLE')])
    after = trace([0, 1, 2], [(1, 2, 'TRIPLE'), (0, 2, 'SINGLE')])
    delta = compare_scenes(before, after)
    assert delta['added_slots'] == [1]
    assert delta['added_edges'][0]['pair'] == [1, 2]
    assert not any(delta[k] for k in ('removed_slots', 'changed_old_atoms', 'removed_edges', 'changed_old_edges'))


def test_changed_old_type_and_atom_reported():
    before = trace([0, 2], [(0, 2, 'SINGLE')])
    after = deepcopy(before)
    after['scene_before_fallback']['bond_objects'][0]['chemical_type'] = 'DOUBLE'
    after['scene_before_fallback']['atom_objects'][0]['role'] = 'TEXT'
    delta = compare_scenes(before, after)
    assert delta['changed_old_edges'] == [[0, 2]] and delta['changed_old_atoms'] == [0]


def test_disappearing_old_connection_reported():
    delta = compare_scenes(trace([0, 2], [(0, 2, 'SINGLE')]), trace([0, 1], []))
    assert delta['removed_slots'] == [2] and delta['removed_edges'] == [[0, 2]]


def test_pair_delta_uses_candidate_pair_not_row_number():
    from tools.audit_single_candidate_replay import logit_delta
    before = dict(pairs=[[0, 2]], presence_logits=[1.], chemical_logits=[[1., 2., 3., 4.]])
    after = dict(pairs=[[0, 1], [0, 2]], presence_logits=[2., 1.],
                 chemical_logits=[[0., 0., 0., 0.], [1., 2., 3., 4.]])
    assert logit_delta(before, after) == dict(presence_logits=0., chemical_logits=0.)


def test_scoring_separates_unknown_and_raw_vs_scene_types():
    from types import SimpleNamespace
    from tools.audit_single_candidate_replay import score
    geometry = torch.zeros(1, 3, 8)
    geometry[0, :, 0] = torch.tensor([0., .1, .2])
    atoms = dict(atom_mask=torch.ones(1, 3, dtype=torch.bool), atom_indices=torch.arange(3)[None],
                 atom_geometry=geometry, atom_features=torch.ones(1, 3, 2), atom_scores=torch.ones(1, 3))
    batch = dict(atom_mask=torch.tensor([[True, True, False]]),
        atom_anchors=geometry[..., :2].clone(), bond_mask=torch.tensor([[True]]),
        bond_atom_refs=torch.tensor([[[0, 1]]]), bond_chemical=torch.tensor([[1]]), bond_ids=[[0]])
    forward = (torch.ones(1, 1, 1, 1), {}, {}, atoms, {}, {}, atoms)
    structure = dict(atoms=atoms, pairs=torch.tensor([[0, 1], [0, 2], [1, 2]]),
        output=dict(presence_logits=torch.tensor([4., 4., -4.]),
                    chemical_logits=torch.tensor([[3., 0., 0., 0.]] * 3)), chemistry=torch.tensor([1, 0, 0]))
    record = trace([0, 1, 2], [(0, 1, 'DOUBLE'), (0, 2, 'SINGLE')])
    record.update(roles=[], ocr=[])
    result = score(forward, batch, SimpleNamespace(image_size=768, atom_match_radius_px=6),
        structure, dict(molblock=None, fallback=False), record, 'CC', 2)
    assert result['metrics']['unknown_present'] == 1
    assert result['metrics']['confirmed_false_positives'] == 0
    assert result['metrics']['correct_typed_bond_ids'] == []
    assert result['target_bonds'][0]['scene_typed_correct']
    assert all(not r['known'] for r in result['incident_pairs'])
