"""Synthetic saved-graph/logit tests; no corpus or model checkpoint."""

from copy import deepcopy

import pytest
from rdkit import Chem
import torch

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss
from tools import pair_regression_diagnostics as diagnostic


def record():
    molecule = Chem.MolFromSmiles('CCC')
    conformer = Chem.Conformer(3)
    for i, x in enumerate((1., 3., 5.)):
        conformer.SetAtomPosition(i, (x, -1., 0.))
    molecule.AddConformer(conformer)
    return dict(metrics=dict(key='train:synthetic', candidate_label_sha256='a' * 64,
        text_sha256='b' * 64, target_bonds=2, reachable_positive_pairs=2,
        correct_typed_bond_ids=[0, 1], sdf_exact=True, sdf_valid=True),
        pairs=[[4, 7], [4, 9], [7, 9]], presence_logits=[4., -4., 4.],
        chemical_logits=[[4., 0., 0., 0.]] * 3,
        prediction=dict(molblock=Chem.MolToMolBlock(molecule), fallback=False),
        trace=dict(roles=[], ocr=[], compiled_candidate_indices=[4, 7, 9],
            scene_before_fallback=dict(atom_objects=[dict(id=i, anchor=[100 + i * 200, 100]) for i in range(3)],
                bond_objects=[dict(atom_refs=[0, 1], chemical_type='SINGLE'),
                              dict(atom_refs=[1, 2], chemical_type='SINGLE')])) )


def event():
    return dict(key='train:synthetic', step=1, attempt=0, updated=True,
        sampled_row_ids=[0, 2, 1], sampled_rows=3, positive_rows=2, sampled_unknown_pairs=0,
        losses={'presence_loss': .125})


def test_raw_type_change_cannot_be_called_a_deleted_final_edge():
    before, after = record(), record()
    after['chemical_logits'][0] = [0., 0., 0., 4.]
    after['metrics']['correct_typed_bond_ids'] = [1]
    result = diagnostic.compare_decoding(before, after)
    assert result['raw_typed_bond_ids_lost'] == [0]
    assert result['final_structure_equal'] and result['compiled_edge_maps_equal']
    assert result['raw_presence_removed'] == 0
    changed = result['changed_pairs'][0]
    assert changed['raw_type_after'] == 'AROMATIC' and changed['compiled_type_after'] == 'SINGLE'
    assert result['original_acceptance_changed'] is False


def test_two_failed_compilations_are_not_the_same_valid_molecule():
    before, after = record(), record()
    for r in (before, after):
        r['prediction']['molblock'] = None
        r['metrics'].update(sdf_exact=False, sdf_valid=False)
    result = diagnostic.compare_decoding(before, after)
    assert not result['final_structure_equal'] and not result['compiled_edge_maps_equal']
    assert result['final_graphs_available'] == [False, False]


def test_compiled_only_bond_changes_are_not_hidden_by_unchanged_raw_outputs():
    before, after = record(), record()
    molecule = Chem.MolFromSmiles('C=CC')
    old = Chem.MolFromMolBlock(before['prediction']['molblock'], removeHs=False)
    molecule.AddConformer(Chem.Conformer(old.GetConformer()))
    after['prediction']['molblock'] = Chem.MolToMolBlock(molecule)
    after['metrics']['sdf_exact'] = False
    result = diagnostic.compare_decoding(before, after)
    assert result['raw_chemical_changes_on_present_pairs'] == 0
    assert not result['final_structure_equal'] and not result['compiled_edge_maps_equal']
    assert len(result['changed_pairs']) == 1
    changed = result['changed_pairs'][0]
    assert changed['raw_type_before'] == changed['raw_type_after'] == 'SINGLE'
    assert changed['scene_type_before'] == changed['scene_type_after'] == 'SINGLE'
    assert changed['compiled_type_before'] == 'SINGLE' and changed['compiled_type_after'] == 'DOUBLE'


def test_real_new_invalid_and_fallback_remain_visible():
    before, after = record(), record()
    after['prediction']['molblock'] = None
    after['prediction']['fallback'] = True
    after['metrics'].update(sdf_exact=False, sdf_valid=False)
    after['presence_logits'][1] = 5.
    result = diagnostic.compare_decoding(before, after)
    assert result['raw_presence_added'] == 1 and result['new_invalid_sdf'] and result['new_fallback']
    assert result['baseline_exact_broken']


@pytest.mark.parametrize('change', ['key', 'signature', 'text', 'lineage', 'pairs', 'nan', 'shape'])
def test_noncomparable_saved_inputs_rejected(change):
    before, after = record(), record()
    if change == 'key': after['metrics']['key'] = 'other'
    elif change == 'signature': after['metrics']['candidate_label_sha256'] = 'c' * 64
    elif change == 'text': after['trace']['ocr'] = ['changed']
    elif change == 'lineage': after['trace']['compiled_candidate_indices'].reverse()
    elif change == 'pairs': after['pairs'].reverse()
    elif change == 'nan': after['presence_logits'][0] = float('nan')
    else: after['chemical_logits'].pop()
    with pytest.raises(ValueError): diagnostic.compare_decoding(before, after)


def test_quantized_lineage_must_be_exact():
    item = record()
    item['trace']['scene_before_fallback']['atom_objects'][0]['anchor'][0] += 1
    with pytest.raises(ValueError): diagnostic.compiled_edges(item)


def test_direct_logit_replay_matches_existing_presence_loss_without_chemical_labels():
    before, after = record(), record()
    after['presence_logits'] = [5., 3., 5.]
    result = diagnostic.presence_replay(before, after, event())
    labels = torch.tensor([True, True, False])
    selected = event()['sampled_row_ids']
    expected = direct_atom_pair_loss(dict(presence_logits=torch.tensor(before['presence_logits'])[selected],
        chemical_logits=torch.zeros(3, 4)), labels, torch.zeros(3, dtype=torch.long))['presence_loss']
    assert result['arms']['initial_logits']['presence_loss'] == float(expected)
    negative0 = result['arms']['initial_logits']['per_sampled_row'][-1]
    negative1 = result['arms']['final_logits']['per_sampled_row'][-1]
    assert negative0['label'] is negative1['label'] is False
    assert negative1['probability'] > negative0['probability']
    assert negative1['direct_logit_gradient'] > negative0['direct_logit_gradient'] > 0
    assert result['historical_parameter_gradients_recovered'] is False


def test_unsampled_pairs_are_not_recovered_as_negative_labels():
    e = event(); e.update(sampled_row_ids=[0, 2], sampled_rows=2)
    labels = diagnostic.recovered_sample_labels(e, 3)
    assert labels == {0: True, 2: True} and 1 not in labels
    result = diagnostic.presence_replay(record(), record(), e)
    assert all(arm['unsampled_gradient_zero'] for arm in result['arms'].values())


@pytest.mark.parametrize('change', ['duplicate', 'out_of_range', 'count', 'unknown', 'bool', 'wrong_sample'])
def test_bad_sampling_receipts_rejected(change):
    e = event()
    if change == 'duplicate': e['sampled_row_ids'] = [0, 0, 1]
    elif change == 'out_of_range': e['sampled_row_ids'] = [0, 1, 8]
    elif change == 'count': e['positive_rows'] = 4
    elif change == 'unknown': e['sampled_unknown_pairs'] = 1
    elif change == 'bool': e['positive_rows'] = True
    else: e['key'] = 'wrong'
    with pytest.raises(ValueError): diagnostic.presence_replay(record(), record(), e)


def test_replay_never_mutates_records_or_rng():
    before, after, e = record(), record(), event()
    snapshots = deepcopy((before, after, e)); rng = torch.get_rng_state().clone()
    diagnostic.presence_replay(before, after, e)
    assert (before, after, e) == snapshots and torch.equal(rng, torch.get_rng_state())
