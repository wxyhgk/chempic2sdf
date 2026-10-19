from copy import deepcopy
from itertools import combinations

import pytest
from rdkit import Chem

from chemvector.experiments.id_graph_prediction.adapter import from_prediction, assert_connectivity_preserved
from chemvector.experiments.id_graph_prediction.scoring import geometry_matches, component_errors, compile_score
from chemvector.experiments.id_graph.roundtrip import signature


def prediction(ids=(7, 20), xy=((100., 200.), (120., 200.)), text='N'):
    return {'sample_key': 'test', 'atoms': [{'id': i, 'xy': list(p), 'score': .8} for i, p in zip(ids, xy)],
            'pairs': [{'id': j, 'atom_ids': list(pair), 'presence_logit': 1., 'chemical_logits': [2., 0., 0., 0.]}
                      for j, pair in enumerate(combinations(ids, 2))],
            'ocr': [{'id': ids[-1], 'text': text, 'score': .9, 'bbox': [0, 0, 1, 1]}] if ids else []}


def expected(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {'core': signature(mol, False), 'stereo': signature(mol)}


def test_preserves_sparse_ids_bonds_coordinates_and_inputs():
    raw = prediction()
    before = deepcopy(raw)
    graph, stats = from_prediction(raw, .5)
    assert_connectivity_preserved(raw, graph, .5)
    assert [a['id'] for a in graph['atoms']] == [7, 20]
    assert graph['bonds'][0]['atom_ids'] == [7, 20]
    assert graph['atoms'][1]['element'] == 'N'
    assert stats['default_carbon_atoms'] == 1 and stats['parsed_text'] == 1
    row, sdf = compile_score(graph, expected('CN'))
    assert row['core_graph_exact'] and row['stereo_graph_exact'] and sdf
    assert raw == before


def test_overlapping_coordinates_do_not_merge_atoms_or_remove_bond():
    raw = prediction(xy=((100., 200.), (100., 200.)))
    graph, _ = from_prediction(raw, 0.)
    assert_connectivity_preserved(raw, graph, 0.)
    assert len(graph['atoms']) == 2 and len(graph['bonds']) == 1
    assert compile_score(graph, expected('CN'))[0]['core_graph_exact']


def test_threshold_boundary_and_class_argmax_tie_match_torch_convention():
    raw = prediction()
    raw['pairs'][0]['chemical_logits'] = [0., 2., 2., 0.]
    graph, _ = from_prediction(raw, 1.)
    assert graph['bonds'][0]['type'] == 'DOUBLE'
    assert_connectivity_preserved(raw, graph, 1.)
    assert not from_prediction(raw, 1.001)[0]['bonds']


@pytest.mark.parametrize('text,status', [('Me', 'unparsed_text_carbon_default'), ('', 'no_text_carbon_default')])
def test_explicit_carbon_defaults_not_hidden(text, status):
    graph, stats = from_prediction(prediction(text=text), 0.)
    assert stats['default_carbon_atoms'] == 2
    assert graph['observations']['chemical_provenance'][1]['source'] == status
    assert graph['observations']['text_predictions'][0]['text'] == text


@pytest.mark.parametrize('text,smiles', [('NH2', 'CN'), ('[N]', 'CC'), ('O-', 'C[O-]'), ('13CH3', 'C[13CH3]')])
def test_frozen_ocr_interpretation(text, smiles):
    graph, _ = from_prediction(prediction(text=text), 0.)
    assert compile_score(graph, expected(smiles))[0]['core_graph_exact']


@pytest.mark.parametrize('case', ['duplicate_atom', 'duplicate_pair', 'dangling', 'self', 'missing_pair',
                                 'bad_class_count', 'nan_score', 'nan_logit', 'bool_id', 'outside_frame', 'orphan_ocr'])
def test_invalid_raw_records_fail_closed(case):
    raw = prediction()
    if case == 'duplicate_atom':
        raw['atoms'].append(deepcopy(raw['atoms'][0]))
    elif case == 'duplicate_pair':
        raw['pairs'].append({**raw['pairs'][0], 'id': 99})
    elif case == 'dangling':
        raw['pairs'][0]['atom_ids'][0] = 999
    elif case == 'self':
        raw['pairs'][0]['atom_ids'] = [7, 7]
    elif case == 'missing_pair':
        raw['pairs'] = []
    elif case == 'bad_class_count':
        raw['pairs'][0]['chemical_logits'] = [1., 0.]
    elif case == 'nan_score':
        raw['atoms'][0]['score'] = float('nan')
    elif case == 'nan_logit':
        raw['pairs'][0]['presence_logit'] = float('nan')
    elif case == 'bool_id':
        raw['atoms'][0]['id'] = True
    elif case == 'outside_frame':
        raw['atoms'][0]['xy'] = [-1., 0.]
    elif case == 'orphan_ocr':
        raw['ocr'][0]['id'] = 999
    with pytest.raises(ValueError):
        from_prediction(raw, 0.)


def test_adapter_guard_detects_changed_bond_type():
    raw = prediction()
    graph, _ = from_prediction(raw, 0.)
    graph['bonds'][0]['type'] = 'DOUBLE'
    with pytest.raises(ValueError, match='bonds'):
        assert_connectivity_preserved(raw, graph, 0.)


def test_raw_order_does_not_change_graph():
    raw = prediction(ids=(2, 7, 20), xy=((100., 0.), (200., 0.), (300., 0.)))
    graph, _ = from_prediction(raw, 0.)
    raw['atoms'].reverse()
    raw['pairs'].reverse()
    assert from_prediction(raw, 0.)[0] == graph


def test_missing_and_extra_atoms_and_bonds_are_not_rescued():
    graph, _ = from_prediction(prediction(text='C'), 2.)
    truth, _ = from_prediction(prediction(), 0.)
    errors = component_errors(graph, truth)
    assert errors['missing_bonds'] == 1 and errors['attribute_errors']['element'] == 1
    assert errors['unmatched_truth_atoms'] == 0
    single, _ = from_prediction(prediction(ids=(7,), xy=((100., 200.),), text=''), 0.)
    assert component_errors(single, truth)['unmatched_truth_atoms'] == 1
    assert component_errors(truth, single)['unmatched_predicted_atoms'] == 1


def test_geometry_ambiguity_excludes_attribute_bond_claims_but_not_node_counts():
    graph, _ = from_prediction(prediction(xy=((100., 200.), (100., 200.))), 0.)
    mapping, safe, ambiguity = geometry_matches(graph, graph)
    assert len(mapping) == 2 and safe == {}
    assert ambiguity['predicted_ambiguous'] == 2 and ambiguity['truth_ambiguous'] == 2
    errors = component_errors(graph, graph)
    assert errors['unambiguous_matched_atoms'] == 0
    assert errors['excluded_truth_bonds'] == 1 and errors['excluded_predicted_bonds'] == 1
    assert errors['unmatched_truth_atoms'] == 0


def test_geometry_uses_unquantized_coordinates_and_ignores_labels():
    graph, _ = from_prediction(prediction(text='C'), 0.)
    truth, _ = from_prediction(prediction(ids=(50, 80), xy=((120., 200.), (100., 200.))), 0.)
    assert geometry_matches(graph, truth)[0] == {7: 80, 20: 50}
    graph['observations']['atom_positions'][0]['xy'][0] += 6.01
    assert len(geometry_matches(graph, truth)[0]) == 1


def test_empty_prediction_retained_and_counts_full_truth_missing():
    graph, _ = from_prediction(prediction(ids=(), xy=()), 0.)
    truth, _ = from_prediction(prediction(), 0.)
    assert component_errors(graph, truth)['unmatched_truth_atoms'] == 2
    row, sdf = compile_score(graph, expected('CN'))
    assert not row['sdf_valid'] and not row['core_graph_exact'] and sdf is None
    assert not row['carbon_fallback']


def test_invalid_valence_is_reported_without_chemical_fallback():
    raw = prediction(ids=tuple(range(6)), xy=tuple((float(i * 30), 0.) for i in range(6)), text='')
    for pair in raw['pairs']:
        pair['presence_logit'] = 1. if 0 in pair['atom_ids'] else -1.
    graph, _ = from_prediction(raw, 0.)
    before = deepcopy(graph)
    row, sdf = compile_score(graph, expected('C'))
    assert not row['sdf_valid'] and row['compile_error'] and sdf is None
    assert not row['carbon_fallback'] and graph == before


def test_output_directory_cannot_be_reused(tmp_path):
    from chemvector.experiments.id_graph_prediction.runtime import freeze

    with pytest.raises(FileExistsError):
        freeze(tmp_path)
