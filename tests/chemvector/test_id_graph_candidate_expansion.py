from copy import deepcopy
from itertools import combinations
import inspect

import pytest

from chemvector.experiments.id_graph_prediction.adapter import from_prediction
from chemvector.experiments.id_graph_candidates.decode import Budget, decode, compiled
from chemvector.experiments.id_graph_candidate_expansion.decode import ExpansionBudget, apply_edit, element_actions, extend
from chemvector.experiments.id_graph_candidate_expansion.evaluate import heavy_inventory, score, summarize


def fixture(labels=('N', None, None, None, None), edges=None):
    if edges is None:
        edges = {(0, i) for i in range(1, len(labels))}
    raw = {'sample_key': 'fixture',
           'atoms': [{'id': i, 'xy': [i * 40., 0.], 'score': .8} for i in range(len(labels))],
           'pairs': [{'id': i, 'atom_ids': list(p), 'presence_logit': 1. if p in edges else -2.,
                      'chemical_logits': [2., 0., 0., 0.]} for i, p in enumerate(combinations(range(len(labels)), 2))],
           'ocr': [{'id': i, 'text': t, 'score': .99, 'bbox': [0., 0., 1., 1.]} for i, t in enumerate(labels) if t is not None]}
    return raw, from_prediction(raw, 0.)[0]


def evidence(texts=('N', 'C', 'O', 'N+')):
    return {0: {'object_key': 'fixture:p000', 'crop_sha256': 'a' * 64, 'model_id': 'ctc',
                'raw_greedy_text': 'N', 'text_role_score': .99, 'error': None,
                'candidates': [{'rank': i, 'text': t, 'score': -float(i), 'score_kind': 'ctc_log_probability'} for i, t in enumerate(texts)]}}


def test_evidence_backed_element_substitution_makes_new_candidate_without_adoption():
    raw, graph = fixture()
    ev = evidence()
    old = decode(raw, graph, ev, 0., Budget(0, 0, 0))
    before = deepcopy((raw, graph, ev, old))
    result = extend(raw, graph, ev, old, 0., ExpansionBudget(64, 0))
    assert any(a['action']['values']['element'] == 'C' and a['valid'] for a in result['extension_attempts'])
    assert not result['automatic_replacement']
    assert (raw, graph, ev, old) == before
    assert len(result['groups']) == 1
    candidate = result['groups'][0]['representative_graph']
    assert candidate['bonds'] == graph['bonds']
    assert [a['id'] for a in candidate['atoms']] == [a['id'] for a in graph['atoms']]


def test_node_deletion_keeps_remaining_ids_and_removes_all_incident_edges():
    _, graph = fixture()
    result = apply_edit(graph, {'kind': 'DELETE_NODE', 'atom_id': 0})
    assert [a['id'] for a in result['atoms']] == [1, 2, 3, 4]
    assert result['bonds'] == []
    assert result['observations'] == graph['observations']
    assert any(p['atom_id'] == 0 for p in result['observations']['atom_positions'])
    assert all(a['stereo']['neighbor_ids'] == [] for a in result['atoms'])


def test_deleting_leaf_does_not_retype_surviving_edges():
    _, graph = fixture()
    result = apply_edit(graph, {'kind': 'DELETE_NODE', 'atom_id': 4})
    assert result['bonds'] == [b for b in graph['bonds'] if 4 not in b['atom_ids']]
    assert 4 not in result['atoms'][0]['stereo']['neighbor_ids']
    assert compiled(result)['valid']


def test_no_empty_molecule_or_unknown_node_deletion():
    _, graph = fixture(('C',), set())
    for atom_id in (0, 99):
        with pytest.raises(ValueError):
            apply_edit(graph, {'kind': 'DELETE_NODE', 'atom_id': atom_id})


@pytest.mark.parametrize('labels,edges', [(('C',), set()), (('Na+', 'Cl-'), set()), (('C', 'N'), {(0, 1)})])
def test_valid_graphs_are_not_edited(labels, edges):
    raw, graph = fixture(labels, edges)
    old = decode(raw, graph, {}, 0.)
    result = extend(raw, graph, {}, old, 0.)
    assert result['baseline']['valid'] and result['extension_attempts'] == []
    assert result['groups'] == []


def test_old_candidates_and_provenance_are_not_lost():
    raw, graph = fixture()
    ev = evidence()
    old = decode(raw, graph, ev, 0.)
    result = extend(raw, graph, ev, old, 0.)
    new = {g['stereo']: g for g in result['groups']}
    assert all(g['stereo'] in new and all(a in new[g['stereo']]['actions'] for a in g['actions']) for g in old['groups'])


def test_equivalent_node_deletions_merge_without_voting():
    raw, graph = fixture()
    old = decode(raw, graph, {}, 0., Budget(0, 0, 0))
    result = extend(raw, graph, {}, old, 0., ExpansionBudget(0, 128))
    assert len(result['groups']) == 2
    assert sorted(len(g['actions']) for g in result['groups']) == [1, 4]


def test_caps_are_explicit_and_do_not_expand_after_failure():
    raw, graph = fixture()
    ev = evidence()
    old = decode(raw, graph, ev, 0., Budget(0, 0, 0))
    result = extend(raw, graph, ev, old, 0., ExpansionBudget(1, 1))
    assert len(result['extension_attempts']) == 2
    assert result['element_truncated'] == 1 and result['deletions_truncated'] == 4


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_bad_budget_rejected(value):
    with pytest.raises(ValueError):
        ExpansionBudget(value, 0)


def test_element_candidates_do_not_change_isotopes_radicals_or_hydrogen_node_status():
    _, graph = fixture()
    actions, _ = element_actions(graph, evidence(('N', 'H', '15C', 'C.', '*', 'C', 'N+')), 64)
    assert [a['values']['element'] for a in actions] == ['C']
    assert actions[0]['evidence']['text'] == 'C'


def test_without_text_evidence_no_element_alternative_is_invented():
    _, graph = fixture()
    assert element_actions(graph, {}, 64) == ([], 0)


def test_mismatched_frozen_baseline_rejected():
    raw, graph = fixture()
    old = decode(raw, graph, {}, 0.)
    old['baseline']['valid'] = True
    with pytest.raises(ValueError, match='replay'):
        extend(raw, graph, {}, old, 0.)


def test_duplicate_coordinates_survive_element_and_node_actions():
    raw, _ = fixture()
    for a in raw['atoms']:
        a['xy'] = [0., 0.]
    graph = from_prediction(raw, 0.)[0]
    old = decode(raw, graph, {}, 0., Budget(0, 0, 0))
    result = extend(raw, graph, {}, old, 0.)
    assert all(len(g['representative_graph']['atoms']) == 4 for g in result['groups'])
    assert all(g['representative_graph']['observations'] == graph['observations'] for g in result['groups'])


def test_arms_are_union_not_sequential_edits_and_coverage_is_not_accuracy():
    raw, graph = fixture()
    ev = evidence()
    old = decode(raw, graph, ev, 0., Budget(0, 0, 0))
    result = extend(raw, graph, ev, old, 0.)
    correct = next(g for g in result['groups'] if any(a['kind'] == 'ELEMENT' for a in g['actions']))
    scores = {r['arm']: r for r in score(result, {'core': correct['core']})}
    assert not scores['v1']['covered_core'] and scores['v1_element']['covered_core']
    assert not scores['v1_node_delete']['covered_core'] and scores['union']['covered_core']
    assert all(not r['operational_core_exact'] and not r['automatic_replacement'] for r in scores.values())
    assert all('action' in a and 'actions' not in a for a in result['extension_attempts'])


def test_heavy_inventory_keeps_isotopes_but_not_charge():
    _, graph = fixture(('C', 'N'), {(0, 1)})
    changed = deepcopy(graph)
    changed['atoms'][1]['formal_charge'] = 1
    assert heavy_inventory(changed) == heavy_inventory(graph)
    changed['atoms'][0]['isotope'] = 13
    assert heavy_inventory(changed) != heavy_inventory(graph)


def test_generation_has_no_truth_interface():
    assert set(inspect.signature(extend).parameters) == {'raw', 'graph', 'evidence', 'previous', 'threshold', 'budget'}
    import chemvector.experiments.id_graph_candidate_expansion.decode as module
    assert 'geometry_matches' not in inspect.getsource(module)
    assert 'TRUTH' not in inspect.getsource(module)


def test_incomplete_denominator_rejected():
    with pytest.raises(ValueError):
        summarize([], [])


def test_existing_output_rejected(tmp_path):
    from chemvector.experiments.id_graph_candidate_expansion.__main__ import freeze
    with pytest.raises(FileExistsError):
        freeze(tmp_path)
