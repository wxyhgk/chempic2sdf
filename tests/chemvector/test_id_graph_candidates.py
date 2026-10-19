from copy import deepcopy
from itertools import combinations
import inspect
import json

import pytest
from rdkit import Chem

from chemvector.experiments.id_graph_candidates.decode import Budget, apply_action, attribute_actions, bond_actions, compiled, decode
from chemvector.experiments.id_graph_candidates.evidence import join_evidence, read_selected_beams
from chemvector.experiments.id_graph_candidates.evaluate import edge_agreement, score_candidates, summarize
from chemvector.experiments.id_graph_prediction.adapter import BOND_TYPES, from_prediction


def raw_graph(labels, edges, types=None):
    pairs = []
    for i, pair in enumerate(combinations(range(len(labels)), 2)):
        kind = (types or {}).get(pair, 'SINGLE')
        logits = [0.] * 4
        logits[BOND_TYPES.index(kind)] = 2.
        pairs.append({'id': i, 'atom_ids': list(pair), 'presence_logit': 1. + i / 100 if pair in edges else -2.,
                      'chemical_logits': logits})
    raw = {'sample_key': 'fixture', 'atoms': [{'id': i, 'xy': [float(i * 40), 0.], 'score': .8} for i in range(len(labels))],
           'pairs': pairs, 'ocr': [{'id': i, 'text': label, 'score': .99, 'bbox': [0., 0., 1., 1.]}
                                 for i, label in enumerate(labels) if label is not None]}
    return raw, from_prediction(raw, 0.)[0]


def evidence_for(atom_id, texts, raw='N'):
    return {'object_key': f'fixture:p{atom_id:03d}', 'crop_sha256': 'a' * 64, 'model_id': 'fixture-ctc',
            'raw_greedy_text': raw, 'text_role_score': .99, 'error': None,
            'candidates': [{'rank': i, 'text': text, 'score': -float(i), 'score_kind': 'ctc_log_probability'}
                           for i, text in enumerate(texts)]}


def quaternary():
    return raw_graph(['N', None, None, None, None], {(0, i) for i in range(1, 5)})


def pyrrole():
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    return raw_graph(['N', None, None, None, None], edges, {p: 'AROMATIC' for p in edges})


def test_charge_candidate_and_wrong_legal_edge_edits_coexist_without_autoadoption():
    raw, graph = quaternary()
    evidence = {0: evidence_for(0, ['N', 'N+'])}
    before = deepcopy((raw, graph, evidence))
    result = decode(raw, graph, evidence, 0.)
    assert not result['baseline']['valid']
    assert len(result['groups']) >= 2 and result['disposition'] == 'REVIEW_MULTIPLE_CHEMICAL_GRAPHS'
    assert not result['automatic_replacement']
    assert any('[N+]' in g['core'] for g in result['groups'])
    assert (raw, graph, evidence) == before
    for g in result['groups']:
        assert [a['id'] for a in g['representative_graph']['atoms']] == list(range(5))
        assert g['representative_graph']['observations'] == graph['observations']


def test_single_valid_candidate_is_still_unproven():
    raw, graph = quaternary()
    result = decode(raw, graph, {0: evidence_for(0, ['N', 'N+'])}, 0., Budget(64, 0, 0))
    assert len(result['groups']) == 1
    assert result['disposition'] == 'REVIEW_SINGLE_UNPROVEN_CANDIDATE'
    assert not result['automatic_replacement']


def test_hydrogen_and_negative_charge_are_not_merged():
    raw, graph = pyrrole()
    result = decode(raw, graph, {0: evidence_for(0, ['N', 'NH', 'N-'])}, 0., Budget(64, 0, 0))
    assert not result['baseline']['valid'] and len(result['groups']) == 2
    assert len({g['stereo'] for g in result['groups']}) == 2


def test_equivalent_edit_paths_merge_and_preserve_all_provenance():
    raw, graph = quaternary()
    result = decode(raw, graph, {}, 0., Budget(0, 4, 0))
    assert len(result['groups']) == 1
    assert len(result['groups'][0]['actions']) == 4
    assert all(a['kind'] == 'DELETE' for a in result['groups'][0]['actions'])


@pytest.mark.parametrize('labels,edges', [(['C', 'N'], {(0, 1)}), (['Na+', 'Cl-'], set()), (['C'], set())])
def test_valid_graphs_salts_and_isolated_atoms_are_kept(labels, edges):
    raw, graph = raw_graph(labels, edges)
    result = decode(raw, graph, {}, 0.)
    assert result['baseline']['valid'] and result['disposition'] == 'KEEP_VALID_BASELINE'
    assert result['attempts'] == result['groups'] == []


def test_empty_prediction_is_retained_and_abstained():
    raw, graph = raw_graph([], set())
    result = decode(raw, graph, {}, 0.)
    assert not result['baseline']['valid'] and result['disposition'] == 'ABSTAIN_NO_VALID_CANDIDATE'


def test_zero_budgets_abstain_and_count_omitted_attribute_edits():
    raw, graph = quaternary()
    result = decode(raw, graph, {0: evidence_for(0, ['N', 'N+'])}, 0., Budget(0, 0, 0))
    assert result['attribute_truncated'] == 1 and result['attempts'] == []


def test_attribute_proposals_do_not_change_element_isotope_or_radicals():
    _, graph = quaternary()
    actions, _ = attribute_actions(graph, {0: evidence_for(0, ['N', 'C', '15N', 'N.', 'N+', 'NH'])}, 64)
    assert [a['evidence']['text'] for a in actions] == ['N+', 'NH']
    assert all(set(a['values']) == {'formal_charge', 'explicit_h_count', 'no_implicit'} for a in actions)


def test_attribute_order_is_breadth_first_and_not_role_confidence():
    _, graph = raw_graph(['N', 'N'], {(0, 1)})
    evidence = {i: evidence_for(i, ['N', 'N+', 'NH']) for i in (0, 1)}
    evidence[0]['text_role_score'] = .01
    actions, total = attribute_actions(graph, evidence, 2)
    assert total == 4 and [a['atom_id'] for a in actions] == [0, 1]
    assert all(a['evidence']['rank'] == 1 for a in actions)


def test_bond_budget_uses_frozen_selected_set_and_type_margin():
    raw, graph = quaternary()
    actions = bond_actions(raw, graph, .5, Budget(0, 1, 1))
    assert len(actions) == 4
    assert actions[0]['kind'] == 'DELETE' and actions[0]['bond_id'] == graph['bonds'][0]['id']
    assert actions[0]['evidence']['presence_margin_to_frozen_threshold'] == .5


def test_retype_updates_bond_aromatic_but_not_atom_fields():
    _, graph = raw_graph(['C', 'C'], {(0, 1)})
    changed = apply_action(graph, {'kind': 'RETYPE', 'bond_id': 0, 'bond_type': 'AROMATIC'})
    assert changed['bonds'][0]['aromatic'] and changed['bonds'][0]['type'] == 'AROMATIC'
    assert changed['atoms'] == graph['atoms']


def test_duplicate_coordinates_stay_distinct():
    raw, _ = quaternary()
    for a in raw['atoms']:
        a['xy'] = [0., 0.]
    graph = from_prediction(raw, 0.)[0]
    result = decode(raw, graph, {}, 0., Budget(0, 4, 0))
    assert len(result['groups'][0]['representative_graph']['atoms']) == 5


def test_raw_graph_mismatch_fails_closed():
    raw, graph = quaternary()
    graph['atoms'][0]['formal_charge'] = 1
    with pytest.raises(ValueError, match='reproduce'):
        decode(raw, graph, {}, 0.)


@pytest.mark.parametrize('field,value', [('object_key', 'other:p000'), ('raw_greedy_text', 'O')])
def test_foreign_evidence_fails_closed(field, value):
    raw, graph = quaternary()
    evidence = evidence_for(0, ['N', 'N+'])
    evidence[field] = value
    with pytest.raises(ValueError):
        decode(raw, graph, {0: evidence}, 0.)


def joined_fixture():
    base = {'arm': 'atom_conditioned', 'object_key': 'A:fixture:p000', 'text': 'N', 'score': .99, 'crop_sha256': 'a' * 64}
    beam = {'arm': 'ctc_v0109_top8', **evidence_for(0, ['N', 'N+'])}
    return base, beam


def test_join_keeps_role_and_ctc_scores_distinct():
    base, beam = joined_fixture()
    result = join_evidence([base], [beam])['fixture:p000']
    assert result['text_role_score'] == .99
    assert result['candidates'][1]['score'] == -1.
    assert result['candidates'][1]['score_kind'] == 'ctc_log_probability'


@pytest.mark.parametrize('case', ['crop', 'duplicate', 'score_kind', 'nan', 'rank', 'error', 'source', 'score_order'])
def test_bad_beam_rejected(case):
    base, beam = joined_fixture()
    beams = [beam]
    if case == 'crop': beam['crop_sha256'] = 'b' * 64
    if case == 'duplicate': beams.append(deepcopy(beam))
    if case == 'score_kind': beam['candidates'][0]['score_kind'] = 'role_score'
    if case == 'nan': beam['candidates'][0]['score'] = float('nan')
    if case == 'rank': beam['candidates'][0]['rank'] = 7
    if case == 'error': beam['error'] = 'failed'
    if case == 'source': beam['arm'] = 'another_model'
    if case == 'score_order': beam['candidates'][1]['score'] = 1.
    with pytest.raises(ValueError):
        join_evidence([base], beams)


def test_missing_beam_explicit_and_unselected_rows_ignored():
    base, beam = joined_fixture()
    beam['object_key'] = 'unselected:p000'
    result = join_evidence([base], [beam])['fixture:p000']
    assert result['error'] == 'missing_beam_record' and result['candidates'] == []


def test_selected_reader_does_not_decode_unselected_payload(tmp_path):
    path = tmp_path / 'beams.jsonl'
    _, beam = joined_fixture()
    path.write_text('{"object_key":"unselected:p000", invalid payload}\n' + json.dumps(beam) + '\n')
    assert list(read_selected_beams(path, {'fixture:p000'})) == [beam]


def test_candidate_coverage_is_not_operational_accuracy():
    raw, graph = quaternary()
    result = decode(raw, graph, {0: evidence_for(0, ['N', 'N+'])}, 0.)
    correct = next(g for g in result['groups'] if '[N+]' in g['core'])
    row = score_candidates(result, {'core': correct['core'], 'stereo': correct['stereo']})
    assert row['candidate_contains_correct_core'] and row['candidate_contains_wrong_core']
    assert not row['operational_core_exact'] and row['automatic_replacements'] == 0
    assert row['selected_candidate'] is None


def test_single_wrong_legal_candidate_is_counted():
    raw, graph = quaternary()
    result = decode(raw, graph, {}, 0., Budget(0, 4, 0))
    row = score_candidates(result, {'core': 'C[N+](C)(C)C', 'stereo': 'C[N+](C)(C)C'})
    assert row['single_group_wrong_core'] and row['candidate_only_wrong_core']


def test_summaries_require_full_denominator():
    with pytest.raises(ValueError):
        summarize([])


def test_all_common_edges_scored_including_raw_equal_normalized_different():
    # A partially aromatic cyclohexadiene sanitizes its raw aromatic connecting
    # bond to single; truth benzene keeps that same raw aromatic edge aromatic.
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)}
    _, truth = raw_graph([None] * 6, edges, {p: 'AROMATIC' for p in edges})
    types = {p: 'SINGLE' for p in edges}
    types.update({(0, 1): 'DOUBLE', (2, 3): 'DOUBLE', (3, 4): 'AROMATIC'})
    _, pred = raw_graph([None] * 6, edges, types)
    result = edge_agreement(pred, truth)
    assert result['common_edges'] == 6
    assert result.get('raw_equal_normalized_different', 0) == 1


def test_equivalent_aromatic_kekule_encodings_have_same_compiled_identity():
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)}
    _, aromatic = raw_graph([None] * 6, edges, {p: 'AROMATIC' for p in edges})
    types = {p: ('DOUBLE' if p in {(0, 1), (2, 3), (4, 5)} else 'SINGLE') for p in edges}
    _, kekule = raw_graph([None] * 6, edges, types)
    assert compiled(aromatic)['stereo'] == compiled(kekule)['stereo']


def test_generator_has_no_truth_or_score_selection_parameters():
    assert set(inspect.signature(decode).parameters) == {'raw', 'graph', 'evidence', 'threshold', 'budget'}
    import chemvector.experiments.id_graph_candidates.decode as module
    source = inspect.getsource(module)
    assert 'geometry_matches' not in source and 'compile_score' not in source and 'TRUTH' not in source


def test_existing_output_rejected(tmp_path):
    from chemvector.experiments.id_graph_candidates.protocol import freeze

    with pytest.raises(FileExistsError):
        freeze(tmp_path)
